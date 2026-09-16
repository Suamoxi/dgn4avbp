from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Subset

from dgn4avbp.data import (
    AVBPHDF5FixedMeshDataset,
    HITReferenceScales,
    HierarchyAttachedDataset,
    PreprocessedFixedMeshDataset,
    dimensionalize_positions,
    dimensionalize_state,
    load_data_config,
    load_hierarchy_artifact,
    load_hierarchy_manifest,
    load_preprocessing_manifest,
    load_reference_config,
    load_split_manifest,
    split_indices_from_manifest,
    standardizer_from_manifest,
    validate_hierarchy_manifest,
    validate_preprocessing_manifest,
    validate_split_manifest,
)
from dgn4avbp.dgn_model import DiffusionGraphNet
from dgn4avbp.diffusion_loss import CanonicalHybridLoss
from dgn4avbp.diffusion_policy import deterministic_validation_corruption_batch
from dgn4avbp.diffusion_process import DiffusionProcess
from dgn4avbp.step_sampler import ImportanceStepSampler


@dataclass
class HITDataBundle:
    raw_dataset: AVBPHDF5FixedMeshDataset
    processed_dataset: PreprocessedFixedMeshDataset
    hierarchy_dataset: HierarchyAttachedDataset
    train_dataset: Subset
    val_dataset: Subset
    test_dataset: Subset
    split_indices: dict[str, list[int]]
    refs: HITReferenceScales
    standardizer: Any
    hierarchy: dict
    split_manifest: dict
    preprocessing_manifest: dict


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in YAML config '{path}'.")
    return value


def build_hit_data_bundle(
    *,
    data_config: str | Path,
    split_manifest: str | Path,
    reference_config: str | Path,
    preprocessing_manifest: str | Path,
    hierarchy_artifact: str | Path,
    hierarchy_manifest: str | Path,
) -> HITDataBundle:
    data_cfg = load_data_config(data_config)
    split = load_split_manifest(split_manifest)
    reference_cfg = load_reference_config(reference_config)
    preprocessing = load_preprocessing_manifest(preprocessing_manifest)
    hierarchy = load_hierarchy_artifact(hierarchy_artifact)
    hierarchy_meta = load_hierarchy_manifest(hierarchy_manifest)

    raw_dataset = AVBPHDF5FixedMeshDataset.from_config(data_cfg)
    validate_split_manifest(split, raw_dataset.files)
    validate_preprocessing_manifest(preprocessing, split)
    validate_hierarchy_manifest(hierarchy_meta, hierarchy)

    refs = HITReferenceScales.from_config(reference_cfg)
    standardizer = standardizer_from_manifest(preprocessing)
    processed = PreprocessedFixedMeshDataset(raw_dataset, refs, standardizer)
    hierarchy_dataset = HierarchyAttachedDataset(processed, hierarchy)
    indices = split_indices_from_manifest(split, raw_dataset.files)

    return HITDataBundle(
        raw_dataset=raw_dataset,
        processed_dataset=processed,
        hierarchy_dataset=hierarchy_dataset,
        train_dataset=Subset(hierarchy_dataset, indices["train"]),
        val_dataset=Subset(hierarchy_dataset, indices["val"]),
        test_dataset=Subset(hierarchy_dataset, indices["test"]),
        split_indices=indices,
        refs=refs,
        standardizer=standardizer,
        hierarchy=hierarchy,
        split_manifest=split,
        preprocessing_manifest=preprocessing,
    )


def build_diffusion_process(config: dict) -> DiffusionProcess:
    return DiffusionProcess(
        num_steps=int(config["num_steps"]),
        schedule_type=str(config["schedule_type"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
        max_beta=float(config["max_beta"]),
    )


def build_hit_model(
    model_config: dict,
    diffusion_process: DiffusionProcess,
    *,
    device: torch.device,
) -> DiffusionGraphNet:
    return DiffusionGraphNet(
        diffusion_process=diffusion_process,
        learnable_variance=bool(model_config["learnable_variance"]),
        arch=model_config["arch"],
        device=device,
    )


def build_training_sampler(policy_config: dict, num_steps: int) -> ImportanceStepSampler:
    training = policy_config["training"]
    if training["sampler"] != "loss_second_moment":
        raise ValueError("D12 canonical HIT training requires loss_second_moment sampling.")
    return ImportanceStepSampler(
        num_diffusion_steps=num_steps,
        min_history_length=int(training["history_per_term"]),
        uniform_prob=float(training["uniform_prob"]),
    )


def build_hybrid_loss(diffusion_config: dict) -> CanonicalHybridLoss:
    return CanonicalHybridLoss(lambda_vlb=float(diffusion_config["lambda_vlb"]))


def sample_ids_from_batch(graph) -> list[str]:
    batch_size = int(graph.batch.max().item()) + 1
    value = getattr(graph, "sample_id", None)
    if isinstance(value, str):
        sample_ids = [value]
    elif isinstance(value, (list, tuple)):
        sample_ids = [str(item) for item in value]
    else:
        raise ValueError("Batched graph does not expose sample_id values required by D8 validation.")
    if len(sample_ids) != batch_size:
        raise ValueError(
            f"Expected {batch_size} sample ids for validation batch, got {len(sample_ids)}."
        )
    return sample_ids


def prepare_training_batch(
    graph,
    *,
    diffusion_process: DiffusionProcess,
    step_sampler: ImportanceStepSampler,
    device: torch.device,
):
    graph = graph.to(device)
    batch_size = int(graph.batch.max().item()) + 1
    graph.r, importance_weight = step_sampler(batch_size, device)
    graph.field_start = graph.target
    graph.field_r, graph.noise = diffusion_process(
        field_start=graph.field_start,
        r=graph.r,
        batch=graph.batch,
    )
    return graph, importance_weight


def prepare_validation_batch(
    graph,
    *,
    diffusion_process: DiffusionProcess,
    base_seed: int,
    device: torch.device,
):
    graph = graph.to(device)
    result = deterministic_validation_corruption_batch(
        diffusion_process,
        graph.target,
        graph.batch,
        sample_ids=sample_ids_from_batch(graph),
        base_seed=int(base_seed),
    )
    graph.field_start = graph.target
    graph.field_r = result["field_r"]
    graph.noise = result["noise"]
    graph.r = result["r"]
    return graph


def train_one_batch(
    *,
    model: DiffusionGraphNet,
    graph,
    diffusion_process: DiffusionProcess,
    step_sampler: ImportanceStepSampler,
    criterion: CanonicalHybridLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    graph, importance_weight = prepare_training_batch(
        graph,
        diffusion_process=diffusion_process,
        step_sampler=step_sampler,
        device=device,
    )
    per_graph_loss = criterion(model, graph)
    step_sampler.update(graph.r, per_graph_loss.detach())
    weighted_loss = (per_graph_loss * importance_weight).mean()
    weighted_loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(grad_clip_norm) if grad_clip_norm is not None else float("inf"),
    )
    optimizer.step()

    return {
        "weighted_loss": float(weighted_loss.detach().cpu().item()),
        "unweighted_loss": float(per_graph_loss.detach().mean().cpu().item()),
        "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
    }


@torch.no_grad()
def validate_one_batch(
    *,
    model: DiffusionGraphNet,
    graph,
    diffusion_process: DiffusionProcess,
    criterion: CanonicalHybridLoss,
    validation_base_seed: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    graph = prepare_validation_batch(
        graph,
        diffusion_process=diffusion_process,
        base_seed=validation_base_seed,
        device=device,
    )
    per_graph_loss = criterion(model, graph)
    return {
        "loss": float(per_graph_loss.mean().cpu().item()),
        "num_graphs": int(per_graph_loss.numel()),
    }


def inverse_generated_state(
    state_standardized: torch.Tensor,
    *,
    standardizer,
    refs: HITReferenceScales,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_nondimensional = standardizer.inverse(state_standardized)
    state_dimensional = dimensionalize_state(state_nondimensional, refs)
    return state_nondimensional, state_dimensional


def dimensional_positions(bundle: HITDataBundle) -> torch.Tensor:
    return dimensionalize_positions(bundle.processed_dataset.pos, bundle.refs)
