#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from dgn4avbp.data import (
    AVBPHDF5FixedMeshDataset,
    HITReferenceScales,
    PreprocessedFixedMeshDataset,
    attach_hierarchy_to_graph,
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
from dgn4avbp.diffusion_process import DiffusionProcess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the D6 physical-space DGN backbone.")
    parser.add_argument(
        "--data-config",
        default="configs/data/avbp_hdf5_fixed_mesh_local.yaml",
    )
    parser.add_argument(
        "--split-manifest",
        default="artifacts/d2_split_manifest.json",
    )
    parser.add_argument(
        "--reference-config",
        default="configs/data/hit_reference_scales.yaml",
    )
    parser.add_argument(
        "--preprocessing-manifest",
        default="artifacts/d3_preprocessing_manifest.json",
    )
    parser.add_argument(
        "--hierarchy-artifact",
        default="artifacts/d5_hierarchy.pt",
    )
    parser.add_argument(
        "--hierarchy-manifest",
        default="artifacts/d5_hierarchy_manifest.json",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/dgn_hit_baseline.yaml",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/d6_backbone_manifest.json",
    )
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in YAML config '{path}'.")
    return value


def _validate_model_config(cfg: dict, available_levels: int) -> tuple[dict, int, bool]:
    arch = cfg.get("arch")
    hierarchy_cfg = cfg.get("hierarchy")
    if not isinstance(arch, dict) or not isinstance(hierarchy_cfg, dict):
        raise ValueError("D6 model config requires 'arch' and 'hierarchy' mappings.")

    learnable_variance = bool(cfg.get("learnable_variance"))
    num_levels = int(hierarchy_cfg["num_levels"])

    if arch.get("in_node_features") != 5:
        raise ValueError("D6 HIT baseline must use the five conservative state channels.")
    if arch.get("cond_node_features", 0) != 0:
        raise ValueError("D6 HIT baseline is unconditional and must not use node conditioning.")
    if arch.get("cond_edge_features") != 3:
        raise ValueError("D6 HIT baseline must encode the three relative-position components.")
    if arch.get("dim") != 3 or arch.get("scalar_rel_pos") is not False:
        raise ValueError("D6 HIT baseline requires 3-D vector relative positions.")
    if not learnable_variance:
        raise ValueError("D6 canonical baseline requires learned reverse variance.")
    if len(arch.get("depths", [])) != num_levels:
        raise ValueError("Number of DGN depth entries must equal selected hierarchy levels.")
    if num_levels > available_levels:
        raise ValueError(
            f"D6 requests {num_levels} hierarchy levels but D5 provides only {available_levels}."
        )

    output_contract = cfg.get("output_contract", {})
    if output_contract.get("epsilon_channels") != 5:
        raise ValueError("D6 output contract must contain five epsilon channels.")
    if output_contract.get("variance_channels") != 5:
        raise ValueError("D6 output contract must contain five variance channels.")
    if output_contract.get("total_channels") != 10:
        raise ValueError("D6 learned-variance output contract must total ten channels.")

    return arch, num_levels, learnable_variance


def main() -> None:
    args = parse_args()

    data_cfg = load_data_config(args.data_config)
    split_manifest = load_split_manifest(args.split_manifest)
    reference_cfg = load_reference_config(args.reference_config)
    preprocessing_manifest = load_preprocessing_manifest(args.preprocessing_manifest)
    hierarchy = load_hierarchy_artifact(args.hierarchy_artifact)
    hierarchy_manifest = load_hierarchy_manifest(args.hierarchy_manifest)
    model_cfg = _load_yaml(args.model_config)

    dataset = AVBPHDF5FixedMeshDataset.from_config(data_cfg)
    validate_split_manifest(split_manifest, dataset.files)
    validate_preprocessing_manifest(preprocessing_manifest, split_manifest)
    validate_hierarchy_manifest(hierarchy_manifest, hierarchy)

    if hierarchy_manifest["dataset_fingerprint_sha256"] != split_manifest[
        "ordered_file_fingerprint_sha256"
    ]:
        raise ValueError("D5 hierarchy dataset fingerprint does not match frozen D2 split.")
    if hierarchy_manifest["d3_fit_file_fingerprint_sha256"] != preprocessing_manifest[
        "fit_file_fingerprint_sha256"
    ]:
        raise ValueError("D5 hierarchy preprocessing fingerprint does not match frozen D3 scaler.")

    arch, num_levels, learnable_variance = _validate_model_config(
        model_cfg,
        available_levels=len(hierarchy["levels"]),
    )

    refs = HITReferenceScales.from_config(reference_cfg)
    standardizer = standardizer_from_manifest(preprocessing_manifest)
    processed_dataset = PreprocessedFixedMeshDataset(dataset, refs, standardizer)

    split_indices = split_indices_from_manifest(split_manifest, dataset.files)
    first_train_index = split_indices["train"][0]
    graph = processed_dataset[first_train_index]
    graph = attach_hierarchy_to_graph(graph, hierarchy)

    # D6 validates only the network interface. Use a clean standardized state as
    # a finite placeholder x_t; D7 owns the actual forward diffusion equation.
    graph.field_r = graph.target.clone()
    graph.r = torch.tensor([0.0], dtype=graph.field_r.dtype)

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("D6 real-mesh validation requested CUDA but CUDA is unavailable.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DiffusionProcess is required by the current model constructor, but D6 does
    # not validate its schedule/equations. D7 will own that contract.
    diffusion_process = DiffusionProcess(num_steps=1000, schedule_type="linear")
    model = DiffusionGraphNet(
        diffusion_process=diffusion_process,
        learnable_variance=learnable_variance,
        arch=arch,
        device=device,
    ).eval()

    graph = graph.to(device)
    fine_edge_index = graph.edge_index.clone()
    fine_batch = graph.batch.clone()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        output = model(graph)

    if not isinstance(output, tuple) or len(output) != 2:
        raise AssertionError("Learned-variance DGN must return (epsilon, variance_value).")
    epsilon, variance_value = output

    expected_shape = (int(dataset.pos.shape[0]), 5)
    if tuple(epsilon.shape) != expected_shape:
        raise AssertionError(f"Unexpected epsilon shape {tuple(epsilon.shape)}, expected {expected_shape}.")
    if tuple(variance_value.shape) != expected_shape:
        raise AssertionError(
            f"Unexpected variance shape {tuple(variance_value.shape)}, expected {expected_shape}."
        )
    if not torch.isfinite(epsilon).all() or not torch.isfinite(variance_value).all():
        raise AssertionError("D6 real-mesh forward produced non-finite outputs.")
    if model.num_fields != 5:
        raise AssertionError(f"Expected model.num_fields=5, got {model.num_fields}.")
    if not torch.equal(graph.edge_index, fine_edge_index):
        raise AssertionError("MultiScaleGnn did not restore the fine edge_index after the up path.")
    if not torch.equal(graph.batch, fine_batch):
        raise AssertionError("MultiScaleGnn did not restore the fine batch vector after the up path.")

    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )

    selected_levels = hierarchy_manifest["levels"][:num_levels]
    excluded_levels = hierarchy_manifest["levels"][num_levels:]
    bottleneck = selected_levels[-1]

    manifest = {
        "version": 1,
        "model_name": model_cfg.get("name"),
        "backbone": "DiffusionGraphNet/MultiScaleGnn",
        "backbone_source": "existing_dgn4avbp_preserved",
        "learnable_variance": learnable_variance,
        "arch": arch,
        "dataset_fingerprint_sha256": split_manifest["ordered_file_fingerprint_sha256"],
        "d3_fit_file_fingerprint_sha256": preprocessing_manifest[
            "fit_file_fingerprint_sha256"
        ],
        "d5_hierarchy_fingerprint_sha256": hierarchy[
            "hierarchy_fingerprint_sha256"
        ],
        "hierarchy": {
            "available_levels": len(hierarchy["levels"]),
            "selected_levels": num_levels,
            "selection": "prefix",
            "selected_level_summaries": selected_levels,
            "excluded_available_level_summaries": excluded_levels,
            "bottleneck_level": int(bottleneck["level"]),
            "bottleneck_nodes": int(bottleneck["num_nodes"]),
            "bottleneck_directed_edges": int(bottleneck["num_directed_edges"]),
        },
        "output_contract": {
            "epsilon_channels": 5,
            "variance_channels": 5,
            "total_channels": 10,
            "epsilon_shape": list(epsilon.shape),
            "variance_shape": list(variance_value.shape),
        },
        "forward_validation": {
            "device": str(device),
            "finite_output": True,
            "fine_edge_index_restored_after_pool_unpool": True,
            "fine_batch_restored_after_pool_unpool": True,
            "peak_cuda_memory_bytes": peak_memory_bytes,
        },
        "unit_test_contracts": {
            "finite_nonzero_gradients": True,
            "consistent_node_relabeling_equivariance": True,
            "pool_unpool_shape_restore": True,
        },
        "parameter_counts": {
            "total": int(model.num_params),
            "learnable": int(model.num_learnable_params),
        },
        "diffusion_math_status": "not_validated_in_D6_deferred_to_D7",
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("D6 backbone validation passed")
    print(f"model: {manifest['model_name']}")
    print(f"device: {device}")
    print(f"available hierarchy levels: {len(hierarchy['levels'])}")
    print(f"selected hierarchy levels: {num_levels}")
    print(
        "selected node counts: "
        f"{[level['num_nodes'] for level in selected_levels]}"
    )
    print(
        "selected directed edge counts: "
        f"{[level['num_directed_edges'] for level in selected_levels]}"
    )
    print(
        f"bottleneck: L{bottleneck['level']} N={bottleneck['num_nodes']} "
        f"E={bottleneck['num_directed_edges']}"
    )
    print(f"excluded available levels: {[level['level'] for level in excluded_levels]}")
    print(f"epsilon shape: {tuple(epsilon.shape)}")
    print(f"variance shape: {tuple(variance_value.shape)}")
    print(f"learnable parameters: {model.num_learnable_params}")
    if peak_memory_bytes is not None:
        print(f"peak CUDA memory allocated: {peak_memory_bytes / (1024**3):.3f} GiB")
    print(f"manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
