#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from copy import deepcopy
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
from dgn4avbp.diffusion_policy import (
    deterministic_validation_corruption,
    deterministic_validation_corruption_batch,
)
from dgn4avbp.diffusion_process import DiffusionProcess
from dgn4avbp.loader import Collater


NUMERICAL_NRMSE_THRESHOLD = 1.0e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate D9 hierarchy-aware multi-sample batching.")
    parser.add_argument("--data-config", default="configs/data/avbp_hdf5_fixed_mesh_local.yaml")
    parser.add_argument("--split-manifest", default="artifacts/d2_split_manifest.json")
    parser.add_argument("--reference-config", default="configs/data/hit_reference_scales.yaml")
    parser.add_argument("--preprocessing-manifest", default="artifacts/d3_preprocessing_manifest.json")
    parser.add_argument("--hierarchy-artifact", default="artifacts/d5_hierarchy.pt")
    parser.add_argument("--hierarchy-manifest", default="artifacts/d5_hierarchy_manifest.json")
    parser.add_argument("--model-config", default="configs/model/dgn_hit_baseline.yaml")
    parser.add_argument("--diffusion-config", default="configs/diffusion/improved_ddpm_hit.yaml")
    parser.add_argument("--policy-config", default="configs/diffusion/hit_timestep_policy.yaml")
    parser.add_argument("--manifest", default="artifacts/d9_batching_manifest.json")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in YAML config '{path}'.")
    return value


def _prepare_single_validation_graph(
    graph,
    diffusion_process: DiffusionProcess,
    base_seed: int,
):
    result = deterministic_validation_corruption(
        diffusion_process,
        graph.target,
        graph.batch,
        sample_id=graph.sample_id,
        base_seed=base_seed,
    )
    graph.field_start = graph.target
    graph.field_r = result["field_r"]
    graph.noise = result["noise"]
    graph.r = result["r"]
    return graph


def _difference_stats(observed: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    observed = observed.float()
    reference = reference.float()
    delta = observed - reference
    abs_delta = delta.abs()
    rmse = torch.sqrt(torch.mean(delta.square()))
    reference_rms = torch.sqrt(torch.mean(reference.square()))
    denominator = torch.clamp(reference_rms, min=torch.finfo(reference.dtype).eps)
    return {
        "max_abs": float(abs_delta.max().item()),
        "mean_abs": float(abs_delta.mean().item()),
        "rmse": float(rmse.item()),
        "reference_rms": float(reference_rms.item()),
        "nrmse": float((rmse / denominator).item()),
    }


def main() -> None:
    args = parse_args()

    data_cfg = load_data_config(args.data_config)
    split_manifest = load_split_manifest(args.split_manifest)
    reference_cfg = load_reference_config(args.reference_config)
    preprocessing_manifest = load_preprocessing_manifest(args.preprocessing_manifest)
    hierarchy = load_hierarchy_artifact(args.hierarchy_artifact)
    hierarchy_manifest = load_hierarchy_manifest(args.hierarchy_manifest)
    model_cfg = _load_yaml(args.model_config)
    diffusion_cfg = _load_yaml(args.diffusion_config)
    policy_cfg = _load_yaml(args.policy_config)

    dataset = AVBPHDF5FixedMeshDataset.from_config(data_cfg)
    validate_split_manifest(split_manifest, dataset.files)
    validate_preprocessing_manifest(preprocessing_manifest, split_manifest)
    validate_hierarchy_manifest(hierarchy_manifest, hierarchy)

    refs = HITReferenceScales.from_config(reference_cfg)
    standardizer = standardizer_from_manifest(preprocessing_manifest)
    processed_dataset = PreprocessedFixedMeshDataset(dataset, refs, standardizer)

    split_indices = split_indices_from_manifest(split_manifest, dataset.files)
    validation_indices = split_indices["val"][:2]
    if len(validation_indices) != 2:
        raise AssertionError("D9 real validation requires at least two frozen validation samples.")

    graphs = [
        attach_hierarchy_to_graph(processed_dataset[index], hierarchy)
        for index in validation_indices
    ]
    sample_ids = [graph.sample_id for graph in graphs]

    diffusion_process = DiffusionProcess(
        num_steps=int(diffusion_cfg["num_steps"]),
        schedule_type=diffusion_cfg["schedule_type"],
        beta_start=float(diffusion_cfg["beta_start"]),
        beta_end=float(diffusion_cfg["beta_end"]),
        max_beta=float(diffusion_cfg["max_beta"]),
    )
    base_seed = int(policy_cfg["validation"]["base_seed"])

    singles_for_corruption = [
        _prepare_single_validation_graph(deepcopy(graph), diffusion_process, base_seed)
        for graph in graphs
    ]

    batched = Collater().collate([deepcopy(graph) for graph in graphs])
    batched_corruption = deterministic_validation_corruption_batch(
        diffusion_process,
        batched.target,
        batched.batch,
        sample_ids=sample_ids,
        base_seed=base_seed,
    )
    batched.field_start = batched.target
    batched.field_r = batched_corruption["field_r"]
    batched.noise = batched_corruption["noise"]
    batched.r = batched_corruption["r"]

    fine_num_nodes = int(hierarchy["levels"][0]["pos"].shape[0])
    fine_num_cells = int(dataset.cells.shape[0])
    for graph_id, single in enumerate(singles_for_corruption):
        node_slice = slice(graph_id * fine_num_nodes, (graph_id + 1) * fine_num_nodes)
        torch.testing.assert_close(
            batched.field_r[node_slice],
            single.field_r,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            batched.noise[node_slice],
            single.noise,
            rtol=0.0,
            atol=0.0,
        )
        if int(batched.r[graph_id].item()) != int(single.r[0].item()):
            raise AssertionError("D8 timestep changed when the sample was placed in a batch.")

    level_checks: list[dict] = []
    for level_number, level in enumerate(hierarchy["levels"], start=1):
        pos = batched.pos if level_number == 1 else getattr(batched, f"pos_{level_number}")
        edge_index = (
            batched.edge_index
            if level_number == 1
            else getattr(batched, f"edge_index_{level_number}")
        )
        batch_index = (
            batched.batch
            if level_number == 1
            else getattr(batched, f"batch_{level_number}")
        )

        nodes_per_graph = int(level["pos"].shape[0])
        edges_per_graph = int(level["edge_index"].shape[1])
        expected_nodes = 2 * nodes_per_graph
        expected_edges = 2 * edges_per_graph
        if int(pos.shape[0]) != expected_nodes:
            raise AssertionError(
                f"L{level_number}: expected {expected_nodes} batched nodes, got {pos.shape[0]}."
            )
        if int(edge_index.shape[1]) != expected_edges:
            raise AssertionError(
                f"L{level_number}: expected {expected_edges} batched edges, got {edge_index.shape[1]}."
            )

        counts = torch.bincount(batch_index, minlength=2)
        expected_counts = torch.tensor(
            [nodes_per_graph, nodes_per_graph],
            dtype=counts.dtype,
            device=counts.device,
        )
        if not torch.equal(counts, expected_counts):
            raise AssertionError(
                f"L{level_number}: invalid graph-id counts {counts.tolist()}, "
                f"expected {expected_counts.tolist()}."
            )
        if edge_index.numel() and not torch.equal(
            batch_index[edge_index[0]], batch_index[edge_index[1]]
        ):
            raise AssertionError(f"L{level_number}: found an edge crossing physical graphs.")

        level_checks.append(
            {
                "level": level_number,
                "nodes_per_graph": nodes_per_graph,
                "batched_nodes": int(pos.shape[0]),
                "directed_edges_per_graph": edges_per_graph,
                "batched_directed_edges": int(edge_index.shape[1]),
                "graph_id_counts": counts.cpu().tolist(),
                "no_cross_graph_edges": True,
            }
        )

    for transition in hierarchy["transitions"]:
        hr = int(transition["from_level"])
        lr = int(transition["to_level"])
        parent = getattr(batched, f"idx{hr}_to_idx{lr}")
        batch_hr = batched.batch if hr == 1 else getattr(batched, f"batch_{hr}")
        batch_lr = getattr(batched, f"batch_{lr}")
        if not torch.equal(batch_hr, batch_lr[parent]):
            raise AssertionError(f"Parent map {hr}->{lr} crosses physical graphs.")

    if batched.cells.shape[0] != 2 * fine_num_cells:
        raise AssertionError("Batched cell table has an unexpected number of cells.")
    first_cells = batched.cells[:fine_num_cells]
    second_cells = batched.cells[fine_num_cells:]
    if int(first_cells.max().item()) >= fine_num_nodes:
        raise AssertionError("First cell block escaped graph 0 node range.")
    if int(second_cells.min().item()) < fine_num_nodes:
        raise AssertionError("Second cell block was not offset into graph 1 node range.")
    if int(second_cells.max().item()) >= 2 * fine_num_nodes:
        raise AssertionError("Second cell block escaped graph 1 node range.")

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("D9 real-mesh validation requested CUDA but CUDA is unavailable.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    model = DiffusionGraphNet(
        diffusion_process=diffusion_process,
        learnable_variance=bool(model_cfg["learnable_variance"]),
        arch=model_cfg["arch"],
        device=device,
    ).eval()

    independent_outputs = []
    with torch.no_grad():
        for graph in singles_for_corruption:
            graph_device = deepcopy(graph).to(device)
            epsilon, variance = model(graph_device)
            independent_outputs.append((epsilon.cpu(), variance.cpu()))
            del graph_device, epsilon, variance
    if device.type == "cuda":
        torch.cuda.empty_cache()

    batched = batched.to(device)
    fine_edge_index_before = batched.edge_index.clone()
    fine_batch_before = batched.batch.clone()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        epsilon_batch, variance_batch = model(batched)

    if not torch.equal(batched.edge_index, fine_edge_index_before):
        raise AssertionError("D9 batched DGN forward did not restore fine edge_index.")
    if not torch.equal(batched.batch, fine_batch_before):
        raise AssertionError("D9 batched DGN forward did not restore fine batch vector.")

    epsilon_batch_cpu = epsilon_batch.cpu()
    variance_batch_cpu = variance_batch.cpu()
    del epsilon_batch, variance_batch

    with torch.no_grad():
        epsilon_repeat, variance_repeat = model(batched)
    epsilon_repeat_cpu = epsilon_repeat.cpu()
    variance_repeat_cpu = variance_repeat.cpu()
    del epsilon_repeat, variance_repeat

    epsilon_repeat_stats = _difference_stats(epsilon_repeat_cpu, epsilon_batch_cpu)
    variance_repeat_stats = _difference_stats(variance_repeat_cpu, variance_batch_cpu)

    per_graph_numerics: list[dict] = []
    epsilon_max_nrmse = 0.0
    variance_max_nrmse = 0.0
    for graph_id, (epsilon_single, variance_single) in enumerate(independent_outputs):
        node_slice = slice(graph_id * fine_num_nodes, (graph_id + 1) * fine_num_nodes)
        epsilon_piece = epsilon_batch_cpu[node_slice]
        variance_piece = variance_batch_cpu[node_slice]
        epsilon_stats = _difference_stats(epsilon_piece, epsilon_single)
        variance_stats = _difference_stats(variance_piece, variance_single)
        epsilon_max_nrmse = max(epsilon_max_nrmse, epsilon_stats["nrmse"])
        variance_max_nrmse = max(variance_max_nrmse, variance_stats["nrmse"])
        per_graph_numerics.append(
            {
                "graph_id": graph_id,
                "sample_id": sample_ids[graph_id],
                "epsilon": epsilon_stats,
                "variance": variance_stats,
            }
        )

    numerical_equivalence_passed = (
        epsilon_max_nrmse <= NUMERICAL_NRMSE_THRESHOLD
        and variance_max_nrmse <= NUMERICAL_NRMSE_THRESHOLD
    )

    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )

    manifest = {
        "version": 2,
        "contract": "hierarchy_aware_multi_sample_batching",
        "dataset_fingerprint_sha256": split_manifest["ordered_file_fingerprint_sha256"],
        "hierarchy_fingerprint_sha256": hierarchy["hierarchy_fingerprint_sha256"],
        "computational_batch_size_validated": 2,
        "sample_ids": sample_ids,
        "available_hierarchy_levels": len(hierarchy["levels"]),
        "model_selected_hierarchy_levels": int(model_cfg["hierarchy"]["num_levels"]),
        "collation": {
            "backend": "torch_geometric.Batch.from_data_list",
            "edge_index_level_compensation": "cumulative_level_nodes_minus_cumulative_fine_nodes",
            "parent_map_offset": "cumulative_destination_level_nodes",
            "batch_level_offset": "native_pyg_batch_key_increment",
            "cell_connectivity_offset": "cumulative_fine_nodes",
        },
        "level_checks": level_checks,
        "parent_maps_preserve_graph_identity": True,
        "cell_connectivity_preserves_graph_identity": True,
        "deterministic_validation_batching_invariant": True,
        "forward_validation": {
            "device": str(device),
            "epsilon_shape": list(epsilon_batch_cpu.shape),
            "variance_shape": list(variance_batch_cpu.shape),
            "structural_batching_valid": True,
            "numerical_comparison": {
                "metric": "NRMSE = RMSE(batched-independent) / RMS(independent)",
                "threshold": NUMERICAL_NRMSE_THRESHOLD,
                "passed": numerical_equivalence_passed,
                "max_epsilon_nrmse": epsilon_max_nrmse,
                "max_variance_nrmse": variance_max_nrmse,
                "per_graph": per_graph_numerics,
                "same_shape_repeat_epsilon": epsilon_repeat_stats,
                "same_shape_repeat_variance": variance_repeat_stats,
                "note": (
                    "GPU scatter/GEMM reductions may differ at roundoff level when the same "
                    "disconnected graphs are evaluated independently versus as one larger batch."
                ),
            },
            "fine_edge_index_restored_after_pool_unpool": True,
            "fine_batch_restored_after_pool_unpool": True,
            "peak_cuda_memory_bytes": peak_memory_bytes,
        },
        "deferred": [
            "DDP synchronization (D10)",
            "checkpoint/reproducibility state (D11)",
            "production training/generation integration (D12)",
        ],
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("D9 hierarchy-aware batching structural checks passed")
    print(f"samples: {sample_ids}")
    print(f"available hierarchy levels: {len(hierarchy['levels'])}")
    print(f"selected model levels: {model_cfg['hierarchy']['num_levels']}")
    print(f"batched fine nodes: {batched.pos.shape[0]}")
    print(f"epsilon shape: {tuple(epsilon_batch_cpu.shape)}")
    print(f"variance shape: {tuple(variance_batch_cpu.shape)}")
    print(f"max epsilon NRMSE: {epsilon_max_nrmse:.3e}")
    print(f"max variance NRMSE: {variance_max_nrmse:.3e}")
    print(f"numerical NRMSE threshold: {NUMERICAL_NRMSE_THRESHOLD:.1e}")
    print(f"numerical equivalence passed: {numerical_equivalence_passed}")
    print(f"same-shape repeat epsilon NRMSE: {epsilon_repeat_stats['nrmse']:.3e}")
    print(f"same-shape repeat variance NRMSE: {variance_repeat_stats['nrmse']:.3e}")
    if peak_memory_bytes is not None:
        print(f"peak CUDA memory allocated: {peak_memory_bytes / (1024**3):.3f} GiB")
    print(f"manifest: {manifest_path.resolve()}")

    if not numerical_equivalence_passed:
        raise AssertionError(
            "D9 structural batching is valid, but batched-vs-independent numerical drift "
            f"exceeded NRMSE threshold {NUMERICAL_NRMSE_THRESHOLD:.1e}: "
            f"epsilon={epsilon_max_nrmse:.3e}, variance={variance_max_nrmse:.3e}. "
            f"Diagnostic manifest was written to {manifest_path}."
        )


if __name__ == "__main__":
    main()
