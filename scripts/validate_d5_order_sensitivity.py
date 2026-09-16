#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dgn4avbp.data import build_fixed_mesh_hierarchy, load_hierarchy_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Guillard hierarchy sensitivity to node renumbering."
    )
    parser.add_argument(
        "--hierarchy",
        default="artifacts/d5_hierarchy.pt",
        help="Frozen D5 hierarchy artifact.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/d5_order_sensitivity.json",
        help="JSON diagnostic output.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Deterministic random node-renumbering seeds.",
    )
    return parser.parse_args()


def _renumber_graph(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact same graph under a deterministic node permutation."""

    num_nodes = int(pos.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    new_to_old = torch.randperm(num_nodes, generator=generator)

    old_to_new = torch.empty_like(new_to_old)
    old_to_new[new_to_old] = torch.arange(num_nodes, dtype=torch.long)

    permuted_pos = pos[new_to_old].contiguous()
    permuted_edge_index = old_to_new[edge_index].contiguous()
    return permuted_pos, permuted_edge_index


def _hierarchy_summary(artifact: dict) -> dict:
    levels = []
    for level in artifact["levels"]:
        lengths = torch.linalg.vector_norm(level["edge_attr"].to(torch.float64), dim=1)
        levels.append(
            {
                "level": int(level["level"]),
                "num_nodes": int(level["pos"].shape[0]),
                "num_directed_edges": int(level["edge_index"].shape[1]),
                "mean_edge_length_nd": float(lengths.mean().item()),
                "max_edge_length_nd": float(lengths.max().item()),
            }
        )

    transitions = []
    for transition in artifact["transitions"]:
        parent_distance = torch.linalg.vector_norm(
            transition["e_hr_to_lr"].to(torch.float64), dim=1
        )
        transitions.append(
            {
                "from_level": int(transition["from_level"]),
                "to_level": int(transition["to_level"]),
                "mean_parent_distance_nd": float(parent_distance.mean().item()),
                "max_parent_distance_nd": float(parent_distance.max().item()),
            }
        )

    return {
        "num_levels": len(levels),
        "stop_reason": artifact["stop_reason"],
        "levels": levels,
        "transitions": transitions,
    }


def _relative_difference(value: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return abs(value - reference) / abs(reference)


def _compare(reference: dict, candidate: dict) -> dict:
    common_levels = min(reference["num_levels"], candidate["num_levels"])
    level_differences = []
    for index in range(common_levels):
        ref = reference["levels"][index]
        cur = candidate["levels"][index]
        level_differences.append(
            {
                "level": index + 1,
                "node_count_relative_difference": _relative_difference(
                    cur["num_nodes"], ref["num_nodes"]
                ),
                "edge_count_relative_difference": _relative_difference(
                    cur["num_directed_edges"], ref["num_directed_edges"]
                ),
                "mean_edge_length_relative_difference": _relative_difference(
                    cur["mean_edge_length_nd"], ref["mean_edge_length_nd"]
                ),
            }
        )

    common_transitions = min(
        len(reference["transitions"]), len(candidate["transitions"])
    )
    transition_differences = []
    for index in range(common_transitions):
        ref = reference["transitions"][index]
        cur = candidate["transitions"][index]
        transition_differences.append(
            {
                "from_level": index + 1,
                "to_level": index + 2,
                "mean_parent_distance_relative_difference": _relative_difference(
                    cur["mean_parent_distance_nd"], ref["mean_parent_distance_nd"]
                ),
                "max_parent_distance_relative_difference": _relative_difference(
                    cur["max_parent_distance_nd"], ref["max_parent_distance_nd"]
                ),
            }
        )

    def _maximum(items: list[dict], key: str) -> float:
        return max((float(item[key]) for item in items), default=0.0)

    return {
        "same_number_of_levels": candidate["num_levels"] == reference["num_levels"],
        "reference_num_levels": reference["num_levels"],
        "candidate_num_levels": candidate["num_levels"],
        "level_differences": level_differences,
        "transition_differences": transition_differences,
        "max_node_count_relative_difference": _maximum(
            level_differences, "node_count_relative_difference"
        ),
        "max_edge_count_relative_difference": _maximum(
            level_differences, "edge_count_relative_difference"
        ),
        "max_mean_edge_length_relative_difference": _maximum(
            level_differences, "mean_edge_length_relative_difference"
        ),
        "max_mean_parent_distance_relative_difference": _maximum(
            transition_differences, "mean_parent_distance_relative_difference"
        ),
    }


def _nodes(summary: dict) -> list[int]:
    return [level["num_nodes"] for level in summary["levels"]]


def _edges(summary: dict) -> list[int]:
    return [level["num_directed_edges"] for level in summary["levels"]]


def main() -> None:
    args = parse_args()
    reference_artifact = load_hierarchy_artifact(args.hierarchy)
    reference_summary = _hierarchy_summary(reference_artifact)

    base = reference_artifact["levels"][0]
    results = []

    print("D5.1 Guillard node-renumbering sensitivity")
    print("This is a diagnostic, not a new hierarchy-selection algorithm.")
    print(f"reference fingerprint: {reference_artifact['hierarchy_fingerprint_sha256']}")
    print(f"reference nodes: {_nodes(reference_summary)}")
    print(f"reference edges: {_edges(reference_summary)}")
    print()

    for seed in args.seeds:
        pos, edge_index = _renumber_graph(base["pos"], base["edge_index"], seed=seed)
        candidate_artifact = build_fixed_mesh_hierarchy(
            pos,
            edge_index,
            max_levels=None,
            max_indegree=reference_artifact["max_indegree"],
        )
        candidate_summary = _hierarchy_summary(candidate_artifact)
        comparison = _compare(reference_summary, candidate_summary)

        result = {
            "seed": seed,
            "hierarchy_fingerprint_sha256": candidate_artifact[
                "hierarchy_fingerprint_sha256"
            ],
            "summary": candidate_summary,
            "comparison_to_reference": comparison,
        }
        results.append(result)

        print(f"seed={seed}")
        print(f"  nodes: {_nodes(candidate_summary)}")
        print(f"  edges: {_edges(candidate_summary)}")
        print(f"  stop: {candidate_summary['stop_reason']}")
        print(
            "  max relative differences: "
            f"nodes={comparison['max_node_count_relative_difference']:.3f}, "
            f"edges={comparison['max_edge_count_relative_difference']:.3f}, "
            f"mean_edge_length={comparison['max_mean_edge_length_relative_difference']:.3f}, "
            f"mean_parent_distance={comparison['max_mean_parent_distance_relative_difference']:.3f}"
        )
        print()

    report = {
        "version": 1,
        "purpose": "diagnose_guillard_sensitivity_to_node_renumbering",
        "interpretation": (
            "Every candidate is the same physical graph with only node labels changed. "
            "Exact coarse-node identity is intentionally not required. The report measures "
            "changes in hierarchy size and geometry."
        ),
        "num_permutations": len(args.seeds),
        "seeds": args.seeds,
        "reference": {
            "hierarchy_fingerprint_sha256": reference_artifact[
                "hierarchy_fingerprint_sha256"
            ],
            "summary": reference_summary,
        },
        "permutations": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"report: {output.resolve()}")
    print("D5.1 diagnostic completed successfully")


if __name__ == "__main__":
    main()
