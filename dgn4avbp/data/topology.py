from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch_geometric.utils import coalesce, to_undirected

from .avbp_hdf5 import HEX_EDGES


# Legacy dgn4avbp/utils.py AVBP hexahedron edge list converted from 1-based
# local labels to 0-based local indices.
LEGACY_AVBP_HEX_EDGES = (
    (0, 1),
    (1, 5),
    (5, 4),
    (4, 0),
    (3, 2),
    (2, 6),
    (6, 7),
    (7, 3),
    (0, 3),
    (1, 2),
    (5, 6),
    (4, 7),
)


def canonical_local_edge_set(edges: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges}


def build_hex_edge_index(
    connectivity: torch.Tensor,
    num_nodes: int,
    local_edges: Sequence[tuple[int, int]] = HEX_EDGES,
) -> torch.Tensor:
    """Build a coalesced bidirectional PyG graph from hexahedral connectivity."""

    if connectivity.ndim != 2 or connectivity.shape[1] != 8:
        raise ValueError(
            f"Expected hexahedral connectivity [num_cells, 8], got {tuple(connectivity.shape)}."
        )
    if connectivity.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=connectivity.device)

    src = torch.cat([connectivity[:, a] for a, _ in local_edges])
    dst = torch.cat([connectivity[:, b] for _, b in local_edges])
    edge_index = torch.stack([src, dst], dim=0).long()
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return coalesce(edge_index, num_nodes=num_nodes, sort_by_row=False)


def _canonical_directed_edge_codes(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    codes = edge_index[0].to(torch.int64) * int(num_nodes) + edge_index[1].to(torch.int64)
    return torch.sort(codes).values


def directed_edge_sets_equal(a: torch.Tensor, b: torch.Tensor, num_nodes: int) -> bool:
    if a.shape != b.shape:
        return False
    return bool(
        torch.equal(
            _canonical_directed_edge_codes(a, num_nodes),
            _canonical_directed_edge_codes(b, num_nodes),
        )
    )


def degree_histogram(edge_index: torch.Tensor, num_nodes: int) -> dict[int, int]:
    degree = torch.bincount(edge_index[1], minlength=num_nodes)
    return dict(sorted(Counter(int(value) for value in degree.tolist()).items()))


def edge_geometry_metrics(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    axis_spacing: Sequence[float] | None = None,
) -> dict:
    """Compute generic edge metrics, plus optional Cartesian diagnostics.

    The generic metrics make no assumption about mesh regularity. When
    ``axis_spacing`` is supplied, HIT-specific Cartesian quantities are added for
    diagnostic use only.
    """

    pos64 = pos.to(torch.float64)
    delta = pos64[edge_index[1]] - pos64[edge_index[0]]
    lengths = torch.linalg.vector_norm(delta, dim=1)

    metrics = {
        "num_edges": int(edge_index.shape[1]),
        "min_edge_length": float(lengths.min().item()),
        "max_edge_length": float(lengths.max().item()),
        "mean_edge_length": float(lengths.mean().item()),
        "zero_length_edges": int((lengths == 0.0).sum().item()),
    }

    if axis_spacing is None:
        return metrics

    abs_delta = delta.abs()
    spacing = torch.tensor(axis_spacing, dtype=torch.float64, device=pos64.device)
    max_spacing = float(spacing.max().item())
    zero_tol = max(max_spacing * 1e-4, 1e-12)
    component_tol = torch.maximum(spacing * 1e-4, torch.full_like(spacing, 1e-12))

    near_zero = abs_delta <= zero_tol
    near_step = (abs_delta - spacing).abs() <= component_tol
    component_valid = near_zero | near_step
    changed_axes = (~near_zero).sum(dim=1)
    nearest_neighbor = component_valid.all(dim=1) & (changed_axes == 1)
    cross_box = (abs_delta > (1.5 * spacing)).any(dim=1)

    span = pos64.amax(dim=0) - pos64.amin(dim=0)
    opposite_face = (abs_delta >= (span - component_tol)).any(dim=1)

    metrics.update(
        {
            "nearest_neighbor_edges": int(nearest_neighbor.sum().item()),
            "nearest_neighbor_fraction": float(nearest_neighbor.double().mean().item()),
            "cross_box_edges": int(cross_box.sum().item()),
            "opposite_face_edges": int(opposite_face.sum().item()),
        }
    )
    return metrics


def validate_native_hex_topology(dataset) -> dict:
    """Validate the D4 core graph contract without assuming Cartesian geometry.

    D4 core ownership is connectivity -> graph. The validator therefore checks
    the native hexahedral connectivity, the graph generated from it, graph
    directionality, and relative geometry. It deliberately does *not* require a
    tensor-product grid, uniform spacing, a particular degree distribution, or a
    box-shaped domain.
    """

    pos = dataset.pos
    cells = dataset.cells
    edge_index = dataset.edge_index
    edge_attr = dataset.edge_attr

    if pos.ndim != 2 or pos.shape[0] == 0:
        raise ValueError(f"Expected non-empty node positions [N, dim], got {tuple(pos.shape)}.")
    if not torch.isfinite(pos).all():
        raise ValueError("Mesh positions contain non-finite values.")

    num_nodes = int(pos.shape[0])
    spatial_dim = int(pos.shape[1])

    if cells.ndim != 2 or cells.shape[1] != 8:
        raise ValueError(f"Expected hexahedral cells [num_cells, 8], got {tuple(cells.shape)}.")
    if cells.numel() == 0:
        raise ValueError("Mesh contains no hexahedral cells.")
    if int(cells.min().item()) < 0 or int(cells.max().item()) >= num_nodes:
        raise ValueError("Cell connectivity contains node indices outside the mesh node range.")

    sorted_cells = torch.sort(cells.long(), dim=1).values
    repeated_local_nodes = (sorted_cells[:, 1:] == sorted_cells[:, :-1]).any(dim=1)
    if repeated_local_nodes.any():
        count = int(repeated_local_nodes.sum().item())
        raise AssertionError(f"Found {count} degenerate hexahedra with repeated local node indices.")

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index [2, E], got {tuple(edge_index.shape)}.")
    if edge_index.numel() == 0:
        raise ValueError("Graph contains no edges.")
    if int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= num_nodes:
        raise ValueError("edge_index contains node indices outside the mesh node range.")

    standard_local = canonical_local_edge_set(HEX_EDGES)
    legacy_local = canonical_local_edge_set(LEGACY_AVBP_HEX_EDGES)
    if standard_local != legacy_local:
        raise AssertionError(
            "Current and legacy AVBP local hexahedron edge sets are not equivalent."
        )

    standard_edges = build_hex_edge_index(cells, num_nodes, HEX_EDGES)
    legacy_edges = build_hex_edge_index(cells, num_nodes, LEGACY_AVBP_HEX_EDGES)
    if not directed_edge_sets_equal(standard_edges, legacy_edges, num_nodes):
        raise AssertionError("Current and legacy AVBP mappings produce different global graphs.")
    if not directed_edge_sets_equal(edge_index, standard_edges, num_nodes):
        raise AssertionError(
            "Dataset edge_index does not match the graph derived from native hexahedral connectivity."
        )

    self_loops = int((edge_index[0] == edge_index[1]).sum().item())
    if self_loops != 0:
        raise AssertionError(f"Native graph contains {self_loops} self-loops.")

    reverse_edges = edge_index[[1, 0]]
    if not directed_edge_sets_equal(edge_index, reverse_edges, num_nodes):
        raise AssertionError("Native graph is not exactly bidirectional.")

    expected_edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
    if edge_attr.shape != expected_edge_attr.shape:
        raise AssertionError(
            f"edge_attr shape {tuple(edge_attr.shape)} does not match expected "
            f"{tuple(expected_edge_attr.shape)}."
        )
    if not torch.isfinite(edge_attr).all():
        raise AssertionError("edge_attr contains non-finite values.")
    torch.testing.assert_close(edge_attr, expected_edge_attr, rtol=0.0, atol=0.0)

    geometry = edge_geometry_metrics(pos, edge_index)
    if geometry["zero_length_edges"] != 0:
        raise AssertionError(
            f"Native graph contains {geometry['zero_length_edges']} zero-length geometric edges."
        )

    used_nodes = int(torch.unique(cells).numel())
    observed_degree_hist = degree_histogram(edge_index, num_nodes)

    return {
        "version": 2,
        "topology_contract": "native_hexahedral_connectivity",
        "mesh_id": dataset.mesh_id,
        "num_nodes": num_nodes,
        "num_cells": int(cells.shape[0]),
        "num_directed_edges": int(edge_index.shape[1]),
        "spatial_dim": spatial_dim,
        "num_nodes_referenced_by_cells": used_nodes,
        "num_nodes_not_referenced_by_cells": num_nodes - used_nodes,
        "periodic_closure_added": False,
        "periodicity_interpretation": (
            "The active graph equals the graph derived from Connectivity/hex->node; "
            "D4 core adds no periodic or other extra edges."
        ),
        "core_validation": {
            "cell_indices_in_range": True,
            "no_repeated_local_cell_nodes": True,
            "dataset_graph_matches_native_connectivity": True,
            "graph_bidirectional": True,
            "no_self_loops": True,
            "edge_attr_equals_pos_j_minus_pos_i": True,
            "no_zero_length_edges": True,
        },
        "degree_histogram": {
            "observed": {str(k): v for k, v in observed_degree_hist.items()},
        },
        "edge_geometry": geometry,
        "local_hex_edge_mappings": {
            "current_standard": [list(edge) for edge in HEX_EDGES],
            "legacy_avbp": [list(edge) for edge in LEGACY_AVBP_HEX_EDGES],
            "same_undirected_local_edge_set": True,
            "same_global_directed_edge_set": True,
        },
    }


def _close(value: float, expected: float, *, rtol: float = 1e-4, atol: float = 1e-12) -> bool:
    return math.isclose(value, expected, rel_tol=rtol, abs_tol=atol)


def _cluster_axis_coordinate_values(values: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Merge numerically duplicated coordinate planes for optional HIT diagnostics."""

    if values.ndim != 1:
        raise ValueError(f"Expected one-dimensional axis coordinates, got {tuple(values.shape)}.")
    if not torch.is_floating_point(values):
        raise ValueError("Axis coordinates must be floating point.")

    values64 = values.to(torch.float64)
    exact_values, exact_counts = torch.unique(values64, sorted=True, return_counts=True)
    if exact_values.numel() < 2:
        raise ValueError("Axis contains fewer than two unique coordinate values.")

    exact_diffs = torch.diff(exact_values)
    if not torch.all(exact_diffs > 0):
        raise ValueError("Exact unique axis coordinates are not strictly increasing.")

    max_gap = float(exact_diffs.max().item())
    source_eps = torch.finfo(values.dtype).eps
    coordinate_scale = max(
        abs(float(exact_values[0].item())),
        abs(float(exact_values[-1].item())),
        max_gap,
    )
    merge_tolerance = max(max_gap * 1e-4, 16.0 * source_eps * coordinate_scale)

    new_group = exact_diffs > merge_tolerance
    group_ids = torch.zeros(exact_values.numel(), dtype=torch.long, device=values.device)
    if new_group.numel() > 0:
        group_ids[1:] = torch.cumsum(new_group.to(torch.long), dim=0)
    num_groups = int(group_ids[-1].item()) + 1

    weighted_sum = torch.zeros(num_groups, dtype=torch.float64, device=values.device)
    total_weight = torch.zeros(num_groups, dtype=torch.float64, device=values.device)
    counts64 = exact_counts.to(torch.float64)
    weighted_sum.scatter_add_(0, group_ids, exact_values * counts64)
    total_weight.scatter_add_(0, group_ids, counts64)
    centers = weighted_sum / total_weight

    spread = (exact_values - centers[group_ids]).abs()
    max_cluster_spread = float(spread.max().item()) if spread.numel() else 0.0

    return centers, {
        "exact_unique_count": int(exact_values.numel()),
        "merged_unique_count": int(centers.numel()),
        "merged_exact_values": int(exact_values.numel() - centers.numel()),
        "merge_tolerance": float(merge_tolerance),
        "max_cluster_spread": max_cluster_spread,
    }


def infer_cartesian_grid(pos: torch.Tensor) -> dict:
    """Infer tensor-product grid dimensions for optional HIT diagnostics."""

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected three-dimensional positions [N, 3], got {tuple(pos.shape)}.")
    if not torch.isfinite(pos).all():
        raise ValueError("Mesh positions contain non-finite values.")

    axis_counts: list[int] = []
    axis_min: list[float] = []
    axis_max: list[float] = []
    axis_span: list[float] = []
    axis_spacing: list[float] = []
    axis_spacing_rel_spread: list[float] = []
    axis_exact_unique_counts: list[int] = []
    axis_merged_exact_values: list[int] = []
    axis_coordinate_merge_tolerance: list[float] = []
    axis_max_coordinate_cluster_spread: list[float] = []

    for axis in range(3):
        values, diagnostics = _cluster_axis_coordinate_values(pos[:, axis])
        count = int(values.numel())
        diffs = torch.diff(values)
        spacing = float(diffs.mean().item())
        max_dev = float((diffs - spacing).abs().max().item())
        rel_spread = max_dev / spacing

        axis_counts.append(count)
        axis_min.append(float(values[0].item()))
        axis_max.append(float(values[-1].item()))
        axis_span.append(float((values[-1] - values[0]).item()))
        axis_spacing.append(spacing)
        axis_spacing_rel_spread.append(rel_spread)
        axis_exact_unique_counts.append(diagnostics["exact_unique_count"])
        axis_merged_exact_values.append(diagnostics["merged_exact_values"])
        axis_coordinate_merge_tolerance.append(diagnostics["merge_tolerance"])
        axis_max_coordinate_cluster_spread.append(diagnostics["max_cluster_spread"])

    expected_nodes = math.prod(axis_counts)
    if expected_nodes != int(pos.shape[0]):
        raise ValueError(
            "Coordinates do not form a complete tensor-product grid after round-off clustering: "
            f"axis_counts={axis_counts} imply {expected_nodes} nodes, observed {pos.shape[0]}."
        )

    return {
        "axis_counts": axis_counts,
        "axis_min": axis_min,
        "axis_max": axis_max,
        "axis_span": axis_span,
        "axis_spacing": axis_spacing,
        "axis_spacing_rel_spread": axis_spacing_rel_spread,
        "axis_exact_unique_counts": axis_exact_unique_counts,
        "axis_merged_exact_values": axis_merged_exact_values,
        "axis_coordinate_merge_tolerance": axis_coordinate_merge_tolerance,
        "axis_max_coordinate_cluster_spread": axis_max_coordinate_cluster_spread,
    }


def expected_nonperiodic_cartesian_degree_histogram(
    axis_counts: Sequence[int],
) -> dict[int, int]:
    nx, ny, nz = (int(value) for value in axis_counts)
    if min(nx, ny, nz) < 2:
        raise ValueError(f"Expected at least two grid points per axis, got {axis_counts}.")

    hist = {
        3: 8,
        4: 4 * ((nx - 2) + (ny - 2) + (nz - 2)),
        5: 2
        * (
            (nx - 2) * (ny - 2)
            + (nx - 2) * (nz - 2)
            + (ny - 2) * (nz - 2)
        ),
        6: (nx - 2) * (ny - 2) * (nz - 2),
    }
    return {degree: count for degree, count in hist.items() if count > 0}


def expected_nonperiodic_cartesian_directed_edges(axis_counts: Sequence[int]) -> int:
    nx, ny, nz = (int(value) for value in axis_counts)
    undirected = (
        (nx - 1) * ny * nz
        + nx * (ny - 1) * nz
        + nx * ny * (nz - 1)
    )
    return 2 * undirected


def validate_hit_cartesian_diagnostics(dataset, L_ref: float) -> dict:
    """Strict HIT Cartesian diagnostic.

    This function intentionally contains case-specific assumptions. Production D4
    validity must come from ``validate_native_hex_topology`` instead. Callers that
    want non-fatal diagnostics should use ``run_hit_cartesian_diagnostics``.
    """

    if not math.isfinite(L_ref) or L_ref <= 0.0:
        raise ValueError(f"L_ref must be finite and > 0, got {L_ref}.")

    grid = infer_cartesian_grid(dataset.pos)
    axis_counts = grid["axis_counts"]
    expected_cells = math.prod(count - 1 for count in axis_counts)
    if int(dataset.cells.shape[0]) != expected_cells:
        raise AssertionError(
            f"Expected {expected_cells} hexahedra from axis_counts={axis_counts}, "
            f"got {dataset.cells.shape[0]}."
        )

    for axis, rel_spread in enumerate(grid["axis_spacing_rel_spread"]):
        if rel_spread > 1e-4:
            raise AssertionError(
                f"Axis {axis} mesh spacing is not uniform enough: relative spread={rel_spread}."
            )
    for axis, span in enumerate(grid["axis_span"]):
        if not _close(span, L_ref, rtol=1e-4, atol=1e-10):
            raise AssertionError(f"Axis {axis} span {span} does not match L_ref={L_ref}.")

    expected_edges = expected_nonperiodic_cartesian_directed_edges(axis_counts)
    if int(dataset.edge_index.shape[1]) != expected_edges:
        raise AssertionError(
            f"Expected {expected_edges} directed nonperiodic Cartesian edges, "
            f"got {dataset.edge_index.shape[1]}."
        )

    observed_degree_hist = degree_histogram(dataset.edge_index, int(dataset.pos.shape[0]))
    expected_degree_hist = expected_nonperiodic_cartesian_degree_histogram(axis_counts)
    if observed_degree_hist != expected_degree_hist:
        raise AssertionError(
            f"Degree histogram mismatch: observed={observed_degree_hist}, expected={expected_degree_hist}."
        )

    geometry = edge_geometry_metrics(dataset.pos, dataset.edge_index, grid["axis_spacing"])
    if geometry["nearest_neighbor_edges"] != geometry["num_edges"]:
        raise AssertionError(
            "At least one graph edge is not an axis-aligned Cartesian nearest-neighbour edge."
        )
    if geometry["cross_box_edges"] != 0:
        raise AssertionError(f"Found {geometry['cross_box_edges']} cross-box edges.")
    if geometry["opposite_face_edges"] != 0:
        raise AssertionError(
            f"Found {geometry['opposite_face_edges']} direct opposite-face edges."
        )

    return {
        "status": "passed",
        "diagnostic_only": True,
        "assumption": "HIT mesh is a uniform nonperiodic Cartesian tensor-product box",
        "cartesian_grid": {
            **grid,
            "expected_num_cells": expected_cells,
            "expected_num_directed_edges": expected_edges,
            "nondimensional_span": [span / L_ref for span in grid["axis_span"]],
            "nondimensional_spacing": [spacing / L_ref for spacing in grid["axis_spacing"]],
        },
        "degree_histogram": {
            "observed": {str(k): v for k, v in observed_degree_hist.items()},
            "expected_nonperiodic": {str(k): v for k, v in expected_degree_hist.items()},
        },
        "edge_geometry": geometry,
    }


def run_hit_cartesian_diagnostics(dataset, L_ref: float) -> dict:
    """Run HIT Cartesian checks without making them a D4 pass/fail criterion."""

    try:
        return validate_hit_cartesian_diagnostics(dataset, L_ref)
    except (AssertionError, ValueError, RuntimeError) as exc:
        return {
            "status": "failed",
            "diagnostic_only": True,
            "assumption": "HIT mesh is a uniform nonperiodic Cartesian tensor-product box",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def write_topology_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
