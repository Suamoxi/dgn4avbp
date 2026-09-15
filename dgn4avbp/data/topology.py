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
    """Build a coalesced directed PyG graph from local hexahedron edges."""

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


def _close(value: float, expected: float, *, rtol: float = 1e-4, atol: float = 1e-12) -> bool:
    return math.isclose(value, expected, rel_tol=rtol, abs_tol=atol)


def infer_cartesian_grid(pos: torch.Tensor) -> dict:
    """Infer the tensor-product grid dimensions and uniform spacing from coordinates."""

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected positions [N, 3], got {tuple(pos.shape)}.")
    if not torch.isfinite(pos).all():
        raise ValueError("Mesh positions contain non-finite values.")

    pos64 = pos.to(torch.float64)
    axis_counts: list[int] = []
    axis_min: list[float] = []
    axis_max: list[float] = []
    axis_span: list[float] = []
    axis_spacing: list[float] = []
    axis_spacing_rel_spread: list[float] = []

    for axis in range(3):
        values = torch.sort(torch.unique(pos64[:, axis])).values
        count = int(values.numel())
        if count < 2:
            raise ValueError(f"Axis {axis} contains fewer than two unique coordinate values.")
        diffs = torch.diff(values)
        if not torch.all(diffs > 0):
            raise ValueError(f"Axis {axis} coordinates are not strictly increasing after uniquing.")
        spacing = float(diffs.mean().item())
        max_dev = float((diffs - spacing).abs().max().item())
        rel_spread = max_dev / spacing

        axis_counts.append(count)
        axis_min.append(float(values[0].item()))
        axis_max.append(float(values[-1].item()))
        axis_span.append(float((values[-1] - values[0]).item()))
        axis_spacing.append(spacing)
        axis_spacing_rel_spread.append(rel_spread)

    expected_nodes = math.prod(axis_counts)
    if expected_nodes != int(pos.shape[0]):
        raise ValueError(
            "Coordinates do not form a complete tensor-product grid: "
            f"axis_counts={axis_counts} imply {expected_nodes} nodes, observed {pos.shape[0]}."
        )

    return {
        "axis_counts": axis_counts,
        "axis_min": axis_min,
        "axis_max": axis_max,
        "axis_span": axis_span,
        "axis_spacing": axis_spacing,
        "axis_spacing_rel_spread": axis_spacing_rel_spread,
    }


def expected_nonperiodic_cartesian_degree_histogram(
    axis_counts: Sequence[int],
) -> dict[int, int]:
    nx, ny, nz = (int(value) for value in axis_counts)
    if min(nx, ny, nz) < 2:
        raise ValueError(f"Expected at least two grid points per axis, got {axis_counts}.")

    # For a complete 3-D Cartesian grid with only axial nearest-neighbour edges:
    # corners have degree 3, edge-interior nodes 4, face-interior nodes 5,
    # and volume-interior nodes 6.
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


def degree_histogram(edge_index: torch.Tensor, num_nodes: int) -> dict[int, int]:
    degree = torch.bincount(edge_index[1], minlength=num_nodes)
    return dict(sorted(Counter(int(value) for value in degree.tolist()).items()))


def edge_geometry_metrics(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    axis_spacing: Sequence[float],
) -> dict:
    pos64 = pos.to(torch.float64)
    delta = pos64[edge_index[1]] - pos64[edge_index[0]]
    abs_delta = delta.abs()
    lengths = torch.linalg.vector_norm(delta, dim=1)
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

    return {
        "num_edges": int(edge_index.shape[1]),
        "min_edge_length": float(lengths.min().item()),
        "max_edge_length": float(lengths.max().item()),
        "mean_edge_length": float(lengths.mean().item()),
        "nearest_neighbor_edges": int(nearest_neighbor.sum().item()),
        "nearest_neighbor_fraction": float(nearest_neighbor.double().mean().item()),
        "cross_box_edges": int(cross_box.sum().item()),
        "opposite_face_edges": int(opposite_face.sum().item()),
    }


def validate_native_hit_topology(dataset, L_ref: float) -> dict:
    """Validate the current HIT mesh as a native, nonperiodic Cartesian graph."""

    if not math.isfinite(L_ref) or L_ref <= 0.0:
        raise ValueError(f"L_ref must be finite and > 0, got {L_ref}.")

    pos = dataset.pos
    cells = dataset.cells
    edge_index = dataset.edge_index
    num_nodes = int(pos.shape[0])

    grid = infer_cartesian_grid(pos)
    axis_counts = grid["axis_counts"]
    expected_cells = math.prod(count - 1 for count in axis_counts)
    if int(cells.shape[0]) != expected_cells:
        raise AssertionError(
            f"Expected {expected_cells} hexahedra from axis_counts={axis_counts}, got {cells.shape[0]}."
        )

    for axis, rel_spread in enumerate(grid["axis_spacing_rel_spread"]):
        if rel_spread > 1e-4:
            raise AssertionError(
                f"Axis {axis} mesh spacing is not uniform enough: relative spread={rel_spread}."
            )
    for axis, span in enumerate(grid["axis_span"]):
        if not _close(span, L_ref, rtol=1e-4, atol=1e-10):
            raise AssertionError(
                f"Axis {axis} span {span} does not match L_ref={L_ref}."
            )

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
        raise AssertionError("Dataset edge_index does not match the validated native hexahedral graph.")

    expected_edges = expected_nonperiodic_cartesian_directed_edges(axis_counts)
    if int(edge_index.shape[1]) != expected_edges:
        raise AssertionError(
            f"Expected {expected_edges} directed nonperiodic Cartesian edges, got {edge_index.shape[1]}."
        )

    self_loops = int((edge_index[0] == edge_index[1]).sum().item())
    if self_loops != 0:
        raise AssertionError(f"Native graph contains {self_loops} self-loops.")

    reverse_edges = edge_index[[1, 0]]
    if not directed_edge_sets_equal(edge_index, reverse_edges, num_nodes):
        raise AssertionError("Native graph is not exactly bidirectional.")

    observed_degree_hist = degree_histogram(edge_index, num_nodes)
    expected_degree_hist = expected_nonperiodic_cartesian_degree_histogram(axis_counts)
    if observed_degree_hist != expected_degree_hist:
        raise AssertionError(
            f"Degree histogram mismatch: observed={observed_degree_hist}, expected={expected_degree_hist}."
        )

    geometry = edge_geometry_metrics(pos, edge_index, grid["axis_spacing"])
    if geometry["nearest_neighbor_edges"] != geometry["num_edges"]:
        raise AssertionError(
            "At least one graph edge is not an axis-aligned native nearest-neighbour mesh edge."
        )
    if geometry["cross_box_edges"] != 0:
        raise AssertionError(f"Found {geometry['cross_box_edges']} cross-box edges.")
    if geometry["opposite_face_edges"] != 0:
        raise AssertionError(
            f"Found {geometry['opposite_face_edges']} direct opposite-face edges; periodic closure is disabled."
        )

    expected_edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
    torch.testing.assert_close(dataset.edge_attr, expected_edge_attr, rtol=0.0, atol=0.0)

    h_nd = [spacing / L_ref for spacing in grid["axis_spacing"]]
    span_nd = [span / L_ref for span in grid["axis_span"]]

    return {
        "version": 1,
        "mesh_id": dataset.mesh_id,
        "num_nodes": num_nodes,
        "num_cells": int(cells.shape[0]),
        "num_directed_edges": int(edge_index.shape[1]),
        "periodic_closure_added": False,
        "cartesian_grid": {
            **grid,
            "expected_num_cells": expected_cells,
            "expected_num_directed_edges": expected_edges,
            "nondimensional_span": span_nd,
            "nondimensional_spacing": h_nd,
        },
        "degree_histogram": {
            "observed": {str(k): v for k, v in observed_degree_hist.items()},
            "expected_nonperiodic": {str(k): v for k, v in expected_degree_hist.items()},
        },
        "edge_geometry": geometry,
        "local_hex_edge_mappings": {
            "current_standard": [list(edge) for edge in HEX_EDGES],
            "legacy_avbp": [list(edge) for edge in LEGACY_AVBP_HEX_EDGES],
            "same_undirected_local_edge_set": True,
            "same_global_directed_edge_set": True,
        },
    }


def write_topology_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
