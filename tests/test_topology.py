from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dgn4avbp.data.avbp_hdf5 import HEX_EDGES
from dgn4avbp.data.topology import (
    LEGACY_AVBP_HEX_EDGES,
    build_hex_edge_index,
    canonical_local_edge_set,
    expected_nonperiodic_cartesian_degree_histogram,
    validate_native_hit_topology,
)


def _structured_grid_dataset(n: int = 3, L: float = 2.0):
    coords = torch.linspace(0.0, L, n)
    pos = torch.tensor(
        [[x, y, z] for z in coords for y in coords for x in coords],
        dtype=torch.float32,
    )

    def node(i: int, j: int, k: int) -> int:
        return k * n * n + j * n + i

    cells = []
    for k in range(n - 1):
        for j in range(n - 1):
            for i in range(n - 1):
                cells.append(
                    [
                        node(i, j, k),
                        node(i + 1, j, k),
                        node(i + 1, j + 1, k),
                        node(i, j + 1, k),
                        node(i, j, k + 1),
                        node(i + 1, j, k + 1),
                        node(i + 1, j + 1, k + 1),
                        node(i, j + 1, k + 1),
                    ]
                )
    cells = torch.tensor(cells, dtype=torch.long)
    edge_index = build_hex_edge_index(cells, num_nodes=pos.shape[0])
    edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
    return SimpleNamespace(
        pos=pos,
        cells=cells,
        edge_index=edge_index,
        edge_attr=edge_attr,
        mesh_id="synthetic.mesh.h5",
    )


def test_legacy_and_current_local_hex_edges_are_identical_sets() -> None:
    assert canonical_local_edge_set(HEX_EDGES) == canonical_local_edge_set(
        LEGACY_AVBP_HEX_EDGES
    )


def test_expected_degree_histogram_for_3x3x3_grid() -> None:
    assert expected_nonperiodic_cartesian_degree_histogram([3, 3, 3]) == {
        3: 8,
        4: 12,
        5: 6,
        6: 1,
    }


def test_structured_native_topology_passes() -> None:
    dataset = _structured_grid_dataset()
    manifest = validate_native_hit_topology(dataset, L_ref=2.0)

    assert manifest["num_nodes"] == 27
    assert manifest["num_cells"] == 8
    assert manifest["num_directed_edges"] == 108
    assert manifest["periodic_closure_added"] is False
    assert manifest["edge_geometry"]["cross_box_edges"] == 0
    assert manifest["edge_geometry"]["opposite_face_edges"] == 0
    assert manifest["edge_geometry"]["nearest_neighbor_fraction"] == 1.0
    assert manifest["cartesian_grid"]["nondimensional_span"] == [1.0, 1.0, 1.0]
    assert manifest["cartesian_grid"]["nondimensional_spacing"] == [0.5, 0.5, 0.5]
    assert manifest["local_hex_edge_mappings"]["same_global_directed_edge_set"] is True


def test_coordinate_roundoff_does_not_create_fake_grid_plane() -> None:
    dataset = _structured_grid_dataset()

    # Split one nominal x=1 plane into two exact float32 coordinate values by a
    # perturbation tiny compared with the native cell width. This reproduces the
    # real HIT failure where exact unique counts were 34,34,33.
    plane = torch.nonzero(dataset.pos[:, 0] == 1.0, as_tuple=False).flatten()
    dataset.pos[plane[0], 0] += 1.0e-6
    dataset.edge_attr = dataset.pos[dataset.edge_index[1]] - dataset.pos[dataset.edge_index[0]]

    manifest = validate_native_hit_topology(dataset, L_ref=2.0)

    grid = manifest["cartesian_grid"]
    assert grid["axis_counts"] == [3, 3, 3]
    assert grid["axis_exact_unique_counts"] == [4, 3, 3]
    assert grid["axis_merged_exact_values"] == [1, 0, 0]
    assert grid["axis_max_coordinate_cluster_spread"][0] > 0.0


def test_extra_opposite_face_edge_is_rejected() -> None:
    dataset = _structured_grid_dataset()
    extra = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
    dataset.edge_index = torch.cat([dataset.edge_index, extra], dim=1)
    dataset.edge_attr = dataset.pos[dataset.edge_index[1]] - dataset.pos[dataset.edge_index[0]]

    with pytest.raises(AssertionError, match="does not match the validated native hexahedral graph"):
        validate_native_hit_topology(dataset, L_ref=2.0)


def test_bad_edge_attr_is_rejected() -> None:
    dataset = _structured_grid_dataset()
    dataset.edge_attr = dataset.edge_attr.clone()
    dataset.edge_attr[0, 0] += 1.0

    with pytest.raises(AssertionError):
        validate_native_hit_topology(dataset, L_ref=2.0)


def test_wrong_reference_length_is_rejected() -> None:
    dataset = _structured_grid_dataset()
    with pytest.raises(AssertionError, match="does not match L_ref"):
        validate_native_hit_topology(dataset, L_ref=1.0)
