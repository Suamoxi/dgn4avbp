from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from dgn4avbp.data.hierarchy import (
    GEOMETRY_CONVENTION,
    attach_hierarchy_to_graph,
    build_fixed_mesh_hierarchy,
    build_hierarchy_manifest,
    load_hierarchy_artifact,
    load_hierarchy_manifest,
    save_hierarchy_artifact,
    validate_fixed_mesh_hierarchy,
    validate_hierarchy_manifest,
    write_hierarchy_manifest,
)
from dgn4avbp.data.topology import build_hex_edge_index


def _structured_grid(n: int = 4) -> SimpleNamespace:
    coords = torch.linspace(0.0, 1.0, n)
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
    return SimpleNamespace(pos=pos, cells=cells, edge_index=edge_index, edge_attr=edge_attr)


def test_hierarchy_builds_multiple_levels_without_assuming_five() -> None:
    mesh = _structured_grid()
    artifact = build_fixed_mesh_hierarchy(mesh.pos, mesh.edge_index)

    assert artifact["geometry_convention"] == GEOMETRY_CONVENTION
    assert artifact["max_levels_requested"] is None
    assert len(artifact["levels"]) >= 2
    assert len(artifact["transitions"]) == len(artifact["levels"]) - 1

    counts = [level["pos"].shape[0] for level in artifact["levels"]]
    assert all(a > b for a, b in zip(counts, counts[1:]))
    validate_fixed_mesh_hierarchy(artifact)


def test_hierarchy_respects_explicit_prefix_limit() -> None:
    mesh = _structured_grid()
    artifact = build_fixed_mesh_hierarchy(mesh.pos, mesh.edge_index, max_levels=2)
    assert len(artifact["levels"]) == 2
    assert artifact["stop_reason"] == "max_levels_reached"


def test_every_level_uses_same_coordinate_convention() -> None:
    mesh = _structured_grid()
    artifact = build_fixed_mesh_hierarchy(mesh.pos, mesh.edge_index, max_levels=3)

    torch.testing.assert_close(artifact["levels"][0]["pos"], mesh.pos, rtol=0.0, atol=0.0)
    for level in artifact["levels"]:
        expected_edge_attr = level["pos"][level["edge_index"][1]] - level["pos"][
            level["edge_index"][0]
        ]
        torch.testing.assert_close(level["edge_attr"], expected_edge_attr, rtol=0.0, atol=0.0)

    for transition, hr, lr in zip(
        artifact["transitions"], artifact["levels"][:-1], artifact["levels"][1:]
    ):
        torch.testing.assert_close(
            lr["pos"], hr["pos"][transition["coarse_mask"]], rtol=0.0, atol=0.0
        )
        expected_e = lr["pos"][transition["idx_to_parent"]] - hr["pos"]
        torch.testing.assert_close(transition["e_hr_to_lr"], expected_e, rtol=0.0, atol=0.0)


def test_hierarchy_is_deterministic() -> None:
    mesh = _structured_grid()
    first = build_fixed_mesh_hierarchy(mesh.pos, mesh.edge_index)
    second = build_fixed_mesh_hierarchy(mesh.pos, mesh.edge_index)
    assert first["hierarchy_fingerprint_sha256"] == second["hierarchy_fingerprint_sha256"]


def test_hierarchy_serialization_round_trip(tmp_path) -> None:
    mesh = _structured_grid()
    artifact = build_fixed_mesh_hierarchy(mesh.pos, mesh.edge_index)
    manifest = build_hierarchy_manifest(
        artifact,
        mesh_id="synthetic.mesh.h5",
        dataset_fingerprint_sha256="dataset-fingerprint",
        d3_fit_file_fingerprint_sha256="train-fingerprint",
        d4_topology_contract="native_hexahedral_connectivity",
    )

    artifact_path = tmp_path / "hierarchy.pt"
    manifest_path = tmp_path / "hierarchy.json"
    save_hierarchy_artifact(artifact, artifact_path)
    write_hierarchy_manifest(manifest, manifest_path)

    loaded_artifact = load_hierarchy_artifact(artifact_path)
    loaded_manifest = load_hierarchy_manifest(manifest_path)
    validate_hierarchy_manifest(loaded_manifest, loaded_artifact)
    assert loaded_artifact["hierarchy_fingerprint_sha256"] == artifact[
        "hierarchy_fingerprint_sha256"
    ]


def test_hierarchy_fingerprint_detects_tensor_mutation() -> None:
    mesh = _structured_grid()
    artifact = build_fixed_mesh_hierarchy(mesh.pos, mesh.edge_index, max_levels=2)
    artifact["levels"][1]["pos"] = artifact["levels"][1]["pos"].clone()
    artifact["levels"][1]["pos"][0, 0] += 0.01

    with pytest.raises((AssertionError, ValueError)):
        validate_fixed_mesh_hierarchy(artifact)


def test_attach_hierarchy_uses_multiscale_gnn_attribute_names() -> None:
    mesh = _structured_grid()
    artifact = build_fixed_mesh_hierarchy(mesh.pos, mesh.edge_index, max_levels=3)
    graph = Data(
        pos=mesh.pos.clone(),
        edge_index=mesh.edge_index.clone(),
        edge_attr=mesh.edge_attr.clone(),
        target=torch.zeros(mesh.pos.shape[0], 5),
    )

    graph = attach_hierarchy_to_graph(graph, artifact)
    assert graph.batch.shape == (mesh.pos.shape[0],)

    for transition, level in zip(artifact["transitions"], artifact["levels"][1:]):
        hr = transition["from_level"]
        lr = transition["to_level"]
        assert hasattr(graph, f"coarse_mask_{lr}")
        assert hasattr(graph, f"pos_{lr}")
        assert hasattr(graph, f"idx{hr}_to_idx{lr}")
        assert hasattr(graph, f"edge_index_{lr}")
        assert hasattr(graph, f"edge_attr_{lr}")
        assert hasattr(graph, f"e_{hr}{lr}")
        assert hasattr(graph, f"batch_{lr}")
        torch.testing.assert_close(getattr(graph, f"pos_{lr}"), level["pos"])


def test_noncartesian_geometry_can_build_same_connectivity_hierarchy() -> None:
    mesh = _structured_grid()
    sheared = mesh.pos.clone()
    sheared[:, 0] = sheared[:, 0] + 0.2 * sheared[:, 1]

    artifact = build_fixed_mesh_hierarchy(sheared, mesh.edge_index, max_levels=2)
    assert len(artifact["levels"]) == 2
    torch.testing.assert_close(artifact["levels"][0]["pos"], sheared)
