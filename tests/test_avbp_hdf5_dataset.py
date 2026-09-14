from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dgn4avbp.data.avbp_hdf5 import (
    AVBPHDF5FixedMeshDataset,
    EXPECTED_CHANNEL_PATHS,
    discover_solution_files,
    validate_snapshot_dataset,
)


COORDS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


def _write_mesh(path: Path) -> None:
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("Coordinates/x", data=COORDS[:, 0])
        h5f.create_dataset("Coordinates/y", data=COORDS[:, 1])
        h5f.create_dataset("Coordinates/z", data=COORDS[:, 2])
        # Use 1-based AVBP-style connectivity to test normalization.
        h5f.create_dataset("Connectivity/hex->node", data=np.arange(1, 9, dtype=np.int64).reshape(1, 8))


def _write_solution(path: Path, offset: float) -> None:
    with h5py.File(path, "w") as h5f:
        for channel, hdf5_path in enumerate(EXPECTED_CHANNEL_PATHS):
            values = np.arange(8, dtype=np.float32) + offset + 100.0 * channel
            h5f.create_dataset(hdf5_path, data=values)


@pytest.fixture
def tiny_avbp_case(tmp_path: Path) -> tuple[Path, list[str]]:
    mesh_file = tmp_path / "mesh.mesh.h5"
    _write_mesh(mesh_file)
    solution_dir = tmp_path / "SOLUT"
    solution_dir.mkdir()
    _write_solution(solution_dir / "sol_0002.h5", offset=2.0)
    _write_solution(solution_dir / "sol_0001.h5", offset=1.0)
    files = discover_solution_files(solution_dir)
    return mesh_file, files


def _dataset(mesh_file: Path, files: list[str]) -> AVBPHDF5FixedMeshDataset:
    return AVBPHDF5FixedMeshDataset(
        files=files,
        mesh_file=str(mesh_file),
        channel_paths=list(EXPECTED_CHANNEL_PATHS),
        coord_paths=["Coordinates/x", "Coordinates/y", "Coordinates/z"],
        connectivity_path="Connectivity/hex->node",
    )


def test_discovery_is_sorted_and_each_file_is_one_sample(tiny_avbp_case) -> None:
    mesh_file, files = tiny_avbp_case
    assert [Path(path).name for path in files] == ["sol_0001.h5", "sol_0002.h5"]

    dataset = _dataset(mesh_file, files)
    assert len(dataset) == 2
    assert dataset[0].sample_id == "sol_0001.h5"
    assert dataset[1].sample_id == "sol_0002.h5"


def test_sample_contains_exactly_five_conservative_channels(tiny_avbp_case) -> None:
    mesh_file, files = tiny_avbp_case
    sample = _dataset(mesh_file, files)[0]

    assert sample.target.shape == (8, 5)
    assert torch.equal(sample.target[:, 0], torch.arange(8, dtype=torch.float32) + 1.0)
    assert torch.equal(sample.target[:, 4], torch.arange(8, dtype=torch.float32) + 401.0)


def test_fixed_mesh_is_reused_across_snapshots(tiny_avbp_case) -> None:
    mesh_file, files = tiny_avbp_case
    dataset = _dataset(mesh_file, files)

    first = dataset[0]
    second = dataset[1]
    assert first.pos.data_ptr() == second.pos.data_ptr() == dataset.pos.data_ptr()
    assert first.edge_index.data_ptr() == second.edge_index.data_ptr() == dataset.edge_index.data_ptr()
    assert torch.equal(first.cells, second.cells)


def test_native_single_hex_topology_has_no_added_diagonal_or_wrap_edges(tiny_avbp_case) -> None:
    mesh_file, files = tiny_avbp_case
    dataset = _dataset(mesh_file, files)

    # One hexahedron has 12 undirected native edges -> 24 directed PyG edges.
    assert dataset.edge_index.shape == (2, 24)
    edges = {tuple(edge) for edge in dataset.edge_index.t().tolist()}

    # Native cube edge is present.
    assert (0, 1) in edges and (1, 0) in edges
    # Opposite/diagonal vertices are not connected by any artificial periodic or clique edge.
    assert (0, 6) not in edges and (6, 0) not in edges


def test_full_validation_visits_every_file_once(tiny_avbp_case) -> None:
    mesh_file, files = tiny_avbp_case
    dataset = _dataset(mesh_file, files)

    manifest = validate_snapshot_dataset(dataset, full_scan=True)
    assert manifest["num_samples"] == 2
    assert manifest["files"] == files
    assert manifest["periodic_edges"] is False
    assert manifest["temporal_windowing"] is False


def test_wrong_channel_order_is_rejected(tiny_avbp_case) -> None:
    mesh_file, files = tiny_avbp_case
    bad_channels = list(EXPECTED_CHANNEL_PATHS)
    bad_channels[0], bad_channels[1] = bad_channels[1], bad_channels[0]

    with pytest.raises(ValueError, match="conservative channel order"):
        AVBPHDF5FixedMeshDataset(
            files=files,
            mesh_file=str(mesh_file),
            channel_paths=bad_channels,
            coord_paths=["Coordinates/x", "Coordinates/y", "Coordinates/z"],
            connectivity_path="Connectivity/hex->node",
        )


def test_node_count_mismatch_is_rejected(tiny_avbp_case, tmp_path: Path) -> None:
    mesh_file, _ = tiny_avbp_case
    bad_solution = tmp_path / "bad.h5"
    with h5py.File(bad_solution, "w") as h5f:
        for hdf5_path in EXPECTED_CHANNEL_PATHS:
            h5f.create_dataset(hdf5_path, data=np.zeros(7, dtype=np.float32))

    dataset = _dataset(mesh_file, [str(bad_solution)])
    with pytest.raises(ValueError, match="Node-count mismatch"):
        _ = dataset[0]
