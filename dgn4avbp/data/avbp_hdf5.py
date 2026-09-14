from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_undirected


EXPECTED_CHANNEL_PATHS = (
    "GaseousPhase/rho",
    "GaseousPhase/rhou",
    "GaseousPhase/rhov",
    "GaseousPhase/rhow",
    "GaseousPhase/rhoE",
)

HEX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def load_data_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping in data config '{path}'.")
    return cfg


def discover_solution_files(
    data_dir: str | Path,
    file_pattern: str = "*.h5",
    recursive: bool = True,
) -> list[str]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"AVBP data directory does not exist: {root}")
    iterator = root.rglob(file_pattern) if recursive else root.glob(file_pattern)
    files = sorted(str(path) for path in iterator if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No AVBP solution files matching '{file_pattern}' under '{root}'."
        )
    return files


def _read_hdf5_array(h5f: h5py.File, path: str) -> torch.Tensor:
    if path not in h5f:
        raise KeyError(f"HDF5 path '{path}' not found in '{h5f.filename}'.")
    return torch.as_tensor(h5f[path][...])


def _normalise_hex_connectivity(connectivity: torch.Tensor, num_nodes: int) -> torch.Tensor:
    conn = connectivity.long()
    if conn.numel() == 0:
        return conn.reshape(0, 8)
    if conn.ndim == 1:
        if conn.numel() % 8 != 0:
            raise ValueError(
                f"Flattened hexahedral connectivity must contain a multiple of 8 values, got {conn.numel()}."
            )
        conn = conn.reshape(-1, 8)
    elif conn.ndim == 2:
        if conn.shape[1] == 8:
            pass
        elif conn.shape[0] == 8:
            conn = conn.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                "Hexahedral connectivity must have shape [num_cells, 8] or [8, num_cells], "
                f"got {tuple(conn.shape)}."
            )
    else:
        raise ValueError(f"Connectivity must be rank 1 or 2, got shape {tuple(conn.shape)}.")

    min_idx = int(conn.min().item())
    if min_idx == 1:
        conn = conn - 1
    elif min_idx != 0:
        raise ValueError(f"Expected 0-based or 1-based connectivity, observed minimum index {min_idx}.")

    max_idx = int(conn.max().item())
    if max_idx >= num_nodes:
        raise ValueError(
            f"Connectivity index {max_idx} is outside node range [0, {num_nodes - 1}]."
        )
    return conn


def _hex_connectivity_to_edge_index(connectivity: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if connectivity.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    src: list[torch.Tensor] = []
    dst: list[torch.Tensor] = []
    for a, b in HEX_EDGES:
        src.append(connectivity[:, a])
        dst.append(connectivity[:, b])
    edge_index = torch.stack([torch.cat(src), torch.cat(dst)], dim=0)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return coalesce(edge_index, num_nodes=num_nodes, sort_by_row=False)


def _read_fixed_mesh(
    mesh_file: str | Path,
    coord_paths: Iterable[str],
    connectivity_path: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mesh_file = Path(mesh_file)
    if not mesh_file.exists():
        raise FileNotFoundError(f"AVBP mesh file does not exist: {mesh_file}")

    with h5py.File(mesh_file, "r") as h5f:
        coords = torch.stack(
            [_read_hdf5_array(h5f, path).reshape(-1).float() for path in coord_paths],
            dim=1,
        )
        connectivity = _normalise_hex_connectivity(
            _read_hdf5_array(h5f, connectivity_path),
            num_nodes=coords.shape[0],
        )

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected mesh coordinates with shape [N, 3], got {tuple(coords.shape)}.")
    if not torch.isfinite(coords).all():
        raise ValueError("Mesh coordinates contain non-finite values.")

    edge_index = _hex_connectivity_to_edge_index(connectivity, num_nodes=coords.shape[0])
    return coords, connectivity, edge_index


class AVBPHDF5FixedMeshDataset(torch.utils.data.Dataset):
    """Map-style AVBP dataset where one HDF5 solution file is one physical sample.

    The shared mesh is loaded once at construction. Samples contain the five raw
    conservative fields and native hexahedral graph connectivity. No temporal
    windowing, nondimensionalization, statistical scaling, or periodic closure
    is performed here.
    """

    def __init__(
        self,
        files: list[str],
        mesh_file: str,
        channel_paths: list[str],
        coord_paths: list[str],
        connectivity_path: str,
    ) -> None:
        if not files:
            raise ValueError("AVBPHDF5FixedMeshDataset requires at least one solution file.")
        if tuple(channel_paths) != EXPECTED_CHANNEL_PATHS:
            raise ValueError(
                "D1 requires the conservative channel order "
                f"{list(EXPECTED_CHANNEL_PATHS)}, got {channel_paths}."
            )
        if len(coord_paths) != 3:
            raise ValueError(f"Expected exactly three coordinate paths, got {coord_paths}.")

        self.files = [str(Path(path)) for path in files]
        self.mesh_file = str(Path(mesh_file))
        self.channel_paths = list(channel_paths)
        self.coord_paths = list(coord_paths)
        self.connectivity_path = connectivity_path

        for file_path in self.files:
            if not Path(file_path).is_file():
                raise FileNotFoundError(f"AVBP solution file does not exist: {file_path}")

        self.pos, self.cells, self.edge_index = _read_fixed_mesh(
            mesh_file=self.mesh_file,
            coord_paths=self.coord_paths,
            connectivity_path=self.connectivity_path,
        )
        self.edge_attr = self.pos[self.edge_index[1]] - self.pos[self.edge_index[0]]
        self.mesh_id = Path(self.mesh_file).name

    @classmethod
    def from_config(cls, config: dict) -> "AVBPHDF5FixedMeshDataset":
        explicit_files = [str(path) for path in config.get("files", [])]
        files = explicit_files or discover_solution_files(
            data_dir=config["data_dir"],
            file_pattern=config.get("file_pattern", "*.h5"),
            recursive=bool(config.get("recursive", True)),
        )
        return cls(
            files=files,
            mesh_file=config["mesh_file"],
            channel_paths=list(config["channel_paths"]),
            coord_paths=list(config["coord_paths"]),
            connectivity_path=config["connectivity_path"],
        )

    def __len__(self) -> int:
        return len(self.files)

    def _read_state(self, file_path: str) -> torch.Tensor:
        with h5py.File(file_path, "r") as h5f:
            channels = [
                _read_hdf5_array(h5f, path).reshape(-1).float()
                for path in self.channel_paths
            ]
        lengths = [channel.numel() for channel in channels]
        if len(set(lengths)) != 1:
            raise ValueError(f"Field channels have inconsistent node counts in '{file_path}': {lengths}.")
        target = torch.stack(channels, dim=1)
        if target.shape[0] != self.pos.shape[0]:
            raise ValueError(
                f"Node-count mismatch for '{file_path}': fields have N={target.shape[0]}, "
                f"shared mesh has N={self.pos.shape[0]}."
            )
        if target.shape[1] != len(EXPECTED_CHANNEL_PATHS):
            raise ValueError(f"Expected five state channels, got shape {tuple(target.shape)}.")
        if not torch.isfinite(target).all():
            raise ValueError(f"Non-finite field value found in '{file_path}'.")
        return target

    def __getitem__(self, index: int) -> Data:
        file_path = self.files[index]
        target = self._read_state(file_path)
        return Data(
            target=target,
            pos=self.pos,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            cells=self.cells,
            sample_id=Path(file_path).name,
            source_path=file_path,
            mesh_id=self.mesh_id,
        )

    def manifest(self) -> dict:
        return {
            "dataset_type": "avbp_hdf5_fixed_mesh",
            "sample_semantics": "one_hdf5_file_one_physical_sample",
            "num_samples": len(self),
            "files": list(self.files),
            "mesh_file": self.mesh_file,
            "mesh_id": self.mesh_id,
            "num_nodes": int(self.pos.shape[0]),
            "num_cells": int(self.cells.shape[0]),
            "num_edges": int(self.edge_index.shape[1]),
            "channel_paths": list(self.channel_paths),
            "coord_paths": list(self.coord_paths),
            "connectivity_path": self.connectivity_path,
            "periodic_edges": False,
            "temporal_windowing": False,
            "nondimensionalized": False,
            "standardized": False,
        }


def validate_snapshot_dataset(dataset: AVBPHDF5FixedMeshDataset, full_scan: bool = True) -> dict:
    if len(dataset) != len(dataset.files):
        raise AssertionError("Dataset length differs from file manifest length.")
    if len(set(dataset.files)) != len(dataset.files):
        raise AssertionError("Solution file manifest contains duplicates.")
    if dataset.pos.shape[1] != 3:
        raise AssertionError("Expected three-dimensional coordinates.")
    if dataset.cells.ndim != 2 or dataset.cells.shape[1] != 8:
        raise AssertionError(f"Expected hexahedral cells [num_cells, 8], got {tuple(dataset.cells.shape)}.")

    indices = range(len(dataset)) if full_scan else range(min(len(dataset), 1))
    seen: list[str] = []
    for index in indices:
        sample = dataset[index]
        if sample.target.shape != (dataset.pos.shape[0], 5):
            raise AssertionError(
                f"Sample '{sample.sample_id}' has target shape {tuple(sample.target.shape)}, "
                f"expected {(dataset.pos.shape[0], 5)}."
            )
        if sample.pos.data_ptr() != dataset.pos.data_ptr():
            raise AssertionError("Fixed-mesh dataset did not reuse the shared coordinate tensor.")
        if sample.edge_index.data_ptr() != dataset.edge_index.data_ptr():
            raise AssertionError("Fixed-mesh dataset did not reuse the shared edge tensor.")
        seen.append(sample.source_path)

    if full_scan and seen != dataset.files:
        raise AssertionError("Full dataset iteration did not visit each solution file exactly once in manifest order.")

    return dataset.manifest()


def write_data_manifest(dataset: AVBPHDF5FixedMeshDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset.manifest(), indent=2) + "\n", encoding="utf-8")
