from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import yaml
from torch_geometric.data import Data

from .splits import file_list_fingerprint


STATE_CHANNEL_NAMES = ("rho", "rhou", "rhov", "rhow", "rhoE")


def load_reference_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping in reference config '{path}'.")
    return cfg


@dataclass(frozen=True)
class HITReferenceScales:
    rho_ref: float
    U_ref: float
    L_ref: float
    T_ref: float

    @classmethod
    def from_config(cls, config: dict) -> "HITReferenceScales":
        references = config.get("references")
        if not isinstance(references, dict):
            raise ValueError("Reference config is missing a 'references' mapping.")

        required = ("rho_ref", "U_ref", "L_ref", "T_ref")
        values: dict[str, float] = {}
        for name in required:
            entry = references.get(name)
            if not isinstance(entry, dict) or "value" not in entry:
                raise ValueError(f"Reference config is missing references.{name}.value.")
            value = float(entry["value"])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Reference '{name}' must be finite and > 0, got {value}.")
            values[name] = value

        return cls(**values)

    def state_scales(self, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        momentum_scale = self.rho_ref * self.U_ref
        energy_scale = self.rho_ref * self.U_ref**2
        return torch.tensor(
            [
                self.rho_ref,
                momentum_scale,
                momentum_scale,
                momentum_scale,
                energy_scale,
            ],
            dtype=dtype,
            device=device,
        )


def _validate_state_shape(state: torch.Tensor) -> None:
    if state.ndim < 1 or state.shape[-1] != len(STATE_CHANNEL_NAMES):
        raise ValueError(
            f"Expected state with final dimension {len(STATE_CHANNEL_NAMES)}, got {tuple(state.shape)}."
        )
    if not torch.isfinite(state).all():
        raise ValueError("State contains non-finite values.")


def nondimensionalize_state(state: torch.Tensor, refs: HITReferenceScales) -> torch.Tensor:
    _validate_state_shape(state)
    scales = refs.state_scales(dtype=state.dtype, device=state.device)
    return state / scales


def dimensionalize_state(state_nd: torch.Tensor, refs: HITReferenceScales) -> torch.Tensor:
    _validate_state_shape(state_nd)
    scales = refs.state_scales(dtype=state_nd.dtype, device=state_nd.device)
    return state_nd * scales


def nondimensionalize_positions(pos: torch.Tensor, refs: HITReferenceScales) -> torch.Tensor:
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected positions with shape [N, 3], got {tuple(pos.shape)}.")
    if not torch.isfinite(pos).all():
        raise ValueError("Positions contain non-finite values.")
    return pos / refs.L_ref


def dimensionalize_positions(pos_nd: torch.Tensor, refs: HITReferenceScales) -> torch.Tensor:
    if pos_nd.ndim != 2 or pos_nd.shape[1] != 3:
        raise ValueError(f"Expected positions with shape [N, 3], got {tuple(pos_nd.shape)}.")
    if not torch.isfinite(pos_nd).all():
        raise ValueError("Positions contain non-finite values.")
    return pos_nd * refs.L_ref


@dataclass(frozen=True)
class ChannelStandardizer:
    mean: torch.Tensor
    std: torch.Tensor
    num_fit_samples: int
    method: str = "sample_balanced_population"

    def __post_init__(self) -> None:
        if self.mean.shape != (len(STATE_CHANNEL_NAMES),):
            raise ValueError(f"Expected mean shape (5,), got {tuple(self.mean.shape)}.")
        if self.std.shape != (len(STATE_CHANNEL_NAMES),):
            raise ValueError(f"Expected std shape (5,), got {tuple(self.std.shape)}.")
        if self.num_fit_samples <= 0:
            raise ValueError("num_fit_samples must be positive.")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("Standardization statistics contain non-finite values.")
        if not torch.all(self.std > 0):
            raise ValueError("All standard deviations must be strictly positive.")
        if self.method != "sample_balanced_population":
            raise ValueError(f"Unsupported standardization method '{self.method}'.")

    def transform(self, state_nd: torch.Tensor) -> torch.Tensor:
        _validate_state_shape(state_nd)
        mean = self.mean.to(dtype=state_nd.dtype, device=state_nd.device)
        std = self.std.to(dtype=state_nd.dtype, device=state_nd.device)
        return (state_nd - mean) / std

    def inverse(self, state_std: torch.Tensor) -> torch.Tensor:
        _validate_state_shape(state_std)
        mean = self.mean.to(dtype=state_std.dtype, device=state_std.device)
        std = self.std.to(dtype=state_std.dtype, device=state_std.device)
        return state_std * std + mean


def fit_sample_balanced_standardizer(
    dataset,
    indices: Sequence[int],
    refs: HITReferenceScales,
) -> ChannelStandardizer:
    """Fit channel statistics from the selected physical samples only.

    Each physical sample has weight 1 regardless of its number of graph nodes.
    Within each sample, nodes are averaged uniformly. For the current fixed HIT
    mesh this is numerically equivalent to a global node-wise population moment,
    while making the intended sample-weighting contract explicit.
    """

    fit_indices = [int(index) for index in indices]
    if not fit_indices:
        raise ValueError("Cannot fit preprocessing statistics from an empty index set.")
    if len(set(fit_indices)) != len(fit_indices):
        raise ValueError("Fit indices contain duplicates.")

    sum_sample_mean = torch.zeros(len(STATE_CHANNEL_NAMES), dtype=torch.float64)
    sum_sample_second_moment = torch.zeros(len(STATE_CHANNEL_NAMES), dtype=torch.float64)

    for index in fit_indices:
        sample = dataset[index]
        state = sample.target.to(dtype=torch.float64)
        state_nd = nondimensionalize_state(state, refs)
        sum_sample_mean += state_nd.mean(dim=0)
        sum_sample_second_moment += state_nd.square().mean(dim=0)

    count = len(fit_indices)
    mean = sum_sample_mean / count
    second_moment = sum_sample_second_moment / count
    variance = second_moment - mean.square()

    # Round-off can make an exactly zero variance very slightly negative.
    variance = variance.clamp_min(0.0)
    std = variance.sqrt()
    if not torch.all(std > 0):
        raise ValueError(
            "At least one nondimensional state channel has zero variance in the fitting split."
        )

    return ChannelStandardizer(
        mean=mean,
        std=std,
        num_fit_samples=count,
    )


def compute_sample_balanced_standardized_moments(
    dataset,
    indices: Sequence[int],
    refs: HITReferenceScales,
    standardizer: ChannelStandardizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute population mean/std after standardization for validation."""

    selected = [int(index) for index in indices]
    if not selected:
        raise ValueError("Cannot validate standardized moments on an empty index set.")

    sum_mean = torch.zeros(len(STATE_CHANNEL_NAMES), dtype=torch.float64)
    sum_second = torch.zeros(len(STATE_CHANNEL_NAMES), dtype=torch.float64)

    for index in selected:
        state_nd = nondimensionalize_state(dataset[index].target.to(torch.float64), refs)
        state_std = standardizer.transform(state_nd)
        sum_mean += state_std.mean(dim=0)
        sum_second += state_std.square().mean(dim=0)

    mean = sum_mean / len(selected)
    variance = (sum_second / len(selected) - mean.square()).clamp_min(0.0)
    return mean, variance.sqrt()


class PreprocessedFixedMeshDataset(torch.utils.data.Dataset):
    """Apply frozen D3 preprocessing without mutating the raw D1 dataset."""

    def __init__(
        self,
        raw_dataset,
        refs: HITReferenceScales,
        standardizer: ChannelStandardizer,
    ) -> None:
        self.raw_dataset = raw_dataset
        self.refs = refs
        self.standardizer = standardizer
        self.files = raw_dataset.files

        self.pos = nondimensionalize_positions(raw_dataset.pos, refs)
        self.edge_index = raw_dataset.edge_index
        self.cells = raw_dataset.cells
        self.edge_attr = self.pos[self.edge_index[1]] - self.pos[self.edge_index[0]]

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> Data:
        raw = self.raw_dataset[index]
        state_nd = nondimensionalize_state(raw.target, self.refs)
        target = self.standardizer.transform(state_nd)
        return Data(
            target=target,
            pos=self.pos,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            cells=self.cells,
            sample_id=raw.sample_id,
            source_path=raw.source_path,
            mesh_id=raw.mesh_id,
        )

    def inverse_target(self, target: torch.Tensor) -> torch.Tensor:
        state_nd = self.standardizer.inverse(target)
        return dimensionalize_state(state_nd, self.refs)


def build_preprocessing_manifest(
    reference_config: dict,
    split_manifest: dict,
    standardizer: ChannelStandardizer,
) -> dict:
    train_files = split_manifest["files"]["train"]
    return {
        "version": 1,
        "case_id": reference_config.get("case_id"),
        "reference_scheme": reference_config.get("reference_scheme"),
        "dataset_fingerprint_sha256": split_manifest["ordered_file_fingerprint_sha256"],
        "fit_split": "train",
        "fit_num_samples": len(train_files),
        "fit_file_fingerprint_sha256": file_list_fingerprint(train_files),
        "channel_names": list(STATE_CHANNEL_NAMES),
        "physical_nondimensionalization": {
            "state_formulas": {
                "rho": "rho / rho_ref",
                "rhou": "rhou / (rho_ref * U_ref)",
                "rhov": "rhov / (rho_ref * U_ref)",
                "rhow": "rhow / (rho_ref * U_ref)",
                "rhoE": "rhoE / (rho_ref * U_ref^2)",
            },
            "geometry_formula": "x / L_ref",
            "edge_geometry_formula": "(x_j - x_i) / L_ref",
            "reference_config": reference_config,
        },
        "statistical_standardization": {
            "scope": "state_channels_only",
            "fit_split": "train",
            "method": standardizer.method,
            "population_std": True,
            "sample_balanced": True,
            "coordinates_standardized": False,
            "mean": standardizer.mean.tolist(),
            "std": standardizer.std.tolist(),
        },
    }


def validate_preprocessing_manifest(manifest: dict, split_manifest: dict) -> None:
    if manifest.get("dataset_fingerprint_sha256") != split_manifest.get(
        "ordered_file_fingerprint_sha256"
    ):
        raise ValueError("D3 preprocessing manifest does not match the D2 dataset fingerprint.")
    if manifest.get("fit_split") != "train":
        raise ValueError("D3 statistics must be fitted on the D2 training split only.")

    train_files = split_manifest["files"]["train"]
    if manifest.get("fit_num_samples") != len(train_files):
        raise ValueError("D3 fit sample count does not match the D2 training split.")
    if manifest.get("fit_file_fingerprint_sha256") != file_list_fingerprint(train_files):
        raise ValueError("D3 fit-file fingerprint does not match the D2 training split.")

    stats = manifest.get("statistical_standardization")
    if not isinstance(stats, dict):
        raise ValueError("D3 preprocessing manifest is missing statistical_standardization.")
    if stats.get("fit_split") != "train" or stats.get("sample_balanced") is not True:
        raise ValueError("D3 standardization contract must be train-only and sample-balanced.")
    if stats.get("coordinates_standardized") is not False:
        raise ValueError("D3 must not statistically standardize coordinates.")

    mean = torch.tensor(stats.get("mean", []), dtype=torch.float64)
    std = torch.tensor(stats.get("std", []), dtype=torch.float64)
    ChannelStandardizer(
        mean=mean,
        std=std,
        num_fit_samples=int(manifest["fit_num_samples"]),
        method=str(stats.get("method")),
    )


def standardizer_from_manifest(manifest: dict) -> ChannelStandardizer:
    stats = manifest["statistical_standardization"]
    return ChannelStandardizer(
        mean=torch.tensor(stats["mean"], dtype=torch.float64),
        std=torch.tensor(stats["std"], dtype=torch.float64),
        num_fit_samples=int(manifest["fit_num_samples"]),
        method=str(stats["method"]),
    )


def write_preprocessing_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_preprocessing_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected mapping in preprocessing manifest '{path}'.")
    return manifest
