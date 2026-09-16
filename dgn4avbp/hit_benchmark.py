from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from dgn4avbp.data.preprocessing import STATE_CHANNEL_NAMES


@dataclass(frozen=True)
class CartesianGrid3D:
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    node_to_flat: np.ndarray
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    periodic_shape: tuple[int, int, int]
    domain_lengths: tuple[float, float, float]

    @property
    def k_nyquist_min(self) -> float:
        return float(min(np.pi / value for value in self.spacing))

    @property
    def fundamental_k(self) -> float:
        return float(min(2.0 * np.pi / value for value in self.domain_lengths))


def _cluster_axis(values: np.ndarray, *, tol: float) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0:
        raise ValueError("Cannot infer a Cartesian axis from an empty coordinate array.")
    span = max(float(ordered[-1] - ordered[0]), 1.0)
    absolute_tol = float(tol) * span
    groups: list[list[float]] = [[float(ordered[0])]]
    for value in ordered[1:]:
        if abs(float(value) - groups[-1][-1]) <= absolute_tol:
            groups[-1].append(float(value))
        else:
            groups.append([float(value)])
    return np.asarray([np.mean(group) for group in groups], dtype=np.float64)


def infer_cartesian_grid_3d(pos: torch.Tensor | np.ndarray, *, tol: float = 1.0e-6) -> CartesianGrid3D:
    coords = pos.detach().cpu().numpy() if isinstance(pos, torch.Tensor) else np.asarray(pos)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected positions [N,3], got {coords.shape}.")
    if not np.isfinite(coords).all():
        raise ValueError("Positions contain non-finite values.")

    axes = tuple(_cluster_axis(coords[:, axis], tol=tol) for axis in range(3))
    shape = tuple(int(axis.size) for axis in axes)
    if int(np.prod(shape)) != coords.shape[0]:
        raise ValueError(f"Coordinates are not a complete Cartesian tensor product: shape={shape}.")

    indices: list[np.ndarray] = []
    spacing: list[float] = []
    domain_lengths: list[float] = []
    for axis_id, axis_values in enumerate(axes):
        if axis_values.size < 3:
            raise ValueError("D13 spectrum requires at least three nodes on every Cartesian axis.")
        delta = np.diff(axis_values)
        median = float(np.median(delta))
        if median <= 0.0 or not np.allclose(delta, median, rtol=5.0e-5, atol=5.0e-8):
            raise ValueError(f"Axis {axis_id} is not uniformly spaced.")
        spacing.append(median)
        domain_lengths.append(float(axis_values[-1] - axis_values[0]))
        nearest = np.abs(coords[:, axis_id, None] - axis_values[None, :]).argmin(axis=1)
        error = np.abs(coords[:, axis_id] - axis_values[nearest])
        if np.max(error) > max(tol * max(domain_lengths[-1], 1.0), 1.0e-10):
            raise ValueError(f"Could not map nodes uniquely to Cartesian axis {axis_id}.")
        indices.append(nearest)

    node_to_flat = np.ravel_multi_index(tuple(indices), shape)
    if np.unique(node_to_flat).size != coords.shape[0]:
        raise ValueError("Cartesian node mapping contains duplicate grid slots.")

    periodic_shape = tuple(value - 1 for value in shape)
    return CartesianGrid3D(
        axes=axes,
        node_to_flat=node_to_flat.astype(np.int64),
        shape=shape,
        spacing=tuple(spacing),
        periodic_shape=periodic_shape,
        domain_lengths=tuple(domain_lengths),
    )


def reshape_nodes_to_grid(field: torch.Tensor | np.ndarray, grid: CartesianGrid3D) -> np.ndarray:
    values = field.detach().cpu().numpy() if isinstance(field, torch.Tensor) else np.asarray(field)
    values = np.asarray(values)
    if values.shape[0] != grid.node_to_flat.size:
        raise ValueError("Field node count does not match the Cartesian grid.")
    trailing = values.shape[1:]
    flat = np.empty((grid.node_to_flat.size,) + trailing, dtype=values.dtype)
    flat[grid.node_to_flat] = values
    return flat.reshape(grid.shape + trailing)


def periodic_unique_grid(field: torch.Tensor | np.ndarray, grid: CartesianGrid3D) -> np.ndarray:
    full = reshape_nodes_to_grid(field, grid)
    return full[:-1, :-1, :-1, ...]


def empirical_wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    lhs = np.sort(np.asarray(left, dtype=np.float64).reshape(-1))
    rhs = np.sort(np.asarray(right, dtype=np.float64).reshape(-1))
    if lhs.size == 0 or rhs.size == 0:
        raise ValueError("Wasserstein distance requires non-empty samples.")
    all_values = np.sort(np.concatenate([lhs, rhs]))
    if all_values.size < 2:
        return 0.0
    deltas = np.diff(all_values)
    cdf_l = np.searchsorted(lhs, all_values[:-1], side="right") / lhs.size
    cdf_r = np.searchsorted(rhs, all_values[:-1], side="right") / rhs.size
    return float(np.sum(np.abs(cdf_l - cdf_r) * deltas))


def _deterministic_subsample(values: np.ndarray, max_values: int, seed: int) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if flat.size <= max_values:
        return flat
    rng = np.random.default_rng(seed)
    indices = rng.choice(flat.size, size=max_values, replace=False)
    return flat[indices]


def _safe_velocity(state: np.ndarray, rho_floor: float) -> tuple[np.ndarray, np.ndarray]:
    rho = state[..., 0]
    invalid = rho <= rho_floor
    denominator = np.where(np.abs(rho) > rho_floor, rho, np.where(rho >= 0.0, rho_floor, -rho_floor))
    velocity = state[..., 1:4] / denominator[..., None]
    return velocity, invalid


def sample_physical_statistics(states: np.ndarray, *, rho_floor: float = 1.0e-8) -> dict[str, np.ndarray]:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 5:
        raise ValueError(f"Expected states [S,N,5], got {values.shape}.")
    velocity, invalid_rho = _safe_velocity(values, rho_floor)
    rho = values[..., 0]
    momentum2 = np.sum(values[..., 1:4] ** 2, axis=-1)
    kinetic_density = 0.5 * momentum2 / np.where(np.abs(rho) > rho_floor, rho, np.nan)
    internal_energy_density = values[..., 4] - kinetic_density

    metrics: dict[str, np.ndarray] = {
        "rho_mean": np.mean(rho, axis=1),
        "rho_std": np.std(rho, axis=1),
        "rho_nonpositive_fraction": np.mean(rho <= 0.0, axis=1),
        "internal_energy_nonpositive_fraction": np.mean(
            (~np.isfinite(internal_energy_density)) | (internal_energy_density <= 0.0), axis=1
        ),
    }
    component_rms = []
    for component, name in enumerate(("u", "v", "w")):
        component_values = velocity[..., component]
        metrics[f"{name}_mean"] = np.mean(component_values, axis=1)
        metrics[f"{name}_std"] = np.std(component_values, axis=1)
        rms = np.sqrt(np.mean(component_values**2, axis=1))
        metrics[f"{name}_rms"] = rms
        component_rms.append(rms)
    component_rms_array = np.stack(component_rms, axis=1)
    metrics["tke"] = 0.5 * np.mean(np.sum(velocity**2, axis=-1), axis=1)
    metrics["isotropy_rms_ratio"] = np.max(component_rms_array, axis=1) / np.maximum(
        np.min(component_rms_array, axis=1), np.finfo(np.float64).eps
    )
    metrics["invalid_density_for_velocity_fraction"] = np.mean(invalid_rho, axis=1)
    return metrics


def isotropic_kinetic_energy_spectrum(
    state: torch.Tensor | np.ndarray,
    grid: CartesianGrid3D,
    *,
    L_ref: float,
    rho_floor: float = 1.0e-8,
) -> dict[str, np.ndarray | float | bool]:
    values = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError(f"Expected one state [N,5], got {values.shape}.")

    periodic_state = periodic_unique_grid(values, grid)
    velocity, invalid_rho = _safe_velocity(periodic_state, rho_floor)
    velocity = velocity - np.mean(velocity, axis=(0, 1, 2), keepdims=True)
    fft_velocity = np.fft.fftn(velocity, axes=(0, 1, 2), norm="forward")
    modal_energy = 0.5 * np.sum(np.abs(fft_velocity) ** 2, axis=-1)

    k_axes = [2.0 * np.pi * np.fft.fftfreq(n, d=dx) for n, dx in zip(grid.periodic_shape, grid.spacing)]
    kx, ky, kz = np.meshgrid(*k_axes, indexing="ij")
    k_magnitude = np.sqrt(kx**2 + ky**2 + kz**2)
    k_nyquist = grid.k_nyquist_min
    k0 = grid.fundamental_k
    num_shells = int(np.floor(k_nyquist / k0 + 1.0e-10))
    centers = np.arange(1, num_shells + 1, dtype=np.float64) * k0
    spectrum = np.zeros(num_shells, dtype=np.float64)
    shell_index = np.floor(k_magnitude / k0 + 0.5).astype(np.int64) - 1
    valid_shell = (k_magnitude > 0.0) & (k_magnitude <= k_nyquist + 1.0e-10)
    for shell in range(num_shells):
        mask = valid_shell & (shell_index == shell)
        spectrum[shell] = float(np.sum(modal_energy[mask]))

    fluctuation_tke = float(0.5 * np.mean(np.sum(velocity**2, axis=-1)))
    full_modal_tke = float(np.sum(modal_energy[k_magnitude > 0.0]))
    resolved_spherical_tke = float(np.sum(spectrum))
    corner_fraction = (
        max(full_modal_tke - resolved_spherical_tke, 0.0) / full_modal_tke
        if full_modal_tke > 0.0
        else 0.0
    )
    return {
        "k_Lref": centers,
        "k_over_k_nyquist": centers / k_nyquist,
        "k_per_m": centers / float(L_ref),
        "energy": spectrum,
        "k_nyquist_Lref": float(k_nyquist),
        "k_nyquist_per_m": float(k_nyquist / L_ref),
        "fluctuation_tke": fluctuation_tke,
        "full_modal_tke": full_modal_tke,
        "resolved_spherical_tke": resolved_spherical_tke,
        "corner_mode_energy_fraction": float(corner_fraction),
        "parseval_relative_error": float(
            abs(full_modal_tke - fluctuation_tke) / max(fluctuation_tke, np.finfo(np.float64).eps)
        ),
        "physical_density_valid": bool(not np.any(invalid_rho)),
        "invalid_density_fraction": float(np.mean(invalid_rho)),
    }


def ensemble_energy_spectrum(
    states: np.ndarray,
    grid: CartesianGrid3D,
    *,
    L_ref: float,
    rho_floor: float = 1.0e-8,
) -> dict[str, np.ndarray | float | int]:
    values = np.asarray(states, dtype=np.float64)
    spectra = [
        isotropic_kinetic_energy_spectrum(sample, grid, L_ref=L_ref, rho_floor=rho_floor)
        for sample in values
    ]
    energy = np.stack([np.asarray(item["energy"]) for item in spectra], axis=0)
    first = spectra[0]
    return {
        "k_Lref": np.asarray(first["k_Lref"]),
        "k_over_k_nyquist": np.asarray(first["k_over_k_nyquist"]),
        "k_per_m": np.asarray(first["k_per_m"]),
        "energy_mean": np.mean(energy, axis=0),
        "energy_std": np.std(energy, axis=0),
        "k_nyquist_Lref": float(first["k_nyquist_Lref"]),
        "k_nyquist_per_m": float(first["k_nyquist_per_m"]),
        "mean_fluctuation_tke": float(np.mean([item["fluctuation_tke"] for item in spectra])),
        "mean_resolved_spherical_tke": float(
            np.mean([item["resolved_spherical_tke"] for item in spectra])
        ),
        "mean_corner_mode_energy_fraction": float(
            np.mean([item["corner_mode_energy_fraction"] for item in spectra])
        ),
        "max_parseval_relative_error": float(
            np.max([item["parseval_relative_error"] for item in spectra])
        ),
        "num_density_valid_samples": int(sum(bool(item["physical_density_valid"]) for item in spectra)),
        "num_samples": int(values.shape[0]),
    }


def _population_channel_rows(
    generated: np.ndarray,
    reference: np.ndarray,
    channel_names: Sequence[str],
    *,
    quantiles: Sequence[float],
    max_pooled_values: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for channel, name in enumerate(channel_names):
        gen = generated[..., channel]
        ref = reference[..., channel]
        gen_sample = _deterministic_subsample(gen, max_pooled_values, 1000 + channel)
        ref_sample = _deterministic_subsample(ref, max_pooled_values, 2000 + channel)
        ref_std = float(np.std(ref))
        row: dict[str, float | str] = {
            "channel": name,
            "generated_mean": float(np.mean(gen)),
            "reference_mean": float(np.mean(ref)),
            "generated_std": float(np.std(gen)),
            "reference_std": ref_std,
            "mean_bias": float(np.mean(gen) - np.mean(ref)),
            "mean_bias_over_reference_std": float(
                (np.mean(gen) - np.mean(ref)) / max(ref_std, np.finfo(np.float64).eps)
            ),
            "std_ratio": float(np.std(gen) / max(ref_std, np.finfo(np.float64).eps)),
            "pooled_wasserstein_1_approx": empirical_wasserstein_1d(gen_sample, ref_sample),
            "snapshot_mean_wasserstein_1": empirical_wasserstein_1d(
                np.mean(gen, axis=1), np.mean(ref, axis=1)
            ),
            "snapshot_std_wasserstein_1": empirical_wasserstein_1d(
                np.std(gen, axis=1), np.std(ref, axis=1)
            ),
        }
        for q in quantiles:
            label = f"q{int(round(100 * q)):02d}"
            row[f"generated_{label}"] = float(np.quantile(gen_sample, q))
            row[f"reference_{label}"] = float(np.quantile(ref_sample, q))
        rows.append(row)
    return rows


def _physical_rows(generated: np.ndarray, reference: np.ndarray) -> list[dict[str, float | str]]:
    gen_metrics = sample_physical_statistics(generated)
    ref_metrics = sample_physical_statistics(reference)
    rows: list[dict[str, float | str]] = []
    for name in gen_metrics:
        gen = np.asarray(gen_metrics[name], dtype=np.float64)
        ref = np.asarray(ref_metrics[name], dtype=np.float64)
        ref_mean = float(np.mean(ref))
        ref_std = float(np.std(ref))
        rows.append(
            {
                "metric": name,
                "generated_mean": float(np.mean(gen)),
                "reference_mean": ref_mean,
                "generated_std": float(np.std(gen)),
                "reference_std": ref_std,
                "mean_bias": float(np.mean(gen) - ref_mean),
                "mean_bias_over_reference_std": float(
                    (np.mean(gen) - ref_mean) / max(ref_std, np.finfo(np.float64).eps)
                ),
                "wasserstein_1": empirical_wasserstein_1d(gen, ref),
            }
        )
    return rows


def _spectral_rows(generated_spectrum: dict, reference_spectrum: dict) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    gen = np.asarray(generated_spectrum["energy_mean"])
    ref = np.asarray(reference_spectrum["energy_mean"])
    for index in range(gen.size):
        rows.append(
            {
                "k_Lref": float(generated_spectrum["k_Lref"][index]),
                "k_per_m": float(generated_spectrum["k_per_m"][index]),
                "k_over_k_nyquist": float(generated_spectrum["k_over_k_nyquist"][index]),
                "generated_energy": float(gen[index]),
                "reference_energy": float(ref[index]),
                "energy_ratio": float(gen[index] / max(ref[index], np.finfo(np.float64).eps)),
            }
        )
    return rows


def _spectral_band_rows(
    generated_spectrum: dict,
    reference_spectrum: dict,
    bands: dict[str, Sequence[float]],
) -> list[dict[str, float | str]]:
    k_norm = np.asarray(generated_spectrum["k_over_k_nyquist"])
    gen = np.asarray(generated_spectrum["energy_mean"])
    ref = np.asarray(reference_spectrum["energy_mean"])
    rows: list[dict[str, float | str]] = []
    for name, bounds in bands.items():
        lower, upper = float(bounds[0]), float(bounds[1])
        if lower < 0.0 or upper <= lower or upper > 1.0 + 1.0e-12:
            raise ValueError(f"Invalid spectral band {name}: {bounds}.")
        mask = (k_norm >= lower) & (k_norm < upper)
        if abs(upper - 1.0) < 1.0e-12:
            mask = (k_norm >= lower) & (k_norm <= upper + 1.0e-12)
        generated_energy = float(np.sum(gen[mask]))
        reference_energy = float(np.sum(ref[mask]))
        rows.append(
            {
                "band": name,
                "lower_k_over_k_nyquist": lower,
                "upper_k_over_k_nyquist": upper,
                "generated_energy": generated_energy,
                "reference_energy": reference_energy,
                "energy_ratio": generated_energy / max(reference_energy, np.finfo(np.float64).eps),
            }
        )
    return rows


def benchmark_hit_populations(
    generated: torch.Tensor | np.ndarray,
    reference: torch.Tensor | np.ndarray,
    pos_nondimensional: torch.Tensor | np.ndarray,
    *,
    L_ref: float,
    channel_names: Sequence[str] = STATE_CHANNEL_NAMES,
    quantiles: Sequence[float] = (0.01, 0.05, 0.5, 0.95, 0.99),
    spectral_bands: dict[str, Sequence[float]] | None = None,
    max_pooled_values: int = 500_000,
) -> dict:
    generated_np = generated.detach().cpu().numpy() if isinstance(generated, torch.Tensor) else np.asarray(generated)
    reference_np = reference.detach().cpu().numpy() if isinstance(reference, torch.Tensor) else np.asarray(reference)
    generated_np = np.asarray(generated_np, dtype=np.float64)
    reference_np = np.asarray(reference_np, dtype=np.float64)
    if generated_np.ndim != 3 or reference_np.ndim != 3:
        raise ValueError("Generated/reference populations must have shape [S,N,C].")
    if generated_np.shape[1:] != reference_np.shape[1:]:
        raise ValueError("Generated/reference node and channel shapes must match.")
    if generated_np.shape[-1] != len(channel_names):
        raise ValueError("Channel-name count does not match population tensors.")
    if not np.isfinite(generated_np).all() or not np.isfinite(reference_np).all():
        raise ValueError("Benchmark populations contain non-finite values.")
    if max_pooled_values <= 0:
        raise ValueError("max_pooled_values must be positive.")

    grid = infer_cartesian_grid_3d(pos_nondimensional)
    generated_spectrum = ensemble_energy_spectrum(generated_np, grid, L_ref=L_ref)
    reference_spectrum = ensemble_energy_spectrum(reference_np, grid, L_ref=L_ref)
    np.testing.assert_allclose(
        generated_spectrum["k_Lref"], reference_spectrum["k_Lref"], rtol=0.0, atol=0.0
    )
    bands = spectral_bands or {
        "low": (0.0, 0.25),
        "mid": (0.25, 0.5),
        "high": (0.5, 1.0),
    }

    return {
        "summary": {
            "benchmark": "d13_hit_3d_unpaired_population",
            "comparison_mode": "unpaired_population",
            "generated_reference_pairing": False,
            "state_space": "physically_nondimensional_conservative_variables",
            "num_generated_samples": int(generated_np.shape[0]),
            "num_reference_samples": int(reference_np.shape[0]),
            "nodes_per_sample": int(generated_np.shape[1]),
            "channel_names": list(channel_names),
            "grid_shape_with_periodic_endpoint": list(grid.shape),
            "fft_grid_shape_unique_periodic_nodes": list(grid.periodic_shape),
            "grid_spacing_nondimensional": list(grid.spacing),
            "k_nyquist_Lref": float(grid.k_nyquist_min),
            "k_nyquist_per_m": float(grid.k_nyquist_min / L_ref),
            "spectrum_primary_coordinate": "k_Lref_and_k_per_m",
            "spectrum_secondary_coordinate": "k_over_k_nyquist",
            "spectrum_range": "0 < k <= k_nyquist_min",
            "periodic_endpoint_handling": "drop_max_coordinate_plane_on_each_axis_before_fft",
            "generated_density_valid_spectra": int(generated_spectrum["num_density_valid_samples"]),
            "reference_density_valid_spectra": int(reference_spectrum["num_density_valid_samples"]),
            "generated_max_parseval_relative_error": float(
                generated_spectrum["max_parseval_relative_error"]
            ),
            "reference_max_parseval_relative_error": float(
                reference_spectrum["max_parseval_relative_error"]
            ),
            "generated_mean_corner_mode_energy_fraction": float(
                generated_spectrum["mean_corner_mode_energy_fraction"]
            ),
            "reference_mean_corner_mode_energy_fraction": float(
                reference_spectrum["mean_corner_mode_energy_fraction"]
            ),
        },
        "channel_rows": _population_channel_rows(
            generated_np,
            reference_np,
            channel_names,
            quantiles=quantiles,
            max_pooled_values=max_pooled_values,
        ),
        "physical_rows": _physical_rows(generated_np, reference_np),
        "spectra_rows": _spectral_rows(generated_spectrum, reference_spectrum),
        "spectral_band_rows": _spectral_band_rows(generated_spectrum, reference_spectrum, bands),
    }
