#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dgn4avbp.data.preprocessing import STATE_CHANNEL_NAMES
from dgn4avbp.hit_benchmark import (
    _deterministic_subsample,
    empirical_wasserstein_1d,
    infer_cartesian_grid_3d,
    periodic_unique_grid,
)
from dgn4avbp.hit_pipeline import build_hit_data_bundle, load_yaml


METHODS = ("ancestral", "vp_sde_euler", "vp_probability_flow_ode_euler")
COMPONENTS = ("u", "v", "w")
BANDS = {
    "low": (0.0, 0.25),
    "mid": (0.25, 0.50),
    "high": (0.50, 1.00),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate controlled reverse-sampler comparison diagnostics.")
    p.add_argument("--training-config", default="configs/training/dgn_hit_baseline_c0_30e.yaml")
    p.add_argument("--generation-root", required=True)
    p.add_argument("--benchmark-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-pooled-values", type=int, default=500_000)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def keyed(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_generated(directory: Path) -> np.ndarray:
    files = sorted(directory.glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"No sample_*.pt files in {directory}.")
    states = []
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        states.append(payload["state_nondimensional"].to(torch.float64).numpy())
    return np.stack(states)


def velocity(states: np.ndarray) -> np.ndarray:
    rho = states[..., 0]
    if np.any(np.abs(rho) <= 1.0e-8):
        raise ValueError("Density too close to zero for centered-velocity diagnostic.")
    return states[..., 1:4] / rho[..., None]


def centered_metrics(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    max_pooled_values: int,
) -> dict[str, float]:
    gv = velocity(generated)
    rv = velocity(reference)
    gc = gv - np.mean(gv, axis=1, keepdims=True)
    rc = rv - np.mean(rv, axis=1, keepdims=True)

    result = {}
    for component, name in enumerate(COMPONENTS):
        g = _deterministic_subsample(gc[..., component], max_pooled_values, 4100 + component)
        r = _deterministic_subsample(rc[..., component], max_pooled_values, 4200 + component)
        result[f"centered_{name}_w1"] = empirical_wasserstein_1d(g, r)

    gen_rms = np.sqrt(np.mean(gc**2, axis=1))
    ref_rms = np.sqrt(np.mean(rc**2, axis=1))
    gen_tke = 0.5 * np.mean(np.sum(gc**2, axis=-1), axis=1)
    ref_tke = 0.5 * np.mean(np.sum(rc**2, axis=-1), axis=1)
    gen_iso = np.max(gen_rms, axis=1) / np.maximum(np.min(gen_rms, axis=1), np.finfo(np.float64).eps)
    ref_iso = np.max(ref_rms, axis=1) / np.maximum(np.min(ref_rms, axis=1), np.finfo(np.float64).eps)

    result.update(
        {
            "fluctuation_tke_generated_mean": float(np.mean(gen_tke)),
            "fluctuation_tke_reference_mean": float(np.mean(ref_tke)),
            "fluctuation_tke_w1": empirical_wasserstein_1d(gen_tke, ref_tke),
            "fluctuation_isotropy_generated_mean": float(np.mean(gen_iso)),
            "fluctuation_isotropy_reference_mean": float(np.mean(ref_iso)),
            "fluctuation_isotropy_w1": empirical_wasserstein_1d(gen_iso, ref_iso),
        }
    )
    return result


def spectral_geometry(pos: torch.Tensor):
    grid = infer_cartesian_grid_3d(pos)
    k_axes = [
        2.0 * np.pi * np.fft.fftfreq(n, d=dx)
        for n, dx in zip(grid.periodic_shape, grid.spacing)
    ]
    kx, ky, kz = np.meshgrid(*k_axes, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    return grid, kmag, float(grid.k_nyquist_min)


def scalar_band_energies(field: np.ndarray, grid, kmag: np.ndarray, knyq: float) -> dict[str, float]:
    values = periodic_unique_grid(field, grid).astype(np.float64)
    values -= np.mean(values)
    modal = np.abs(np.fft.fftn(values, norm="forward")) ** 2
    k_norm = kmag / knyq
    result = {}
    for name, (lo, hi) in BANDS.items():
        mask = (k_norm >= lo) & (k_norm < hi) & (kmag > 0.0)
        if hi == 1.0:
            mask = (k_norm >= lo) & (k_norm <= hi + 1.0e-12) & (kmag > 0.0)
        result[name] = float(np.sum(modal[mask]))
    return result


def reference_standardized_band_energy(bundle) -> dict[str, dict[str, float]]:
    grid, kmag, knyq = spectral_geometry(bundle.processed_dataset.pos.cpu())
    accum = {
        channel: {band: [] for band in BANDS}
        for channel in STATE_CHANNEL_NAMES
    }
    for index in bundle.split_indices["test"]:
        state = bundle.processed_dataset[index].target.to(torch.float64).numpy()
        for channel_index, channel in enumerate(STATE_CHANNEL_NAMES):
            energy = scalar_band_energies(state[:, channel_index], grid, kmag, knyq)
            for band in BANDS:
                accum[channel][band].append(energy[band])
    return {
        channel: {
            band: float(np.mean(values))
            for band, values in band_values.items()
        }
        for channel, band_values in accum.items()
    }


def aggregate_trajectory(
    trajectory_path: Path,
    reference_band_energy: dict[str, dict[str, float]],
) -> list[dict]:
    raw = read_csv(trajectory_path)
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[(row["method"], int(row["reverse_t"]), row["channel"])].append(row)

    result = []
    scalar_keys = (
        "x_t_mean",
        "x_t_std",
        "x0_hat_mean",
        "x0_hat_std",
        "epsilon_mean",
        "epsilon_std",
        "model_v_mean",
        "model_v_std",
        "learned_variance_mean",
        "x0_hat_low_band_energy",
        "x0_hat_mid_band_energy",
        "x0_hat_high_band_energy",
    )
    for (method, reverse_t, channel), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], -item[0][1], item[0][2])
    ):
        item = {
            "method": method,
            "reverse_t": reverse_t,
            "channel": channel,
            "num_samples": len(rows),
        }
        for key in scalar_keys:
            item[key] = float(np.mean([float(row[key]) for row in rows]))
        for band in BANDS:
            energy = item[f"x0_hat_{band}_band_energy"]
            ref = reference_band_energy[channel][band]
            item[f"x0_hat_{band}_energy_ratio_to_test"] = energy / max(
                ref, np.finfo(np.float64).eps
            )
        result.append(item)
    return result


def plot_trajectory(rows: list[dict], output_dir: Path) -> None:
    for channel in STATE_CHANNEL_NAMES:
        selected_channel = [row for row in rows if row["channel"] == channel]

        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for method in METHODS:
            selected = [row for row in selected_channel if row["method"] == method]
            selected.sort(key=lambda row: row["reverse_t"], reverse=True)
            ax.plot(
                [row["reverse_t"] for row in selected],
                [row["x0_hat_mean"] for row in selected],
                marker="o",
                label=method,
            )
        ax.set_xlabel("Reverse diffusion timestep")
        ax.set_ylabel("Mean predicted standardized x0")
        ax.set_title(f"Predicted DC mode during sampling: {channel}")
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"trajectory_x0_mean_{channel}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for method in METHODS:
            selected = [row for row in selected_channel if row["method"] == method]
            selected.sort(key=lambda row: row["reverse_t"], reverse=True)
            ax.plot(
                [row["reverse_t"] for row in selected],
                [row["x0_hat_high_energy_ratio_to_test"] for row in selected],
                marker="o",
                label=method,
            )
        ax.axhline(1.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Reverse diffusion timestep")
        ax.set_ylabel("Predicted x0 high-band energy / test")
        ax.set_title(f"Predicted high-k energy during sampling: {channel}")
        ax.invert_xaxis()
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"trajectory_high_k_{channel}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.training_config)
    bundle = build_hit_data_bundle(**cfg["data"])

    reference_states = []
    for index in bundle.split_indices["test"]:
        processed = bundle.processed_dataset[index]
        reference_states.append(
            bundle.standardizer.inverse(processed.target).to(torch.float64).numpy()
        )
    reference = np.stack(reference_states)

    gen_root = Path(args.generation_root)
    bench_root = Path(args.benchmark_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    for method in METHODS:
        generated = load_generated(gen_root / method)
        centered = centered_metrics(
            generated,
            reference,
            max_pooled_values=args.max_pooled_values,
        )
        channels = keyed(read_csv(bench_root / method / "channel_metrics.csv"), "channel")
        physical = keyed(read_csv(bench_root / method / "physical_metrics.csv"), "metric")
        bands = keyed(read_csv(bench_root / method / "spectral_bands.csv"), "band")

        row = {"method": method, **centered}
        for channel in STATE_CHANNEL_NAMES:
            row[f"{channel}_pooled_w1"] = float(channels[channel]["pooled_wasserstein_1_approx"])
            row[f"{channel}_mean_bias"] = float(channels[channel]["mean_bias"])
        for component in COMPONENTS:
            row[f"{component}_mean_w1"] = float(physical[f"{component}_mean"]["wasserstein_1"])
        row["d13_tke_w1"] = float(physical["tke"]["wasserstein_1"])
        row["d13_isotropy_w1"] = float(physical["isotropy_rms_ratio"]["wasserstein_1"])
        row["invalid_density_fraction"] = float(
            physical["invalid_density_for_velocity_fraction"]["generated_mean"]
        )
        for band in BANDS:
            row[f"{band}_energy_ratio"] = float(bands[band]["energy_ratio"])
        comparison_rows.append(row)

    write_csv(output_dir / "reverse_sampler_comparison.csv", comparison_rows)

    ref_band = reference_standardized_band_energy(bundle)
    trajectory_rows = aggregate_trajectory(
        gen_root / "reverse_trajectory.csv",
        ref_band,
    )
    write_csv(output_dir / "reverse_trajectory_summary.csv", trajectory_rows)
    plot_trajectory(trajectory_rows, output_dir)

    print("REVERSE-SAMPLER COMPARISON AGGREGATION COMPLETE")
    print("final comparison:", output_dir / "reverse_sampler_comparison.csv")
    print("trajectory:", output_dir / "reverse_trajectory_summary.csv")
    for row in comparison_rows:
        print(row)


if __name__ == "__main__":
    main()
