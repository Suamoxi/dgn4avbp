#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dgn4avbp.hit_benchmark import empirical_wasserstein_1d, _deterministic_subsample
from dgn4avbp.hit_pipeline import build_hit_data_bundle, load_yaml


COMPONENTS = ("u", "v", "w")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate fixed-seed milestone generation diagnostics.")
    p.add_argument("--training-config", default="configs/training/dgn_hit_baseline_c0_30e.yaml")
    p.add_argument("--generation-root", required=True)
    p.add_argument("--benchmark-root", required=True)
    p.add_argument("--c0-metrics", required=True)
    p.add_argument("--c1-metrics", required=True)
    p.add_argument("--epochs", nargs="+", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-pooled-values", type=int, default=500_000)
    return p.parse_args()


def read_csv_dict(path: Path, key: str) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as s:
        return {row[key]: row for row in csv.DictReader(s)}


def metrics_by_epoch(path: Path) -> dict[int, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[int(row["epoch"])] = row
    return out


def load_generated(directory: Path) -> np.ndarray:
    files = sorted(directory.glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"No generation files in {directory}.")
    states = []
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        states.append(payload["state_nondimensional"].to(torch.float64).numpy())
    return np.stack(states)


def velocity(states: np.ndarray) -> np.ndarray:
    rho = states[..., 0]
    if np.any(np.abs(rho) <= 1.0e-8):
        raise ValueError("Density too close to zero in milestone generation.")
    return states[..., 1:4] / rho[..., None]


def centered_w1(gen: np.ndarray, ref: np.ndarray, max_values: int, seed: int) -> dict[str, float]:
    gv = velocity(gen)
    rv = velocity(ref)
    gv = gv - np.mean(gv, axis=1, keepdims=True)
    rv = rv - np.mean(rv, axis=1, keepdims=True)
    out = {}
    for c, name in enumerate(COMPONENTS):
        g = _deterministic_subsample(gv[..., c], max_values, seed + 10 + c)
        r = _deterministic_subsample(rv[..., c], max_values, seed + 20 + c)
        out[name] = empirical_wasserstein_1d(g, r)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def line_plot(rows: list[dict], keys: list[str], title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    epochs = [row["epoch"] for row in rows]
    for key in keys:
        ax.plot(epochs, [row[key] for row in rows], marker="o", label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.training_config)
    bundle = build_hit_data_bundle(**cfg["data"])

    ref = []
    for index in bundle.split_indices["test"]:
        sample = bundle.processed_dataset[index]
        ref.append(bundle.standardizer.inverse(sample.target).to(torch.float64).numpy())
    reference = np.stack(ref)

    c0 = metrics_by_epoch(Path(args.c0_metrics))
    c1 = metrics_by_epoch(Path(args.c1_metrics))

    gen_root = Path(args.generation_root)
    bench_root = Path(args.benchmark_root)
    rows = []

    for epoch in args.epochs:
        label = f"epoch{epoch}"
        generated = load_generated(gen_root / label)
        cw1 = centered_w1(generated, reference, args.max_pooled_values, seed=10000 + epoch)

        physical = read_csv_dict(bench_root / label / "physical_metrics.csv", "metric")
        bands = read_csv_dict(bench_root / label / "spectral_bands.csv", "band")
        validation = c0.get(epoch, c1.get(epoch, {})).get("validation_loss")

        row = {
            "epoch": epoch,
            "validation_loss": float(validation) if validation is not None else np.nan,
            "centered_u_w1": cw1["u"],
            "centered_v_w1": cw1["v"],
            "centered_w_w1": cw1["w"],
            "u_mean_w1": float(physical["u_mean"]["wasserstein_1"]),
            "v_mean_w1": float(physical["v_mean"]["wasserstein_1"]),
            "w_mean_w1": float(physical["w_mean"]["wasserstein_1"]),
            "tke_generated_mean": float(physical["tke"]["generated_mean"]),
            "tke_reference_mean": float(physical["tke"]["reference_mean"]),
            "tke_w1": float(physical["tke"]["wasserstein_1"]),
            "isotropy_generated_mean": float(physical["isotropy_rms_ratio"]["generated_mean"]),
            "isotropy_reference_mean": float(physical["isotropy_rms_ratio"]["reference_mean"]),
            "isotropy_w1": float(physical["isotropy_rms_ratio"]["wasserstein_1"]),
            "low_energy_ratio": float(bands["low"]["energy_ratio"]),
            "mid_energy_ratio": float(bands["mid"]["energy_ratio"]),
            "high_energy_ratio": float(bands["high"]["energy_ratio"]),
        }
        rows.append(row)

    rows.sort(key=lambda row: row["epoch"])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "milestone_trajectory.csv", rows)

    line_plot(
        rows,
        ["centered_u_w1", "centered_v_w1", "centered_w_w1"],
        "Centered velocity Wasserstein distance",
        "W1",
        out / "centered_velocity_w1_vs_epoch.png",
    )
    line_plot(
        rows,
        ["low_energy_ratio", "mid_energy_ratio", "high_energy_ratio"],
        "Spectral energy ratio vs epoch",
        "Generated / reference energy",
        out / "spectral_energy_ratio_vs_epoch.png",
    )
    line_plot(
        rows,
        ["tke_w1", "isotropy_w1"],
        "Physical population errors vs epoch",
        "W1",
        out / "physical_w1_vs_epoch.png",
    )

    valid_rows = [row for row in rows if np.isfinite(row["validation_loss"])]
    if valid_rows:
        line_plot(
            valid_rows,
            ["validation_loss"],
            "Deterministic validation loss",
            "Validation loss",
            out / "validation_loss_vs_epoch.png",
        )

    print("MILESTONE TRAJECTORY AGGREGATION COMPLETE")
    print("output:", out / "milestone_trajectory.csv")
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
