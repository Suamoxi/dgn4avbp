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


BINS = [
    (0, 100, "0-99"),
    (100, 250, "100-249"),
    (250, 500, "250-499"),
    (500, 750, "500-749"),
    (750, 1000, "750-999"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect saved loss-second-moment timestep sampler states.")
    p.add_argument("--checkpoint", action="append", required=True, help="label=path; repeatable")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def parse_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected label=path, got {spec!r}")
    label, path = spec.split("=", 1)
    return label, Path(path)


def sampler_from_checkpoint(path: Path) -> tuple[int, np.ndarray, np.ndarray, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    epoch = int(ckpt["progress"]["epoch"])
    state = ckpt["step_sampler"]
    hist = np.asarray(state["loss_history"], dtype=np.float64)
    counts = np.asarray(state["loss_counts"], dtype=np.int64)
    uniform_prob = float(state["uniform_prob"])

    warmed = np.all(counts == int(state["min_history_length"]))
    if warmed:
        rms = np.sqrt(np.mean(hist**2, axis=1))
        probability = rms / np.sum(rms)
        probability *= 1.0 - uniform_prob
        probability += uniform_prob / len(probability)
    else:
        rms = np.sqrt(np.mean(hist**2, axis=1))
        probability = np.full(hist.shape[0], 1.0 / hist.shape[0], dtype=np.float64)
    return epoch, counts, rms, probability


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestep_rows = []
    bin_rows = []
    summary = {}

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for spec in args.checkpoint:
        label, path = parse_spec(spec)
        epoch, counts, rms, p = sampler_from_checkpoint(path)
        warmed = bool(np.all(counts == counts.max()) and counts.min() == 10)

        for t in range(len(p)):
            timestep_rows.append({
                "label": label,
                "epoch": epoch,
                "timestep": t,
                "history_count": int(counts[t]),
                "loss_second_moment_rms": float(rms[t]),
                "sampling_probability": float(p[t]),
                "importance_weight": float(1.0 / (len(p) * p[t])),
            })

        bins = {}
        for lo, hi, name in BINS:
            prob_mass = float(np.sum(p[lo:hi]))
            mean_rms = float(np.mean(rms[lo:hi]))
            bins[name] = {
                "probability_mass": prob_mass,
                "mean_loss_rms": mean_rms,
            }
            bin_rows.append({
                "label": label,
                "epoch": epoch,
                "bin": name,
                "probability_mass": prob_mass,
                "mean_loss_rms": mean_rms,
            })

        summary[label] = {
            "checkpoint": str(path),
            "epoch": epoch,
            "warmed_up": warmed,
            "min_history_count": int(counts.min()),
            "max_history_count": int(counts.max()),
            "min_probability": float(p.min()),
            "max_probability": float(p.max()),
            "max_over_uniform": float(p.max() * len(p)),
            "min_over_uniform": float(p.min() * len(p)),
            "bins": bins,
        }

        ax.plot(np.arange(len(p)), p * len(p), label=f"{label} (e{epoch})")

    ax.axhline(1.0, linestyle="--", linewidth=1.0, label="uniform")
    ax.set_xlabel("Diffusion timestep")
    ax.set_ylabel("Sampling probability / uniform probability")
    ax.set_title("Loss-second-moment timestep sampler")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "sampler_probability_vs_timestep.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    write_csv(out / "sampler_timestep.csv", timestep_rows)
    write_csv(out / "sampler_bins.csv", bin_rows)
    (out / "sampler_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print("SAMPLER DIAGNOSTIC COMPLETE")
    print("output:", out)
    for label, values in summary.items():
        print(label, values)


if __name__ == "__main__":
    main()
