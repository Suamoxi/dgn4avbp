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

from dgn4avbp.data.preprocessing import STATE_CHANNEL_NAMES
from dgn4avbp.hit_pipeline import build_hit_data_bundle, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit D3 normalization and per-sample DC modes for real and generated HIT populations."
    )
    parser.add_argument("--config", default="configs/training/dgn_hit_baseline_c0_30e.yaml")
    parser.add_argument("--epoch22-dir", required=True)
    parser.add_argument("--epoch84-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_generated(directory: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    files = sorted(directory.glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"No sample_*.pt files found in {directory}.")
    standardized = []
    nondimensional = []
    ids = []
    for path in files:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        state_std = payload.get("state_standardized")
        state_nd = payload.get("state_nondimensional")
        if not isinstance(state_std, torch.Tensor) or state_std.ndim != 2 or state_std.shape[1] != 5:
            raise ValueError(f"Invalid state_standardized in {path}.")
        if not isinstance(state_nd, torch.Tensor) or state_nd.ndim != 2 or state_nd.shape[1] != 5:
            raise ValueError(f"Invalid state_nondimensional in {path}.")
        standardized.append(state_std.to(torch.float64).numpy())
        nondimensional.append(state_nd.to(torch.float64).numpy())
        ids.append(path.name)
    return np.stack(standardized), np.stack(nondimensional), ids


def real_population(bundle, indices: list[int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    standardized = []
    nondimensional = []
    ids = []
    for index in indices:
        processed = bundle.processed_dataset[index]
        state_std = processed.target.to(torch.float64)
        state_nd = bundle.standardizer.inverse(state_std)
        standardized.append(state_std.numpy())
        nondimensional.append(state_nd.numpy())
        ids.append(str(processed.sample_id))
    return np.stack(standardized), np.stack(nondimensional), ids


def population_rows(label: str, ids: list[str], state_std: np.ndarray, state_nd: np.ndarray) -> list[dict]:
    rows = []
    for sample_index, sample_id in enumerate(ids):
        for channel, name in enumerate(STATE_CHANNEL_NAMES):
            rows.append(
                {
                    "population": label,
                    "sample_id": sample_id,
                    "channel": name,
                    "standardized_mean": float(np.mean(state_std[sample_index, :, channel])),
                    "standardized_std": float(np.std(state_std[sample_index, :, channel])),
                    "nondimensional_mean": float(np.mean(state_nd[sample_index, :, channel])),
                    "nondimensional_std": float(np.std(state_nd[sample_index, :, channel])),
                }
            )
    return rows


def summary_for_population(state_std: np.ndarray, state_nd: np.ndarray) -> dict:
    out = {}
    for channel, name in enumerate(STATE_CHANNEL_NAMES):
        sample_std_mean = np.mean(state_std[:, :, channel], axis=1)
        sample_std_std = np.std(state_std[:, :, channel], axis=1)
        sample_nd_mean = np.mean(state_nd[:, :, channel], axis=1)
        sample_nd_std = np.std(state_nd[:, :, channel], axis=1)
        out[name] = {
            "standardized_sample_mean_average": float(np.mean(sample_std_mean)),
            "standardized_sample_mean_std": float(np.std(sample_std_mean)),
            "standardized_sample_std_average": float(np.mean(sample_std_std)),
            "nondimensional_sample_mean_average": float(np.mean(sample_nd_mean)),
            "nondimensional_sample_mean_std": float(np.std(sample_nd_mean)),
            "nondimensional_sample_std_average": float(np.mean(sample_nd_std)),
        }
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_sample_means(output_dir: Path, populations: dict[str, tuple[np.ndarray, np.ndarray, list[str]]]) -> None:
    for channel, name in enumerate(STATE_CHANNEL_NAMES):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for label, (state_std, _, _) in populations.items():
            means = np.mean(state_std[:, :, channel], axis=1)
            ax.hist(
                means,
                bins=30,
                density=True,
                histtype="step",
                linewidth=1.5,
                label=label,
            )
        ax.set_xlabel(f"Per-sample mean of standardized {name}")
        ax.set_ylabel("Probability density")
        ax.set_title(f"DC-mode audit: {name}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"standardized_sample_mean_{name}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    bundle = build_hit_data_bundle(**cfg["data"])

    train = real_population(bundle, list(bundle.split_indices["train"]))
    test = real_population(bundle, list(bundle.split_indices["test"]))
    epoch22 = load_generated(Path(args.epoch22_dir))
    epoch84 = load_generated(Path(args.epoch84_dir))

    populations = {
        "train": train,
        "test": test,
        "epoch22": epoch22,
        "epoch84": epoch84,
    }

    # Exact transform audit. Real and generated states must survive std -> nd -> std.
    roundtrip = {}
    for label, (state_std_np, state_nd_np, _) in populations.items():
        state_std = torch.from_numpy(state_std_np)
        state_nd = torch.from_numpy(state_nd_np)
        nd_from_std = bundle.standardizer.inverse(state_std)
        std_from_nd = bundle.standardizer.transform(state_nd)
        roundtrip[label] = {
            "max_abs_nd_saved_minus_inverse_std": float(torch.max(torch.abs(state_nd - nd_from_std)).item()),
            "max_abs_std_saved_minus_transform_nd": float(torch.max(torch.abs(state_std - std_from_nd)).item()),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, (state_std, state_nd, ids) in populations.items():
        rows.extend(population_rows(label, ids, state_std, state_nd))
    write_csv(output_dir / "per_sample_channel_moments.csv", rows)

    plot_sample_means(output_dir, populations)

    stats = {
        label: summary_for_population(state_std, state_nd)
        for label, (state_std, state_nd, _) in populations.items()
    }

    artifact = {
        "version": 1,
        "contract": "d3_normalization_and_dc_mode_audit",
        "channel_names": list(STATE_CHANNEL_NAMES),
        "standardizer": {
            "mean": bundle.standardizer.mean.tolist(),
            "std": bundle.standardizer.std.tolist(),
            "fit_num_samples": int(bundle.standardizer.num_fit_samples),
        },
        "population_sizes": {
            label: int(state_std.shape[0])
            for label, (state_std, _, _) in populations.items()
        },
        "roundtrip": roundtrip,
        "population_statistics": stats,
        "outputs": {
            "per_sample_channel_moments": "per_sample_channel_moments.csv",
            "standardized_sample_mean_plots": [
                f"standardized_sample_mean_{name}.png" for name in STATE_CHANNEL_NAMES
            ],
        },
    }
    (output_dir / "normalization_dc_audit.json").write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("NORMALIZATION / DC-MODE AUDIT COMPLETE")
    print("output:", output_dir)
    for label in ("train", "test", "epoch22", "epoch84"):
        print("\n", label)
        print("  roundtrip:", roundtrip[label])
        for name in STATE_CHANNEL_NAMES:
            values = stats[label][name]
            print(
                f"  {name}: standardized mean={values['standardized_sample_mean_average']:.6g} "
                f"+/-{values['standardized_sample_mean_std']:.6g}; "
                f"nondim mean={values['nondimensional_sample_mean_average']:.6g}"
            )


if __name__ == "__main__":
    main()
