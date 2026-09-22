#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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


def rows_from_pair(population: str, sample_id: str, state_std: torch.Tensor, state_nd: torch.Tensor) -> list[dict]:
    rows = []
    for channel, name in enumerate(STATE_CHANNEL_NAMES):
        rows.append(
            {
                "population": population,
                "sample_id": sample_id,
                "channel": name,
                "standardized_mean": float(state_std[:, channel].mean().item()),
                "standardized_std": float(state_std[:, channel].std(unbiased=False).item()),
                "nondimensional_mean": float(state_nd[:, channel].mean().item()),
                "nondimensional_std": float(state_nd[:, channel].std(unbiased=False).item()),
            }
        )
    return rows


def audit_real_population(bundle, population: str, indices: list[int]) -> tuple[list[dict], dict]:
    rows = []
    max_nd_error = 0.0
    max_std_error = 0.0
    for index in indices:
        processed = bundle.processed_dataset[index]
        state_std = processed.target.to(torch.float64)
        state_nd = bundle.standardizer.inverse(state_std)
        std_roundtrip = bundle.standardizer.transform(state_nd)
        nd_roundtrip = bundle.standardizer.inverse(std_roundtrip)
        max_std_error = max(max_std_error, float(torch.max(torch.abs(state_std - std_roundtrip)).item()))
        max_nd_error = max(max_nd_error, float(torch.max(torch.abs(state_nd - nd_roundtrip)).item()))
        rows.extend(rows_from_pair(population, str(processed.sample_id), state_std, state_nd))
    return rows, {
        "num_samples": len(indices),
        "max_abs_std_roundtrip_error": max_std_error,
        "max_abs_nd_roundtrip_error": max_nd_error,
    }


def audit_generated_population(bundle, population: str, directory: Path) -> tuple[list[dict], dict]:
    files = sorted(directory.glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"No sample_*.pt files found in {directory}.")

    rows = []
    max_saved_nd_inverse_error = 0.0
    max_saved_std_transform_error = 0.0

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

        state_std = state_std.to(torch.float64)
        state_nd = state_nd.to(torch.float64)
        inverse_std = bundle.standardizer.inverse(state_std)
        transform_nd = bundle.standardizer.transform(state_nd)
        max_saved_nd_inverse_error = max(
            max_saved_nd_inverse_error,
            float(torch.max(torch.abs(state_nd - inverse_std)).item()),
        )
        max_saved_std_transform_error = max(
            max_saved_std_transform_error,
            float(torch.max(torch.abs(state_std - transform_nd)).item()),
        )

        sample_id = f"seed={payload.get('seed', 'unknown')}:sample={payload.get('sample_index', path.stem)}"
        rows.extend(rows_from_pair(population, sample_id, state_std, state_nd))

    return rows, {
        "num_samples": len(files),
        "max_abs_saved_nd_minus_inverse_std": max_saved_nd_inverse_error,
        "max_abs_saved_std_minus_transform_nd": max_saved_std_transform_error,
    }


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["population"], row["channel"])].append(row)

    result: dict[str, dict] = {}
    for (population, channel), values in grouped.items():
        result.setdefault(population, {})
        result[population][channel] = {
            "standardized_sample_mean_average": float(np.mean([v["standardized_mean"] for v in values])),
            "standardized_sample_mean_std": float(np.std([v["standardized_mean"] for v in values])),
            "standardized_sample_std_average": float(np.mean([v["standardized_std"] for v in values])),
            "nondimensional_sample_mean_average": float(np.mean([v["nondimensional_mean"] for v in values])),
            "nondimensional_sample_mean_std": float(np.std([v["nondimensional_mean"] for v in values])),
            "nondimensional_sample_std_average": float(np.mean([v["nondimensional_std"] for v in values])),
        }
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_sample_means(output_dir: Path, rows: list[dict]) -> None:
    populations = sorted({row["population"] for row in rows})
    for channel in STATE_CHANNEL_NAMES:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for population in populations:
            values = [
                row["standardized_mean"]
                for row in rows
                if row["population"] == population and row["channel"] == channel
            ]
            ax.hist(values, bins=30, density=True, histtype="step", linewidth=1.5, label=population)
        ax.set_xlabel(f"Per-sample mean of standardized {channel}")
        ax.set_ylabel("Probability density")
        ax.set_title(f"DC-mode audit: {channel}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"standardized_sample_mean_{channel}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    bundle = build_hit_data_bundle(**cfg["data"])

    all_rows = []
    contracts = {}

    rows, contracts["train"] = audit_real_population(
        bundle, "train", list(bundle.split_indices["train"])
    )
    all_rows.extend(rows)

    rows, contracts["test"] = audit_real_population(
        bundle, "test", list(bundle.split_indices["test"])
    )
    all_rows.extend(rows)

    rows, contracts["epoch22"] = audit_generated_population(
        bundle, "epoch22", Path(args.epoch22_dir)
    )
    all_rows.extend(rows)

    rows, contracts["epoch84"] = audit_generated_population(
        bundle, "epoch84", Path(args.epoch84_dir)
    )
    all_rows.extend(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_sample_channel_moments.csv", all_rows)
    plot_sample_means(output_dir, all_rows)

    stats = summarize(all_rows)
    artifact = {
        "version": 1,
        "contract": "d3_normalization_and_dc_mode_audit",
        "channel_names": list(STATE_CHANNEL_NAMES),
        "standardizer": {
            "mean": bundle.standardizer.mean.tolist(),
            "std": bundle.standardizer.std.tolist(),
            "fit_num_samples": int(bundle.standardizer.num_fit_samples),
        },
        "transform_checks": contracts,
        "population_statistics": stats,
        "outputs": {
            "per_sample_channel_moments": "per_sample_channel_moments.csv",
            "standardized_sample_mean_plots": [
                f"standardized_sample_mean_{name}.png" for name in STATE_CHANNEL_NAMES
            ],
        },
    }
    (output_dir / "normalization_dc_audit.json").write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print("NORMALIZATION / DC-MODE AUDIT COMPLETE")
    print("output:", output_dir)
    for population in ("train", "test", "epoch22", "epoch84"):
        print("\n", population, contracts[population])
        for name in STATE_CHANNEL_NAMES:
            values = stats[population][name]
            print(
                f"  {name}: standardized sample mean={values['standardized_sample_mean_average']:.6g} "
                f"+/-{values['standardized_sample_mean_std']:.6g}; "
                f"nondim mean={values['nondimensional_sample_mean_average']:.6g}"
            )


if __name__ == "__main__":
    main()
