#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dgn4avbp.data.preprocessing import STATE_CHANNEL_NAMES
from dgn4avbp.hit_benchmark import infer_cartesian_grid_3d, reshape_nodes_to_grid
from dgn4avbp.hit_pipeline import build_hit_data_bundle, load_yaml


AXES = ("x", "y", "z")
FIELDS = (*STATE_CHANNEL_NAMES, "u", "v", "w")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure periodic endpoint-plane mismatch in real and generated HIT fields."
    )
    parser.add_argument("--config", default="configs/training/dgn_hit_baseline_c0_30e.yaml")
    parser.add_argument(
        "--population",
        action="append",
        default=[],
        help="Generated population as label=directory; repeatable.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rho-floor", type=float, default=1.0e-8)
    return parser.parse_args()


def parse_population(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected label=directory, got {spec!r}.")
    label, directory = spec.split("=", 1)
    if not label:
        raise ValueError(f"Population label is empty in {spec!r}.")
    return label, Path(directory)


def fields_from_state(state_nd: np.ndarray, rho_floor: float) -> dict[str, np.ndarray]:
    state = np.asarray(state_nd, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 5:
        raise ValueError(f"Expected state [N,5], got {state.shape}.")
    result = {
        name: state[:, index]
        for index, name in enumerate(STATE_CHANNEL_NAMES)
    }
    rho = state[:, 0]
    if np.any(np.abs(rho) <= rho_floor):
        raise ValueError("Density is too close to zero for velocity seam diagnostics.")
    velocity = state[:, 1:4] / rho[:, None]
    for index, name in enumerate(("u", "v", "w")):
        result[name] = velocity[:, index]
    return result


def seam_rows(
    *,
    population: str,
    sample_id: str,
    state_nd: np.ndarray,
    grid,
    rho_floor: float,
) -> list[dict]:
    rows = []
    for field_name, values in fields_from_state(state_nd, rho_floor).items():
        full = reshape_nodes_to_grid(values, grid).astype(np.float64)
        field_std = float(np.std(values))
        denominator = max(field_std, np.finfo(np.float64).eps)

        plane_pairs = (
            (full[0, :, :], full[-1, :, :]),
            (full[:, 0, :], full[:, -1, :]),
            (full[:, :, 0], full[:, :, -1]),
        )
        for axis, (minimum, maximum) in zip(AXES, plane_pairs):
            difference = minimum - maximum
            rms = float(np.sqrt(np.mean(difference**2)))
            mean_abs = float(np.mean(np.abs(difference)))
            max_abs = float(np.max(np.abs(difference)))
            rows.append(
                {
                    "population": population,
                    "sample_id": sample_id,
                    "field": field_name,
                    "axis": axis,
                    "field_std": field_std,
                    "seam_rms": rms,
                    "seam_mean_abs": mean_abs,
                    "seam_max_abs": max_abs,
                    "seam_rms_over_field_std": rms / denominator,
                }
            )
    return rows


def load_generated(directory: Path) -> list[tuple[str, np.ndarray]]:
    files = sorted(directory.glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"No sample_*.pt files found in {directory}.")
    samples = []
    for path in files:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        state = payload.get("state_nondimensional")
        if not isinstance(state, torch.Tensor):
            raise ValueError(f"Missing state_nondimensional in {path}.")
        samples.append((path.name, state.to(torch.float64).numpy()))
    return samples


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["population"], row["field"], row["axis"])].append(row)

    summary = []
    for (population, field, axis), values in sorted(grouped.items()):
        normalized = np.asarray(
            [item["seam_rms_over_field_std"] for item in values],
            dtype=np.float64,
        )
        raw = np.asarray([item["seam_rms"] for item in values], dtype=np.float64)
        summary.append(
            {
                "population": population,
                "field": field,
                "axis": axis,
                "num_samples": len(values),
                "mean_seam_rms": float(np.mean(raw)),
                "mean_normalized_seam_rms": float(np.mean(normalized)),
                "median_normalized_seam_rms": float(np.median(normalized)),
                "q95_normalized_seam_rms": float(np.quantile(normalized, 0.95)),
                "max_normalized_seam_rms": float(np.max(normalized)),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    bundle = build_hit_data_bundle(**cfg["data"])
    grid = infer_cartesian_grid_3d(bundle.processed_dataset.pos.cpu())

    rows = []
    for index in bundle.split_indices["test"]:
        processed = bundle.processed_dataset[index]
        state_nd = bundle.standardizer.inverse(processed.target).to(torch.float64).numpy()
        rows.extend(
            seam_rows(
                population="test",
                sample_id=str(processed.sample_id),
                state_nd=state_nd,
                grid=grid,
                rho_floor=args.rho_floor,
            )
        )

    for spec in args.population:
        label, directory = parse_population(spec)
        for sample_id, state_nd in load_generated(directory):
            rows.extend(
                seam_rows(
                    population=label,
                    sample_id=sample_id,
                    state_nd=state_nd,
                    grid=grid,
                    rho_floor=args.rho_floor,
                )
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "periodic_seam_samples.csv", rows)
    summary = summarize(rows)
    write_csv(output_dir / "periodic_seam_summary.csv", summary)

    print("PERIODIC SEAM DIAGNOSTIC COMPLETE")
    print("sample rows:", output_dir / "periodic_seam_samples.csv")
    print("summary:", output_dir / "periodic_seam_summary.csv")
    for row in summary:
        if row["field"] in ("u", "v", "w"):
            print(row)


if __name__ == "__main__":
    main()
