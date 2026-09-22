#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dgn4avbp.data.preprocessing import STATE_CHANNEL_NAMES
from dgn4avbp.hit_benchmark import infer_cartesian_grid_3d, periodic_unique_grid
from dgn4avbp.hit_pipeline import build_diffusion_process, build_hit_data_bundle, build_hit_model, load_yaml
from dgn4avbp.loader import Collater


BANDS = {
    "low": (0.0, 0.25),
    "mid": (0.25, 0.50),
    "high": (0.50, 1.00),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure DGN denoising error versus diffusion timestep and wavenumber.")
    p.add_argument("--training-config", default="configs/training/dgn_hit_baseline_c0_30e.yaml")
    p.add_argument("--checkpoint", action="append", required=True, help="label=path; repeatable")
    p.add_argument("--timesteps", nargs="+", type=int, default=[10, 50, 100, 250, 500, 750, 900])
    p.add_argument("--num-test-samples", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=424242)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def checkpoint_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected label=path, got {spec!r}")
    label, value = spec.split("=", 1)
    return label, Path(value)


def spectral_geometry(pos: torch.Tensor):
    grid = infer_cartesian_grid_3d(pos)
    shape = grid.periodic_shape
    k_axes = [2.0 * np.pi * np.fft.fftfreq(n, d=dx) for n, dx in zip(shape, grid.spacing)]
    kx, ky, kz = np.meshgrid(*k_axes, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    knyq = grid.k_nyquist_min
    return grid, kmag, knyq


def band_error_ratios(true_field: np.ndarray, error_field: np.ndarray, grid, kmag, knyq) -> dict[str, float]:
    true_grid = periodic_unique_grid(true_field, grid).astype(np.float64)
    err_grid = periodic_unique_grid(error_field, grid).astype(np.float64)

    true_grid = true_grid - np.mean(true_grid)
    err_grid = err_grid - np.mean(err_grid)

    true_fft = np.fft.fftn(true_grid, norm="forward")
    err_fft = np.fft.fftn(err_grid, norm="forward")
    true_energy = np.abs(true_fft) ** 2
    err_energy = np.abs(err_fft) ** 2

    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (kmag > lo * knyq) & (kmag <= hi * knyq)
        denominator = float(np.sum(true_energy[mask]))
        numerator = float(np.sum(err_energy[mask]))
        out[name] = numerator / denominator if denominator > 0.0 else float("nan")
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.num_test_samples <= 0:
        raise ValueError("--num-test-samples must be positive.")

    cfg = load_yaml(args.training_config)
    bundle = build_hit_data_bundle(**cfg["data"])
    diffusion_cfg = load_yaml(cfg["diffusion_config"])
    model_cfg = load_yaml(cfg["model_config"])
    diffusion = build_diffusion_process(diffusion_cfg)

    for t in args.timesteps:
        if t < 0 or t >= diffusion.num_steps:
            raise ValueError(f"Timestep {t} outside [0,{diffusion.num_steps - 1}].")

    if not torch.cuda.is_available():
        raise RuntimeError("Denoising spectrum diagnostic requires CUDA.")
    device = torch.device("cuda")

    grid, kmag, knyq = spectral_geometry(bundle.processed_dataset.pos.cpu())

    rows = []
    summaries = {}

    sample_count = min(args.num_test_samples, len(bundle.test_dataset))

    for spec in args.checkpoint:
        label, path = checkpoint_spec(spec)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        epoch = int(checkpoint["progress"]["epoch"])

        model = build_hit_model(model_cfg, diffusion, device=device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        accum: dict[tuple[int, int], list[dict]] = {}

        for local_index in range(sample_count):
            sample = deepcopy(bundle.test_dataset[local_index])
            true_std_cpu = sample.target.to(torch.float32).cpu()
            true_std_np = true_std_cpu.to(torch.float64).numpy()

            for t in args.timesteps:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(args.base_seed + 100000 * local_index + t))
                noise_cpu = torch.randn(
                    true_std_cpu.shape,
                    generator=generator,
                    dtype=true_std_cpu.dtype,
                )

                sqrt_ab = float(diffusion.sqrt_alphas_cumprod[t])
                sqrt_omab = float(diffusion.sqrt_one_minus_alphas_cumprod[t])
                xt_cpu = sqrt_ab * true_std_cpu + sqrt_omab * noise_cpu

                graph = Collater().collate([deepcopy(sample)]).to(device)
                graph.field_start = graph.target
                graph.field_r = xt_cpu.to(device)
                graph.noise = noise_cpu.to(device)
                graph.r = torch.tensor([t], dtype=torch.long, device=device)

                eps_pred, _ = model(graph)
                x0_hat = (
                    graph.field_r - sqrt_omab * eps_pred
                ) / sqrt_ab

                eps_pred_np = eps_pred.detach().cpu().to(torch.float64).numpy()
                x0_hat_np = x0_hat.detach().cpu().to(torch.float64).numpy()
                noise_np = noise_cpu.to(torch.float64).numpy()

                for channel, name in enumerate(STATE_CHANNEL_NAMES):
                    error = x0_hat_np[:, channel] - true_std_np[:, channel]
                    true_std_value = float(np.std(true_std_np[:, channel]))
                    ratios = band_error_ratios(
                        true_std_np[:, channel],
                        error,
                        grid,
                        kmag,
                        knyq,
                    )
                    record = {
                        "epsilon_mse": float(np.mean((eps_pred_np[:, channel] - noise_np[:, channel]) ** 2)),
                        "x0_mse": float(np.mean(error**2)),
                        "dc_bias_over_true_std": (
                            float(np.mean(error)) / true_std_value if true_std_value > 0.0 else float("nan")
                        ),
                        "low_relative_error_energy": ratios["low"],
                        "mid_relative_error_energy": ratios["mid"],
                        "high_relative_error_energy": ratios["high"],
                    }
                    accum.setdefault((t, channel), []).append(record)

        summary_rows = []
        for (t, channel), records in sorted(accum.items()):
            row = {
                "label": label,
                "checkpoint_epoch": epoch,
                "timestep": t,
                "channel": STATE_CHANNEL_NAMES[channel],
            }
            for key in records[0]:
                row[key] = float(np.mean([record[key] for record in records]))
            rows.append(row)
            summary_rows.append(row)

        summaries[label] = {
            "checkpoint": str(path),
            "epoch": epoch,
            "num_test_samples": sample_count,
        }

        del model
        torch.cuda.empty_cache()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "denoising_spectrum_by_timestep.csv", rows)
    (out / "denoising_spectrum_summary.json").write_text(
        json.dumps(
            {
                "version": 1,
                "contract": "standardized_x0_reconstruction_error_by_timestep_and_wavenumber",
                "timesteps": args.timesteps,
                "bands_k_over_knyquist": BANDS,
                "checkpoints": summaries,
            },
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    for channel in STATE_CHANNEL_NAMES:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for label in summaries:
            selected = [r for r in rows if r["label"] == label and r["channel"] == channel]
            selected.sort(key=lambda r: r["timestep"])
            for band, key in (
                ("low", "low_relative_error_energy"),
                ("mid", "mid_relative_error_energy"),
                ("high", "high_relative_error_energy"),
            ):
                ax.plot(
                    [r["timestep"] for r in selected],
                    [r[key] for r in selected],
                    marker="o",
                    label=f"{label} {band}",
                )
        ax.set_xlabel("Diffusion timestep")
        ax.set_ylabel("Error spectral energy / true spectral energy")
        ax.set_yscale("log")
        ax.set_title(f"Denoising spectral error: {channel}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"denoising_spectral_error_{channel}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    print("DENOISING SPECTRUM DIAGNOSTIC COMPLETE")
    print("output:", out / "denoising_spectrum_by_timestep.csv")


if __name__ == "__main__":
    main()
