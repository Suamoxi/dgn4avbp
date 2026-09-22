#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from dgn4avbp.hit_benchmark import infer_cartesian_grid_3d, periodic_unique_grid
from dgn4avbp.hit_pipeline import (
    build_diffusion_process,
    build_hit_data_bundle,
    build_hit_model,
    dimensional_positions,
    inverse_generated_state,
    load_yaml,
)
from dgn4avbp.improved_ddpm import ancestral_sample_step, learned_range_log_variance
from dgn4avbp.loader import Collater
from dgn4avbp.reverse_sampling import (
    epsilon_to_x0,
    vp_probability_flow_ode_euler_step,
    vp_reverse_sde_euler_step,
)


METHODS = ("ancestral", "vp_sde_euler", "vp_probability_flow_ode_euler")
CHANNELS = ("rho", "rhou", "rhov", "rhow", "rhoE")
BANDS = {
    "low": (0.0, 0.25),
    "mid": (0.25, 0.50),
    "high": (0.50, 1.00),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled reverse-sampler comparison using one trained DDPM epsilon network: "
            "Improved-DDPM ancestral, VP reverse-SDE Euler-Maruyama, and VP probability-flow ODE Euler."
        )
    )
    parser.add_argument("--config", default="configs/training/dgn_hit_baseline_c0_30e.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=220030)
    parser.add_argument(
        "--trajectory-steps",
        nargs="+",
        type=int,
        default=[999, 900, 750, 500, 250, 100, 50, 10, 0],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    return generator


def spectral_geometry(pos: torch.Tensor):
    grid = infer_cartesian_grid_3d(pos)
    k_axes = [
        2.0 * np.pi * np.fft.fftfreq(n, d=dx)
        for n, dx in zip(grid.periodic_shape, grid.spacing)
    ]
    kx, ky, kz = np.meshgrid(*k_axes, indexing="ij")
    k_magnitude = np.sqrt(kx**2 + ky**2 + kz**2)
    return grid, k_magnitude, float(grid.k_nyquist_min)


def scalar_band_energies(
    field: np.ndarray,
    *,
    grid,
    k_magnitude: np.ndarray,
    k_nyquist: float,
) -> dict[str, float]:
    values = periodic_unique_grid(field, grid).astype(np.float64)
    values = values - np.mean(values)
    fft_values = np.fft.fftn(values, norm="forward")
    modal_energy = np.abs(fft_values) ** 2

    result = {}
    k_norm = k_magnitude / k_nyquist
    for name, (lower, upper) in BANDS.items():
        mask = (k_norm >= lower) & (k_norm < upper)
        if upper == 1.0:
            mask = (k_norm >= lower) & (k_norm <= upper + 1.0e-12)
        mask &= k_magnitude > 0.0
        result[name] = float(np.sum(modal_energy[mask]))
    return result


def trajectory_rows(
    *,
    method: str,
    batch_start: int,
    reverse_t: int,
    graph,
    x0_hat: torch.Tensor,
    model_epsilon: torch.Tensor,
    model_v: torch.Tensor,
    learned_variance: torch.Tensor,
    num_nodes: int,
    grid,
    k_magnitude: np.ndarray,
    k_nyquist: float,
) -> list[dict]:
    field_r = graph.field_r.detach().cpu().to(torch.float64)
    x0_cpu = x0_hat.detach().cpu().to(torch.float64)
    eps_cpu = model_epsilon.detach().cpu().to(torch.float64)
    v_cpu = model_v.detach().cpu().to(torch.float64)
    variance_cpu = learned_variance.detach().cpu().to(torch.float64)

    batch_size = int(graph.batch.max().item()) + 1
    rows = []
    for local_index in range(batch_size):
        node_slice = slice(local_index * num_nodes, (local_index + 1) * num_nodes)
        sample_index = batch_start + local_index
        for channel, channel_name in enumerate(CHANNELS):
            x0_channel = x0_cpu[node_slice, channel].numpy()
            bands = scalar_band_energies(
                x0_channel,
                grid=grid,
                k_magnitude=k_magnitude,
                k_nyquist=k_nyquist,
            )
            rows.append(
                {
                    "method": method,
                    "sample_index": sample_index,
                    "reverse_t": reverse_t,
                    "channel": channel_name,
                    "x_t_mean": float(field_r[node_slice, channel].mean().item()),
                    "x_t_std": float(field_r[node_slice, channel].std(unbiased=False).item()),
                    "x0_hat_mean": float(x0_cpu[node_slice, channel].mean().item()),
                    "x0_hat_std": float(x0_cpu[node_slice, channel].std(unbiased=False).item()),
                    "epsilon_mean": float(eps_cpu[node_slice, channel].mean().item()),
                    "epsilon_std": float(eps_cpu[node_slice, channel].std(unbiased=False).item()),
                    "model_v_mean": float(v_cpu[node_slice, channel].mean().item()),
                    "model_v_std": float(v_cpu[node_slice, channel].std(unbiased=False).item()),
                    "learned_variance_mean": float(
                        variance_cpu[node_slice, channel].mean().item()
                    ),
                    "x0_hat_low_band_energy": bands["low"],
                    "x0_hat_mid_band_energy": bands["mid"],
                    "x0_hat_high_band_energy": bands["high"],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No trajectory rows were produced.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def generate_method(
    *,
    method: str,
    model,
    diffusion,
    template,
    bundle,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    seed: int,
    trajectory_steps: set[int],
    grid,
    k_magnitude: np.ndarray,
    k_nyquist: float,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=False)
    num_nodes = int(template.pos.shape[0])
    pos_nd = bundle.processed_dataset.pos.cpu()
    pos_dim = dimensional_positions(bundle).cpu()
    trajectory = []

    for batch_start in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - batch_start)
        graph = Collater().collate([deepcopy(template) for _ in range(current_batch)])
        graph = graph.to(model.device)

        initial_seed = int(seed + 1_000_000 + batch_start)
        reverse_seed = int(seed + 2_000_000 + batch_start)
        initial_generator = cuda_generator(initial_seed)
        reverse_generator = cuda_generator(reverse_seed)

        graph.field_r = torch.randn(
            (graph.batch.shape[0], model.num_fields),
            dtype=graph.pos.dtype,
            device=model.device,
            generator=initial_generator,
        )

        for step in range(diffusion.num_steps - 1, -1, -1):
            graph.r = torch.full(
                (current_batch,),
                step,
                dtype=torch.long,
                device=model.device,
            )
            model_epsilon, model_v = model(graph)

            if step in trajectory_steps:
                x0_hat = epsilon_to_x0(
                    diffusion,
                    graph.field_r,
                    model_epsilon,
                    graph.batch,
                    graph.r,
                )
                learned_variance = torch.exp(
                    learned_range_log_variance(
                        diffusion,
                        model_v,
                        graph.batch,
                        graph.r,
                    )
                )
                trajectory.extend(
                    trajectory_rows(
                        method=method,
                        batch_start=batch_start,
                        reverse_t=step,
                        graph=graph,
                        x0_hat=x0_hat,
                        model_epsilon=model_epsilon,
                        model_v=model_v,
                        learned_variance=learned_variance,
                        num_nodes=num_nodes,
                        grid=grid,
                        k_magnitude=k_magnitude,
                        k_nyquist=k_nyquist,
                    )
                )

            gaussian_noise = None
            if method in ("ancestral", "vp_sde_euler"):
                gaussian_noise = torch.randn(
                    graph.field_r.shape,
                    dtype=graph.field_r.dtype,
                    device=model.device,
                    generator=reverse_generator,
                )

            if method == "ancestral":
                graph.field_r = ancestral_sample_step(
                    diffusion,
                    field_r=graph.field_r,
                    model_epsilon=model_epsilon,
                    model_v=model_v,
                    batch=graph.batch,
                    r=graph.r,
                    gaussian_noise=gaussian_noise,
                )
            elif method == "vp_sde_euler":
                assert gaussian_noise is not None
                graph.field_r = vp_reverse_sde_euler_step(
                    diffusion,
                    field_r=graph.field_r,
                    model_epsilon=model_epsilon,
                    batch=graph.batch,
                    r=graph.r,
                    gaussian_noise=gaussian_noise,
                )
            elif method == "vp_probability_flow_ode_euler":
                graph.field_r = vp_probability_flow_ode_euler_step(
                    diffusion,
                    field_r=graph.field_r,
                    model_epsilon=model_epsilon,
                    batch=graph.batch,
                    r=graph.r,
                )
            else:
                raise ValueError(f"Unknown method {method!r}.")

        state_std = graph.field_r.detach().cpu()
        for local_index in range(current_batch):
            sample_index = batch_start + local_index
            node_slice = slice(local_index * num_nodes, (local_index + 1) * num_nodes)
            state_std_one = state_std[node_slice]
            state_nd, state_dim = inverse_generated_state(
                state_std_one,
                standardizer=bundle.standardizer,
                refs=bundle.refs,
            )
            payload = {
                "state_standardized": state_std_one,
                "state_nondimensional": state_nd,
                "state_dimensional": state_dim,
                "pos_nondimensional": pos_nd,
                "pos_dimensional": pos_dim,
                "cells": bundle.raw_dataset.cells.cpu(),
                "channel_names": list(CHANNELS),
                "seed": seed,
                "sample_index": sample_index,
                "diffusion_steps": diffusion.num_steps,
                "sampling_steps": "full_1000",
                "sampling_method": method,
                "initial_noise_seed": initial_seed,
                "reverse_noise_seed": reverse_seed if method != "vp_probability_flow_ode_euler" else None,
                "checkpoint": str(model._diagnostic_checkpoint_path),
                "task": "reverse_sampler_diagnostic_unconditional_HIT",
            }
            torch.save(payload, output_dir / f"sample_{sample_index:05d}.pt")

    return trajectory


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("--num-samples and --batch-size must be positive.")

    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Reverse-sampler comparison requires one CUDA GPU.")
    device = torch.device("cuda")

    cfg = load_yaml(args.config)
    bundle = build_hit_data_bundle(**cfg["data"])
    model_cfg = load_yaml(cfg["model_config"])
    diffusion_cfg = load_yaml(cfg["diffusion_config"])
    diffusion = build_diffusion_process(diffusion_cfg)
    model = build_hit_model(model_cfg, diffusion, device=device)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model._diagnostic_checkpoint_path = checkpoint_path

    trajectory_steps = set(int(step) for step in args.trajectory_steps)
    invalid_steps = sorted(
        step for step in trajectory_steps if step < 0 or step >= diffusion.num_steps
    )
    if invalid_steps:
        raise ValueError(f"Invalid trajectory steps: {invalid_steps}")

    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root exists: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    template = bundle.test_dataset[0]
    grid, k_magnitude, k_nyquist = spectral_geometry(bundle.processed_dataset.pos.cpu())

    all_trajectory_rows = []
    for method in METHODS:
        print(f"Generating method: {method}", flush=True)
        rows = generate_method(
            method=method,
            model=model,
            diffusion=diffusion,
            template=template,
            bundle=bundle,
            output_dir=output_root / method,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            trajectory_steps=trajectory_steps,
            grid=grid,
            k_magnitude=k_magnitude,
            k_nyquist=k_nyquist,
        )
        all_trajectory_rows.extend(rows)

    write_csv(output_root / "reverse_trajectory.csv", all_trajectory_rows)

    metadata = {
        "version": 1,
        "contract": "controlled_reverse_sampler_comparison",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["progress"]["epoch"]),
        "methods": list(METHODS),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "base_seed": int(args.seed),
        "common_initial_noise_across_methods": True,
        "common_reverse_gaussian_noise_between_ancestral_and_vp_sde": True,
        "probability_flow_ode_is_deterministic_given_initial_noise": True,
        "vp_mapping": {
            "score": "-epsilon_theta / sqrt(1-alpha_bar_t)",
            "integrated_beta_per_interval": "-log(alpha_t)",
            "vp_sde_solver": "backward Euler-Maruyama over all 1000 DDPM intervals",
            "vp_ode_solver": "backward Euler over all 1000 DDPM intervals",
            "learned_variance_used_by": ["ancestral"],
            "learned_variance_ignored_by": ["vp_sde_euler", "vp_probability_flow_ode_euler"],
        },
        "trajectory_steps": sorted(trajectory_steps, reverse=True),
        "state_space": "standardized during reverse process; saved outputs also inverse-transformed",
    }
    (output_root / "comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("CONTROLLED REVERSE-SAMPLER GENERATION COMPLETE")
    print("output root:", output_root)


if __name__ == "__main__":
    main()
