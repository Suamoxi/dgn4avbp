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

from dgn4avbp.data.preprocessing import STATE_CHANNEL_NAMES
from dgn4avbp.hit_benchmark import (
    ensemble_energy_spectrum,
    infer_cartesian_grid_3d,
    isotropic_kinetic_energy_spectrum,
    reshape_nodes_to_grid,
)
from dgn4avbp.hit_pipeline import (
    build_diffusion_process,
    build_hit_data_bundle,
    build_hit_model,
    dimensional_positions,
    inverse_generated_state,
    load_yaml,
)
from dgn4avbp.improved_ddpm import ancestral_sample_step
from dgn4avbp.loader import Collater
from dgn4avbp.reverse_sampling import epsilon_to_x0


CHANNELS = tuple(STATE_CHANNEL_NAMES)
FIELDS = (*CHANNELS, "u", "v", "w")
AXES = ("x", "y", "z")
BANDS = {
    "low": (0.0, 0.25),
    "mid": (0.25, 0.50),
    "high": (0.50, 1.00),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ancestral rollout from a free-generated x_t with rollout from a "
            "true forward-corrupted test x_t, using identical reverse Gaussian noise."
        )
    )
    parser.add_argument("--config", default="configs/training/dgn_hit_baseline_c0_30e.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--restart-t", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=220030)
    parser.add_argument(
        "--trajectory-steps",
        nargs="+",
        type=int,
        default=[500, 250, 100, 50, 10, 0],
    )
    parser.add_argument("--output-root", required=True)
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows produced for {path}.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fields_from_state_nd(state_nd: np.ndarray) -> dict[str, np.ndarray]:
    state = np.asarray(state_nd, dtype=np.float64)
    result = {name: state[:, i] for i, name in enumerate(CHANNELS)}
    rho = state[:, 0]
    denominator = np.where(
        np.abs(rho) > 1.0e-8,
        rho,
        np.where(rho >= 0.0, 1.0e-8, -1.0e-8),
    )
    velocity = state[:, 1:4] / denominator[:, None]
    for i, name in enumerate(("u", "v", "w")):
        result[name] = velocity[:, i]
    return result


def seam_metrics(state_nd: np.ndarray, grid) -> list[dict]:
    rows = []
    for field, values in fields_from_state_nd(state_nd).items():
        full = reshape_nodes_to_grid(values, grid).astype(np.float64)
        std = float(np.std(values))
        denominator = max(std, np.finfo(np.float64).eps)
        axis_views = (
            np.moveaxis(full, 0, 0),
            np.moveaxis(full, 1, 0),
            np.moveaxis(full, 2, 0),
        )
        for axis, axis_values in zip(AXES, axis_views):
            duplicate_difference = axis_values[0] - axis_values[-1]
            wrap_difference = axis_values[0] - axis_values[-2]
            interior_difference = np.diff(axis_values[:-1], axis=0)

            duplicate_rms = float(np.sqrt(np.mean(duplicate_difference**2)))
            wrap_rms = float(np.sqrt(np.mean(wrap_difference**2)))
            interior_rms = float(np.sqrt(np.mean(interior_difference**2)))
            rows.append(
                {
                    "field": field,
                    "axis": axis,
                    "duplicate_endpoint_rms": duplicate_rms,
                    "duplicate_endpoint_rms_over_field_std": duplicate_rms / denominator,
                    "fft_wrap_jump_rms": wrap_rms,
                    "interior_adjacent_jump_rms": interior_rms,
                    "fft_wrap_over_interior_jump": wrap_rms
                    / max(interior_rms, np.finfo(np.float64).eps),
                }
            )
    return rows


def spectrum_band_energies(spectrum: dict) -> dict[str, float]:
    k = np.asarray(spectrum["k_over_k_nyquist"], dtype=np.float64)
    energy = np.asarray(spectrum["energy"], dtype=np.float64)
    result = {}
    for band, (lower, upper) in BANDS.items():
        mask = (k >= lower) & (k < upper)
        if upper == 1.0:
            mask = (k >= lower) & (k <= upper + 1.0e-12)
        result[band] = float(np.sum(energy[mask]))
    return result


def reference_band_energies(bundle, grid) -> dict[str, float]:
    reference = []
    for index in bundle.split_indices["test"]:
        sample = bundle.processed_dataset[index]
        reference.append(
            bundle.standardizer.inverse(sample.target).to(torch.float64).numpy()
        )
    spectrum = ensemble_energy_spectrum(
        np.stack(reference),
        grid,
        L_ref=bundle.refs.L_ref,
    )
    k = np.asarray(spectrum["k_over_k_nyquist"], dtype=np.float64)
    energy = np.asarray(spectrum["energy_mean"], dtype=np.float64)
    result = {}
    for band, (lower, upper) in BANDS.items():
        mask = (k >= lower) & (k < upper)
        if upper == 1.0:
            mask = (k >= lower) & (k <= upper + 1.0e-12)
        result[band] = float(np.sum(energy[mask]))
    return result


def trajectory_diagnostics(
    *,
    mode: str,
    sample_index: int,
    reverse_t: int,
    field_r: torch.Tensor,
    x0_hat: torch.Tensor,
    standardizer,
    grid,
    L_ref: float,
    reference_bands: dict[str, float],
) -> tuple[list[dict], list[dict], list[dict]]:
    state_rows = []
    seam_rows_out = []
    spectrum_rows = []

    x_t_cpu = field_r.detach().cpu().to(torch.float64)
    x0_cpu = x0_hat.detach().cpu().to(torch.float64)
    x0_nd = standardizer.inverse(x0_cpu).numpy()

    for channel, name in enumerate(CHANNELS):
        state_rows.append(
            {
                "mode": mode,
                "sample_index": sample_index,
                "reverse_t": reverse_t,
                "channel": name,
                "x_t_standardized_mean": float(x_t_cpu[:, channel].mean().item()),
                "x_t_standardized_std": float(x_t_cpu[:, channel].std(unbiased=False).item()),
                "x0_hat_standardized_mean": float(x0_cpu[:, channel].mean().item()),
                "x0_hat_standardized_std": float(x0_cpu[:, channel].std(unbiased=False).item()),
            }
        )

    for row in seam_metrics(x0_nd, grid):
        seam_rows_out.append(
            {
                "mode": mode,
                "sample_index": sample_index,
                "reverse_t": reverse_t,
                **row,
            }
        )

    spectrum = isotropic_kinetic_energy_spectrum(
        x0_nd,
        grid,
        L_ref=L_ref,
    )
    bands = spectrum_band_energies(spectrum)
    for band in BANDS:
        spectrum_rows.append(
            {
                "mode": mode,
                "sample_index": sample_index,
                "reverse_t": reverse_t,
                "band": band,
                "predicted_x0_energy": bands[band],
                "reference_energy": reference_bands[band],
                "energy_ratio_to_test": bands[band]
                / max(reference_bands[band], np.finfo(np.float64).eps),
            }
        )

    return state_rows, seam_rows_out, spectrum_rows


def save_final_population(
    *,
    output_dir: Path,
    state_std: torch.Tensor,
    sample_start: int,
    num_nodes: int,
    bundle,
    seed: int,
    checkpoint_path: Path,
    mode: str,
    restart_t: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pos_nd = bundle.processed_dataset.pos.cpu()
    pos_dim = dimensional_positions(bundle).cpu()
    batch_size = state_std.shape[0] // num_nodes

    for local_index in range(batch_size):
        sample_index = sample_start + local_index
        node_slice = slice(local_index * num_nodes, (local_index + 1) * num_nodes)
        one_std = state_std[node_slice].detach().cpu()
        state_nd, state_dim = inverse_generated_state(
            one_std,
            standardizer=bundle.standardizer,
            refs=bundle.refs,
        )
        torch.save(
            {
                "state_standardized": one_std,
                "state_nondimensional": state_nd,
                "state_dimensional": state_dim,
                "pos_nondimensional": pos_nd,
                "pos_dimensional": pos_dim,
                "cells": bundle.raw_dataset.cells.cpu(),
                "channel_names": list(CHANNELS),
                "seed": seed,
                "sample_index": sample_index,
                "checkpoint": str(checkpoint_path),
                "sampling_method": "ancestral",
                "restart_mode": mode,
                "restart_t": restart_t,
                "task": "on_manifold_restart_diagnostic",
            },
            output_dir / f"sample_{sample_index:05d}.pt",
        )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("--num-samples and --batch-size must be positive.")

    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("On-manifold restart diagnostic requires one CUDA GPU.")
    device = torch.device("cuda")

    cfg = load_yaml(args.config)
    bundle = build_hit_data_bundle(**cfg["data"])
    diffusion_cfg = load_yaml(cfg["diffusion_config"])
    model_cfg = load_yaml(cfg["model_config"])
    diffusion = build_diffusion_process(diffusion_cfg)

    if args.restart_t <= 0 or args.restart_t >= diffusion.num_steps:
        raise ValueError("--restart-t must be between 1 and T-1.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_hit_model(model_cfg, diffusion, device=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root exists: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    grid = infer_cartesian_grid_3d(bundle.processed_dataset.pos.cpu())
    reference_bands = reference_band_energies(bundle, grid)
    trajectory_steps = set(args.trajectory_steps)
    if args.restart_t not in trajectory_steps:
        trajectory_steps.add(args.restart_t)

    num_nodes = int(bundle.processed_dataset.pos.shape[0])
    state_rows = []
    seam_rows_all = []
    spectrum_rows_all = []

    for batch_start in range(0, args.num_samples, args.batch_size):
        current_batch = min(args.batch_size, args.num_samples - batch_start)
        real_samples = [
            deepcopy(bundle.test_dataset[batch_start + local_index])
            for local_index in range(current_batch)
        ]

        free_graph = Collater().collate([deepcopy(sample) for sample in real_samples]).to(device)
        real_graph = Collater().collate([deepcopy(sample) for sample in real_samples]).to(device)

        initial_generator = cuda_generator(args.seed + 1_000_000 + batch_start)
        reverse_generator = cuda_generator(args.seed + 2_000_000 + batch_start)
        forward_generator = cuda_generator(args.seed + 3_000_000 + batch_start)

        free_graph.field_r = torch.randn(
            (free_graph.batch.shape[0], model.num_fields),
            dtype=free_graph.pos.dtype,
            device=device,
            generator=initial_generator,
        )

        # Reproduce the free ancestral trajectory down to x_restart_t.
        for step in range(diffusion.num_steps - 1, args.restart_t, -1):
            free_graph.r = torch.full(
                (current_batch,), step, dtype=torch.long, device=device
            )
            epsilon, model_v = model(free_graph)
            gaussian_noise = torch.randn(
                free_graph.field_r.shape,
                dtype=free_graph.field_r.dtype,
                device=device,
                generator=reverse_generator,
            )
            free_graph.field_r = ancestral_sample_step(
                diffusion,
                field_r=free_graph.field_r,
                model_epsilon=epsilon,
                model_v=model_v,
                batch=free_graph.batch,
                r=free_graph.r,
                gaussian_noise=gaussian_noise,
            )

        # Construct an on-manifold x_restart_t by exactly forward-corrupting real x0.
        real_graph.field_start = real_graph.target
        real_graph.r = torch.full(
            (current_batch,), args.restart_t, dtype=torch.long, device=device
        )
        forward_noise = torch.randn(
            real_graph.target.shape,
            dtype=real_graph.target.dtype,
            device=device,
            generator=forward_generator,
        )
        sqrt_ab = diffusion.get_index_from_list(
            diffusion.sqrt_alphas_cumprod,
            real_graph.batch,
            real_graph.r,
        )
        sqrt_omab = diffusion.get_index_from_list(
            diffusion.sqrt_one_minus_alphas_cumprod,
            real_graph.batch,
            real_graph.r,
        )
        real_graph.field_r = sqrt_ab * real_graph.target + sqrt_omab * forward_noise

        # From restart_t to zero, both states receive the exact same Gaussian draw
        # at each reverse step. Differences therefore come from the state entering
        # the rollout, not from different reverse-noise realizations.
        for step in range(args.restart_t, -1, -1):
            shared_noise = torch.randn(
                free_graph.field_r.shape,
                dtype=free_graph.field_r.dtype,
                device=device,
                generator=reverse_generator,
            )

            for mode, graph in (
                ("free_x500", free_graph),
                ("real_forward_x500", real_graph),
            ):
                graph.r = torch.full(
                    (current_batch,), step, dtype=torch.long, device=device
                )
                epsilon, model_v = model(graph)

                if step in trajectory_steps:
                    x0_hat = epsilon_to_x0(
                        diffusion,
                        graph.field_r,
                        epsilon,
                        graph.batch,
                        graph.r,
                    )
                    for local_index in range(current_batch):
                        node_slice = slice(
                            local_index * num_nodes,
                            (local_index + 1) * num_nodes,
                        )
                        state_part, seam_part, spectrum_part = trajectory_diagnostics(
                            mode=mode,
                            sample_index=batch_start + local_index,
                            reverse_t=step,
                            field_r=graph.field_r[node_slice],
                            x0_hat=x0_hat[node_slice],
                            standardizer=bundle.standardizer,
                            grid=grid,
                            L_ref=bundle.refs.L_ref,
                            reference_bands=reference_bands,
                        )
                        state_rows.extend(state_part)
                        seam_rows_all.extend(seam_part)
                        spectrum_rows_all.extend(spectrum_part)

                graph.field_r = ancestral_sample_step(
                    diffusion,
                    field_r=graph.field_r,
                    model_epsilon=epsilon,
                    model_v=model_v,
                    batch=graph.batch,
                    r=graph.r,
                    gaussian_noise=shared_noise,
                )

        save_final_population(
            output_dir=output_root / "free_x500",
            state_std=free_graph.field_r,
            sample_start=batch_start,
            num_nodes=num_nodes,
            bundle=bundle,
            seed=args.seed,
            checkpoint_path=checkpoint_path,
            mode="free_x500",
            restart_t=args.restart_t,
        )
        save_final_population(
            output_dir=output_root / "real_forward_x500",
            state_std=real_graph.field_r,
            sample_start=batch_start,
            num_nodes=num_nodes,
            bundle=bundle,
            seed=args.seed,
            checkpoint_path=checkpoint_path,
            mode="real_forward_x500",
            restart_t=args.restart_t,
        )

    write_csv(output_root / "restart_state_trajectory.csv", state_rows)
    write_csv(output_root / "restart_seam_trajectory.csv", seam_rows_all)
    write_csv(output_root / "restart_spectrum_trajectory.csv", spectrum_rows_all)

    metadata = {
        "version": 1,
        "contract": "paired_on_manifold_restart_at_fixed_t",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["progress"]["epoch"]),
        "restart_t": int(args.restart_t),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "base_seed": int(args.seed),
        "free_initial_noise_matches_reverse_sampler_comparison_convention": True,
        "free_reverse_noise_matches_reverse_sampler_comparison_convention": True,
        "same_reverse_noise_from_restart_t_to_zero": True,
        "real_restart_definition": (
            "x_t = sqrt(alpha_bar_t) * real_test_x0 + "
            "sqrt(1-alpha_bar_t) * fixed_gaussian_noise"
        ),
        "trajectory_steps": sorted(trajectory_steps, reverse=True),
        "reference_spectral_bands": reference_bands,
    }
    (output_root / "restart_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("ON-MANIFOLD RESTART DIAGNOSTIC COMPLETE")
    print("output root:", output_root)


if __name__ == "__main__":
    main()
