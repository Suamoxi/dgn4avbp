#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from dgn4avbp.diffusion_process import DiffusionStepsGenerator
from dgn4avbp.hit_pipeline import (
    build_diffusion_process,
    build_hit_data_bundle,
    build_hit_model,
    dimensional_positions,
    inverse_generated_state,
    load_yaml,
)
from dgn4avbp.improved_ddpm import sample_unconditional_physical_dgn
from dgn4avbp.loader import Collater


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate unconditional HIT samples with the D12 DGN.")
    parser.add_argument("--config", default="configs/training/dgn_hit_baseline.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-spaced-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    generation_cfg = cfg["generation"]
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    _seed_everything(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("D12 canonical generation currently requires one CUDA GPU.")
    device = torch.device("cuda")

    bundle = build_hit_data_bundle(**cfg["data"])
    model_cfg = load_yaml(cfg["model_config"])
    diffusion_cfg = load_yaml(cfg["diffusion_config"])
    diffusion = build_diffusion_process(diffusion_cfg)
    model = build_hit_model(model_cfg, diffusion, device=device)

    checkpoint_path = Path(args.checkpoint or generation_cfg["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Generation checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    num_samples = int(
        args.num_samples if args.num_samples is not None else generation_cfg["num_samples"]
    )
    batch_size = int(args.batch_size if args.batch_size is not None else generation_cfg["batch_size"])
    if num_samples <= 0 or batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive.")

    configured_spaced = generation_cfg.get("num_spaced_steps")
    num_spaced_steps = args.num_spaced_steps if args.num_spaced_steps is not None else configured_spaced
    steps = None
    if num_spaced_steps is not None:
        num_spaced_steps = int(num_spaced_steps)
        if num_spaced_steps < 2:
            raise ValueError("num_spaced_steps must be at least 2 when specified.")
        generator = DiffusionStepsGenerator(
            type=str(generation_cfg.get("spaced_schedule", "linear")),
            base_diffusion_steps=diffusion.num_steps,
        )
        steps = generator(num_spaced_steps)

    output_dir = Path(args.output_dir or generation_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # The task is unconditional; the chosen test snapshot contributes only the
    # frozen physical mesh/hierarchy used as the generation domain.
    template = bundle.test_dataset[0]
    num_nodes = int(template.pos.shape[0])
    pos_nd = bundle.processed_dataset.pos.cpu()
    pos_dim = dimensional_positions(bundle).cpu()

    sample_index = 0
    while sample_index < num_samples:
        current_batch = min(batch_size, num_samples - sample_index)
        graph_batch = Collater().collate([deepcopy(template) for _ in range(current_batch)])
        state_std = sample_unconditional_physical_dgn(model, graph_batch, steps=steps).cpu()

        for local_index in range(current_batch):
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
                "channel_names": ["rho", "rhou", "rhov", "rhow", "rhoE"],
                "seed": seed,
                "sample_index": sample_index,
                "diffusion_steps": diffusion.num_steps,
                "sampling_steps": steps if steps is not None else "full_ancestral_1000",
                "checkpoint": str(checkpoint_path),
                "task": "unconditional_equilibrium_HIT_generation",
            }
            output_path = output_dir / f"sample_{sample_index:05d}.pt"
            torch.save(payload, output_path)
            print(output_path)
            sample_index += 1


if __name__ == "__main__":
    main()
