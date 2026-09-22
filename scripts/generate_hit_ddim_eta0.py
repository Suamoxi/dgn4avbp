#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from dgn4avbp.hit_pipeline import (
    build_diffusion_process,
    build_hit_data_bundle,
    build_hit_model,
    dimensional_positions,
    inverse_generated_state,
    load_yaml,
)
from dgn4avbp.loader import Collater
from dgn4avbp.reverse_sampling import ddim_eta_zero_step


CHANNELS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate unconditional HIT samples with deterministic DDIM eta=0."
    )
    parser.add_argument("--config", default="configs/training/dgn_hit_baseline_c0_30e.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=220030)
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


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("--num-samples and --batch-size must be positive.")

    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("DDIM eta=0 diagnostic requires one CUDA GPU.")
    device = torch.device("cuda")

    cfg = load_yaml(args.config)
    bundle = build_hit_data_bundle(**cfg["data"])
    model_cfg = load_yaml(cfg["model_config"])
    diffusion_cfg = load_yaml(cfg["diffusion_config"])
    diffusion = build_diffusion_process(diffusion_cfg)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_hit_model(model_cfg, diffusion, device=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    template = bundle.test_dataset[0]
    num_nodes = int(template.pos.shape[0])
    pos_nd = bundle.processed_dataset.pos.cpu()
    pos_dim = dimensional_positions(bundle).cpu()

    for batch_start in range(0, args.num_samples, args.batch_size):
        current_batch = min(args.batch_size, args.num_samples - batch_start)
        graph = Collater().collate([deepcopy(template) for _ in range(current_batch)]).to(device)

        # Match the initial-noise convention used by compare_reverse_samplers.py,
        # so DDIM and the prior ancestral/SDE/ODE populations start from the same x_T.
        initial_seed = int(args.seed + 1_000_000 + batch_start)
        initial_generator = cuda_generator(initial_seed)
        graph.field_r = torch.randn(
            (graph.batch.shape[0], model.num_fields),
            dtype=graph.pos.dtype,
            device=device,
            generator=initial_generator,
        )

        for step in range(diffusion.num_steps - 1, -1, -1):
            graph.r = torch.full(
                (current_batch,),
                step,
                dtype=torch.long,
                device=device,
            )
            model_epsilon, _ = model(graph)
            graph.field_r = ddim_eta_zero_step(
                diffusion,
                field_r=graph.field_r,
                model_epsilon=model_epsilon,
                batch=graph.batch,
                r=graph.r,
            )

        state_std = graph.field_r.detach().cpu()
        for local_index in range(current_batch):
            sample_index = batch_start + local_index
            node_slice = slice(local_index * num_nodes, (local_index + 1) * num_nodes)
            one_std = state_std[node_slice]
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
                    "seed": args.seed,
                    "sample_index": sample_index,
                    "diffusion_steps": diffusion.num_steps,
                    "sampling_steps": "full_1000",
                    "sampling_method": "ddim_eta_zero",
                    "initial_noise_seed": initial_seed,
                    "checkpoint": str(checkpoint_path),
                    "task": "deterministic_ddim_eta_zero_HIT_diagnostic",
                },
                output_dir / f"sample_{sample_index:05d}.pt",
            )

    print("DDIM ETA=0 GENERATION COMPLETE")
    print("output:", output_dir)


if __name__ == "__main__":
    main()
