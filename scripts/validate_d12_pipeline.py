#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from dgn4avbp.checkpointing import load_training_checkpoint, save_training_checkpoint
from dgn4avbp.diffusion_process import DiffusionStepsGenerator
from dgn4avbp.hit_pipeline import (
    build_diffusion_process,
    build_hit_data_bundle,
    build_hit_model,
    build_hybrid_loss,
    build_training_sampler,
    dimensional_positions,
    inverse_generated_state,
    load_yaml,
    prepare_validation_batch,
    train_one_batch,
    validate_one_batch,
)
from dgn4avbp.improved_ddpm import sample_unconditional_physical_dgn
from dgn4avbp.loader import Collater, DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the D12 HIT train/val/generation pipeline.")
    parser.add_argument("--config", default="configs/training/dgn_hit_baseline.yaml")
    parser.add_argument("--checkpoint", default="artifacts/d12_smoke_checkpoint.pt")
    parser.add_argument("--sample", default="artifacts/d12_smoke_generation.pt")
    parser.add_argument("--manifest", default="artifacts/d12_pipeline_manifest.json")
    parser.add_argument("--generation-steps", type=int, default=4)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _state_dict_exact(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[key].detach().cpu(), b[key].detach().cpu()) for key in a)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    _seed_everything(seed)

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("D12 validation requested CUDA but CUDA is unavailable.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = build_hit_data_bundle(**cfg["data"])
    if [len(bundle.train_dataset), len(bundle.val_dataset), len(bundle.test_dataset)] != [1009, 126, 126]:
        raise AssertionError("D12 did not recover the frozen D2 split sizes 1009/126/126.")

    model_cfg = load_yaml(cfg["model_config"])
    diffusion_cfg = load_yaml(cfg["diffusion_config"])
    policy_cfg = load_yaml(cfg["policy_config"])
    diffusion = build_diffusion_process(diffusion_cfg)
    criterion = build_hybrid_loss(diffusion_cfg)
    model = build_hit_model(model_cfg, diffusion, device=device)
    sampler = build_training_sampler(policy_cfg, diffusion.num_steps)

    lr = float(cfg["training"]["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_cfg = cfg["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(scheduler_cfg["factor"]),
        patience=int(scheduler_cfg["patience"]),
        min_lr=float(scheduler_cfg["min_lr"]),
        eps=0.0,
    )

    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(seed)
    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        generator=data_generator,
    )
    val_loader = DataLoader(
        bundle.val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=1,
    )

    train_graph = next(iter(train_loader))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_metrics = train_one_batch(
        model=model,
        graph=train_graph,
        diffusion_process=diffusion,
        step_sampler=sampler,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        grad_clip_norm=float(cfg["training"]["grad_clip_norm"]),
    )
    if not all(np.isfinite(value) for value in training_metrics.values()):
        raise AssertionError("D12 training step returned a non-finite metric.")
    scheduler.step(training_metrics["weighted_loss"])
    train_peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None

    val_graph = next(iter(val_loader))
    prepared_a = prepare_validation_batch(
        deepcopy(val_graph),
        diffusion_process=diffusion,
        base_seed=int(policy_cfg["validation"]["base_seed"]),
        device=device,
    )
    prepared_b = prepare_validation_batch(
        deepcopy(val_graph),
        diffusion_process=diffusion,
        base_seed=int(policy_cfg["validation"]["base_seed"]),
        device=device,
    )
    if not torch.equal(prepared_a.r.cpu(), prepared_b.r.cpu()):
        raise AssertionError("D12 deterministic validation timestep changed between repeats.")
    if not torch.equal(prepared_a.noise.cpu(), prepared_b.noise.cpu()):
        raise AssertionError("D12 deterministic validation noise changed between repeats.")
    if not torch.equal(prepared_a.field_r.cpu(), prepared_b.field_r.cpu()):
        raise AssertionError("D12 deterministic validation corruption changed between repeats.")

    validation_metrics = validate_one_batch(
        model=model,
        graph=deepcopy(val_graph),
        diffusion_process=diffusion,
        criterion=criterion,
        validation_base_seed=int(policy_cfg["validation"]["base_seed"]),
        device=device,
    )
    if not np.isfinite(validation_metrics["loss"]):
        raise AssertionError("D12 deterministic validation loss is non-finite.")

    checkpoint_path = Path(args.checkpoint)
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        step_sampler=sampler,
        epoch=1,
        global_step=1,
        batch_in_epoch=0,
        data_generator=data_generator,
        metadata={
            "contract": "d12_real_HIT_pipeline_smoke",
            "dataset_fingerprint_sha256": bundle.split_manifest["ordered_file_fingerprint_sha256"],
            "hierarchy_fingerprint_sha256": bundle.hierarchy["hierarchy_fingerprint_sha256"],
        },
    )

    model_restored = build_hit_model(model_cfg, diffusion, device=device)
    optimizer_restored = torch.optim.Adam(model_restored.parameters(), lr=lr)
    scheduler_restored = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_restored,
        factor=float(scheduler_cfg["factor"]),
        patience=int(scheduler_cfg["patience"]),
        min_lr=float(scheduler_cfg["min_lr"]),
        eps=0.0,
    )
    sampler_restored = build_training_sampler(policy_cfg, diffusion.num_steps)
    data_generator_restored = torch.Generator(device="cpu")
    data_generator_restored.manual_seed(seed + 999)
    progress = load_training_checkpoint(
        checkpoint_path,
        model=model_restored,
        optimizer=optimizer_restored,
        scheduler=scheduler_restored,
        scaler=None,
        step_sampler=sampler_restored,
        data_generator=data_generator_restored,
        map_location=device,
        restore_rng=True,
    )
    checkpoint_model_exact = _state_dict_exact(model.state_dict(), model_restored.state_dict())
    if not checkpoint_model_exact or progress["epoch"] != 1 or progress["global_step"] != 1:
        raise AssertionError("D12 checkpoint did not restore the trained smoke state exactly.")

    if args.generation_steps < 2:
        raise ValueError("generation-steps must be >= 2.")
    spaced_steps = DiffusionStepsGenerator("linear", diffusion.num_steps)(args.generation_steps)
    template = bundle.test_dataset[0]
    generation_graph = Collater().collate([deepcopy(template)])
    model_restored.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    state_std = sample_unconditional_physical_dgn(
        model_restored,
        generation_graph,
        steps=spaced_steps,
    ).cpu()
    generation_peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    state_nd, state_dim = inverse_generated_state(
        state_std,
        standardizer=bundle.standardizer,
        refs=bundle.refs,
    )
    if not torch.isfinite(state_std).all() or not torch.isfinite(state_nd).all() or not torch.isfinite(state_dim).all():
        raise AssertionError("D12 smoke generation contains non-finite values.")

    sample_path = Path(args.sample)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_standardized": state_std,
            "state_nondimensional": state_nd,
            "state_dimensional": state_dim,
            "pos_nondimensional": bundle.processed_dataset.pos.cpu(),
            "pos_dimensional": dimensional_positions(bundle).cpu(),
            "cells": bundle.raw_dataset.cells.cpu(),
            "sampling_steps": spaced_steps,
            "note": "D12 integration smoke sample from a one-update model; not a scientific result.",
        },
        sample_path,
    )

    manifest = {
        "version": 1,
        "contract": "d12_train_validation_generation_integration",
        "device": str(device),
        "dataset_fingerprint_sha256": bundle.split_manifest["ordered_file_fingerprint_sha256"],
        "hierarchy_fingerprint_sha256": bundle.hierarchy["hierarchy_fingerprint_sha256"],
        "split_sizes": {
            "train": len(bundle.train_dataset),
            "val": len(bundle.val_dataset),
            "test": len(bundle.test_dataset),
        },
        "training": {
            "batch_size": 2,
            "one_real_batch_completed": True,
            "metrics": training_metrics,
            "peak_cuda_memory_bytes": train_peak_bytes,
            "sampler_history_updates": int(sampler._loss_counts.sum()),
        },
        "validation": {
            "batch_size": 2,
            "deterministic_corruption_repeat_exact": True,
            "loss": validation_metrics["loss"],
            "num_graphs": validation_metrics["num_graphs"],
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "model_state_exact_after_restore": checkpoint_model_exact,
            "progress": progress,
        },
        "generation": {
            "num_samples": 1,
            "num_spaced_steps_for_smoke": args.generation_steps,
            "steps": spaced_steps,
            "standardized_shape": list(state_std.shape),
            "nondimensional_shape": list(state_nd.shape),
            "dimensional_shape": list(state_dim.shape),
            "all_finite": True,
            "peak_cuda_memory_bytes": generation_peak_bytes,
            "artifact": str(sample_path),
            "production_default": "full_1000_step_ancestral_ddpm",
        },
        "d10_ddp": "deferred",
        "all_passed": True,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("D12 real HIT pipeline validation passed")
    print(f"training metrics: {training_metrics}")
    print(f"validation loss: {validation_metrics['loss']:.6e}")
    print(f"checkpoint restore exact: {checkpoint_model_exact}")
    print(f"generation steps: {spaced_steps}")
    print(f"generation shape: {tuple(state_dim.shape)}")
    if train_peak_bytes is not None:
        print(f"training peak CUDA memory: {train_peak_bytes / (1024**3):.3f} GiB")
        print(f"generation peak CUDA memory: {generation_peak_bytes / (1024**3):.3f} GiB")
    print(f"manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
