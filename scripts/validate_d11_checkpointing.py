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
from dgn4avbp.step_sampler import ImportanceStepSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate D11 exact-resume checkpoint state.")
    parser.add_argument("--checkpoint", default="artifacts/d11_probe_checkpoint.pt")
    parser.add_argument("--manifest", default="artifacts/d11_checkpoint_manifest.json")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def _make_scaler(device: torch.device):
    if device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def _build_objects(device: torch.device):
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 16),
        torch.nn.SiLU(),
        torch.nn.Linear(16, 2),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2
    )
    scaler = _make_scaler(device)
    sampler = ImportanceStepSampler(
        num_diffusion_steps=16,
        min_history_length=3,
        uniform_prob=0.001,
    )
    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(2026)
    return model, optimizer, scheduler, scaler, sampler, data_generator


def _one_training_step(model, optimizer, scheduler, scaler, sampler, device):
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(32, 4, device=device)
    target = torch.randn(32, 2, device=device)

    if scaler is not None:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(x)
            loss = (prediction - target).square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        prediction = model(x)
        loss = (prediction - target).square().mean()
        loss.backward()
        optimizer.step()

    scheduler.step(float(loss.detach().cpu()))
    rs = torch.tensor([0, 1, 1, 5, 9, 15], device=device)
    losses = torch.tensor([1.0, 2.0, 3.0, 0.5, 4.0, 1.5], device=device)
    sampler.update(rs, losses)


def _probe_rng(sampler, data_generator, device):
    return {
        "python": random.random(),
        "numpy": float(np.random.standard_normal()),
        "torch_cpu": torch.rand(8),
        "torch_device": torch.rand(8, device=device).cpu(),
        "data_generator": torch.rand(8, generator=data_generator),
        "sampler": sampler.sample(batch_size=8, device=torch.device("cpu")),
    }


def _state_dict_tensors_equal(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        a = left[key]
        b = right[key]
        if isinstance(a, dict) and isinstance(b, dict):
            if not _state_dict_tensors_equal(a, b):
                return False
        elif torch.is_tensor(a) and torch.is_tensor(b):
            if not torch.equal(a.cpu(), b.cpu()):
                return False
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            for x, y in zip(a, b):
                if isinstance(x, dict) and isinstance(y, dict):
                    if not _state_dict_tensors_equal(x, y):
                        return False
                elif x != y:
                    return False
        elif a != b:
            return False
    return True


def main() -> None:
    args = parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("D11 validation requested CUDA but CUDA is unavailable.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    model, optimizer, scheduler, scaler, sampler, data_generator = _build_objects(device)
    _one_training_step(model, optimizer, scheduler, scaler, sampler, device)

    checkpoint_path = Path(args.checkpoint)
    metadata = {
        "contract": "d11_single_process_epoch_boundary_exact_resume",
        "probe_device": str(device),
    }
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step_sampler=sampler,
        epoch=5,
        global_step=321,
        batch_in_epoch=0,
        data_generator=data_generator,
        metadata=metadata,
    )

    saved_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    saved_optimizer = deepcopy(optimizer.state_dict())
    saved_scheduler = deepcopy(scheduler.state_dict())
    saved_scaler = deepcopy(scaler.state_dict()) if scaler is not None else None
    saved_sampler = sampler.state_dict()
    expected_probe = _probe_rng(sampler, data_generator, device)

    # Deliberately perturb all RNGs and rebuild all stateful training objects.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(999)
    resumed_model, resumed_optimizer, resumed_scheduler, resumed_scaler, resumed_sampler, resumed_generator = _build_objects(device)
    resumed_generator.manual_seed(999)

    progress = load_training_checkpoint(
        checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        step_sampler=resumed_sampler,
        data_generator=resumed_generator,
        map_location=device,
        restore_rng=True,
    )
    actual_probe = _probe_rng(resumed_sampler, resumed_generator, device)

    model_exact = all(
        torch.equal(resumed_model.state_dict()[key].detach().cpu(), value)
        for key, value in saved_model.items()
    )
    optimizer_exact = _state_dict_tensors_equal(
        resumed_optimizer.state_dict(), saved_optimizer
    )
    scheduler_exact = resumed_scheduler.state_dict() == saved_scheduler
    scaler_exact = (
        resumed_scaler.state_dict() == saved_scaler if resumed_scaler is not None else saved_scaler is None
    )
    sampler_exact = (
        np.array_equal(resumed_sampler._loss_history, saved_sampler["loss_history"])
        and np.array_equal(resumed_sampler._loss_counts, saved_sampler["loss_counts"])
    )

    rng_exact = {
        "python": actual_probe["python"] == expected_probe["python"],
        "numpy": actual_probe["numpy"] == expected_probe["numpy"],
        "torch_cpu": torch.equal(actual_probe["torch_cpu"], expected_probe["torch_cpu"]),
        "torch_device": torch.equal(actual_probe["torch_device"], expected_probe["torch_device"]),
        "data_generator": torch.equal(
            actual_probe["data_generator"], expected_probe["data_generator"]
        ),
        "step_sampler_timesteps": torch.equal(
            actual_probe["sampler"][0], expected_probe["sampler"][0]
        ),
        "step_sampler_importance_weights": torch.equal(
            actual_probe["sampler"][1], expected_probe["sampler"][1]
        ),
    }

    expected_progress = {
        "epoch": 5,
        "global_step": 321,
        "batch_in_epoch": 0,
        "metadata": metadata,
    }
    progress_exact = progress == expected_progress
    all_passed = (
        model_exact
        and optimizer_exact
        and scheduler_exact
        and scaler_exact
        and sampler_exact
        and progress_exact
        and all(rng_exact.values())
    )

    manifest = {
        "version": 1,
        "contract": "single_process_epoch_boundary_exact_resume",
        "checkpoint_version": 1,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "restored": {
            "model_state_exact": model_exact,
            "optimizer_state_exact": optimizer_exact,
            "scheduler_state_exact": scheduler_exact,
            "amp_scaler_state_exact": scaler_exact,
            "loss_second_moment_sampler_state_exact": sampler_exact,
            "progress_exact": progress_exact,
            "rng": rng_exact,
        },
        "progress": progress,
        "atomic_save": not checkpoint_path.with_name(checkpoint_path.name + ".tmp").exists(),
        "exact_resume_scope": "epoch_boundary_single_process_single_gpu",
        "mid_epoch_shuffle_resume": "not_claimed_requires_stateful_data_sampler_or_saved_permutation",
        "ddp_resume": "deferred_D10",
        "all_passed": all_passed,
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("D11 checkpoint/reproducibility validation")
    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path.resolve()}")
    print(f"checkpoint bytes: {manifest['checkpoint_size_bytes']}")
    print(f"model state exact: {model_exact}")
    print(f"optimizer state exact: {optimizer_exact}")
    print(f"scheduler state exact: {scheduler_exact}")
    print(f"AMP scaler state exact: {scaler_exact}")
    print(f"step sampler state exact: {sampler_exact}")
    print(f"progress exact: {progress_exact}")
    for key, value in rng_exact.items():
        print(f"RNG {key} exact: {value}")
    print(f"atomic save: {manifest['atomic_save']}")
    print(f"manifest: {manifest_path.resolve()}")

    if not all_passed:
        raise AssertionError("D11 checkpoint exact-resume contract failed; see manifest.")


if __name__ == "__main__":
    main()
