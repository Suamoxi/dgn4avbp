from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_VERSION = 1


def capture_rng_state(data_generator: torch.Generator | None = None) -> dict[str, Any]:
    """Capture all stochastic state needed by the single-process D11 contract."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
        "data_generator": (
            data_generator.get_state().clone() if data_generator is not None else None
        ),
    }


def restore_rng_state(
    state: dict[str, Any],
    data_generator: torch.Generator | None = None,
) -> None:
    """Restore Python, NumPy, PyTorch CPU/CUDA and optional loader-generator RNGs."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable during restore."
            )
        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError(
                "Checkpoint CUDA RNG-state count does not match the current CUDA device count."
            )
        torch.cuda.set_rng_state_all(cuda_state)

    generator_state = state.get("data_generator")
    if generator_state is not None:
        if data_generator is None:
            raise ValueError(
                "Checkpoint contains data-generator RNG state but no generator was provided."
            )
        data_generator.set_state(generator_state)


def build_training_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step_sampler,
    epoch: int,
    global_step: int,
    batch_in_epoch: int = 0,
    scheduler: Any = None,
    scaler: Any = None,
    data_generator: torch.Generator | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the D11 single-process exact-resume checkpoint payload."""

    if epoch < 0 or global_step < 0 or batch_in_epoch < 0:
        raise ValueError("epoch, global_step and batch_in_epoch must be non-negative.")

    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step_sampler": step_sampler.state_dict(),
        "progress": {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "batch_in_epoch": int(batch_in_epoch),
        },
        "rng_state": capture_rng_state(data_generator),
        "metadata": dict(metadata or {}),
    }


def save_training_checkpoint(
    path: str | Path,
    **checkpoint_kwargs,
) -> None:
    """Atomically save a D11 training checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = build_training_checkpoint(**checkpoint_kwargs)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step_sampler,
    scheduler: Any = None,
    scaler: Any = None,
    data_generator: torch.Generator | None = None,
    map_location: str | torch.device | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load the D11 checkpoint and restore all supplied training objects."""

    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    version = int(checkpoint.get("checkpoint_version", -1))
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {version}; expected {CHECKPOINT_VERSION}."
        )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step_sampler.load_state_dict(checkpoint["step_sampler"])

    scheduler_state = checkpoint.get("scheduler")
    if scheduler_state is not None:
        if scheduler is None:
            raise ValueError("Checkpoint contains scheduler state but no scheduler was provided.")
        scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler")
    if scaler_state is not None:
        if scaler is None:
            raise ValueError("Checkpoint contains AMP scaler state but no scaler was provided.")
        scaler.load_state_dict(scaler_state)

    if restore_rng:
        restore_rng_state(checkpoint["rng_state"], data_generator=data_generator)

    progress = checkpoint["progress"]
    return {
        "epoch": int(progress["epoch"]),
        "global_step": int(progress["global_step"]),
        "batch_in_epoch": int(progress.get("batch_in_epoch", 0)),
        "metadata": checkpoint.get("metadata", {}),
    }
