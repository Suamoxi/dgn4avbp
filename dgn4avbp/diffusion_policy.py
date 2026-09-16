from __future__ import annotations

import hashlib

import torch

from dgn4avbp.diffusion_process import DiffusionProcess


_MAX_TORCH_SEED = 2**63 - 1


def _stable_seed(namespace: str, base_seed: int, sample_id: str) -> int:
    """Return a process-independent 63-bit seed for one sample and purpose."""

    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string.")
    payload = f"{namespace}\0{int(base_seed)}\0{sample_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)
    return value % _MAX_TORCH_SEED


def deterministic_validation_assignment(
    sample_id: str,
    *,
    num_steps: int,
    base_seed: int = 42,
) -> tuple[int, int]:
    """Assign one fixed diffusion timestep and Gaussian seed to ``sample_id``."""

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    timestep_seed = _stable_seed("d8-validation-timestep", base_seed, sample_id)
    noise_seed = _stable_seed("d8-validation-noise", base_seed, sample_id)
    return timestep_seed % num_steps, noise_seed


def deterministic_validation_noise(
    shape: torch.Size | tuple[int, ...],
    *,
    dtype: torch.dtype,
    sample_id: str,
    base_seed: int = 42,
) -> tuple[torch.Tensor, int]:
    """Generate validation noise on CPU so the same sample is device-independent."""

    _, noise_seed = deterministic_validation_assignment(
        sample_id,
        num_steps=1,
        base_seed=base_seed,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(noise_seed)
    noise = torch.randn(shape, dtype=dtype, device="cpu", generator=generator)
    return noise, noise_seed


def deterministic_validation_corruption(
    diffusion_process: DiffusionProcess,
    field_start: torch.Tensor,
    batch: torch.Tensor,
    *,
    sample_id: str,
    base_seed: int = 42,
) -> dict[str, torch.Tensor | int]:
    """Apply one fixed DDPM forward corruption to one validation graph.

    D8 intentionally supports one physical graph at a time. Hierarchy-aware
    multi-sample batching is owned by D9. The timestep and Gaussian realization
    depend only on ``(base_seed, sample_id)`` and are therefore fixed across
    epochs and validation iteration order.
    """

    if field_start.ndim != 2:
        raise ValueError(f"field_start must have shape [N, C], got {tuple(field_start.shape)}.")
    if batch.ndim != 1 or batch.shape[0] != field_start.shape[0]:
        raise ValueError("batch must have shape [N] matching field_start.")
    if batch.numel() == 0 or int(batch.min().item()) != 0 or int(batch.max().item()) != 0:
        raise ValueError("D8 deterministic validation corruption expects exactly one graph.")

    timestep, noise_seed = deterministic_validation_assignment(
        sample_id,
        num_steps=diffusion_process.num_steps,
        base_seed=base_seed,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(noise_seed)
    noise = torch.randn(
        field_start.shape,
        dtype=field_start.dtype,
        device="cpu",
        generator=generator,
    ).to(field_start.device)

    r = torch.tensor([timestep], dtype=torch.long, device=batch.device)
    sqrt_alpha_bar = diffusion_process.get_index_from_list(
        diffusion_process.sqrt_alphas_cumprod,
        batch,
        r,
    )
    sqrt_one_minus_alpha_bar = diffusion_process.get_index_from_list(
        diffusion_process.sqrt_one_minus_alphas_cumprod,
        batch,
        r,
    )
    field_r = sqrt_alpha_bar * field_start + sqrt_one_minus_alpha_bar * noise

    return {
        "field_r": field_r,
        "noise": noise,
        "r": r,
        "timestep": timestep,
        "noise_seed": noise_seed,
    }
