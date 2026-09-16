from __future__ import annotations

import hashlib
from collections.abc import Sequence

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

    The timestep and Gaussian realization depend only on ``(base_seed,
    sample_id)`` and are therefore fixed across epochs and validation iteration
    order. D9 adds the batched equivalent below without changing this single-
    graph contract.
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


def deterministic_validation_corruption_batch(
    diffusion_process: DiffusionProcess,
    field_start: torch.Tensor,
    batch: torch.Tensor,
    *,
    sample_ids: Sequence[str],
    base_seed: int = 42,
) -> dict[str, torch.Tensor | list[int]]:
    """Apply the frozen D8 validation corruption independently to each graph.

    Nodes are assumed to be grouped by PyG's batch vector, but no equal-node-
    count assumption is made. Noise for every physical sample is generated on
    CPU from that sample's D8 seed and then moved to the field device. Hence a
    sample receives the same corruption whether evaluated alone or inside a
    different computational batch.
    """

    if field_start.ndim != 2:
        raise ValueError(f"field_start must have shape [N, C], got {tuple(field_start.shape)}.")
    if batch.ndim != 1 or batch.shape[0] != field_start.shape[0]:
        raise ValueError("batch must have shape [N] matching field_start.")
    if batch.numel() == 0:
        raise ValueError("batch must contain at least one node.")

    batch_size = int(batch.max().item()) + 1
    expected_graph_ids = torch.arange(batch_size, device=batch.device)
    observed_graph_ids = torch.unique(batch, sorted=True)
    if not torch.equal(observed_graph_ids, expected_graph_ids):
        raise ValueError("batch graph ids must be contiguous and start at zero.")
    if len(sample_ids) != batch_size:
        raise ValueError(
            f"Expected {batch_size} sample_ids for the computational batch, got {len(sample_ids)}."
        )

    noise = torch.empty_like(field_start)
    timesteps: list[int] = []
    noise_seeds: list[int] = []

    for graph_id, sample_id in enumerate(sample_ids):
        timestep, noise_seed = deterministic_validation_assignment(
            sample_id,
            num_steps=diffusion_process.num_steps,
            base_seed=base_seed,
        )
        mask = batch == graph_id
        num_nodes = int(mask.sum().item())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(noise_seed)
        sample_noise = torch.randn(
            (num_nodes, field_start.shape[1]),
            dtype=field_start.dtype,
            device="cpu",
            generator=generator,
        ).to(field_start.device)
        noise[mask] = sample_noise
        timesteps.append(timestep)
        noise_seeds.append(noise_seed)

    r = torch.tensor(timesteps, dtype=torch.long, device=batch.device)
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
        "timesteps": timesteps,
        "noise_seeds": noise_seeds,
    }
