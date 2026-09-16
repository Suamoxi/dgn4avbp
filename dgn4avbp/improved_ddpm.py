from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from .diffusion_process import DiffusionProcess, DiffusionProcessSubSet

if TYPE_CHECKING:
    from .dgn_model import DiffusionModel


def batch_wise_mean(field: torch.Tensor, batch: torch.LongTensor) -> torch.Tensor:
    """Mean over node and channel dimensions, separately for each physical graph."""

    if field.dim() not in (1, 2):
        raise ValueError(f"field must be one- or two-dimensional, got {field.dim()} dimensions.")
    if field.dim() == 2:
        field = field.mean(dim=1)
    batch_size = int(batch.max().item()) + 1
    return scatter(field, batch, dim=0, dim_size=batch_size, reduce="mean")


def normal_kl_divergence(
    mean1: torch.Tensor,
    variance1: torch.Tensor,
    mean2: torch.Tensor,
    variance2: torch.Tensor,
) -> torch.Tensor:
    """KL[N(mean1, variance1) || N(mean2, variance2)] in nats per component."""

    return 0.5 * (
        torch.log(variance2)
        - torch.log(variance1)
        + variance1 / variance2
        + (mean1 - mean2).square() / variance2
        - 1.0
    )


def learned_range_log_variance(
    diffusion_process: DiffusionProcess,
    model_v: torch.Tensor,
    batch: torch.LongTensor,
    r: torch.LongTensor,
) -> torch.Tensor:
    """Improved-DDPM learned-range log variance.

    The network output is intentionally unconstrained, matching the released
    Improved-DDPM implementation. ``model_v=-1`` corresponds to the clipped
    posterior variance and ``model_v=+1`` corresponds to beta_t.
    """

    frac = (model_v + 1.0) / 2.0
    min_log = diffusion_process.get_index_from_list(
        diffusion_process.posterior_log_variance_clipped, batch, r
    )
    max_log = torch.log(
        diffusion_process.get_index_from_list(diffusion_process.betas, batch, r)
    )
    return frac * max_log + (1.0 - frac) * min_log


def epsilon_reverse_mean(
    diffusion_process: DiffusionProcess,
    field_r: torch.Tensor,
    model_epsilon: torch.Tensor,
    batch: torch.LongTensor,
    r: torch.LongTensor,
) -> torch.Tensor:
    """DDPM reverse mean when the model predicts epsilon."""

    betas_r = diffusion_process.get_index_from_list(diffusion_process.betas, batch, r)
    sqrt_one_minus_alphas_cumprod_r = diffusion_process.get_index_from_list(
        diffusion_process.sqrt_one_minus_alphas_cumprod, batch, r
    )
    sqrt_recip_alphas_r = diffusion_process.get_index_from_list(
        diffusion_process.sqrt_recip_alphas, batch, r
    )
    return sqrt_recip_alphas_r * (
        field_r - betas_r * model_epsilon / sqrt_one_minus_alphas_cumprod_r
    )


def model_mean_and_variance(
    diffusion_process: DiffusionProcess,
    field_r: torch.Tensor,
    model_epsilon: torch.Tensor,
    model_v: torch.Tensor,
    batch: torch.LongTensor,
    r: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the canonical Improved-DDPM reverse mean and learned variance."""

    mean = epsilon_reverse_mean(diffusion_process, field_r, model_epsilon, batch, r)
    log_variance = learned_range_log_variance(diffusion_process, model_v, batch, r)
    return mean, torch.exp(log_variance)


def continuous_vlb_term(
    diffusion_process: DiffusionProcess,
    *,
    field_start: torch.Tensor,
    field_r: torch.Tensor,
    model_epsilon: torch.Tensor,
    model_v: torch.Tensor,
    batch: torch.LongTensor,
    r: torch.LongTensor,
) -> torch.Tensor:
    """Per-graph VLB term in bits/dimension for continuous CFD states.

    For t>0 this is the Gaussian KL between the exact forward posterior and
    the learned reverse posterior. At t=0 we use a continuous Gaussian decoder
    NLL rather than the discretized image likelihood used by Improved-DDPM.
    Both branches are expressed in bits/dimension.
    """

    true_mean, _ = diffusion_process.get_posterior_mean_and_variance(
        field_start, field_r, batch, r
    )
    model_mean, model_variance = model_mean_and_variance(
        diffusion_process, field_r, model_epsilon, model_v, batch, r
    )

    # The exact posterior variance is zero at t=0. Use the same clipped value
    # used by Improved-DDPM to keep the unselected KL branch finite; t=0 uses
    # the decoder likelihood below.
    true_variance_for_kl = torch.exp(
        diffusion_process.get_index_from_list(
            diffusion_process.posterior_log_variance_clipped, batch, r
        )
    )
    kl = normal_kl_divergence(
        true_mean,
        true_variance_for_kl,
        model_mean,
        model_variance,
    )
    kl = batch_wise_mean(kl, batch) / math.log(2.0)

    decoder_nll = F.gaussian_nll_loss(
        model_mean,
        field_start,
        model_variance.expand_as(field_start),
        full=True,
        reduction="none",
    )
    decoder_nll = batch_wise_mean(decoder_nll, batch) / math.log(2.0)

    return torch.where(r == 0, decoder_nll, kl)


def hybrid_loss_terms(
    diffusion_process: DiffusionProcess,
    *,
    field_start: torch.Tensor,
    field_r: torch.Tensor,
    noise: torch.Tensor,
    model_epsilon: torch.Tensor,
    model_v: torch.Tensor,
    batch: torch.LongTensor,
    r: torch.LongTensor,
    lambda_vlb: float = 0.001,
) -> dict[str, torch.Tensor | float]:
    """Improved-DDPM hybrid objective for a sampled diffusion timestep.

    ``L_vlb`` in the paper is a sum over diffusion timesteps while
    ``L_simple`` is an expectation over sampled timesteps. The training loop's
    importance weight ``1 / (T p(t))`` estimates a timestep average, so the
    sampled VLB contribution must be multiplied by ``lambda_vlb * T``.

    For the canonical DGN baseline, T=1000 and lambda_vlb=0.001, hence the
    sampled VLB multiplier is exactly 1.0.
    """

    if lambda_vlb < 0.0:
        raise ValueError(f"lambda_vlb must be non-negative, got {lambda_vlb}.")
    if model_epsilon.shape != noise.shape:
        raise ValueError(
            f"epsilon/noise shape mismatch: {tuple(model_epsilon.shape)} vs {tuple(noise.shape)}."
        )
    if model_v.shape != noise.shape:
        raise ValueError(
            f"variance-head/noise shape mismatch: {tuple(model_v.shape)} vs {tuple(noise.shape)}."
        )

    mse = batch_wise_mean((model_epsilon - noise).square(), batch)

    # Stop-gradient on the reverse mean for the VLB term exactly as specified
    # in Improved-DDPM: VLB trains the variance head, MSE trains epsilon.
    vb = continuous_vlb_term(
        diffusion_process,
        field_start=field_start,
        field_r=field_r,
        model_epsilon=model_epsilon.detach(),
        model_v=model_v,
        batch=batch,
        r=r,
    )
    vlb_multiplier = float(lambda_vlb * diffusion_process.num_steps)
    loss = mse + vlb_multiplier * vb
    return {
        "mse": mse,
        "vb": vb,
        "vlb_multiplier": vlb_multiplier,
        "loss": loss,
    }


def ancestral_sample_step(
    diffusion_process: DiffusionProcess,
    *,
    field_r: torch.Tensor,
    model_epsilon: torch.Tensor,
    model_v: torch.Tensor,
    batch: torch.LongTensor,
    r: torch.LongTensor,
    gaussian_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """One ancestral Improved-DDPM reverse step.

    The stochastic term is explicitly zero for graphs at t=0. This is required
    by DDPM sampling even though the learned-range variance itself is non-zero
    there because the posterior log variance is clipped for numerical reasons.
    """

    mean, variance = model_mean_and_variance(
        diffusion_process, field_r, model_epsilon, model_v, batch, r
    )
    if gaussian_noise is None:
        gaussian_noise = torch.randn_like(field_r)
    if gaussian_noise.shape != field_r.shape:
        raise ValueError(
            f"gaussian_noise shape {tuple(gaussian_noise.shape)} does not match field {tuple(field_r.shape)}."
        )

    nonzero_mask = (r[batch] != 0).to(field_r.dtype).unsqueeze(-1)
    return mean + nonzero_mask * torch.sqrt(variance) * gaussian_noise


@torch.no_grad()
def sample_unconditional_physical_dgn(
    model: "DiffusionModel",
    graph: Data,
    *,
    steps: list[int] | None = None,
) -> torch.Tensor:
    """Canonical ancestral sampler for the D0 physical-space HIT baseline.

    This intentionally supports only the current scientific contract:
    unconditional physical-space diffusion with no Dirichlet boundary values.
    Generic latent/conditional generation remains outside D7.
    """

    if getattr(model, "is_latent_diffusion", False):
        raise ValueError("D7 physical HIT sampler does not support latent diffusion.")
    if hasattr(graph, "dirichlet_mask"):
        raise ValueError("D7 physical HIT sampler does not support Dirichlet conditioning.")
    if not model.learnable_variance:
        raise ValueError("D7 canonical baseline requires learned-range variance.")

    if not hasattr(graph, "batch") or graph.batch is None:
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=graph.pos.device)

    if graph.pos.device != model.device:
        graph = graph.to(model.device)

    if steps is not None:
        if not steps or not all(isinstance(step, int) for step in steps):
            raise ValueError("steps must be a non-empty list of integer base diffusion indices.")
        steps = sorted(steps)
        if len(steps) != len(set(steps)):
            raise ValueError("steps must not contain duplicates.")
        if steps[0] < 0 or steps[-1] >= model.diffusion_process.num_steps:
            raise ValueError("steps contain an index outside the base diffusion process.")
        diffusion_process = DiffusionProcessSubSet(model.diffusion_process, steps)
    else:
        diffusion_process = model.diffusion_process

    batch_size = int(graph.batch.max().item()) + 1
    graph.field_r = torch.randn(
        graph.batch.shape[0], model.num_fields, device=model.device
    )

    for step in diffusion_process.steps[::-1]:
        graph.r = torch.full(
            (batch_size,), step, dtype=torch.long, device=model.device
        )
        model_epsilon, model_v = model(graph)
        graph.field_r = ancestral_sample_step(
            diffusion_process,
            field_r=graph.field_r,
            model_epsilon=model_epsilon,
            model_v=model_v,
            batch=graph.batch,
            r=graph.r,
        )

    return graph.field_r
