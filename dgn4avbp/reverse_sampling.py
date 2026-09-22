from __future__ import annotations

import torch

from .diffusion_process import DiffusionProcess


def epsilon_to_score(
    diffusion_process: DiffusionProcess,
    model_epsilon: torch.Tensor,
    batch: torch.LongTensor,
    r: torch.LongTensor,
) -> torch.Tensor:
    """Convert DDPM epsilon prediction to the VP score at the same discrete noise level.

    For x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon,
    score(x_t,t) = -epsilon_theta / sqrt(1-alpha_bar_t).
    """

    sigma = diffusion_process.get_index_from_list(
        diffusion_process.sqrt_one_minus_alphas_cumprod,
        batch,
        r,
    )
    return -model_epsilon / sigma


def epsilon_to_x0(
    diffusion_process: DiffusionProcess,
    field_r: torch.Tensor,
    model_epsilon: torch.Tensor,
    batch: torch.LongTensor,
    r: torch.LongTensor,
) -> torch.Tensor:
    """Recover the epsilon-parameterized x0 estimate at a discrete DDPM noise level."""

    alpha = diffusion_process.get_index_from_list(
        diffusion_process.sqrt_alphas_cumprod,
        batch,
        r,
    )
    sigma = diffusion_process.get_index_from_list(
        diffusion_process.sqrt_one_minus_alphas_cumprod,
        batch,
        r,
    )
    return (field_r - sigma * model_epsilon) / alpha


def vp_integrated_beta_interval(
    diffusion_process: DiffusionProcess,
    batch: torch.LongTensor,
    r: torch.LongTensor,
) -> torch.Tensor:
    """Integrated VP-SDE beta over one DDPM interval.

    The discrete forward transition has alpha_t = 1-beta_t. A piecewise-constant
    VP-SDE matches that transition exactly when

        integral beta(s) ds = -log(alpha_t).

    Returning this integrated quantity avoids introducing an arbitrary continuous
    time normalization in the diagnostic Euler/Euler-Maruyama samplers.
    """

    alpha = diffusion_process.get_index_from_list(
        diffusion_process.alphas,
        batch,
        r,
    )
    return -torch.log(alpha)


def vp_reverse_sde_euler_step(
    diffusion_process: DiffusionProcess,
    *,
    field_r: torch.Tensor,
    model_epsilon: torch.Tensor,
    batch: torch.LongTensor,
    r: torch.LongTensor,
    gaussian_noise: torch.Tensor,
) -> torch.Tensor:
    """One backward Euler-Maruyama step for the VP reverse-time SDE.

    Forward VP-SDE:
        dx = -0.5 beta(t) x dt + sqrt(beta(t)) dW

    Reverse SDE:
        dx = [-0.5 beta(t) x - beta(t) score(x,t)] dt
             + sqrt(beta(t)) dW_bar

    We integrate backward over one DDPM interval and use the exact integrated
    beta = -log(alpha_t) for that interval. The trained epsilon head supplies
    the score. The Improved-DDPM learned variance head is intentionally not
    used by this diagnostic sampler.
    """

    if gaussian_noise.shape != field_r.shape:
        raise ValueError(
            f"gaussian_noise shape {tuple(gaussian_noise.shape)} does not match "
            f"field shape {tuple(field_r.shape)}."
        )
    integrated_beta = vp_integrated_beta_interval(diffusion_process, batch, r)
    score = epsilon_to_score(diffusion_process, model_epsilon, batch, r)
    return (
        field_r
        + 0.5 * integrated_beta * field_r
        + integrated_beta * score
        + torch.sqrt(integrated_beta) * gaussian_noise
    )


def vp_probability_flow_ode_euler_step(
    diffusion_process: DiffusionProcess,
    *,
    field_r: torch.Tensor,
    model_epsilon: torch.Tensor,
    batch: torch.LongTensor,
    r: torch.LongTensor,
) -> torch.Tensor:
    """One backward Euler step for the VP probability-flow ODE.

    Probability-flow ODE:
        dx = [-0.5 beta(t) x - 0.5 beta(t) score(x,t)] dt

    Integrated backward over one matched DDPM interval. This path is fully
    deterministic conditional on the initial Gaussian field and ignores the
    Improved-DDPM learned variance head.
    """

    integrated_beta = vp_integrated_beta_interval(diffusion_process, batch, r)
    score = epsilon_to_score(diffusion_process, model_epsilon, batch, r)
    return field_r + 0.5 * integrated_beta * field_r + 0.5 * integrated_beta * score


def ddim_eta_zero_step(
    diffusion_process: DiffusionProcess,
    *,
    field_r: torch.Tensor,
    model_epsilon: torch.Tensor,
    batch: torch.LongTensor,
    r: torch.LongTensor,
) -> torch.Tensor:
    """One deterministic DDIM step with eta=0 for adjacent DDPM timesteps.

    The model is evaluated at the same discrete timestep used during training.
    For t>0,

        x_{t-1} = sqrt(alpha_bar_{t-1}) x0_hat
                  + sqrt(1-alpha_bar_{t-1}) epsilon_theta.

    At t=0 the previous cumulative alpha is exactly one, so the returned state
    is x0_hat. No learned variance or additional Gaussian noise is used.
    """

    if torch.any(r < 0) or torch.any(r >= diffusion_process.num_steps):
        raise ValueError("DDIM timestep is outside the diffusion process.")

    x0_hat = epsilon_to_x0(
        diffusion_process,
        field_r,
        model_epsilon,
        batch,
        r,
    )

    alpha_bar_prev_graph = torch.ones(
        r.shape,
        dtype=field_r.dtype,
        device=field_r.device,
    )
    nonzero = r > 0
    if torch.any(nonzero):
        alpha_bar_prev_graph[nonzero] = diffusion_process.alphas_cumprod.to(
            device=field_r.device,
            dtype=field_r.dtype,
        )[r[nonzero] - 1]

    alpha_bar_prev = alpha_bar_prev_graph[batch].unsqueeze(-1)
    return (
        torch.sqrt(alpha_bar_prev) * x0_hat
        + torch.sqrt(torch.clamp(1.0 - alpha_bar_prev, min=0.0)) * model_epsilon
    )
