from __future__ import annotations

import torch

from dgn4avbp.diffusion_process import DiffusionProcess
from dgn4avbp.reverse_sampling import (
    epsilon_to_score,
    epsilon_to_x0,
    vp_integrated_beta_interval,
    vp_probability_flow_ode_euler_step,
    vp_reverse_sde_euler_step,
)


def _process() -> DiffusionProcess:
    return DiffusionProcess(
        num_steps=1000,
        schedule_type="linear",
        beta_start=1.0e-4,
        beta_end=2.0e-2,
        max_beta=0.999,
    )


def test_epsilon_to_x0_recovers_exact_start() -> None:
    process = _process()
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    r = torch.tensor([100, 700], dtype=torch.long)
    x0 = torch.randn(5, 5, dtype=torch.float64)
    epsilon = torch.randn_like(x0)
    alpha = process.get_index_from_list(process.sqrt_alphas_cumprod, batch, r)
    sigma = process.get_index_from_list(process.sqrt_one_minus_alphas_cumprod, batch, r)
    xt = alpha * x0 + sigma * epsilon

    reconstructed = epsilon_to_x0(process, xt, epsilon, batch, r)
    torch.testing.assert_close(reconstructed, x0, rtol=1.0e-12, atol=1.0e-12)


def test_epsilon_to_score_matches_vp_identity() -> None:
    process = _process()
    batch = torch.zeros(4, dtype=torch.long)
    r = torch.tensor([250], dtype=torch.long)
    epsilon = torch.randn(4, 5, dtype=torch.float64)
    sigma = process.get_index_from_list(process.sqrt_one_minus_alphas_cumprod, batch, r)

    score = epsilon_to_score(process, epsilon, batch, r)
    torch.testing.assert_close(score, -epsilon / sigma, rtol=0.0, atol=0.0)


def test_integrated_beta_matches_discrete_alpha() -> None:
    process = _process()
    batch = torch.zeros(3, dtype=torch.long)
    r = torch.tensor([999], dtype=torch.long)

    integrated_beta = vp_integrated_beta_interval(process, batch, r)
    alpha = process.get_index_from_list(process.alphas, batch, r)
    torch.testing.assert_close(torch.exp(-integrated_beta), alpha, rtol=1.0e-6, atol=1.0e-7)


def test_reverse_sde_euler_step_formula() -> None:
    process = _process()
    batch = torch.zeros(3, dtype=torch.long)
    r = torch.tensor([500], dtype=torch.long)
    xt = torch.randn(3, 5, dtype=torch.float64)
    epsilon = torch.randn_like(xt)
    noise = torch.randn_like(xt)

    integrated_beta = vp_integrated_beta_interval(process, batch, r)
    score = epsilon_to_score(process, epsilon, batch, r)
    expected = xt + 0.5 * integrated_beta * xt + integrated_beta * score + torch.sqrt(integrated_beta) * noise

    actual = vp_reverse_sde_euler_step(
        process,
        field_r=xt,
        model_epsilon=epsilon,
        batch=batch,
        r=r,
        gaussian_noise=noise,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_probability_flow_ode_euler_step_formula() -> None:
    process = _process()
    batch = torch.zeros(3, dtype=torch.long)
    r = torch.tensor([500], dtype=torch.long)
    xt = torch.randn(3, 5, dtype=torch.float64)
    epsilon = torch.randn_like(xt)

    integrated_beta = vp_integrated_beta_interval(process, batch, r)
    score = epsilon_to_score(process, epsilon, batch, r)
    expected = xt + 0.5 * integrated_beta * xt + 0.5 * integrated_beta * score

    actual = vp_probability_flow_ode_euler_step(
        process,
        field_r=xt,
        model_epsilon=epsilon,
        batch=batch,
        r=r,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
