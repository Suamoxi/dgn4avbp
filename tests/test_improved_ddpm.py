from __future__ import annotations

from pathlib import Path

import torch
import yaml

from dgn4avbp.diffusion_process import DiffusionProcess
from dgn4avbp.improved_ddpm import (
    ancestral_sample_step,
    continuous_vlb_term,
    hybrid_loss_terms,
    learned_range_log_variance,
)


ROOT = Path(__file__).resolve().parents[1]


def _one_graph(num_nodes: int = 8, channels: int = 5):
    batch = torch.zeros(num_nodes, dtype=torch.long)
    field_start = torch.randn(num_nodes, channels)
    return batch, field_start


def test_d7_config_contract() -> None:
    with (ROOT / "configs/diffusion/improved_ddpm_hit.yaml").open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    assert cfg["num_steps"] == 1000
    assert cfg["schedule_type"] == "linear"
    assert cfg["beta_start"] == 0.0001
    assert cfg["beta_end"] == 0.02
    assert cfg["prediction"] == "epsilon"
    assert cfg["variance"] == "learned_range"
    assert cfg["learnable_variance"] is True
    assert cfg["lambda_vlb"] == 0.001
    assert cfg["sampled_vlb_multiplier"] == 1.0
    assert cfg["timestep_embedding"]["input"] == "raw_integer_index"
    assert cfg["sampling"]["suppress_noise_at_t0"] is True


def test_learned_range_variance_endpoints() -> None:
    dp = DiffusionProcess(num_steps=1000, schedule_type="linear")
    batch, field_start = _one_graph()
    r = torch.tensor([500], dtype=torch.long)

    minus_one = -torch.ones_like(field_start)
    plus_one = torch.ones_like(field_start)

    observed_min = learned_range_log_variance(dp, minus_one, batch, r)
    observed_max = learned_range_log_variance(dp, plus_one, batch, r)

    expected_min = dp.get_index_from_list(dp.posterior_log_variance_clipped, batch, r)
    expected_max = torch.log(dp.get_index_from_list(dp.betas, batch, r))

    torch.testing.assert_close(observed_min, expected_min.expand_as(field_start))
    torch.testing.assert_close(observed_max, expected_max.expand_as(field_start))


def test_hybrid_vlb_multiplier_matches_full_vlb_objective() -> None:
    torch.manual_seed(3)
    dp = DiffusionProcess(num_steps=1000, schedule_type="linear")
    batch, field_start = _one_graph()
    r = torch.tensor([400], dtype=torch.long)
    field_r, noise = dp(field_start, r, batch)
    model_epsilon = torch.randn_like(noise)
    model_v = torch.zeros_like(noise)

    terms = hybrid_loss_terms(
        dp,
        field_start=field_start,
        field_r=field_r,
        noise=noise,
        model_epsilon=model_epsilon,
        model_v=model_v,
        batch=batch,
        r=r,
        lambda_vlb=0.001,
    )

    assert terms["vlb_multiplier"] == 1.0
    torch.testing.assert_close(terms["loss"], terms["mse"] + terms["vb"])


def test_vlb_stop_gradient_does_not_train_epsilon_head() -> None:
    torch.manual_seed(4)
    dp = DiffusionProcess(num_steps=1000, schedule_type="linear")
    batch, field_start = _one_graph()
    r = torch.tensor([500], dtype=torch.long)
    field_r, noise = dp(field_start, r, batch)

    # MSE is exactly zero, so any epsilon gradient would have to leak from VLB.
    model_epsilon = noise.detach().clone().requires_grad_(True)
    model_v = torch.zeros_like(noise, requires_grad=True)

    terms = hybrid_loss_terms(
        dp,
        field_start=field_start,
        field_r=field_r,
        noise=noise,
        model_epsilon=model_epsilon,
        model_v=model_v,
        batch=batch,
        r=r,
        lambda_vlb=0.001,
    )
    terms["loss"].sum().backward()

    assert model_epsilon.grad is not None
    assert torch.equal(model_epsilon.grad, torch.zeros_like(model_epsilon.grad))
    assert model_v.grad is not None
    assert torch.isfinite(model_v.grad).all()
    assert float(model_v.grad.abs().sum().item()) > 0.0


def test_t0_continuous_decoder_vlb_is_finite() -> None:
    torch.manual_seed(5)
    dp = DiffusionProcess(num_steps=1000, schedule_type="linear")
    batch, field_start = _one_graph()
    r = torch.tensor([0], dtype=torch.long)
    field_r, noise = dp(field_start, r, batch)

    vb = continuous_vlb_term(
        dp,
        field_start=field_start,
        field_r=field_r,
        model_epsilon=noise,
        model_v=torch.zeros_like(noise),
        batch=batch,
        r=r,
    )

    assert vb.shape == (1,)
    assert torch.isfinite(vb).all()


def test_t0_ancestral_step_suppresses_gaussian_noise() -> None:
    torch.manual_seed(6)
    dp = DiffusionProcess(num_steps=1000, schedule_type="linear")
    batch, field_start = _one_graph()
    r = torch.tensor([0], dtype=torch.long)
    field_r, noise = dp(field_start, r, batch)
    model_v = torch.zeros_like(noise)

    sample_a = ancestral_sample_step(
        dp,
        field_r=field_r,
        model_epsilon=noise,
        model_v=model_v,
        batch=batch,
        r=r,
        gaussian_noise=torch.randn_like(field_r),
    )
    sample_b = ancestral_sample_step(
        dp,
        field_r=field_r,
        model_epsilon=noise,
        model_v=model_v,
        batch=batch,
        r=r,
        gaussian_noise=torch.randn_like(field_r),
    )

    torch.testing.assert_close(sample_a, sample_b, rtol=0.0, atol=0.0)


def test_positive_t_ancestral_step_uses_gaussian_noise() -> None:
    torch.manual_seed(7)
    dp = DiffusionProcess(num_steps=1000, schedule_type="linear")
    batch, field_start = _one_graph()
    r = torch.tensor([500], dtype=torch.long)
    field_r, noise = dp(field_start, r, batch)
    model_v = torch.zeros_like(noise)

    sample_a = ancestral_sample_step(
        dp,
        field_r=field_r,
        model_epsilon=noise,
        model_v=model_v,
        batch=batch,
        r=r,
        gaussian_noise=torch.zeros_like(field_r),
    )
    sample_b = ancestral_sample_step(
        dp,
        field_r=field_r,
        model_epsilon=noise,
        model_v=model_v,
        batch=batch,
        r=r,
        gaussian_noise=torch.ones_like(field_r),
    )

    assert not torch.equal(sample_a, sample_b)
