from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from dgn4avbp.diffusion_policy import (
    deterministic_validation_assignment,
    deterministic_validation_corruption,
    deterministic_validation_noise,
)
from dgn4avbp.diffusion_process import DiffusionProcess
from dgn4avbp.step_sampler import ImportanceStepSampler


ROOT = Path(__file__).resolve().parents[1]


def test_d8_config_contract() -> None:
    with (ROOT / "configs/diffusion/hit_timestep_policy.yaml").open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    training = cfg["training"]
    validation = cfg["validation"]
    assert training["sampler"] == "loss_second_moment"
    assert training["history_per_term"] == 10
    assert training["uniform_prob"] == 0.001
    assert training["importance_weight"] == "inverse_Tp"
    assert validation["deterministic"] is True
    assert validation["key"] == "sample_id"
    assert validation["mutate_training_sampler"] is False


def test_loss_second_moment_sampler_is_uniform_until_warmup() -> None:
    sampler = ImportanceStepSampler(num_diffusion_steps=4, min_history_length=2, uniform_prob=0.001)
    np.testing.assert_allclose(sampler.weights, np.ones(4))
    sampler.update(torch.tensor([0, 1]), torch.tensor([2.0, 3.0]))
    np.testing.assert_allclose(sampler.weights, np.ones(4))


def test_loss_second_moment_sampler_diagnostics_track_warmup() -> None:
    sampler = ImportanceStepSampler(num_diffusion_steps=4, min_history_length=2, uniform_prob=0.001)
    sampler.update(
        torch.tensor([0, 0, 1, 3, 3]),
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    diagnostics = sampler.diagnostics()
    assert diagnostics["warmed_up"] is False
    assert diagnostics["min_history_count"] == 0
    assert diagnostics["max_history_count"] == 2
    assert diagnostics["mean_history_count"] == 1.25
    assert diagnostics["observed_timesteps_fraction"] == 0.75
    assert diagnostics["full_history_fraction"] == 0.5


def test_loss_second_moment_weights_match_reference_formula() -> None:
    sampler = ImportanceStepSampler(num_diffusion_steps=3, min_history_length=2, uniform_prob=0.001)
    timesteps = torch.tensor([0, 0, 1, 1, 2, 2])
    losses = torch.tensor([1.0, 3.0, 2.0, 4.0, 4.0, 8.0])
    sampler.update(timesteps, losses)

    history = np.array([[1.0, 3.0], [2.0, 4.0], [4.0, 8.0]], dtype=np.float64)
    expected = np.sqrt(np.mean(history**2, axis=-1))
    expected /= expected.sum()
    expected *= 0.999
    expected += 0.001 / len(expected)
    np.testing.assert_allclose(sampler.weights, expected, rtol=1e-12, atol=1e-12)


def test_sampler_importance_weights_are_inverse_Tp() -> None:
    sampler = ImportanceStepSampler(num_diffusion_steps=3, min_history_length=1, uniform_prob=0.001)
    sampler.update(torch.tensor([0, 1, 2]), torch.tensor([1.0, 2.0, 4.0]))
    np.random.seed(123)
    timesteps, weights = sampler.sample(batch_size=32)
    p = sampler.weights / sampler.weights.sum()
    expected = torch.tensor(1.0 / (3.0 * p[timesteps.numpy()]), dtype=torch.float32)
    torch.testing.assert_close(weights.cpu(), expected, rtol=0.0, atol=0.0)


def test_deterministic_validation_assignment_is_repeatable_and_order_independent() -> None:
    sample_ids = ["solut_hit_10840.h5", "solut_hit_10850.h5", "solut_hit_10860.h5"]
    forward = {
        sample_id: deterministic_validation_assignment(sample_id, num_steps=1000, base_seed=42)
        for sample_id in sample_ids
    }
    reverse = {
        sample_id: deterministic_validation_assignment(sample_id, num_steps=1000, base_seed=42)
        for sample_id in reversed(sample_ids)
    }
    assert forward == reverse


def test_deterministic_validation_noise_differs_across_sample_ids() -> None:
    noise_a, seed_a = deterministic_validation_noise(
        (8, 5), dtype=torch.float32, sample_id="sample_a.h5", base_seed=42
    )
    noise_b, seed_b = deterministic_validation_noise(
        (8, 5), dtype=torch.float32, sample_id="sample_b.h5", base_seed=42
    )
    assert seed_a != seed_b
    assert not torch.equal(noise_a, noise_b)


def test_deterministic_validation_corruption_repeats_exactly() -> None:
    dp = DiffusionProcess(num_steps=1000, schedule_type="linear")
    field_start = torch.arange(40, dtype=torch.float32).reshape(8, 5) / 10.0
    batch = torch.zeros(8, dtype=torch.long)

    first = deterministic_validation_corruption(
        dp, field_start, batch, sample_id="solut_hit_10840.h5", base_seed=42
    )
    second = deterministic_validation_corruption(
        dp, field_start, batch, sample_id="solut_hit_10840.h5", base_seed=42
    )

    assert first["timestep"] == second["timestep"]
    assert first["noise_seed"] == second["noise_seed"]
    torch.testing.assert_close(first["noise"], second["noise"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first["field_r"], second["field_r"], rtol=0.0, atol=0.0)


def test_deterministic_validation_corruption_matches_forward_equation() -> None:
    dp = DiffusionProcess(num_steps=1000, schedule_type="linear")
    field_start = torch.randn(8, 5)
    batch = torch.zeros(8, dtype=torch.long)
    result = deterministic_validation_corruption(
        dp, field_start, batch, sample_id="solut_hit_12090.h5", base_seed=42
    )

    r = result["r"]
    sqrt_alpha_bar = dp.get_index_from_list(dp.sqrt_alphas_cumprod, batch, r)
    sqrt_one_minus_alpha_bar = dp.get_index_from_list(dp.sqrt_one_minus_alphas_cumprod, batch, r)
    expected = sqrt_alpha_bar * field_start + sqrt_one_minus_alpha_bar * result["noise"]
    torch.testing.assert_close(result["field_r"], expected, rtol=0.0, atol=0.0)
