from __future__ import annotations

import numpy as np
import torch

from dgn4avbp.hit_benchmark import (
    benchmark_hit_populations,
    empirical_wasserstein_1d,
    infer_cartesian_grid_3d,
    isotropic_kinetic_energy_spectrum,
)


def _periodic_endpoint_grid(n_unique: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(0.0, 1.0, n_unique + 1, dtype=torch.float64)
    pos = torch.tensor(
        [[x, y, z] for x in axis for y in axis for z in axis],
        dtype=torch.float64,
    )
    x = pos[:, 0]
    velocity = torch.sin(2.0 * torch.pi * x)
    state = torch.zeros(pos.shape[0], 5, dtype=torch.float64)
    state[:, 0] = 1.0
    state[:, 1] = velocity
    state[:, 4] = 10.0 + 0.5 * velocity.square()
    return pos, state


def test_empirical_wasserstein_supports_unequal_populations() -> None:
    left = np.array([0.0, 1.0])
    right = np.array([0.0, 0.5, 1.0])
    assert empirical_wasserstein_1d(left, left) == 0.0
    assert np.isclose(empirical_wasserstein_1d(left, right), 1.0 / 6.0)
    assert np.isclose(empirical_wasserstein_1d(left, right), empirical_wasserstein_1d(right, left))


def test_cartesian_grid_inference_is_node_order_independent() -> None:
    pos, _ = _periodic_endpoint_grid(n_unique=4)
    permutation = torch.randperm(pos.shape[0], generator=torch.Generator().manual_seed(7))
    grid = infer_cartesian_grid_3d(pos[permutation])
    assert grid.shape == (5, 5, 5)
    assert grid.periodic_shape == (4, 4, 4)
    np.testing.assert_allclose(grid.spacing, (0.25, 0.25, 0.25))


def test_3d_kinetic_energy_spectrum_parseval_and_peak() -> None:
    pos, state = _periodic_endpoint_grid(n_unique=8)
    grid = infer_cartesian_grid_3d(pos)
    spectrum = isotropic_kinetic_energy_spectrum(state, grid, L_ref=2.0)

    assert spectrum["physical_density_valid"] is True
    assert spectrum["parseval_relative_error"] < 1.0e-12
    assert np.isclose(spectrum["fluctuation_tke"], 0.25, rtol=1.0e-12, atol=1.0e-12)
    assert np.isclose(spectrum["full_modal_tke"], 0.25, rtol=1.0e-12, atol=1.0e-12)
    assert np.argmax(spectrum["energy"]) == 0
    assert np.isclose(spectrum["k_Lref"][0], 2.0 * np.pi)
    assert np.isclose(spectrum["k_per_m"][0], np.pi)


def test_identical_populations_have_identity_metrics_without_pairing() -> None:
    pos, state = _periodic_endpoint_grid(n_unique=8)
    states = torch.stack([state, state.clone()], dim=0)
    result = benchmark_hit_populations(
        states,
        states.clone(),
        pos,
        L_ref=1.0,
        max_pooled_values=100_000,
    )

    summary = result["summary"]
    assert summary["comparison_mode"] == "unpaired_population"
    assert summary["generated_reference_pairing"] is False
    assert summary["grid_shape_with_periodic_endpoint"] == [9, 9, 9]
    assert summary["fft_grid_shape_unique_periodic_nodes"] == [8, 8, 8]

    for row in result["channel_rows"]:
        assert abs(row["mean_bias"]) < 1.0e-14
        assert abs(row["pooled_wasserstein_1_approx"]) < 1.0e-14
        if row["reference_std"] > 0.0:
            assert np.isclose(row["std_ratio"], 1.0)

    for row in result["physical_rows"]:
        assert abs(row["mean_bias"]) < 1.0e-14
        assert abs(row["wasserstein_1"]) < 1.0e-14

    for row in result["spectra_rows"]:
        if row["reference_energy"] > 1.0e-20:
            assert np.isclose(row["energy_ratio"], 1.0, rtol=1.0e-12, atol=1.0e-12)
