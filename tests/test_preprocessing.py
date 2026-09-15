from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import Data

from dgn4avbp.data.preprocessing import (
    ChannelStandardizer,
    HITReferenceScales,
    build_preprocessing_manifest,
    dimensionalize_positions,
    dimensionalize_state,
    fit_sample_balanced_standardizer,
    nondimensionalize_positions,
    nondimensionalize_state,
    validate_preprocessing_manifest,
)
from dgn4avbp.data.splits import create_contiguous_split_manifest


class TinyDataset:
    def __init__(self, states: list[torch.Tensor]) -> None:
        self.states = states

    def __getitem__(self, index: int) -> Data:
        return Data(target=self.states[index])

    def __len__(self) -> int:
        return len(self.states)


def _unit_refs() -> HITReferenceScales:
    return HITReferenceScales(rho_ref=1.0, U_ref=1.0, L_ref=1.0, T_ref=1.0)


def _real_refs() -> HITReferenceScales:
    return HITReferenceScales(
        rho_ref=1.17,
        U_ref=17.360947554785138,
        L_ref=0.0005668079275737893,
        T_ref=300.0,
    )


def _state(rows: list[float]) -> torch.Tensor:
    values = torch.tensor(rows, dtype=torch.float64).reshape(-1, 1)
    return values.repeat(1, 5)


def _files(n: int = 10) -> list[str]:
    return [f"/data/solut_hit_{i:08d}.h5" for i in range(10, 10 * (n + 1), 10)]


def _reference_config() -> dict:
    return {
        "case_id": "HIT_LES_FORCED",
        "reference_scheme": "hit_target_rms_box_reference",
        "references": {
            "rho_ref": {"value": 1.17},
            "U_ref": {"value": 17.360947554785138},
            "L_ref": {"value": 0.0005668079275737893},
            "T_ref": {"value": 300.0},
        },
    }


def test_state_nondimensionalization_round_trip() -> None:
    refs = _real_refs()
    state = torch.tensor(
        [[1.2, 20.0, -15.0, 4.0, 3.7e5]],
        dtype=torch.float64,
    )
    recovered = dimensionalize_state(nondimensionalize_state(state, refs), refs)
    torch.testing.assert_close(recovered, state, rtol=1e-12, atol=1e-12)


def test_position_nondimensionalization_round_trip() -> None:
    refs = _real_refs()
    pos = torch.tensor(
        [[0.0, refs.L_ref / 2.0, refs.L_ref]],
        dtype=torch.float64,
    )
    pos_nd = nondimensionalize_positions(pos, refs)
    torch.testing.assert_close(pos_nd, torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64))
    torch.testing.assert_close(
        dimensionalize_positions(pos_nd, refs),
        pos,
        rtol=1e-12,
        atol=1e-12,
    )


def test_sample_balanced_fit_weights_physical_samples_equally() -> None:
    dataset = TinyDataset([
        _state([0.0, 2.0]),
        _state([10.0]),
    ])
    standardizer = fit_sample_balanced_standardizer(dataset, [0, 1], _unit_refs())

    expected_mean = 5.5
    expected_std = math.sqrt(20.75)
    torch.testing.assert_close(
        standardizer.mean,
        torch.full((5,), expected_mean, dtype=torch.float64),
    )
    torch.testing.assert_close(
        standardizer.std,
        torch.full((5,), expected_std, dtype=torch.float64),
    )

    # A node-balanced fit would produce mean=4.0 here, so this checks the
    # equal-physical-sample objective rather than merely checking arithmetic.
    assert standardizer.mean[0].item() != pytest.approx(4.0)


def test_validation_and_test_values_cannot_affect_training_statistics() -> None:
    train_a = _state([1.0, 2.0])
    train_b = _state([3.0, 4.0])
    dataset_a = TinyDataset([train_a, train_b, _state([1.0e6])])
    dataset_b = TinyDataset([train_a, train_b, _state([-1.0e12, 1.0e12])])

    stats_a = fit_sample_balanced_standardizer(dataset_a, [0, 1], _unit_refs())
    stats_b = fit_sample_balanced_standardizer(dataset_b, [0, 1], _unit_refs())

    torch.testing.assert_close(stats_a.mean, stats_b.mean)
    torch.testing.assert_close(stats_a.std, stats_b.std)


def test_standardization_round_trip() -> None:
    standardizer = ChannelStandardizer(
        mean=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64),
        std=torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64),
        num_fit_samples=10,
    )
    x = torch.tensor(
        [[0.0, 4.0, 7.0, -1.0, 11.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(standardizer.inverse(standardizer.transform(x)), x)


def test_preprocessing_manifest_is_bound_to_d2_training_split() -> None:
    split = create_contiguous_split_manifest(_files(10))
    standardizer = ChannelStandardizer(
        mean=torch.zeros(5, dtype=torch.float64),
        std=torch.ones(5, dtype=torch.float64),
        num_fit_samples=split["counts"]["train"],
    )
    manifest = build_preprocessing_manifest(_reference_config(), split, standardizer)
    validate_preprocessing_manifest(manifest, split)

    changed_split = create_contiguous_split_manifest(
        _files(9) + ["/data/solut_hit_99999999.h5"]
    )
    with pytest.raises(ValueError, match="dataset fingerprint"):
        validate_preprocessing_manifest(manifest, changed_split)
