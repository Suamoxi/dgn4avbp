#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dgn4avbp.hit_benchmark import benchmark_hit_populations
from dgn4avbp.hit_pipeline import build_hit_data_bundle, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate D13 on real held-out HIT snapshots.")
    parser.add_argument("--training-config", default="configs/training/dgn_hit_baseline.yaml")
    parser.add_argument("--benchmark-config", default="configs/benchmark/dgn_hit_distribution.yaml")
    parser.add_argument("--num-reference-samples", type=int, default=2)
    parser.add_argument("--manifest", default="artifacts/d13_benchmark_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_cfg = load_yaml(args.training_config)
    benchmark_cfg = load_yaml(args.benchmark_config)
    bundle = build_hit_data_bundle(**training_cfg["data"])

    test_indices = list(bundle.split_indices["test"][: int(args.num_reference_samples)])
    if len(test_indices) < 2:
        raise ValueError("D13 validation requires at least two real test snapshots.")

    states = []
    sample_ids = []
    for index in test_indices:
        sample = bundle.processed_dataset[index]
        states.append(bundle.standardizer.inverse(sample.target).to(torch.float64))
        sample_ids.append(str(sample.sample_id))
    population = torch.stack(states, dim=0)

    result = benchmark_hit_populations(
        population,
        population.clone(),
        bundle.processed_dataset.pos,
        L_ref=bundle.refs.L_ref,
        quantiles=tuple(float(value) for value in benchmark_cfg["quantiles"]),
        spectral_bands=benchmark_cfg["spectra"]["bands"],
        max_pooled_values=min(int(benchmark_cfg["max_pooled_values"]), 200_000),
    )
    summary = result["summary"]

    if summary["grid_shape_with_periodic_endpoint"] != [33, 33, 33]:
        raise AssertionError(f"Unexpected real HIT Cartesian shape: {summary['grid_shape_with_periodic_endpoint']}")
    if summary["fft_grid_shape_unique_periodic_nodes"] != [32, 32, 32]:
        raise AssertionError(f"Unexpected D13 FFT shape: {summary['fft_grid_shape_unique_periodic_nodes']}")
    expected_k_nyquist = 32.0 * np.pi
    if not np.isclose(summary["k_nyquist_Lref"], expected_k_nyquist, rtol=1.0e-5, atol=1.0e-5):
        raise AssertionError(
            f"Unexpected nondimensional Nyquist wavenumber {summary['k_nyquist_Lref']} != {expected_k_nyquist}."
        )
    if summary["reference_max_parseval_relative_error"] > 1.0e-12:
        raise AssertionError("D13 real-HIT kinetic-energy spectrum violates Parseval consistency.")
    if summary["reference_density_valid_spectra"] != len(test_indices):
        raise AssertionError("Real HIT test snapshots contain invalid density for velocity spectra.")

    max_channel_w1 = max(float(row["pooled_wasserstein_1_approx"]) for row in result["channel_rows"])
    max_physical_w1 = max(float(row["wasserstein_1"]) for row in result["physical_rows"])
    max_spectrum_error = max(
        abs(float(row["generated_energy"]) - float(row["reference_energy"]))
        for row in result["spectra_rows"]
    )
    if max_channel_w1 != 0.0 or max_physical_w1 != 0.0 or max_spectrum_error != 0.0:
        raise AssertionError("D13 identity-population validation is not exact.")

    manifest = {
        "version": 1,
        "contract": "d13_unpaired_3d_hit_distribution_benchmark",
        "dataset_fingerprint_sha256": bundle.split_manifest["ordered_file_fingerprint_sha256"],
        "validation_population": "test_self_comparison_for_metric_identity_only",
        "validation_sample_ids": sample_ids,
        "generated_reference_pairing": False,
        "grid_shape_with_periodic_endpoint": summary["grid_shape_with_periodic_endpoint"],
        "fft_grid_shape_unique_periodic_nodes": summary["fft_grid_shape_unique_periodic_nodes"],
        "k_nyquist_Lref": summary["k_nyquist_Lref"],
        "k_nyquist_per_m": summary["k_nyquist_per_m"],
        "spectrum_range": summary["spectrum_range"],
        "periodic_endpoint_handling": summary["periodic_endpoint_handling"],
        "reference_max_parseval_relative_error": summary["reference_max_parseval_relative_error"],
        "reference_mean_corner_mode_energy_fraction": summary[
            "reference_mean_corner_mode_energy_fraction"
        ],
        "identity_checks": {
            "max_channel_wasserstein_1": max_channel_w1,
            "max_physical_wasserstein_1": max_physical_w1,
            "max_spectrum_absolute_error": max_spectrum_error,
        },
        "production_comparison": "generated_population_vs_full_frozen_test_population_unpaired",
        "all_passed": True,
    }
    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("D13 real HIT benchmark validation passed")
    print(f"samples: {sample_ids}")
    print(f"grid: {summary['grid_shape_with_periodic_endpoint']} -> FFT {summary['fft_grid_shape_unique_periodic_nodes']}")
    print(f"k_Nyquist * L_ref: {summary['k_nyquist_Lref']:.9f}")
    print(f"k_Nyquist [1/m]: {summary['k_nyquist_per_m']:.9e}")
    print(f"Parseval max relative error: {summary['reference_max_parseval_relative_error']:.3e}")
    print(f"corner-mode energy fraction: {summary['reference_mean_corner_mode_energy_fraction']:.3e}")
    print(f"manifest: {path.resolve()}")


if __name__ == "__main__":
    main()
