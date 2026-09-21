#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dgn4avbp.hit_benchmark import empirical_wasserstein_1d, _deterministic_subsample
from dgn4avbp.hit_pipeline import build_hit_data_bundle, load_yaml


COMPONENTS = ("u", "v", "w")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether HIT velocity Wasserstein errors are dominated by the sample mean mode."
    )
    parser.add_argument("--config", default="configs/training/dgn_hit_baseline_c0_30e.yaml")
    parser.add_argument("--epoch22-dir", required=True)
    parser.add_argument("--epoch84-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pooled-values", type=int, default=500_000)
    parser.add_argument("--rho-floor", type=float, default=1.0e-8)
    return parser.parse_args()


def load_generated_population(directory: Path) -> np.ndarray:
    files = sorted(directory.glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"No sample_*.pt files found in {directory}.")
    states = []
    for path in files:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        state = payload.get("state_nondimensional")
        if not isinstance(state, torch.Tensor) or state.ndim != 2 or state.shape[1] != 5:
            raise ValueError(f"Invalid state_nondimensional in {path}.")
        states.append(state.to(torch.float64).numpy())
    values = np.stack(states, axis=0)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite generated state in {directory}.")
    return values


def state_to_velocity(states: np.ndarray, rho_floor: float) -> tuple[np.ndarray, float]:
    rho = states[..., 0]
    invalid = np.abs(rho) <= rho_floor
    denominator = np.where(
        np.abs(rho) > rho_floor,
        rho,
        np.where(rho >= 0.0, rho_floor, -rho_floor),
    )
    velocity = states[..., 1:4] / denominator[..., None]
    return velocity, float(np.mean(invalid))


def center_velocity(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spatial_mean = np.mean(velocity, axis=1)
    centered = velocity - spatial_mean[:, None, :]
    return centered, spatial_mean


def fluctuation_metrics(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    component_rms = np.sqrt(np.mean(centered**2, axis=1))
    tke = 0.5 * np.mean(np.sum(centered**2, axis=-1), axis=1)
    isotropy = np.max(component_rms, axis=1) / np.maximum(
        np.min(component_rms, axis=1), np.finfo(np.float64).eps
    )
    return tke, isotropy


def component_metrics(
    generated_velocity: np.ndarray,
    reference_velocity: np.ndarray,
    *,
    max_pooled_values: int,
    seed_offset: int,
) -> dict:
    generated_centered, generated_mean = center_velocity(generated_velocity)
    reference_centered, reference_mean = center_velocity(reference_velocity)

    result = {}
    for component, name in enumerate(COMPONENTS):
        raw_gen = _deterministic_subsample(
            generated_velocity[..., component], max_pooled_values, seed_offset + 10 + component
        )
        raw_ref = _deterministic_subsample(
            reference_velocity[..., component], max_pooled_values, seed_offset + 20 + component
        )
        centered_gen = _deterministic_subsample(
            generated_centered[..., component], max_pooled_values, seed_offset + 30 + component
        )
        centered_ref = _deterministic_subsample(
            reference_centered[..., component], max_pooled_values, seed_offset + 40 + component
        )

        raw_w1 = empirical_wasserstein_1d(raw_gen, raw_ref)
        centered_w1 = empirical_wasserstein_1d(centered_gen, centered_ref)
        ref_centered_std = float(np.std(reference_centered[..., component]))

        result[name] = {
            "raw_pooled_velocity_w1": raw_w1,
            "centered_pooled_velocity_w1": centered_w1,
            "centered_over_raw_w1": centered_w1 / raw_w1 if raw_w1 > 0.0 else None,
            "centered_w1_over_reference_fluctuation_std": (
                centered_w1 / ref_centered_std if ref_centered_std > 0.0 else None
            ),
            "generated_mean_velocity_average": float(np.mean(generated_mean[:, component])),
            "reference_mean_velocity_average": float(np.mean(reference_mean[:, component])),
            "generated_fluctuation_std": float(np.std(generated_centered[..., component])),
            "reference_fluctuation_std": ref_centered_std,
            "fluctuation_std_ratio": float(
                np.std(generated_centered[..., component]) / ref_centered_std
            ),
        }

    gen_tke, gen_iso = fluctuation_metrics(generated_centered)
    ref_tke, ref_iso = fluctuation_metrics(reference_centered)
    result["fluctuation_population"] = {
        "generated_tke_mean": float(np.mean(gen_tke)),
        "reference_tke_mean": float(np.mean(ref_tke)),
        "tke_wasserstein_1": empirical_wasserstein_1d(gen_tke, ref_tke),
        "generated_isotropy_rms_ratio_mean": float(np.mean(gen_iso)),
        "reference_isotropy_rms_ratio_mean": float(np.mean(ref_iso)),
        "isotropy_rms_ratio_wasserstein_1": empirical_wasserstein_1d(gen_iso, ref_iso),
    }
    return result


def main() -> None:
    args = parse_args()
    if args.max_pooled_values <= 0:
        raise ValueError("--max-pooled-values must be positive.")

    cfg = load_yaml(args.config)
    bundle = build_hit_data_bundle(**cfg["data"])

    reference_states = []
    for index in bundle.split_indices["test"]:
        processed = bundle.processed_dataset[index]
        reference_states.append(
            bundle.standardizer.inverse(processed.target).to(torch.float64).numpy()
        )
    reference = np.stack(reference_states, axis=0)

    epoch22 = load_generated_population(Path(args.epoch22_dir))
    epoch84 = load_generated_population(Path(args.epoch84_dir))

    if epoch22.shape[1:] != reference.shape[1:] or epoch84.shape[1:] != reference.shape[1:]:
        raise ValueError(
            f"Generated/reference shapes do not match: "
            f"epoch22={epoch22.shape}, epoch84={epoch84.shape}, reference={reference.shape}."
        )

    reference_velocity, ref_invalid = state_to_velocity(reference, args.rho_floor)
    epoch22_velocity, e22_invalid = state_to_velocity(epoch22, args.rho_floor)
    epoch84_velocity, e84_invalid = state_to_velocity(epoch84, args.rho_floor)

    artifact = {
        "version": 1,
        "contract": "centered_velocity_diagnostic_c0e22_vs_c1e84",
        "interpretation": (
            "Each sample velocity field is centered by subtracting its own spatial mean "
            "before pooled Wasserstein, fluctuation TKE, and isotropy diagnostics."
        ),
        "reference_population": "all_126_frozen_D2_test_snapshots",
        "num_reference_samples": int(reference.shape[0]),
        "num_epoch22_samples": int(epoch22.shape[0]),
        "num_epoch84_samples": int(epoch84.shape[0]),
        "max_pooled_values": int(args.max_pooled_values),
        "rho_floor": float(args.rho_floor),
        "invalid_density_fraction": {
            "reference": ref_invalid,
            "epoch22": e22_invalid,
            "epoch84": e84_invalid,
        },
        "epoch22": component_metrics(
            epoch22_velocity,
            reference_velocity,
            max_pooled_values=args.max_pooled_values,
            seed_offset=1000,
        ),
        "epoch84": component_metrics(
            epoch84_velocity,
            reference_velocity,
            max_pooled_values=args.max_pooled_values,
            seed_offset=2000,
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("CENTERED VELOCITY DIAGNOSTIC COMPLETE")
    print("output:", output)
    for label in ("epoch22", "epoch84"):
        print()
        print(label)
        for name in COMPONENTS:
            values = artifact[label][name]
            print(
                f"  {name}: raw_W1={values['raw_pooled_velocity_w1']:.6g} "
                f"centered_W1={values['centered_pooled_velocity_w1']:.6g} "
                f"centered/raw={values['centered_over_raw_w1']:.6g} "
                f"centered_W1/ref_std={values['centered_w1_over_reference_fluctuation_std']:.6g}"
            )
        population = artifact[label]["fluctuation_population"]
        print(
            "  fluctuation TKE:",
            population["generated_tke_mean"],
            "reference:",
            population["reference_tke_mean"],
            "W1:",
            population["tke_wasserstein_1"],
        )
        print(
            "  fluctuation isotropy:",
            population["generated_isotropy_rms_ratio_mean"],
            "reference:",
            population["reference_isotropy_rms_ratio_mean"],
            "W1:",
            population["isotropy_rms_ratio_wasserstein_1"],
        )


if __name__ == "__main__":
    main()
