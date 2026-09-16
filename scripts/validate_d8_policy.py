#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from dgn4avbp.data import (
    AVBPHDF5FixedMeshDataset,
    load_data_config,
    load_split_manifest,
    split_indices_from_manifest,
    validate_split_manifest,
)
from dgn4avbp.diffusion_policy import (
    deterministic_validation_assignment,
    deterministic_validation_corruption,
)
from dgn4avbp.diffusion_process import DiffusionProcess
from dgn4avbp.step_sampler import ImportanceStepSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate D8 timestep sampling and deterministic validation policy.")
    parser.add_argument("--data-config", default="configs/data/avbp_hdf5_fixed_mesh_local.yaml")
    parser.add_argument("--split-manifest", default="artifacts/d2_split_manifest.json")
    parser.add_argument("--diffusion-config", default="configs/diffusion/improved_ddpm_hit.yaml")
    parser.add_argument("--policy-config", default="configs/diffusion/hit_timestep_policy.yaml")
    parser.add_argument("--manifest", default="artifacts/d8_policy_manifest.json")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in YAML config '{path}'.")
    return value


def main() -> None:
    args = parse_args()
    data_cfg = load_data_config(args.data_config)
    split_manifest = load_split_manifest(args.split_manifest)
    diffusion_cfg = _load_yaml(args.diffusion_config)
    policy_cfg = _load_yaml(args.policy_config)

    dataset = AVBPHDF5FixedMeshDataset.from_config(data_cfg)
    validate_split_manifest(split_manifest, dataset.files)
    split_indices = split_indices_from_manifest(split_manifest, dataset.files)
    val_indices = split_indices["val"]
    val_sample_ids = [Path(dataset.files[index]).name for index in val_indices]

    num_steps = int(diffusion_cfg["num_steps"])
    if num_steps != 1000 or diffusion_cfg["schedule_type"] != "linear":
        raise ValueError("D8 must use the frozen D7 1000-step linear diffusion contract.")

    training_cfg = policy_cfg["training"]
    validation_cfg = policy_cfg["validation"]
    if training_cfg["sampler"] != "loss_second_moment":
        raise ValueError("D8 training sampler must be loss_second_moment.")
    if int(training_cfg["history_per_term"]) != 10:
        raise ValueError("D8 canonical sampler requires history_per_term=10.")
    if float(training_cfg["uniform_prob"]) != 0.001:
        raise ValueError("D8 canonical sampler requires uniform_prob=0.001.")
    if not validation_cfg["deterministic"] or validation_cfg["key"] != "sample_id":
        raise ValueError("D8 validation must be deterministic and keyed by sample_id.")

    base_seed = int(validation_cfg["base_seed"])
    assignments = {
        sample_id: deterministic_validation_assignment(
            sample_id,
            num_steps=num_steps,
            base_seed=base_seed,
        )
        for sample_id in val_sample_ids
    }
    reverse_assignments = {
        sample_id: deterministic_validation_assignment(
            sample_id,
            num_steps=num_steps,
            base_seed=base_seed,
        )
        for sample_id in reversed(val_sample_ids)
    }
    if assignments != reverse_assignments:
        raise AssertionError("Validation assignments changed with iteration order.")

    timesteps = [value[0] for value in assignments.values()]
    if not all(0 <= timestep < num_steps for timestep in timesteps):
        raise AssertionError("Validation policy generated a timestep outside [0, T).")

    history_per_term = int(training_cfg["history_per_term"])
    uniform_prob = float(training_cfg["uniform_prob"])
    sampler = ImportanceStepSampler(
        num_diffusion_steps=num_steps,
        min_history_length=history_per_term,
        uniform_prob=uniform_prob,
    )
    rs = torch.arange(num_steps, dtype=torch.long).repeat_interleave(history_per_term)
    synthetic_losses = (rs.to(torch.float64) + 1.0) / num_steps
    sampler.update(rs, synthetic_losses)
    weights = sampler.weights
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise AssertionError("Loss-second-moment sampler produced invalid weights.")
    if not np.isclose(weights.sum(), 1.0):
        raise AssertionError("Loss-second-moment sampler probabilities do not sum to one.")

    np.random.seed(123)
    sampled_r, sampled_weight = sampler.sample(batch_size=64)
    p = weights / weights.sum()
    expected_importance = torch.tensor(
        1.0 / (num_steps * p[sampled_r.cpu().numpy()]),
        dtype=torch.float32,
    )
    torch.testing.assert_close(sampled_weight.cpu(), expected_importance, rtol=0.0, atol=0.0)

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("D8 validation requested CUDA but CUDA is unavailable.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cuda_update_validated = False
    if device.type == "cuda":
        device_sampler = ImportanceStepSampler(num_diffusion_steps=10)
        device_sampler.update(
            torch.tensor([3], dtype=torch.long, device=device),
            torch.tensor([2.5], dtype=torch.float32, device=device),
        )
        if device_sampler._loss_counts[3] != 1 or device_sampler._loss_history[3, 0] != 2.5:
            raise AssertionError("CUDA loss history update did not reach the NumPy sampler state.")
        cuda_update_validated = True

    diffusion_process = DiffusionProcess(
        num_steps=num_steps,
        schedule_type=diffusion_cfg["schedule_type"],
        beta_start=float(diffusion_cfg["beta_start"]),
        beta_end=float(diffusion_cfg["beta_end"]),
        max_beta=float(diffusion_cfg["max_beta"]),
    )
    field_start = (torch.arange(40, dtype=torch.float32).reshape(8, 5) / 10.0).to(device)
    batch = torch.zeros(8, dtype=torch.long, device=device)
    probe_sample_id = val_sample_ids[0]
    first = deterministic_validation_corruption(
        diffusion_process,
        field_start,
        batch,
        sample_id=probe_sample_id,
        base_seed=base_seed,
    )
    second = deterministic_validation_corruption(
        diffusion_process,
        field_start,
        batch,
        sample_id=probe_sample_id,
        base_seed=base_seed,
    )
    torch.testing.assert_close(first["noise"], second["noise"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first["field_r"], second["field_r"], rtol=0.0, atol=0.0)

    unique_timesteps = len(set(timesteps))
    manifest = {
        "version": 1,
        "contract": "loss_second_moment_training_and_deterministic_validation",
        "dataset_fingerprint_sha256": split_manifest["ordered_file_fingerprint_sha256"],
        "num_diffusion_steps": num_steps,
        "training_sampler": {
            "name": "loss_second_moment",
            "history_per_term": history_per_term,
            "uniform_prob": uniform_prob,
            "importance_weight": "1/(T*p(t))",
            "update_uses_unweighted_per_sample_loss": True,
            "device_safe_numpy_history_update": True,
            "cuda_update_validated": cuda_update_validated,
            "warmed_weight_min": float(weights.min()),
            "warmed_weight_max": float(weights.max()),
            "warmed_weight_sum": float(weights.sum()),
        },
        "validation": {
            "deterministic": True,
            "base_seed": base_seed,
            "key": "sample_id",
            "num_validation_samples": len(val_sample_ids),
            "unique_timesteps": unique_timesteps,
            "timestep_collisions": len(val_sample_ids) - unique_timesteps,
            "timestep_min": min(timesteps),
            "timestep_max": max(timesteps),
            "fixed_across_epochs": True,
            "independent_of_iteration_order": True,
            "mutates_training_sampler": False,
            "interpretation": "stable_monitoring_metric_not_exact_full_T_vlb_estimator",
            "probe_sample_id": probe_sample_id,
            "probe_timestep": int(first["timestep"]),
            "probe_noise_seed": int(first["noise_seed"]),
            "probe_corruption_repeat_exact": True,
            "first_assignments": [
                {
                    "sample_id": sample_id,
                    "timestep": int(assignments[sample_id][0]),
                    "noise_seed": int(assignments[sample_id][1]),
                }
                for sample_id in val_sample_ids[:10]
            ],
        },
        "deferred": [
            "hierarchy-aware multi-sample batching (D9)",
            "DDP synchronization (D10)",
            "checkpointing sampler state and global RNG state (D11)",
        ],
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("D8 timestep/validation policy passed")
    print(f"validation samples: {len(val_sample_ids)}")
    print(f"unique assigned timesteps: {unique_timesteps}")
    print(f"timestep collisions: {len(val_sample_ids) - unique_timesteps}")
    print(f"timestep range: {min(timesteps)} -> {max(timesteps)}")
    print(f"sampler history per term: {history_per_term}")
    print(f"sampler uniform probability: {uniform_prob}")
    print(f"CUDA sampler update validated: {cuda_update_validated}")
    print(f"probe sample: {probe_sample_id}")
    print(f"probe timestep: {first['timestep']}")
    print("deterministic corruption repeated exactly: True")
    print(f"manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
