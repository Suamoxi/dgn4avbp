#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dgn4avbp.hit_pipeline import (
    build_diffusion_process,
    build_hit_data_bundle,
    build_hit_model,
    build_hybrid_loss,
    load_yaml,
    validate_one_batch,
)
from dgn4avbp.loader import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a HIT DGN with the frozen D8 validation policy.")
    parser.add_argument("--config", default="configs/training/dgn_hit_baseline.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("D12 canonical evaluation currently requires one CUDA GPU.")
    device = torch.device("cuda")

    bundle = build_hit_data_bundle(**cfg["data"])
    model_cfg = load_yaml(cfg["model_config"])
    diffusion_cfg = load_yaml(cfg["diffusion_config"])
    policy_cfg = load_yaml(cfg["policy_config"])
    diffusion = build_diffusion_process(diffusion_cfg)
    model = build_hit_model(model_cfg, diffusion, device=device)
    criterion = build_hybrid_loss(diffusion_cfg)

    checkpoint_path = Path(args.checkpoint or cfg["generation"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = bundle.val_dataset if args.split == "val" else bundle.test_dataset
    validation_cfg = cfg["validation"]
    loader = DataLoader(
        dataset,
        batch_size=int(validation_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(validation_cfg["num_workers"]),
    )

    base_seed = int(policy_cfg["validation"]["base_seed"])
    weighted_sum = 0.0
    num_graphs = 0
    num_batches = 0
    for batch_index, graph in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        result = validate_one_batch(
            model=model,
            graph=graph,
            diffusion_process=diffusion,
            criterion=criterion,
            validation_base_seed=base_seed,
            device=device,
        )
        weighted_sum += result["loss"] * result["num_graphs"]
        num_graphs += result["num_graphs"]
        num_batches += 1

    if num_graphs == 0:
        raise RuntimeError("Evaluation produced no graphs.")
    payload = {
        "contract": "d12_deterministic_validation",
        "split": args.split,
        "loss": weighted_sum / num_graphs,
        "num_graphs": num_graphs,
        "num_batches": num_batches,
        "checkpoint": str(checkpoint_path),
        "validation_base_seed": base_seed,
        "interpretation": "stable_monitoring_metric_not_exact_full_T_vlb_estimator",
    }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
