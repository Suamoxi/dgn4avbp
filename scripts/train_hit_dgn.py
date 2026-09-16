#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dgn4avbp.checkpointing import load_training_checkpoint, save_training_checkpoint
from dgn4avbp.hit_pipeline import (
    build_diffusion_process,
    build_hit_data_bundle,
    build_hit_model,
    build_hybrid_loss,
    build_training_sampler,
    load_yaml,
    train_one_batch,
    validate_one_batch,
)
from dgn4avbp.loader import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the canonical physical-space HIT DGN.")
    parser.add_argument("--config", default="configs/training/dgn_hit_baseline.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _average(records: list[dict], key: str) -> float:
    return float(sum(record[key] for record in records) / len(records))


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    _seed_everything(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("D12 canonical training currently requires one CUDA GPU.")
    device = torch.device("cuda")

    if bool(cfg["training"].get("mixed_precision", False)):
        raise NotImplementedError(
            "D12 freezes the initial physical-space baseline in FP32. AMP remains a later runtime ablation."
        )

    data_cfg = cfg["data"]
    bundle = build_hit_data_bundle(**data_cfg)
    model_cfg = load_yaml(cfg["model_config"])
    diffusion_cfg = load_yaml(cfg["diffusion_config"])
    policy_cfg = load_yaml(cfg["policy_config"])

    diffusion = build_diffusion_process(diffusion_cfg)
    model = build_hit_model(model_cfg, diffusion, device=device)
    criterion = build_hybrid_loss(diffusion_cfg)
    step_sampler = build_training_sampler(policy_cfg, diffusion.num_steps)

    training_cfg = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))
    scheduler_cfg = training_cfg["scheduler"]
    if scheduler_cfg["type"] != "reduce_on_plateau":
        raise ValueError("D12 supports only reduce_on_plateau for the frozen baseline.")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(scheduler_cfg["factor"]),
        patience=int(scheduler_cfg["patience"]),
        min_lr=float(scheduler_cfg["min_lr"]),
        eps=0.0,
    )

    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(seed)
    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg["num_workers"]),
        generator=data_generator,
    )
    validation_cfg = cfg["validation"]
    val_loader = DataLoader(
        bundle.val_dataset,
        batch_size=int(validation_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(validation_cfg["num_workers"]),
    )

    checkpoint_path = Path(args.checkpoint or training_cfg["checkpoint_path"])
    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
        progress = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            step_sampler=step_sampler,
            data_generator=data_generator,
            map_location=device,
            restore_rng=True,
        )
        if progress["batch_in_epoch"] != 0:
            raise ValueError("D12 exact resume supports epoch-boundary checkpoints only.")
        start_epoch = int(progress["epoch"])
        global_step = int(progress["global_step"])
        best_validation_loss = float(
            progress["metadata"].get("best_validation_loss", float("inf"))
        )
        print(f"Resuming after epoch {start_epoch}, global_step={global_step}")

    epochs = int(args.epochs if args.epochs is not None else training_cfg["epochs"])
    if epochs <= start_epoch:
        raise ValueError(f"Requested epochs={epochs} is not greater than restored epoch={start_epoch}.")

    metrics_path = Path(cfg["logging"]["metrics_jsonl"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg["logging"]["tensorboard_dir"])
    validation_seed = int(policy_cfg["validation"]["base_seed"])

    for epoch in range(start_epoch + 1, epochs + 1):
        train_records: list[dict] = []
        for batch_index, graph in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            metrics = train_one_batch(
                model=model,
                graph=graph,
                diffusion_process=diffusion,
                step_sampler=step_sampler,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                grad_clip_norm=float(training_cfg["grad_clip_norm"]),
            )
            train_records.append(metrics)
            global_step += 1

        if not train_records:
            raise RuntimeError("Training epoch produced no batches.")
        train_weighted = _average(train_records, "weighted_loss")
        train_unweighted = _average(train_records, "unweighted_loss")
        grad_norm = _average(train_records, "grad_norm")

        validation_loss = None
        if epoch % int(validation_cfg["every_epochs"]) == 0:
            val_records: list[dict] = []
            for batch_index, graph in enumerate(val_loader):
                if args.max_val_batches is not None and batch_index >= args.max_val_batches:
                    break
                val_records.append(
                    validate_one_batch(
                        model=model,
                        graph=graph,
                        diffusion_process=diffusion,
                        criterion=criterion,
                        validation_base_seed=validation_seed,
                        device=device,
                    )
                )
            if not val_records:
                raise RuntimeError("Validation produced no batches.")
            validation_loss = _average(val_records, "loss")
            best_validation_loss = min(best_validation_loss, validation_loss)

        monitor_name = str(scheduler_cfg["monitor"])
        if monitor_name == "train_weighted_loss":
            scheduler_value = train_weighted
        elif monitor_name == "validation_loss":
            if validation_loss is None:
                raise ValueError("Scheduler monitors validation_loss but validation did not run this epoch.")
            scheduler_value = validation_loss
        else:
            raise ValueError(f"Unsupported scheduler monitor '{monitor_name}'.")
        scheduler.step(scheduler_value)

        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_weighted_loss": train_weighted,
            "train_unweighted_loss": train_unweighted,
            "grad_norm": grad_norm,
            "validation_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
        }
        print(json.dumps(epoch_metrics, sort_keys=True))
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(epoch_metrics) + "\n")
        writer.add_scalar("loss/train_weighted", train_weighted, epoch)
        writer.add_scalar("loss/train_unweighted", train_unweighted, epoch)
        writer.add_scalar("optimization/grad_norm", grad_norm, epoch)
        writer.add_scalar("optimization/lr", optimizer.param_groups[0]["lr"], epoch)
        if validation_loss is not None:
            writer.add_scalar("loss/validation_deterministic", validation_loss, epoch)

        if epoch % int(training_cfg["checkpoint_every_epochs"]) == 0:
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                step_sampler=step_sampler,
                epoch=epoch,
                global_step=global_step,
                batch_in_epoch=0,
                data_generator=data_generator,
                metadata={
                    "contract": "d12_hit_physical_dgn_training",
                    "best_validation_loss": best_validation_loss,
                    "training_config": str(Path(args.config)),
                    "dataset_fingerprint_sha256": bundle.split_manifest[
                        "ordered_file_fingerprint_sha256"
                    ],
                    "hierarchy_fingerprint_sha256": bundle.hierarchy[
                        "hierarchy_fingerprint_sha256"
                    ],
                    "truncated_epoch_for_smoke_test": args.max_train_batches is not None,
                },
            )

        if optimizer.param_groups[0]["lr"] <= float(scheduler_cfg["min_lr"]):
            print("Reached configured minimum learning rate; stopping.")
            break

    writer.close()
    print(f"checkpoint: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
