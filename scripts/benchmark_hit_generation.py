#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from dgn4avbp.hit_benchmark import benchmark_hit_populations
from dgn4avbp.hit_pipeline import build_hit_data_bundle, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark unconditional DGN HIT generations as an unpaired population.")
    parser.add_argument("--config", default="configs/benchmark/dgn_hit_distribution.yaml")
    parser.add_argument("--generation-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-reference-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _deterministic_subsample(values: np.ndarray, max_values: int, seed: int) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if flat.size <= max_values:
        return flat
    rng = np.random.default_rng(seed)
    indices = rng.choice(flat.size, size=max_values, replace=False)
    return flat[indices]


def _plot_channel_distributions(
    generated: torch.Tensor,
    reference: torch.Tensor,
    *,
    output_dir: Path,
    channel_names: tuple[str, ...],
    max_pooled_values: int,
    bins: int,
    dpi: int,
) -> dict[str, str]:
    if bins <= 0:
        raise ValueError("Distribution plot bin count must be positive.")
    if dpi <= 0:
        raise ValueError("Distribution plot DPI must be positive.")

    generated_np = generated.detach().cpu().numpy()
    reference_np = reference.detach().cpu().numpy()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    for channel, name in enumerate(channel_names):
        # Match the deterministic pooled samples used by D13 channel W1.
        gen_sample = _deterministic_subsample(
            generated_np[..., channel],
            max_pooled_values,
            1000 + channel,
        )
        ref_sample = _deterministic_subsample(
            reference_np[..., channel],
            max_pooled_values,
            2000 + channel,
        )

        combined = np.concatenate([gen_sample, ref_sample])
        if not np.isfinite(combined).all():
            raise ValueError(f"Non-finite values in distribution plot for channel {name}.")
        value_min = float(np.min(combined))
        value_max = float(np.max(combined))
        if value_max <= value_min:
            pad = max(abs(value_min) * 1.0e-6, 1.0e-12)
            value_min -= pad
            value_max += pad
        edges = np.linspace(value_min, value_max, bins + 1)

        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ax.hist(
            gen_sample,
            bins=edges,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=f"Generated (n={gen_sample.size:,})",
        )
        ax.hist(
            ref_sample,
            bins=edges,
            density=True,
            histtype="step",
            linewidth=1.5,
            linestyle="--",
            label=f"Test (n={ref_sample.size:,})",
        )
        ax.set_xlabel(f"{name} (nondimensional)")
        ax.set_ylabel("Probability density")
        ax.set_title(f"{name}: generated vs test distribution")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()

        filename = f"{name}.png"
        path = output_dir / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        outputs[name] = filename

    return outputs


def _load_generated_population(directory: Path) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    files = sorted(directory.glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"No sample_*.pt generation artifacts found in {directory}.")

    states = []
    generated_ids = []
    pos = None
    checkpoint_values: set[str] = set()
    for path in files:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Generation artifact is not a mapping: {path}")
        state = payload.get("state_nondimensional")
        position = payload.get("pos_nondimensional")
        if not isinstance(state, torch.Tensor) or state.ndim != 2 or state.shape[1] != 5:
            raise ValueError(f"Invalid state_nondimensional in {path}.")
        if not isinstance(position, torch.Tensor) or position.ndim != 2 or position.shape[1] != 3:
            raise ValueError(f"Invalid pos_nondimensional in {path}.")
        if not torch.isfinite(state).all() or not torch.isfinite(position).all():
            raise ValueError(f"Non-finite generated artifact: {path}.")
        if pos is None:
            pos = position
        else:
            torch.testing.assert_close(position, pos, rtol=0.0, atol=0.0)
        states.append(state)
        seed = payload.get("seed", "unknown")
        sample_index = payload.get("sample_index", len(generated_ids))
        generated_ids.append(f"seed={seed}:sample={sample_index}")
        checkpoint_values.add(str(payload.get("checkpoint", "unknown")))

    assert pos is not None
    return torch.stack(states, dim=0), pos, generated_ids, sorted(checkpoint_values)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if str(cfg.get("comparison_mode")) != "unpaired_population":
        raise ValueError("D13 requires comparison_mode=unpaired_population.")
    if str(cfg.get("state_space")) != "nondimensional":
        raise ValueError("D13 currently benchmarks physically nondimensional states only.")

    training_cfg = load_yaml(cfg["training_config"])
    bundle = build_hit_data_bundle(**training_cfg["data"])

    generation_dir = Path(args.generation_dir or cfg["generation_dir"]).expanduser().resolve()
    output_dir = Path(args.output_dir or cfg["output_dir"]).expanduser().resolve()
    generated, generated_pos, generated_ids, checkpoints = _load_generated_population(generation_dir)

    reference_indices = list(bundle.split_indices["test"])
    if args.max_reference_samples is not None:
        reference_indices = reference_indices[: int(args.max_reference_samples)]
    if not reference_indices:
        raise ValueError("D13 reference population is empty.")

    reference_states = []
    reference_ids = []
    for index in reference_indices:
        processed = bundle.processed_dataset[index]
        reference_states.append(bundle.standardizer.inverse(processed.target).to(torch.float32))
        reference_ids.append(str(processed.sample_id))
    reference = torch.stack(reference_states, dim=0)

    torch.testing.assert_close(
        generated_pos,
        bundle.processed_dataset.pos.cpu(),
        rtol=0.0,
        atol=0.0,
    )

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Benchmark output already exists: {output_dir}. Use --overwrite explicitly.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    channel_names = ("rho", "rhou", "rhov", "rhow", "rhoE")
    max_pooled_values = int(cfg["max_pooled_values"])

    result = benchmark_hit_populations(
        generated,
        reference,
        generated_pos,
        L_ref=bundle.refs.L_ref,
        channel_names=channel_names,
        quantiles=tuple(float(value) for value in cfg["quantiles"]),
        spectral_bands=cfg["spectra"]["bands"],
        max_pooled_values=max_pooled_values,
    )

    distribution_plot_outputs: dict[str, str] = {}
    plot_cfg = cfg.get("plots", {}).get("channel_distributions", {})
    if bool(plot_cfg.get("enabled", False)):
        plot_dir_name = str(plot_cfg.get("directory", "channel_distributions"))
        plot_dir = output_dir / plot_dir_name
        plot_files = _plot_channel_distributions(
            generated,
            reference,
            output_dir=plot_dir,
            channel_names=channel_names,
            max_pooled_values=max_pooled_values,
            bins=int(plot_cfg.get("bins", 160)),
            dpi=int(plot_cfg.get("dpi", 220)),
        )
        distribution_plot_outputs = {
            name: str(Path(plot_dir_name) / filename)
            for name, filename in plot_files.items()
        }

    summary = result["summary"]
    summary.update(
        {
            "dataset_fingerprint_sha256": bundle.split_manifest["ordered_file_fingerprint_sha256"],
            "reference_population": "test",
            "generated_ids": generated_ids,
            "reference_ids": reference_ids,
            "generated_ids_semantics": "sampling_seed_and_sample_index_only",
            "generated_reference_pairing": False,
            "source_generation_dir": str(generation_dir),
            "source_checkpoints": checkpoints,
            "outputs": {
                "channel_metrics": "channel_metrics.csv",
                "physical_metrics": "physical_metrics.csv",
                "spectra": "spectra.csv",
                "spectral_bands": "spectral_bands.csv",
                "channel_distribution_plots": distribution_plot_outputs,
            },
        }
    )

    _write_csv(output_dir / cfg["outputs"]["channel_metrics"], result["channel_rows"])
    _write_csv(output_dir / cfg["outputs"]["physical_metrics"], result["physical_rows"])
    _write_csv(output_dir / cfg["outputs"]["spectra"], result["spectra_rows"])
    _write_csv(output_dir / cfg["outputs"]["spectral_bands"], result["spectral_band_rows"])
    (output_dir / cfg["outputs"]["summary"]).write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("D13 unpaired HIT population benchmark complete")
    print(f"generated samples: {generated.shape[0]}")
    print(f"reference test samples: {reference.shape[0]}")
    print(f"grid: {summary['grid_shape_with_periodic_endpoint']} -> FFT {summary['fft_grid_shape_unique_periodic_nodes']}")
    print(f"k_Nyquist * L_ref: {summary['k_nyquist_Lref']:.6g}")
    if distribution_plot_outputs:
        print("channel distribution plots:")
        for name, path in distribution_plot_outputs.items():
            print(f"  {name}: {path}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
