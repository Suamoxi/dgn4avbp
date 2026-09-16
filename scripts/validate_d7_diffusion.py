from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the D7 canonical Improved-DDPM contract.")
    parser.add_argument(
        "--config",
        default="configs/diffusion/improved_ddpm_hit.yaml",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/d7_diffusion_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    manifest_path = Path(args.manifest)

    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    dp = DiffusionProcess(
        num_steps=int(cfg["num_steps"]),
        schedule_type=str(cfg["schedule_type"]),
        beta_start=float(cfg["beta_start"]),
        beta_end=float(cfg["beta_end"]),
        max_beta=float(cfg["max_beta"]),
    )

    if cfg["prediction"] != "epsilon":
        raise ValueError("D7 canonical contract requires epsilon prediction.")
    if cfg["variance"] != "learned_range" or cfg["learnable_variance"] is not True:
        raise ValueError("D7 canonical contract requires learned-range variance.")
    if int(cfg["num_steps"]) != 1000:
        raise ValueError("D7 baseline freezes the original DGN 1000-step schedule.")

    expected_multiplier = float(cfg["lambda_vlb"]) * dp.num_steps
    if expected_multiplier != float(cfg["sampled_vlb_multiplier"]):
        raise ValueError(
            "Configured sampled_vlb_multiplier does not equal lambda_vlb * num_steps."
        )

    torch.manual_seed(17)
    num_nodes = 32
    channels = int(cfg["scientific_scope"]["state_channels"])
    batch = torch.zeros(num_nodes, dtype=torch.long)
    field_start = torch.randn(num_nodes, channels)

    # Learned-range endpoints at a representative interior timestep.
    r_mid = torch.tensor([500], dtype=torch.long)
    v_minus = -torch.ones_like(field_start)
    v_plus = torch.ones_like(field_start)
    min_log = learned_range_log_variance(dp, v_minus, batch, r_mid)
    max_log = learned_range_log_variance(dp, v_plus, batch, r_mid)
    expected_min = dp.get_index_from_list(dp.posterior_log_variance_clipped, batch, r_mid)
    expected_max = torch.log(dp.get_index_from_list(dp.betas, batch, r_mid))
    torch.testing.assert_close(min_log, expected_min.expand_as(field_start))
    torch.testing.assert_close(max_log, expected_max.expand_as(field_start))

    # Hybrid loss at an interior timestep.
    field_mid, noise_mid = dp(field_start, r_mid, batch)
    model_epsilon = noise_mid + 0.1 * torch.randn_like(noise_mid)
    model_v = torch.zeros_like(noise_mid)
    terms_mid = hybrid_loss_terms(
        dp,
        field_start=field_start,
        field_r=field_mid,
        noise=noise_mid,
        model_epsilon=model_epsilon,
        model_v=model_v,
        batch=batch,
        r=r_mid,
        lambda_vlb=float(cfg["lambda_vlb"]),
    )
    if not torch.isfinite(terms_mid["loss"]).all():
        raise AssertionError("D7 interior hybrid loss is non-finite.")

    # Continuous t=0 decoder branch must be finite.
    r_zero = torch.tensor([0], dtype=torch.long)
    field_zero, noise_zero = dp(field_start, r_zero, batch)
    vb_zero = continuous_vlb_term(
        dp,
        field_start=field_start,
        field_r=field_zero,
        model_epsilon=noise_zero,
        model_v=torch.zeros_like(noise_zero),
        batch=batch,
        r=r_zero,
    )
    if not torch.isfinite(vb_zero).all():
        raise AssertionError("D7 t=0 continuous decoder term is non-finite.")

    # The final reverse step must be deterministic with respect to Gaussian noise.
    sample_zero_a = ancestral_sample_step(
        dp,
        field_r=field_zero,
        model_epsilon=noise_zero,
        model_v=torch.zeros_like(noise_zero),
        batch=batch,
        r=r_zero,
        gaussian_noise=torch.randn_like(field_zero),
    )
    sample_zero_b = ancestral_sample_step(
        dp,
        field_r=field_zero,
        model_epsilon=noise_zero,
        model_v=torch.zeros_like(noise_zero),
        batch=batch,
        r=r_zero,
        gaussian_noise=torch.randn_like(field_zero),
    )
    torch.testing.assert_close(sample_zero_a, sample_zero_b, rtol=0.0, atol=0.0)

    manifest = {
        "version": 1,
        "contract": "canonical_improved_ddpm_for_physical_hit_dgn",
        "source_lineage": {
            "dgn4cfd_diffusion_process_preserved": True,
            "prediction": "epsilon",
            "variance": "learned_range",
        },
        "schedule": {
            "num_steps": dp.num_steps,
            "type": cfg["schedule_type"],
            "beta_start": float(dp.betas[0].item()),
            "beta_end": float(dp.betas[-1].item()),
            "alpha_bar_final": float(dp.alphas_cumprod[-1].item()),
        },
        "timestep_embedding": cfg["timestep_embedding"],
        "hybrid_objective": {
            "lambda_vlb": float(cfg["lambda_vlb"]),
            "sampled_vlb_multiplier": expected_multiplier,
            "stop_gradient_on_epsilon_for_vlb": True,
            "interior_mse": float(terms_mid["mse"].item()),
            "interior_vb_bits_per_dimension": float(terms_mid["vb"].item()),
            "interior_hybrid_loss": float(terms_mid["loss"].item()),
        },
        "continuous_t0_decoder": {
            "likelihood": "gaussian_nll",
            "units": "bits_per_dimension",
            "finite": bool(torch.isfinite(vb_zero).all().item()),
            "value": float(vb_zero.item()),
        },
        "sampling": {
            "method": "ancestral_ddpm",
            "noise_suppressed_at_t0": bool(torch.equal(sample_zero_a, sample_zero_b)),
        },
        "corrections_relative_to_legacy_dgn4cfd_path": [
            "sampled VLB term scaled by lambda_vlb * num_steps",
            "continuous t=0 Gaussian decoder NLL expressed in bits/dimension",
            "ancestral Gaussian noise suppressed at t=0",
        ],
        "deferred": [
            "loss-second-moment sampler and deterministic validation policy (D8)",
            "hierarchy-aware multi-sample batching (D9)",
            "training/checkpoint integration",
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("D7 Improved-DDPM validation passed")
    print("num steps:", dp.num_steps)
    print("schedule:", cfg["schedule_type"])
    print("beta range:", float(dp.betas[0]), "->", float(dp.betas[-1]))
    print("alpha_bar(T-1):", float(dp.alphas_cumprod[-1]))
    print("lambda_vlb:", float(cfg["lambda_vlb"]))
    print("sampled VLB multiplier:", expected_multiplier)
    print("t=0 decoder finite:", bool(torch.isfinite(vb_zero).all().item()))
    print("t=0 sampling noise suppressed:", bool(torch.equal(sample_zero_a, sample_zero_b)))
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
