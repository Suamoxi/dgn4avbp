#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dgn4avbp.data import (
    AVBPHDF5FixedMeshDataset,
    HITReferenceScales,
    PreprocessedFixedMeshDataset,
    build_preprocessing_manifest,
    compute_sample_balanced_standardized_moments,
    fit_sample_balanced_standardizer,
    load_data_config,
    load_reference_config,
    load_split_manifest,
    split_indices_from_manifest,
    validate_preprocessing_manifest,
    validate_split_manifest,
    write_preprocessing_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and validate D3 HIT physical/statistical preprocessing."
    )
    parser.add_argument(
        "--data-config",
        default="configs/data/avbp_hdf5_fixed_mesh_local.yaml",
    )
    parser.add_argument(
        "--split-manifest",
        default="artifacts/d2_split_manifest.json",
    )
    parser.add_argument(
        "--reference-config",
        default="configs/data/hit_reference_scales.yaml",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/d3_preprocessing_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = load_data_config(args.data_config)
    reference_cfg = load_reference_config(args.reference_config)
    split_manifest = load_split_manifest(args.split_manifest)

    dataset = AVBPHDF5FixedMeshDataset.from_config(data_cfg)
    validate_split_manifest(split_manifest, dataset.files)
    split_indices = split_indices_from_manifest(split_manifest, dataset.files)
    train_indices = split_indices["train"]

    refs = HITReferenceScales.from_config(reference_cfg)
    standardizer = fit_sample_balanced_standardizer(dataset, train_indices, refs)

    manifest = build_preprocessing_manifest(
        reference_config=reference_cfg,
        split_manifest=split_manifest,
        standardizer=standardizer,
    )
    validate_preprocessing_manifest(manifest, split_manifest)
    write_preprocessing_manifest(manifest, args.manifest)

    # Independent full-train verification of the frozen standardizer.
    standardized_mean, standardized_std = compute_sample_balanced_standardized_moments(
        dataset,
        train_indices,
        refs,
        standardizer,
    )
    torch.testing.assert_close(
        standardized_mean,
        torch.zeros_like(standardized_mean),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        standardized_std,
        torch.ones_like(standardized_std),
        rtol=0.0,
        atol=1e-6,
    )

    # Check dimensional <-> nondimensional <-> standardized round trip.
    processed_dataset = PreprocessedFixedMeshDataset(dataset, refs, standardizer)
    first_train_index = train_indices[0]
    raw = dataset[first_train_index]
    processed = processed_dataset[first_train_index]
    reconstructed = processed_dataset.inverse_target(processed.target)
    torch.testing.assert_close(reconstructed, raw.target, rtol=5e-5, atol=1e-6)

    pos_reconstructed = processed_dataset.pos.to(torch.float64) * refs.L_ref
    torch.testing.assert_close(
        pos_reconstructed,
        dataset.pos.to(torch.float64),
        rtol=1e-6,
        atol=1e-12,
    )

    span_nd = processed_dataset.pos.amax(dim=0) - processed_dataset.pos.amin(dim=0)
    torch.testing.assert_close(
        span_nd,
        torch.ones_like(span_nd),
        rtol=1e-4,
        atol=1e-5,
    )

    expected_edge_attr = (
        processed_dataset.pos[processed_dataset.edge_index[1]]
        - processed_dataset.pos[processed_dataset.edge_index[0]]
    )
    torch.testing.assert_close(processed_dataset.edge_attr, expected_edge_attr)

    print("D3 preprocessing validation passed")
    print(f"dataset fingerprint: {manifest['dataset_fingerprint_sha256']}")
    print(f"fit split: {manifest['fit_split']}")
    print(f"fit samples: {manifest['fit_num_samples']}")
    print(f"fit-file fingerprint: {manifest['fit_file_fingerprint_sha256']}")
    print(
        "references: "
        f"rho_ref={refs.rho_ref}, U_ref={refs.U_ref}, "
        f"L_ref={refs.L_ref}, T_ref={refs.T_ref}"
    )
    print(f"state scales: {refs.state_scales(dtype=torch.float64, device='cpu').tolist()}")
    print(f"nondimensional coordinate span: {span_nd.tolist()}")
    print(f"train nondimensional mean: {standardizer.mean.tolist()}")
    print(f"train nondimensional std: {standardizer.std.tolist()}")
    print(f"post-standardization mean: {standardized_mean.tolist()}")
    print(f"post-standardization std: {standardized_std.tolist()}")
    print(f"manifest: {Path(args.manifest).resolve()}")


if __name__ == "__main__":
    main()
