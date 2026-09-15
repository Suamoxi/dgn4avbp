#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from dgn4avbp.data import (
    AVBPHDF5FixedMeshDataset,
    create_contiguous_split_manifest,
    load_data_config,
    load_split_config,
    validate_split_manifest,
    write_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and validate the persistent D2 contiguous HIT split manifest."
    )
    parser.add_argument(
        "--data-config",
        default="configs/data/avbp_hdf5_fixed_mesh_local.yaml",
        help="Path to the D1 fixed-mesh data YAML.",
    )
    parser.add_argument(
        "--split-config",
        default="configs/data/hit_contiguous_split.yaml",
        help="Path to the D2 split YAML.",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/d2_split_manifest.json",
        help="Output split manifest path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = load_data_config(args.data_config)
    split_cfg = load_split_config(args.split_config)

    strategy = split_cfg.get("strategy")
    if strategy != "contiguous_by_snapshot_iteration":
        raise ValueError(
            "D2 currently supports only strategy='contiguous_by_snapshot_iteration', "
            f"got {strategy!r}."
        )

    dataset = AVBPHDF5FixedMeshDataset.from_config(data_cfg)
    manifest = create_contiguous_split_manifest(
        dataset.files,
        train_ratio=float(split_cfg.get("train_ratio", 0.8)),
        val_ratio=float(split_cfg.get("val_ratio", 0.1)),
        test_ratio=float(split_cfg.get("test_ratio", 0.1)),
    )
    validate_split_manifest(manifest, dataset.files)
    write_split_manifest(manifest, args.manifest)

    print("D2 split validation passed")
    print(f"samples: {manifest['num_samples']}")
    print(f"strategy: {manifest['strategy']}")
    print(f"counts: {manifest['counts']}")
    print(f"actual ratios: {manifest['actual_ratios']}")
    print(f"iteration ranges: {manifest['iteration_ranges']}")
    print(f"boundary gaps: {manifest['boundary_iteration_gaps']}")
    print(f"file fingerprint: {manifest['ordered_file_fingerprint_sha256']}")
    print(f"manifest: {Path(args.manifest).resolve()}")


if __name__ == "__main__":
    main()
