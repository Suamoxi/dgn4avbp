#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from dgn4avbp.data import (
    AVBPHDF5FixedMeshDataset,
    load_data_config,
    validate_snapshot_dataset,
    write_data_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the D1 one-HDF5-snapshot AVBP data contract."
    )
    parser.add_argument(
        "--config",
        default="configs/data/avbp_hdf5_fixed_mesh_local.yaml",
        help="Path to the D1 data YAML.",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/d1_data_manifest.json",
        help="Output JSON manifest path.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Validate only the first field snapshot instead of scanning every solution file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_data_config(args.config)
    dataset = AVBPHDF5FixedMeshDataset.from_config(config)
    manifest = validate_snapshot_dataset(dataset, full_scan=not args.quick)
    write_data_manifest(dataset, args.manifest)

    print("D1 data validation passed")
    print(f"samples: {manifest['num_samples']}")
    print(f"nodes/sample: {manifest['num_nodes']}")
    print(f"cells: {manifest['num_cells']}")
    print(f"directed edges: {manifest['num_edges']}")
    print(f"channels: {manifest['channel_paths']}")
    print(f"mesh: {manifest['mesh_file']}")
    print(f"periodic edges added: {manifest['periodic_edges']}")
    print(f"temporal windowing: {manifest['temporal_windowing']}")
    print(f"manifest: {Path(args.manifest).resolve()}")


if __name__ == "__main__":
    main()
