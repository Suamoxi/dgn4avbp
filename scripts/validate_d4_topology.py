#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dgn4avbp.data import (
    AVBPHDF5FixedMeshDataset,
    HITReferenceScales,
    PreprocessedFixedMeshDataset,
    load_data_config,
    load_preprocessing_manifest,
    load_reference_config,
    load_split_manifest,
    run_hit_cartesian_diagnostics,
    standardizer_from_manifest,
    validate_native_hex_topology,
    validate_preprocessing_manifest,
    validate_split_manifest,
    write_topology_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate D4 native mesh topology with optional HIT Cartesian diagnostics."
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
        "--preprocessing-manifest",
        default="artifacts/d3_preprocessing_manifest.json",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/d4_topology_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = load_data_config(args.data_config)
    split_manifest = load_split_manifest(args.split_manifest)
    reference_cfg = load_reference_config(args.reference_config)
    preprocessing_manifest = load_preprocessing_manifest(args.preprocessing_manifest)

    dataset = AVBPHDF5FixedMeshDataset.from_config(data_cfg)
    validate_split_manifest(split_manifest, dataset.files)
    validate_preprocessing_manifest(preprocessing_manifest, split_manifest)

    frozen_reference_cfg = preprocessing_manifest["physical_nondimensionalization"][
        "reference_config"
    ]
    if frozen_reference_cfg != reference_cfg:
        raise ValueError(
            "Current HIT reference config differs from the reference config frozen in the D3 manifest."
        )

    refs = HITReferenceScales.from_config(reference_cfg)

    # Core D4 pass/fail contract: graph topology follows native mesh
    # connectivity. No Cartesian or structured-grid assumption is made here.
    manifest = validate_native_hex_topology(dataset)

    # D4 also checks that the frozen D3 wrapper expresses exactly the same
    # relative geometry in nondimensional units, without statistical coordinate
    # scaling. This is part of the preprocessing/topology interface contract and
    # remains a hard check.
    standardizer = standardizer_from_manifest(preprocessing_manifest)
    processed = PreprocessedFixedMeshDataset(dataset, refs, standardizer)
    torch.testing.assert_close(
        processed.pos,
        dataset.pos / refs.L_ref,
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        processed.edge_attr,
        dataset.edge_attr / refs.L_ref,
        rtol=1e-6,
        atol=1e-7,
    )

    # HIT happens to use a structured box, so retain the old Cartesian checks as
    # useful diagnostics. They are explicitly non-fatal: an arbitrary AVBP mesh
    # can satisfy D4 even when these case-specific assumptions do not apply.
    hit_cartesian = run_hit_cartesian_diagnostics(dataset, L_ref=refs.L_ref)

    manifest["dataset_fingerprint_sha256"] = split_manifest[
        "ordered_file_fingerprint_sha256"
    ]
    manifest["d3_fit_file_fingerprint_sha256"] = preprocessing_manifest[
        "fit_file_fingerprint_sha256"
    ]
    manifest["d3_reference_config_matches_current"] = True
    manifest["d3_geometry_consistency"] = {
        "coordinates_equal_raw_over_L_ref": True,
        "edge_attr_equal_raw_over_L_ref": True,
        "coordinates_statistically_standardized": False,
    }
    manifest["optional_diagnostics"] = {
        "hit_cartesian": hit_cartesian,
    }

    write_topology_manifest(manifest, args.manifest)

    print("D4 generic topology validation passed")
    print(f"mesh: {manifest['mesh_id']}")
    print(f"nodes: {manifest['num_nodes']}")
    print(f"cells: {manifest['num_cells']}")
    print(f"directed edges: {manifest['num_directed_edges']}")
    print(f"spatial dim: {manifest['spatial_dim']}")
    print(f"degree histogram: {manifest['degree_histogram']['observed']}")
    print(f"edge geometry: {manifest['edge_geometry']}")
    print(
        "legacy/current local mapping equivalent: "
        f"{manifest['local_hex_edge_mappings']['same_undirected_local_edge_set']}"
    )
    print(
        "legacy/current global graph equivalent: "
        f"{manifest['local_hex_edge_mappings']['same_global_directed_edge_set']}"
    )
    print(f"periodic closure added: {manifest['periodic_closure_added']}")
    print("D3 reference config matches current: True")

    print("HIT Cartesian diagnostic status:", hit_cartesian["status"])
    if hit_cartesian["status"] == "passed":
        grid = hit_cartesian["cartesian_grid"]
        geom = hit_cartesian["edge_geometry"]
        print(f"  axis counts: {grid['axis_counts']}")
        print(f"  physical span: {grid['axis_span']}")
        print(f"  physical spacing: {grid['axis_spacing']}")
        print(f"  nondimensional spacing: {grid['nondimensional_spacing']}")
        print(f"  nearest-neighbor fraction: {geom['nearest_neighbor_fraction']}")
        print(f"  cross-box edges: {geom['cross_box_edges']}")
        print(f"  opposite-face edges: {geom['opposite_face_edges']}")
    else:
        print(f"  diagnostic error: {hit_cartesian['error_type']}: {hit_cartesian['message']}")
        print("  This does not invalidate the D4 generic topology contract.")

    print(f"manifest: {Path(args.manifest).resolve()}")


if __name__ == "__main__":
    main()
