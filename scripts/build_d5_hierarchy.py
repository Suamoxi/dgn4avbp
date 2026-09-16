#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from dgn4avbp.data import (
    AVBPHDF5FixedMeshDataset,
    GEOMETRY_CONVENTION,
    HITReferenceScales,
    PreprocessedFixedMeshDataset,
    attach_hierarchy_to_graph,
    build_fixed_mesh_hierarchy,
    build_hierarchy_manifest,
    load_data_config,
    load_hierarchy_artifact,
    load_hierarchy_manifest,
    load_preprocessing_manifest,
    load_reference_config,
    load_split_manifest,
    save_hierarchy_artifact,
    standardizer_from_manifest,
    validate_hierarchy_manifest,
    validate_preprocessing_manifest,
    validate_split_manifest,
    write_hierarchy_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, persist, reload, and validate the D5 fixed HIT Guillard hierarchy."
    )
    parser.add_argument(
        "--data-config",
        default="configs/data/avbp_hdf5_fixed_mesh_local.yaml",
    )
    parser.add_argument(
        "--hierarchy-config",
        default="configs/data/hit_guillard_hierarchy.yaml",
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
        "--topology-manifest",
        default="artifacts/d4_topology_manifest.json",
    )
    parser.add_argument(
        "--artifact",
        default="artifacts/d5_hierarchy.pt",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/d5_hierarchy_manifest.json",
    )
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping in YAML '{path}'.")
    return cfg


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in JSON '{path}'.")
    return value


def main() -> None:
    args = parse_args()

    data_cfg = load_data_config(args.data_config)
    hierarchy_cfg = _load_yaml(args.hierarchy_config)
    split_manifest = load_split_manifest(args.split_manifest)
    reference_cfg = load_reference_config(args.reference_config)
    preprocessing_manifest = load_preprocessing_manifest(args.preprocessing_manifest)
    topology_manifest = _load_json(args.topology_manifest)

    if hierarchy_cfg.get("algorithm") != "guillard":
        raise ValueError(f"D5 requires algorithm='guillard', got {hierarchy_cfg.get('algorithm')!r}.")
    if hierarchy_cfg.get("geometry_convention") != GEOMETRY_CONVENTION:
        raise ValueError(
            "D5 hierarchy config geometry convention does not match the implementation."
        )
    if hierarchy_cfg.get("scalar_rel_pos") is not False:
        raise ValueError("D5 baseline requires vector fine-to-coarse displacements.")
    if hierarchy_cfg.get("periodic_closure_added") is not False:
        raise ValueError("D5 baseline must not add periodic closure.")

    dataset = AVBPHDF5FixedMeshDataset.from_config(data_cfg)
    validate_split_manifest(split_manifest, dataset.files)
    validate_preprocessing_manifest(preprocessing_manifest, split_manifest)

    frozen_reference_cfg = preprocessing_manifest["physical_nondimensionalization"][
        "reference_config"
    ]
    if frozen_reference_cfg != reference_cfg:
        raise ValueError(
            "Current HIT reference config differs from the reference config frozen in D3."
        )

    if topology_manifest.get("topology_contract") != "native_hexahedral_connectivity":
        raise ValueError("D5 requires the D4 native_hexahedral_connectivity contract.")
    if topology_manifest.get("dataset_fingerprint_sha256") != split_manifest.get(
        "ordered_file_fingerprint_sha256"
    ):
        raise ValueError("D4 topology manifest does not match the frozen D2 dataset fingerprint.")
    if topology_manifest.get("d3_fit_file_fingerprint_sha256") != preprocessing_manifest.get(
        "fit_file_fingerprint_sha256"
    ):
        raise ValueError("D4 topology manifest does not match the frozen D3 training fingerprint.")
    if topology_manifest.get("periodic_closure_added") is not False:
        raise ValueError("D4 topology manifest indicates periodic closure was added.")

    refs = HITReferenceScales.from_config(reference_cfg)
    standardizer = standardizer_from_manifest(preprocessing_manifest)
    processed_dataset = PreprocessedFixedMeshDataset(dataset, refs, standardizer)

    raw_max_levels = hierarchy_cfg.get("max_levels")
    max_levels = None if raw_max_levels is None else int(raw_max_levels)
    raw_max_indegree = hierarchy_cfg.get("max_indegree")
    max_indegree = None if raw_max_indegree is None else int(raw_max_indegree)

    artifact = build_fixed_mesh_hierarchy(
        processed_dataset.pos,
        processed_dataset.edge_index,
        max_levels=max_levels,
        max_indegree=max_indegree,
    )

    # The hierarchy base level must be exactly the frozen D3 geometry convention.
    torch.testing.assert_close(
        artifact["levels"][0]["pos"],
        processed_dataset.pos.cpu(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        artifact["levels"][0]["edge_attr"],
        processed_dataset.edge_attr.cpu(),
        rtol=0.0,
        atol=0.0,
    )

    manifest = build_hierarchy_manifest(
        artifact,
        mesh_id=dataset.mesh_id,
        dataset_fingerprint_sha256=split_manifest["ordered_file_fingerprint_sha256"],
        d3_fit_file_fingerprint_sha256=preprocessing_manifest[
            "fit_file_fingerprint_sha256"
        ],
        d4_topology_contract=topology_manifest["topology_contract"],
    )

    save_hierarchy_artifact(artifact, args.artifact)
    write_hierarchy_manifest(manifest, args.manifest)

    # Persistence is part of D5: immediately reload both files and revalidate.
    reloaded_artifact = load_hierarchy_artifact(args.artifact)
    reloaded_manifest = load_hierarchy_manifest(args.manifest)
    validate_hierarchy_manifest(reloaded_manifest, reloaded_artifact)

    # Verify that the persisted hierarchy can be reused on a physical sample
    # without recomputing Guillard coarsening.
    first_train_file = split_manifest["files"]["train"][0]
    first_train_index = dataset.files.index(first_train_file)
    graph = processed_dataset[first_train_index]
    graph = attach_hierarchy_to_graph(graph, reloaded_artifact)

    for transition, level in zip(
        reloaded_artifact["transitions"], reloaded_artifact["levels"][1:]
    ):
        hr = transition["from_level"]
        lr = transition["to_level"]
        torch.testing.assert_close(
            getattr(graph, f"pos_{lr}").cpu(), level["pos"], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            getattr(graph, f"edge_attr_{lr}").cpu(),
            level["edge_attr"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            getattr(graph, f"e_{hr}{lr}").cpu(),
            transition["e_hr_to_lr"],
            rtol=0.0,
            atol=0.0,
        )

    print("D5 hierarchy validation passed")
    print(f"mesh: {manifest['mesh_id']}")
    print(f"algorithm: {manifest['algorithm']}")
    print(f"geometry convention: {manifest['geometry_convention']}")
    print(f"levels built: {manifest['num_levels_built']}")
    print(f"stop reason: {manifest['stop_reason']}")
    print(f"hierarchy fingerprint: {manifest['hierarchy_fingerprint_sha256']}")
    print("level summary:")
    for level in manifest["levels"]:
        print(
            f"  L{level['level']}: N={level['num_nodes']} "
            f"E={level['num_directed_edges']} "
            f"edge_length_nd=[{level['min_edge_length_nd']:.6e}, "
            f"{level['mean_edge_length_nd']:.6e}, {level['max_edge_length_nd']:.6e}]"
        )
    print(f"artifact: {Path(args.artifact).resolve()}")
    print(f"manifest: {Path(args.manifest).resolve()}")


if __name__ == "__main__":
    main()
