from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
from torch_geometric.data import Data

from dgn4avbp.transform_locals import guillard_coarsening, pool_edges

from .topology import directed_edge_sets_equal


HIERARCHY_VERSION = 1
HIERARCHY_ALGORITHM = "guillard"
GEOMETRY_CONVENTION = "nondimensional_by_L_ref_no_level_rescaling"


def _edge_attr(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    return pos[edge_index[1]] - pos[edge_index[0]]


def _tensor_hash_update(hasher: "hashlib._Hash", name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    hasher.update(name.encode("utf-8"))
    hasher.update(str(value.dtype).encode("utf-8"))
    hasher.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
    hasher.update(value.numpy().tobytes())


def hierarchy_fingerprint(artifact: dict) -> str:
    """Logical SHA-256 over all hierarchy tensors and structural metadata."""

    hasher = hashlib.sha256()
    hasher.update(str(artifact.get("version")).encode("utf-8"))
    hasher.update(str(artifact.get("algorithm")).encode("utf-8"))
    hasher.update(str(artifact.get("geometry_convention")).encode("utf-8"))
    hasher.update(str(artifact.get("max_indegree")).encode("utf-8"))

    for level in artifact["levels"]:
        prefix = f"level_{level['level']}"
        _tensor_hash_update(hasher, f"{prefix}.pos", level["pos"])
        _tensor_hash_update(hasher, f"{prefix}.edge_index", level["edge_index"])
        _tensor_hash_update(hasher, f"{prefix}.edge_attr", level["edge_attr"])

    for transition in artifact["transitions"]:
        prefix = f"transition_{transition['from_level']}_{transition['to_level']}"
        _tensor_hash_update(hasher, f"{prefix}.coarse_mask", transition["coarse_mask"])
        _tensor_hash_update(hasher, f"{prefix}.idx_to_parent", transition["idx_to_parent"])
        _tensor_hash_update(hasher, f"{prefix}.e_hr_to_lr", transition["e_hr_to_lr"])

    return hasher.hexdigest()


def build_fixed_mesh_hierarchy(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    max_levels: int | None = None,
    max_indegree: int | None = None,
) -> dict:
    """Build the deterministic Guillard hierarchy once for a fixed mesh.

    ``pos`` must already be in the D3 nondimensional coordinate system. No
    additional level-wise position or relative-vector scaling is applied.

    When ``max_levels`` is ``None`` the hierarchy is extended until another
    Guillard step would produce fewer than two coarse nodes, or until the graph
    no longer has edges. This lets D5 measure the available hierarchy instead of
    assuming the historical five-scale choice. D6 will choose the model prefix.
    """

    if pos.ndim != 2 or pos.shape[0] < 2:
        raise ValueError(f"Expected positions [N, dim] with N >= 2, got {tuple(pos.shape)}.")
    if not torch.isfinite(pos).all():
        raise ValueError("Hierarchy positions contain non-finite values.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] == 0:
        raise ValueError(f"Expected non-empty edge_index [2, E], got {tuple(edge_index.shape)}.")
    if max_levels is not None and max_levels < 1:
        raise ValueError(f"max_levels must be >= 1 or None, got {max_levels}.")
    if max_indegree is not None and max_indegree <= 0:
        raise ValueError(f"max_indegree must be > 0 or None, got {max_indegree}.")

    pos_current = pos.detach().cpu().contiguous()
    edge_index_current = edge_index.detach().cpu().long().contiguous()
    edge_attr_current = _edge_attr(pos_current, edge_index_current)

    levels: list[dict] = [
        {
            "level": 1,
            "pos": pos_current,
            "edge_index": edge_index_current,
            "edge_attr": edge_attr_current,
        }
    ]
    transitions: list[dict] = []
    stop_reason = "unknown"

    while True:
        if max_levels is not None and len(levels) >= max_levels:
            stop_reason = "max_levels_reached"
            break
        if pos_current.shape[0] <= 1:
            stop_reason = "single_node"
            break
        if edge_index_current.shape[1] == 0:
            stop_reason = "no_edges"
            break

        coarse_mask, idx_to_parent = guillard_coarsening(pos_current, edge_index_current)
        num_coarse = int(coarse_mask.sum().item())
        num_fine = int(pos_current.shape[0])

        if num_coarse >= num_fine:
            stop_reason = "no_node_reduction"
            break
        if num_coarse < 2:
            # Existing DGN message-passing blocks require a graph rather than a
            # singleton with no edges. Do not materialize that unusable level.
            stop_reason = "next_level_fewer_than_two_nodes"
            break

        pos_coarse = pos_current[coarse_mask].contiguous()
        edge_index_coarse = pool_edges(
            coarse_mask=coarse_mask,
            idxHR_to_idxLR=idx_to_parent,
            edge_index=edge_index_current,
            max_indegree=max_indegree,
            pos=pos_coarse,
        ).long().contiguous()

        if edge_index_coarse.shape[1] == 0:
            stop_reason = "next_level_has_no_edges"
            break

        edge_attr_coarse = _edge_attr(pos_coarse, edge_index_coarse).contiguous()
        e_hr_to_lr = (pos_coarse[idx_to_parent] - pos_current).contiguous()

        from_level = len(levels)
        to_level = from_level + 1
        transitions.append(
            {
                "from_level": from_level,
                "to_level": to_level,
                "coarse_mask": coarse_mask.cpu().contiguous(),
                "idx_to_parent": idx_to_parent.cpu().long().contiguous(),
                "e_hr_to_lr": e_hr_to_lr.cpu().contiguous(),
            }
        )
        levels.append(
            {
                "level": to_level,
                "pos": pos_coarse.cpu().contiguous(),
                "edge_index": edge_index_coarse.cpu().contiguous(),
                "edge_attr": edge_attr_coarse.cpu().contiguous(),
            }
        )

        pos_current = pos_coarse
        edge_index_current = edge_index_coarse

    artifact = {
        "version": HIERARCHY_VERSION,
        "algorithm": HIERARCHY_ALGORITHM,
        "geometry_convention": GEOMETRY_CONVENTION,
        "max_levels_requested": max_levels,
        "max_indegree": max_indegree,
        "scalar_rel_pos": False,
        "periodic_closure_added": False,
        "stop_reason": stop_reason,
        "levels": levels,
        "transitions": transitions,
    }
    validate_fixed_mesh_hierarchy(artifact)
    artifact["hierarchy_fingerprint_sha256"] = hierarchy_fingerprint(artifact)
    return artifact


def validate_fixed_mesh_hierarchy(artifact: dict) -> None:
    if artifact.get("version") != HIERARCHY_VERSION:
        raise ValueError(f"Unsupported hierarchy version {artifact.get('version')}.")
    if artifact.get("algorithm") != HIERARCHY_ALGORITHM:
        raise ValueError(f"Unsupported hierarchy algorithm {artifact.get('algorithm')!r}.")
    if artifact.get("geometry_convention") != GEOMETRY_CONVENTION:
        raise ValueError(
            f"Unexpected geometry convention {artifact.get('geometry_convention')!r}."
        )
    if artifact.get("scalar_rel_pos") is not False:
        raise ValueError("D5 baseline requires vector fine-to-coarse relative positions.")
    if artifact.get("periodic_closure_added") is not False:
        raise ValueError("D5 must not add periodic closure.")

    levels = artifact.get("levels")
    transitions = artifact.get("transitions")
    if not isinstance(levels, list) or not levels:
        raise ValueError("Hierarchy must contain at least one level.")
    if not isinstance(transitions, list) or len(transitions) != len(levels) - 1:
        raise ValueError("Hierarchy must contain exactly one transition between adjacent levels.")

    max_indegree = artifact.get("max_indegree")
    previous_num_nodes: int | None = None

    for index, level in enumerate(levels, start=1):
        if level.get("level") != index:
            raise ValueError(f"Expected hierarchy level number {index}, got {level.get('level')}.")
        pos = level["pos"]
        edge_index = level["edge_index"]
        edge_attr = level["edge_attr"]

        if pos.ndim != 2 or pos.shape[0] < 2:
            raise ValueError(f"Level {index} positions must have shape [N, dim] with N >= 2.")
        if not torch.isfinite(pos).all():
            raise ValueError(f"Level {index} positions contain non-finite values.")
        num_nodes = int(pos.shape[0])
        if previous_num_nodes is not None and num_nodes >= previous_num_nodes:
            raise AssertionError(
                f"Hierarchy node count did not decrease from {previous_num_nodes} to {num_nodes}."
            )
        previous_num_nodes = num_nodes

        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] == 0:
            raise ValueError(f"Level {index} edge_index must be non-empty [2, E].")
        if int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= num_nodes:
            raise ValueError(f"Level {index} edge_index is outside node range.")
        if int((edge_index[0] == edge_index[1]).sum().item()) != 0:
            raise AssertionError(f"Level {index} contains self-loops.")
        if not directed_edge_sets_equal(edge_index, edge_index[[1, 0]], num_nodes):
            raise AssertionError(f"Level {index} graph is not bidirectional.")

        expected_edge_attr = _edge_attr(pos, edge_index)
        torch.testing.assert_close(edge_attr, expected_edge_attr, rtol=0.0, atol=0.0)
        if not torch.isfinite(edge_attr).all():
            raise AssertionError(f"Level {index} edge_attr contains non-finite values.")
        if torch.linalg.vector_norm(edge_attr.to(torch.float64), dim=1).eq(0).any():
            raise AssertionError(f"Level {index} contains zero-length edges.")

    for index, transition in enumerate(transitions):
        hr = levels[index]
        lr = levels[index + 1]
        expected_from = index + 1
        expected_to = index + 2
        if transition.get("from_level") != expected_from or transition.get("to_level") != expected_to:
            raise ValueError(
                f"Transition {index} expected {expected_from}->{expected_to}, got "
                f"{transition.get('from_level')}->{transition.get('to_level')}."
            )

        coarse_mask = transition["coarse_mask"]
        idx_to_parent = transition["idx_to_parent"]
        e_hr_to_lr = transition["e_hr_to_lr"]
        n_hr = int(hr["pos"].shape[0])
        n_lr = int(lr["pos"].shape[0])

        if coarse_mask.dtype != torch.bool or coarse_mask.shape != (n_hr,):
            raise ValueError(f"Transition {expected_from}->{expected_to} has invalid coarse_mask.")
        if int(coarse_mask.sum().item()) != n_lr:
            raise AssertionError(
                f"Transition {expected_from}->{expected_to} coarse_mask selects "
                f"{coarse_mask.sum().item()} nodes, expected {n_lr}."
            )
        torch.testing.assert_close(lr["pos"], hr["pos"][coarse_mask], rtol=0.0, atol=0.0)

        if idx_to_parent.shape != (n_hr,) or idx_to_parent.dtype != torch.long:
            raise ValueError(f"Transition {expected_from}->{expected_to} has invalid parent map.")
        if int(idx_to_parent.min().item()) < 0 or int(idx_to_parent.max().item()) >= n_lr:
            raise ValueError(f"Transition {expected_from}->{expected_to} parent map is out of range.")

        expected_e = lr["pos"][idx_to_parent] - hr["pos"]
        torch.testing.assert_close(e_hr_to_lr, expected_e, rtol=0.0, atol=0.0)

        expected_lr_edges = pool_edges(
            coarse_mask=coarse_mask,
            idxHR_to_idxLR=idx_to_parent,
            edge_index=hr["edge_index"],
            max_indegree=max_indegree,
            pos=lr["pos"],
        )
        if not directed_edge_sets_equal(
            lr["edge_index"], expected_lr_edges, n_lr
        ):
            raise AssertionError(
                f"Transition {expected_from}->{expected_to} coarse connectivity is not reproducible."
            )

    stored_fingerprint = artifact.get("hierarchy_fingerprint_sha256")
    if stored_fingerprint is not None:
        observed = hierarchy_fingerprint(artifact)
        if stored_fingerprint != observed:
            raise ValueError(
                f"Hierarchy fingerprint mismatch: stored {stored_fingerprint}, observed {observed}."
            )


def _level_summary(level: dict) -> dict:
    lengths = torch.linalg.vector_norm(level["edge_attr"].to(torch.float64), dim=1)
    return {
        "level": int(level["level"]),
        "num_nodes": int(level["pos"].shape[0]),
        "num_directed_edges": int(level["edge_index"].shape[1]),
        "min_edge_length_nd": float(lengths.min().item()),
        "max_edge_length_nd": float(lengths.max().item()),
        "mean_edge_length_nd": float(lengths.mean().item()),
    }


def build_hierarchy_manifest(
    artifact: dict,
    *,
    mesh_id: str,
    dataset_fingerprint_sha256: str,
    d3_fit_file_fingerprint_sha256: str,
    d4_topology_contract: str,
) -> dict:
    validate_fixed_mesh_hierarchy(artifact)

    transition_summaries = []
    for transition, hr, lr in zip(
        artifact["transitions"], artifact["levels"][:-1], artifact["levels"][1:]
    ):
        transition_summaries.append(
            {
                "from_level": transition["from_level"],
                "to_level": transition["to_level"],
                "num_hr_nodes": int(hr["pos"].shape[0]),
                "num_lr_nodes": int(lr["pos"].shape[0]),
                "coarse_node_fraction": float(lr["pos"].shape[0] / hr["pos"].shape[0]),
                "num_hr_directed_edges": int(hr["edge_index"].shape[1]),
                "num_lr_directed_edges": int(lr["edge_index"].shape[1]),
            }
        )

    return {
        "version": HIERARCHY_VERSION,
        "algorithm": artifact["algorithm"],
        "geometry_convention": artifact["geometry_convention"],
        "geometry_statement": (
            "All hierarchy coordinates, coarse edge vectors, and fine-to-coarse displacement "
            "vectors use the D3 coordinates x/L_ref. No h-based or level-wise rescaling is applied."
        ),
        "mesh_id": mesh_id,
        "dataset_fingerprint_sha256": dataset_fingerprint_sha256,
        "d3_fit_file_fingerprint_sha256": d3_fit_file_fingerprint_sha256,
        "d4_topology_contract": d4_topology_contract,
        "hierarchy_fingerprint_sha256": artifact["hierarchy_fingerprint_sha256"],
        "max_levels_requested": artifact["max_levels_requested"],
        "max_indegree": artifact["max_indegree"],
        "scalar_rel_pos": artifact["scalar_rel_pos"],
        "periodic_closure_added": artifact["periodic_closure_added"],
        "num_levels_built": len(artifact["levels"]),
        "stop_reason": artifact["stop_reason"],
        "levels": [_level_summary(level) for level in artifact["levels"]],
        "transitions": transition_summaries,
        "model_scale_selection": "deferred_to_D6_use_a_prefix_of_this_fixed_hierarchy",
    }


def save_hierarchy_artifact(artifact: dict, path: str | Path) -> None:
    validate_fixed_mesh_hierarchy(artifact)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def load_hierarchy_artifact(path: str | Path) -> dict:
    path = Path(path)
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise ValueError(f"Expected hierarchy artifact mapping in '{path}'.")
    validate_fixed_mesh_hierarchy(artifact)
    return artifact


def write_hierarchy_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_hierarchy_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected hierarchy manifest mapping in '{path}'.")
    return manifest


def validate_hierarchy_manifest(manifest: dict, artifact: dict) -> None:
    validate_fixed_mesh_hierarchy(artifact)
    if manifest.get("hierarchy_fingerprint_sha256") != artifact.get(
        "hierarchy_fingerprint_sha256"
    ):
        raise ValueError("D5 hierarchy manifest does not match hierarchy tensor artifact.")
    if manifest.get("num_levels_built") != len(artifact["levels"]):
        raise ValueError("D5 hierarchy manifest level count does not match tensor artifact.")
    if manifest.get("geometry_convention") != GEOMETRY_CONVENTION:
        raise ValueError("D5 hierarchy manifest has an unexpected geometry convention.")
    if manifest.get("periodic_closure_added") is not False:
        raise ValueError("D5 hierarchy manifest must state that no periodic closure was added.")


def attach_hierarchy_to_graph(graph: Data, artifact: dict) -> Data:
    """Attach a persisted single-mesh hierarchy using the names expected by MultiScaleGnn.

    D5 intentionally handles one physical graph here. Replication and correct
    index offsets for computational batches are owned by D9.
    """

    validate_fixed_mesh_hierarchy(artifact)
    if getattr(graph, "num_graphs", 1) != 1:
        raise ValueError("D5 hierarchy attachment expects one physical graph; D9 owns batching.")

    base = artifact["levels"][0]
    if graph.pos.shape != base["pos"].shape:
        raise ValueError("Graph position shape does not match D5 hierarchy base level.")
    if graph.edge_index.shape != base["edge_index"].shape:
        raise ValueError("Graph edge_index shape does not match D5 hierarchy base level.")
    torch.testing.assert_close(graph.pos.cpu(), base["pos"], rtol=1e-6, atol=1e-7)
    if not directed_edge_sets_equal(graph.edge_index.cpu(), base["edge_index"], graph.pos.shape[0]):
        raise ValueError("Graph edge_index does not match D5 hierarchy base level.")
    torch.testing.assert_close(graph.edge_attr.cpu(), base["edge_attr"], rtol=1e-6, atol=1e-7)

    graph.batch = torch.zeros(graph.pos.shape[0], dtype=torch.long, device=graph.pos.device)

    for transition, level in zip(artifact["transitions"], artifact["levels"][1:]):
        hr = int(transition["from_level"])
        lr = int(transition["to_level"])
        device = graph.pos.device
        setattr(graph, f"coarse_mask_{lr}", transition["coarse_mask"].to(device))
        setattr(graph, f"pos_{lr}", level["pos"].to(device))
        setattr(graph, f"idx{hr}_to_idx{lr}", transition["idx_to_parent"].to(device))
        setattr(graph, f"edge_index_{lr}", level["edge_index"].to(device))
        setattr(graph, f"edge_attr_{lr}", level["edge_attr"].to(device))
        setattr(graph, f"e_{hr}{lr}", transition["e_hr_to_lr"].to(device))
        setattr(
            graph,
            f"batch_{lr}",
            torch.zeros(level["pos"].shape[0], dtype=torch.long, device=device),
        )
    return graph


class HierarchyAttachedDataset(torch.utils.data.Dataset):
    """Reuse one persisted fixed-mesh hierarchy for every physical snapshot."""

    def __init__(self, dataset, artifact: dict) -> None:
        validate_fixed_mesh_hierarchy(artifact)
        self.dataset = dataset
        self.artifact = artifact
        self.files = getattr(dataset, "files", None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Data:
        graph = self.dataset[index]
        return attach_hierarchy_to_graph(graph, self.artifact)
