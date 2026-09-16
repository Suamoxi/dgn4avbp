from __future__ import annotations

import torch
from torch_geometric.data import Data

from .hierarchy import validate_fixed_mesh_hierarchy
from .topology import directed_edge_sets_equal


def attach_fixed_hierarchy(
    graph: Data,
    hierarchy: dict,
    *,
    num_levels: int,
) -> Data:
    """Attach a prefix of the persisted D5 hierarchy to one fixed-mesh graph.

    The existing DGN ``MeshDownMP``/``MeshUpMP`` blocks expect attributes named
    ``idx1_to_idx2``, ``e_12``, ``edge_index_2``, ``edge_attr_2``, etc. D5
    already persists exactly those tensors conceptually. This adapter exposes
    them under the historical attribute names without recomputing Guillard
    coarsening for every physical sample.

    D6 deliberately supports one physical graph at a time. Multi-sample batching
    requires offsetting all hierarchy index maps consistently and belongs to the
    later batching stage (D9).
    """

    validate_fixed_mesh_hierarchy(hierarchy)
    levels = hierarchy["levels"]
    transitions = hierarchy["transitions"]

    if not isinstance(num_levels, int) or num_levels < 1:
        raise ValueError(f"num_levels must be a positive integer, got {num_levels!r}.")
    if num_levels > len(levels):
        raise ValueError(
            f"Requested {num_levels} hierarchy levels but D5 provides only {len(levels)}."
        )

    fine = levels[0]
    if graph.pos.shape != fine["pos"].shape:
        raise ValueError(
            f"Graph position shape {tuple(graph.pos.shape)} does not match D5 level 1 "
            f"shape {tuple(fine['pos'].shape)}."
        )
    torch.testing.assert_close(graph.pos.cpu(), fine["pos"], rtol=1e-6, atol=1e-7)

    num_nodes = int(graph.pos.shape[0])
    if not directed_edge_sets_equal(graph.edge_index.cpu(), fine["edge_index"], num_nodes):
        raise ValueError("Graph edge_index does not match D5 hierarchy level 1.")
    torch.testing.assert_close(
        graph.edge_attr.cpu(),
        fine["edge_attr"],
        rtol=1e-6,
        atol=1e-7,
    )

    batch = getattr(graph, "batch", None)
    if batch is None:
        graph.batch = torch.zeros(num_nodes, dtype=torch.long, device=graph.pos.device)
        batch_id = 0
    else:
        if batch.shape != (num_nodes,):
            raise ValueError(f"Expected batch shape ({num_nodes},), got {tuple(batch.shape)}.")
        unique_batch = torch.unique(batch)
        if unique_batch.numel() != 1:
            raise NotImplementedError(
                "D6 hierarchy attachment supports one physical graph at a time; "
                "hierarchy-aware multi-sample batching is deferred to D9."
            )
        batch_id = int(unique_batch.item())

    # Keep the persisted geometry tensors as the source of truth. They are CPU
    # tensors here; PyG ``Data.to(device)`` moves all attached hierarchy tensors
    # together with the sample before the model forward pass.
    for level_number in range(2, num_levels + 1):
        level = levels[level_number - 1]
        transition = transitions[level_number - 2]
        previous = level_number - 1

        setattr(graph, f"coarse_mask_{level_number}", transition["coarse_mask"].clone())
        setattr(graph, f"pos_{level_number}", level["pos"].clone())
        setattr(
            graph,
            f"idx{previous}_to_idx{level_number}",
            transition["idx_to_parent"].clone(),
        )
        setattr(graph, f"edge_index_{level_number}", level["edge_index"].clone())
        setattr(graph, f"edge_attr_{level_number}", level["edge_attr"].clone())
        setattr(
            graph,
            f"e_{previous}{level_number}",
            transition["e_hr_to_lr"].clone(),
        )
        setattr(
            graph,
            f"batch_{level_number}",
            torch.full(
                (int(level["pos"].shape[0]),),
                batch_id,
                dtype=torch.long,
            ),
        )

    graph.d5_hierarchy_levels_attached = int(num_levels)
    graph.d5_hierarchy_fingerprint_sha256 = hierarchy["hierarchy_fingerprint_sha256"]
    return graph
