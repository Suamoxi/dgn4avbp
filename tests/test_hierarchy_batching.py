from __future__ import annotations

from copy import deepcopy

import torch
from torch_geometric.data import Data

from dgn4avbp.data import attach_hierarchy_to_graph, build_fixed_mesh_hierarchy, build_hex_edge_index
from dgn4avbp.dgn_model import DiffusionGraphNet
from dgn4avbp.diffusion_policy import (
    deterministic_validation_corruption,
    deterministic_validation_corruption_batch,
)
from dgn4avbp.diffusion_process import DiffusionProcess
from dgn4avbp.loader import Collater


def _chain_edge_index(num_nodes: int) -> torch.Tensor:
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long)
    forward = torch.stack(
        [torch.arange(num_nodes - 1), torch.arange(1, num_nodes)],
        dim=0,
    )
    return torch.cat([forward, forward.flip(0)], dim=1)


def _manual_hierarchy_graph(n1: int, n2: int, n3: int, sample_id: str) -> Data:
    graph = Data(
        pos=torch.randn(n1, 3),
        edge_index=_chain_edge_index(n1),
        edge_attr=torch.randn(2 * max(n1 - 1, 0), 3),
        target=torch.randn(n1, 5),
        sample_id=sample_id,
    )
    graph.batch = torch.zeros(n1, dtype=torch.long)

    graph.pos_2 = torch.randn(n2, 3)
    graph.edge_index_2 = _chain_edge_index(n2)
    graph.edge_attr_2 = torch.randn(2 * max(n2 - 1, 0), 3)
    graph.idx1_to_idx2 = torch.arange(n1, dtype=torch.long) % n2
    graph.e_12 = torch.randn(n1, 3)
    graph.batch_2 = torch.zeros(n2, dtype=torch.long)

    graph.pos_3 = torch.randn(n3, 3)
    graph.edge_index_3 = _chain_edge_index(n3)
    graph.edge_attr_3 = torch.randn(2 * max(n3 - 1, 0), 3)
    graph.idx2_to_idx3 = torch.arange(n2, dtype=torch.long) % n3
    graph.e_23 = torch.randn(n2, 3)
    graph.batch_3 = torch.zeros(n3, dtype=torch.long)
    return graph


def test_collater_offsets_each_hierarchy_index_space_once() -> None:
    torch.manual_seed(11)
    graph_a = _manual_hierarchy_graph(4, 3, 2, "a")
    graph_b = _manual_hierarchy_graph(5, 2, 1, "b")

    edge_1_b = graph_b.edge_index.clone()
    edge_2_b = graph_b.edge_index_2.clone()
    edge_3_b = graph_b.edge_index_3.clone()
    parent_12_b = graph_b.idx1_to_idx2.clone()
    parent_23_b = graph_b.idx2_to_idx3.clone()

    batch = Collater().collate([graph_a, graph_b])

    expected_edge_1 = torch.cat([graph_a.edge_index, edge_1_b + 4], dim=1)
    expected_edge_2 = torch.cat([graph_a.edge_index_2, edge_2_b + 3], dim=1)
    expected_edge_3 = torch.cat([graph_a.edge_index_3, edge_3_b + 2], dim=1)
    expected_parent_12 = torch.cat([graph_a.idx1_to_idx2, parent_12_b + 3])
    expected_parent_23 = torch.cat([graph_a.idx2_to_idx3, parent_23_b + 2])

    assert torch.equal(batch.edge_index, expected_edge_1)
    assert torch.equal(batch.edge_index_2, expected_edge_2)
    assert torch.equal(batch.edge_index_3, expected_edge_3)
    assert torch.equal(batch.idx1_to_idx2, expected_parent_12)
    assert torch.equal(batch.idx2_to_idx3, expected_parent_23)

    assert torch.equal(batch.batch, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1]))
    assert torch.equal(batch.batch_2, torch.tensor([0, 0, 0, 1, 1]))
    assert torch.equal(batch.batch_3, torch.tensor([0, 0, 1]))


def test_collated_hierarchy_has_no_cross_graph_edges_or_parent_maps() -> None:
    torch.manual_seed(12)
    batch = Collater().collate(
        [
            _manual_hierarchy_graph(6, 4, 2, "a"),
            _manual_hierarchy_graph(5, 3, 2, "b"),
            _manual_hierarchy_graph(4, 2, 1, "c"),
        ]
    )

    for level in (1, 2, 3):
        batch_l = batch.batch if level == 1 else getattr(batch, f"batch_{level}")
        edge_index_l = batch.edge_index if level == 1 else getattr(batch, f"edge_index_{level}")
        if edge_index_l.numel():
            assert torch.equal(batch_l[edge_index_l[0]], batch_l[edge_index_l[1]])

    assert torch.equal(batch.batch, batch.batch_2[batch.idx1_to_idx2])
    assert torch.equal(batch.batch_2, batch.batch_3[batch.idx2_to_idx3])


def test_single_graph_collation_preserves_hierarchy_indices() -> None:
    torch.manual_seed(13)
    graph = _manual_hierarchy_graph(5, 3, 2, "single")
    edge_2 = graph.edge_index_2.clone()
    edge_3 = graph.edge_index_3.clone()
    parent_12 = graph.idx1_to_idx2.clone()
    parent_23 = graph.idx2_to_idx3.clone()

    batch = Collater().collate([graph])

    assert torch.equal(batch.edge_index_2, edge_2)
    assert torch.equal(batch.edge_index_3, edge_3)
    assert torch.equal(batch.idx1_to_idx2, parent_12)
    assert torch.equal(batch.idx2_to_idx3, parent_23)
    assert torch.equal(batch.batch, torch.zeros(5, dtype=torch.long))
    assert torch.equal(batch.batch_2, torch.zeros(3, dtype=torch.long))
    assert torch.equal(batch.batch_3, torch.zeros(2, dtype=torch.long))


def test_deterministic_validation_corruption_is_batching_invariant() -> None:
    torch.manual_seed(14)
    diffusion = DiffusionProcess(num_steps=1000, schedule_type="linear")
    field_a = torch.randn(7, 5)
    field_b = torch.randn(4, 5)
    sample_ids = ["sample_a.h5", "sample_b.h5"]

    single_a = deterministic_validation_corruption(
        diffusion,
        field_a,
        torch.zeros(field_a.shape[0], dtype=torch.long),
        sample_id=sample_ids[0],
        base_seed=42,
    )
    single_b = deterministic_validation_corruption(
        diffusion,
        field_b,
        torch.zeros(field_b.shape[0], dtype=torch.long),
        sample_id=sample_ids[1],
        base_seed=42,
    )

    field = torch.cat([field_a, field_b], dim=0)
    batch_index = torch.cat(
        [
            torch.zeros(field_a.shape[0], dtype=torch.long),
            torch.ones(field_b.shape[0], dtype=torch.long),
        ]
    )
    batched = deterministic_validation_corruption_batch(
        diffusion,
        field,
        batch_index,
        sample_ids=sample_ids,
        base_seed=42,
    )

    n_a = field_a.shape[0]
    torch.testing.assert_close(batched["field_r"][:n_a], single_a["field_r"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(batched["field_r"][n_a:], single_b["field_r"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(batched["noise"][:n_a], single_a["noise"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(batched["noise"][n_a:], single_b["noise"], rtol=0.0, atol=0.0)
    assert batched["r"].tolist() == [single_a["timestep"], single_b["timestep"]]


def _structured_hex_graph(n: int = 4) -> Data:
    coords = torch.linspace(0.0, 1.0, n)
    pos = torch.tensor(
        [[x, y, z] for z in coords for y in coords for x in coords],
        dtype=torch.float32,
    )

    def node(i: int, j: int, k: int) -> int:
        return k * n * n + j * n + i

    cells = []
    for k in range(n - 1):
        for j in range(n - 1):
            for i in range(n - 1):
                cells.append(
                    [
                        node(i, j, k),
                        node(i + 1, j, k),
                        node(i + 1, j + 1, k),
                        node(i, j + 1, k),
                        node(i, j, k + 1),
                        node(i + 1, j, k + 1),
                        node(i + 1, j + 1, k + 1),
                        node(i, j + 1, k + 1),
                    ]
                )
    cells = torch.tensor(cells, dtype=torch.long)
    edge_index = build_hex_edge_index(cells, num_nodes=pos.shape[0])
    edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
    return Data(pos=pos, cells=cells, edge_index=edge_index, edge_attr=edge_attr)


def _small_model(num_levels: int) -> DiffusionGraphNet:
    return DiffusionGraphNet(
        diffusion_process=DiffusionProcess(num_steps=10, schedule_type="linear"),
        learnable_variance=True,
        arch={
            "in_node_features": 5,
            "cond_node_features": 0,
            "cond_edge_features": 3,
            "depths": [1] * num_levels,
            "fnns_depth": 2,
            "fnns_width": 16,
            "aggr": "sum",
            "dropout": 0.0,
            "emb_width": 32,
            "dim": 3,
            "scalar_rel_pos": False,
        },
    ).eval()


def test_batched_backbone_matches_independent_forwards() -> None:
    torch.manual_seed(15)
    base = _structured_hex_graph(n=4)
    hierarchy = build_fixed_mesh_hierarchy(base.pos, base.edge_index, max_levels=3)
    num_levels = len(hierarchy["levels"])
    assert num_levels >= 2

    graph_a = attach_hierarchy_to_graph(deepcopy(base), hierarchy)
    graph_b = attach_hierarchy_to_graph(deepcopy(base), hierarchy)
    graph_a.field_r = torch.randn(base.pos.shape[0], 5)
    graph_b.field_r = torch.randn(base.pos.shape[0], 5)
    graph_a.r = torch.tensor([2], dtype=torch.long)
    graph_b.r = torch.tensor([7], dtype=torch.long)

    model = _small_model(num_levels=num_levels)
    with torch.no_grad():
        eps_a, var_a = model(deepcopy(graph_a))
        eps_b, var_b = model(deepcopy(graph_b))

        graph_batch = Collater().collate([deepcopy(graph_a), deepcopy(graph_b)])
        eps_batch, var_batch = model(graph_batch)

    n = base.pos.shape[0]
    torch.testing.assert_close(eps_batch[:n], eps_a, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(eps_batch[n:], eps_b, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(var_batch[:n], var_a, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(var_batch[n:], var_b, rtol=2e-5, atol=2e-6)
