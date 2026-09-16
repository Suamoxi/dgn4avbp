from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data

from dgn4avbp.data import attach_hierarchy_to_graph, build_fixed_mesh_hierarchy, build_hex_edge_index
from dgn4avbp.dgn_model import DiffusionGraphNet
from dgn4avbp.diffusion_process import DiffusionProcess


ROOT = Path(__file__).resolve().parents[1]


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


def _small_attached_graph(num_levels: int = 3) -> tuple[Data, dict]:
    graph = _structured_hex_graph(n=4)
    hierarchy = build_fixed_mesh_hierarchy(
        graph.pos,
        graph.edge_index,
        max_levels=4,
    )
    assert len(hierarchy["levels"]) >= num_levels
    graph = attach_hierarchy_to_graph(graph, hierarchy)
    torch.manual_seed(7)
    graph.field_r = torch.randn(graph.pos.shape[0], 5)
    graph.r = torch.tensor([3.0])
    return graph, hierarchy


def _small_model(num_levels: int = 3) -> DiffusionGraphNet:
    arch = {
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
    }
    return DiffusionGraphNet(
        diffusion_process=DiffusionProcess(num_steps=10, schedule_type="linear"),
        learnable_variance=True,
        arch=arch,
    )


def _permute_attached_graph(graph: Data, num_levels: int) -> tuple[Data, torch.Tensor]:
    """Relabel every hierarchy level consistently without recomputing coarsening."""

    generator = torch.Generator().manual_seed(19)
    permutations: dict[int, torch.Tensor] = {}
    inverse: dict[int, torch.Tensor] = {}

    for level in range(1, num_levels + 1):
        pos = graph.pos if level == 1 else getattr(graph, f"pos_{level}")
        perm = torch.randperm(pos.shape[0], generator=generator)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel())
        permutations[level] = perm
        inverse[level] = inv

    permuted = Data()
    p1 = permutations[1]
    permuted.pos = graph.pos[p1]
    permuted.field_r = graph.field_r[p1]
    permuted.r = graph.r.clone()
    permuted.batch = graph.batch[p1]
    permuted.edge_index = inverse[1][graph.edge_index]
    # Edge order is unchanged, so the physical edge vector attached to each edge
    # remains unchanged under a pure node-label permutation.
    permuted.edge_attr = graph.edge_attr.clone()

    for level in range(2, num_levels + 1):
        previous = level - 1
        p_hr = permutations[previous]
        p_lr = permutations[level]
        inv_lr = inverse[level]

        setattr(permuted, f"pos_{level}", getattr(graph, f"pos_{level}")[p_lr])
        setattr(
            permuted,
            f"edge_index_{level}",
            inverse[level][getattr(graph, f"edge_index_{level}")],
        )
        setattr(permuted, f"edge_attr_{level}", getattr(graph, f"edge_attr_{level}").clone())
        setattr(permuted, f"batch_{level}", getattr(graph, f"batch_{level}")[p_lr])

        parent_old = getattr(graph, f"idx{previous}_to_idx{level}")
        parent_new = inv_lr[parent_old[p_hr]]
        setattr(permuted, f"idx{previous}_to_idx{level}", parent_new)
        setattr(permuted, f"e_{previous}{level}", getattr(graph, f"e_{previous}{level}")[p_hr])

    return permuted, p1


def test_canonical_hit_backbone_config_contract() -> None:
    with (ROOT / "configs/model/dgn_hit_baseline.yaml").open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    arch = cfg["arch"]
    assert cfg["learnable_variance"] is True
    assert cfg["hierarchy"]["num_levels"] == 7
    assert arch["in_node_features"] == 5
    assert arch["cond_node_features"] == 0
    assert arch["cond_edge_features"] == 3
    assert arch["dim"] == 3
    assert arch["scalar_rel_pos"] is False
    assert arch["depths"] == [2] * 7
    assert arch["fnns_width"] == 128
    assert arch["aggr"] == "sum"
    assert cfg["output_contract"] == {
        "epsilon_channels": 5,
        "variance_channels": 5,
        "total_channels": 10,
    }


def test_learned_variance_forward_shape_and_pool_unpool_restore() -> None:
    graph, _ = _small_attached_graph(num_levels=3)
    model = _small_model(num_levels=3).eval()

    fine_edge_index = graph.edge_index.clone()
    fine_batch = graph.batch.clone()
    with torch.no_grad():
        epsilon, variance = model(graph)

    assert epsilon.shape == (graph.pos.shape[0], 5)
    assert variance.shape == (graph.pos.shape[0], 5)
    assert torch.isfinite(epsilon).all()
    assert torch.isfinite(variance).all()
    assert model.num_fields == 5
    assert torch.equal(graph.edge_index, fine_edge_index)
    assert torch.equal(graph.batch, fine_batch)


def test_backbone_backward_has_finite_nonzero_gradients() -> None:
    graph, _ = _small_attached_graph(num_levels=3)
    model = _small_model(num_levels=3).train()

    epsilon, variance = model(graph)
    loss = epsilon.square().mean() + variance.square().mean()
    loss.backward()

    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert grads
    assert all(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads if grad is not None)
    total_abs_grad = sum(float(grad.abs().sum().item()) for grad in grads if grad is not None)
    assert total_abs_grad > 0.0


def test_full_backbone_is_equivariant_to_consistent_node_relabeling() -> None:
    graph, _ = _small_attached_graph(num_levels=3)
    permuted, fine_permutation = _permute_attached_graph(graph, num_levels=3)
    model = _small_model(num_levels=3).eval()

    with torch.no_grad():
        epsilon, variance = model(graph)
        epsilon_perm, variance_perm = model(permuted)

    torch.testing.assert_close(
        epsilon_perm,
        epsilon[fine_permutation],
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        variance_perm,
        variance[fine_permutation],
        rtol=2e-5,
        atol=2e-6,
    )
