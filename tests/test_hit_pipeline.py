from __future__ import annotations

from copy import deepcopy

import torch
from torch_geometric.data import Data

from dgn4avbp.data import (
    ChannelStandardizer,
    HITReferenceScales,
    attach_hierarchy_to_graph,
    build_fixed_mesh_hierarchy,
    build_hex_edge_index,
)
from dgn4avbp.dgn_model import DiffusionGraphNet
from dgn4avbp.diffusion_loss import CanonicalHybridLoss
from dgn4avbp.diffusion_process import DiffusionProcess
from dgn4avbp.hit_pipeline import (
    inverse_generated_state,
    prepare_validation_batch,
    sample_ids_from_batch,
    train_one_batch,
    validate_one_batch,
)
from dgn4avbp.loader import Collater
from dgn4avbp.step_sampler import ImportanceStepSampler


def _structured_hex_graph(n: int = 4, sample_id: str = "sample.h5") -> Data:
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
    return Data(
        pos=pos,
        cells=cells,
        edge_index=edge_index,
        edge_attr=edge_attr,
        target=torch.randn(pos.shape[0], 5),
        sample_id=sample_id,
    )


def _batch_and_model():
    torch.manual_seed(91)
    base = _structured_hex_graph()
    hierarchy = build_fixed_mesh_hierarchy(base.pos, base.edge_index, max_levels=3)
    graphs = []
    for sample_id in ("a.h5", "b.h5"):
        graph = deepcopy(base)
        graph.sample_id = sample_id
        graph.target = torch.randn_like(base.target)
        graphs.append(attach_hierarchy_to_graph(graph, hierarchy))
    batch = Collater().collate(graphs)

    diffusion = DiffusionProcess(num_steps=1000, schedule_type="linear")
    model = DiffusionGraphNet(
        diffusion_process=diffusion,
        learnable_variance=True,
        arch={
            "in_node_features": 5,
            "cond_node_features": 0,
            "cond_edge_features": 3,
            "depths": [1] * len(hierarchy["levels"]),
            "fnns_depth": 2,
            "fnns_width": 16,
            "aggr": "sum",
            "dropout": 0.0,
            "emb_width": 32,
            "dim": 3,
            "scalar_rel_pos": False,
        },
    )
    return batch, model, diffusion


def test_sample_ids_from_batched_graph() -> None:
    batch, _, _ = _batch_and_model()
    assert sample_ids_from_batch(batch) == ["a.h5", "b.h5"]


def test_deterministic_validation_corruption_is_repeatable_in_pipeline() -> None:
    batch, _, diffusion = _batch_and_model()
    first = prepare_validation_batch(
        deepcopy(batch),
        diffusion_process=diffusion,
        base_seed=42,
        device=torch.device("cpu"),
    )
    second = prepare_validation_batch(
        deepcopy(batch),
        diffusion_process=diffusion,
        base_seed=42,
        device=torch.device("cpu"),
    )
    assert torch.equal(first.r, second.r)
    assert torch.equal(first.noise, second.noise)
    assert torch.equal(first.field_r, second.field_r)


def test_training_and_validation_steps_are_finite_and_update_state() -> None:
    batch, model, diffusion = _batch_and_model()
    criterion = CanonicalHybridLoss(lambda_vlb=0.001)
    sampler = ImportanceStepSampler(num_diffusion_steps=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    before = next(model.parameters()).detach().clone()

    training = train_one_batch(
        model=model,
        graph=deepcopy(batch),
        diffusion_process=diffusion,
        step_sampler=sampler,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        grad_clip_norm=1.0,
    )
    after = next(model.parameters()).detach().clone()

    assert all(torch.isfinite(torch.tensor(value)) for value in training.values())
    assert not torch.equal(before, after)
    assert int(sampler._loss_counts.sum()) == 2

    validation = validate_one_batch(
        model=model,
        graph=deepcopy(batch),
        diffusion_process=diffusion,
        criterion=criterion,
        validation_base_seed=42,
        device=torch.device("cpu"),
    )
    assert validation["num_graphs"] == 2
    assert torch.isfinite(torch.tensor(validation["loss"]))


def test_inverse_generated_state_returns_nondimensional_and_dimensional_fields() -> None:
    standardizer = ChannelStandardizer(
        mean=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        std=torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0]),
        num_fit_samples=2,
    )
    refs = HITReferenceScales(rho_ref=2.0, U_ref=3.0, L_ref=4.0, T_ref=300.0)
    state_std = torch.zeros(3, 5)
    state_nd, state_dim = inverse_generated_state(
        state_std,
        standardizer=standardizer,
        refs=refs,
    )
    assert torch.equal(state_nd, standardizer.mean.expand_as(state_nd))
    expected_scales = torch.tensor([2.0, 6.0, 6.0, 6.0, 18.0])
    torch.testing.assert_close(state_dim, state_nd * expected_scales)
