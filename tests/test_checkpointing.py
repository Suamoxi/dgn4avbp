from __future__ import annotations

import random
from copy import deepcopy

import numpy as np
import pytest
import torch

from dgn4avbp.checkpointing import (
    build_training_checkpoint,
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from dgn4avbp.step_sampler import ImportanceStepSampler


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_training_objects():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.SiLU(),
        torch.nn.Linear(8, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2
    )
    sampler = ImportanceStepSampler(
        num_diffusion_steps=8,
        min_history_length=2,
        uniform_prob=0.001,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)
    return model, optimizer, scheduler, sampler, generator


def _training_step(model, optimizer, scheduler, sampler, generator):
    python_draw = random.random()
    numpy_draw = float(np.random.standard_normal())
    torch_draw = torch.randn(6, 4)
    generator_draw = torch.randn(6, 4, generator=generator)
    timesteps, sample_weights = sampler.sample(batch_size=6)

    x = torch_draw + generator_draw + python_draw + numpy_draw
    target = torch.stack(
        [timesteps.float() / 8.0, sample_weights.float()], dim=1
    )
    prediction = model(x)
    per_sample = (prediction - target).square().mean(dim=1)
    sampler.update(timesteps, per_sample.detach())
    loss = (per_sample * sample_weights).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step(float(loss.detach()))

    return {
        "python_draw": python_draw,
        "numpy_draw": numpy_draw,
        "torch_draw": torch_draw.clone(),
        "generator_draw": generator_draw.clone(),
        "timesteps": timesteps.clone(),
        "sample_weights": sample_weights.clone(),
        "loss": loss.detach().clone(),
        "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
        "sampler_weights": sampler.weights.copy(),
    }


def test_importance_sampler_state_roundtrip() -> None:
    sampler = ImportanceStepSampler(
        num_diffusion_steps=4,
        min_history_length=2,
        uniform_prob=0.001,
    )
    sampler.update(
        torch.tensor([0, 1, 1, 3]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    state = sampler.state_dict()

    restored = ImportanceStepSampler(
        num_diffusion_steps=4,
        min_history_length=2,
        uniform_prob=0.001,
    )
    restored.load_state_dict(state)

    np.testing.assert_array_equal(restored._loss_counts, sampler._loss_counts)
    np.testing.assert_array_equal(restored._loss_history, sampler._loss_history)
    np.testing.assert_array_equal(restored.weights, sampler.weights)


def test_sampler_state_rejects_incompatible_contract() -> None:
    sampler = ImportanceStepSampler(num_diffusion_steps=4, min_history_length=2)
    state = sampler.state_dict()
    incompatible = ImportanceStepSampler(num_diffusion_steps=5, min_history_length=2)
    with pytest.raises(ValueError, match="num_diffusion_steps"):
        incompatible.load_state_dict(state)


def test_rng_state_roundtrip_includes_data_generator() -> None:
    _seed_everything(31)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(77)
    state = capture_rng_state(generator)

    expected = (
        random.random(),
        float(np.random.rand()),
        torch.rand(5),
        torch.rand(5, generator=generator),
    )

    random.random()
    np.random.rand()
    torch.rand(5)
    torch.rand(5, generator=generator)

    restore_rng_state(state, generator)
    observed = (
        random.random(),
        float(np.random.rand()),
        torch.rand(5),
        torch.rand(5, generator=generator),
    )

    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])
    assert torch.equal(observed[3], expected[3])


def test_build_checkpoint_records_progress_and_metadata() -> None:
    _seed_everything(32)
    model, optimizer, scheduler, sampler, generator = _build_training_objects()
    checkpoint = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step_sampler=sampler,
        epoch=7,
        global_step=123,
        batch_in_epoch=4,
        data_generator=generator,
        metadata={"run": "unit-test"},
    )
    assert checkpoint["progress"] == {
        "epoch": 7,
        "global_step": 123,
        "batch_in_epoch": 4,
    }
    assert checkpoint["metadata"] == {"run": "unit-test"}
    assert checkpoint["scheduler"] is not None
    assert checkpoint["step_sampler"]["sampler_class"] == "ImportanceStepSampler"


def test_save_restore_reproduces_uninterrupted_next_step(tmp_path) -> None:
    _seed_everything(33)
    model, optimizer, scheduler, sampler, generator = _build_training_objects()

    _training_step(model, optimizer, scheduler, sampler, generator)
    checkpoint_path = tmp_path / "resume.chk"
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step_sampler=sampler,
        epoch=2,
        global_step=17,
        batch_in_epoch=0,
        data_generator=generator,
        metadata={"contract": "d11-test"},
    )
    assert checkpoint_path.exists()
    assert not (tmp_path / "resume.chk.tmp").exists()

    reference = _training_step(model, optimizer, scheduler, sampler, generator)
    reference_optimizer = deepcopy(optimizer.state_dict())
    reference_scheduler = deepcopy(scheduler.state_dict())

    # Perturb every state before restoring into newly constructed objects.
    _seed_everything(999)
    resumed_model, resumed_optimizer, resumed_scheduler, resumed_sampler, resumed_generator = (
        _build_training_objects()
    )
    resumed_generator.manual_seed(999)

    progress = load_training_checkpoint(
        checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        step_sampler=resumed_sampler,
        data_generator=resumed_generator,
        map_location="cpu",
        restore_rng=True,
    )
    assert progress == {
        "epoch": 2,
        "global_step": 17,
        "batch_in_epoch": 0,
        "metadata": {"contract": "d11-test"},
    }

    resumed = _training_step(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        resumed_sampler,
        resumed_generator,
    )

    assert resumed["python_draw"] == reference["python_draw"]
    assert resumed["numpy_draw"] == reference["numpy_draw"]
    assert torch.equal(resumed["torch_draw"], reference["torch_draw"])
    assert torch.equal(resumed["generator_draw"], reference["generator_draw"])
    assert torch.equal(resumed["timesteps"], reference["timesteps"])
    assert torch.equal(resumed["sample_weights"], reference["sample_weights"])
    assert torch.equal(resumed["loss"], reference["loss"])
    for key in reference["model"]:
        assert torch.equal(resumed["model"][key], reference["model"][key])
    np.testing.assert_array_equal(resumed["sampler_weights"], reference["sampler_weights"])

    resumed_optimizer_state = resumed_optimizer.state_dict()
    assert resumed_optimizer_state["param_groups"] == reference_optimizer["param_groups"]
    for parameter_id, state in reference_optimizer["state"].items():
        for key, value in state.items():
            observed = resumed_optimizer_state["state"][parameter_id][key]
            if torch.is_tensor(value):
                assert torch.equal(observed, value)
            else:
                assert observed == value
    assert resumed_scheduler.state_dict() == reference_scheduler
