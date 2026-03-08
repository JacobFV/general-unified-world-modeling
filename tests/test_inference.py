"""Tests for the inference module."""

import pytest
import torch

from general_unified_world_model.projection.subset import project
from general_unified_world_model.training.backbone import build_world_model
from general_unified_world_model.training.heterogeneous import (
    FieldEncoder, FieldDecoder, DatasetSpec, DataSource, InputSpec, OutputSpec,
)
from general_unified_world_model.inference import WorldModel


@pytest.fixture
def small_model():
    """A small world model for testing inference."""
    bound = project(include=["financial.yield_curves", "regime"], T=1, H=24, W=24, d_model=32)
    backbone = build_world_model(bound, n_layers=2, n_loops=1)
    encoder = FieldEncoder(bound)
    decoder = FieldDecoder(bound)
    return WorldModel(bound, backbone, encoder, decoder, device="cpu")


def test_observe(small_model):
    """Observe should store values."""
    small_model.observe("financial.yield_curves.ten_year", 4.25)
    assert "financial.yield_curves.ten_year" in small_model._observations


def test_clear_observations(small_model):
    """Clear should remove all observations."""
    small_model.observe("financial.yield_curves.ten_year", 4.25)
    small_model.clear_observations()
    assert len(small_model._observations) == 0


def test_predict_returns_dict(small_model):
    """Predict should return dict of field predictions."""
    small_model.observe("financial.yield_curves.ten_year", 4.25)
    predictions = small_model.predict(n_steps=2)
    assert isinstance(predictions, dict)
    assert len(predictions) > 0


def test_predict_field(small_model):
    """predict_field should return tensor for a single field."""
    small_model.observe("financial.yield_curves.ten_year", 4.25)
    pred = small_model.predict_field("regime.growth_regime", n_steps=2)
    assert isinstance(pred, torch.Tensor)


def test_predict_without_observations(small_model):
    """Predict with no observations should still work (unconditional generation)."""
    predictions = small_model.predict(n_steps=2)
    assert len(predictions) > 0


def test_save_and_load(small_model, tmp_path):
    """Save and load should preserve model structure."""
    checkpoint_path = tmp_path / "test_model.pt"
    small_model.save(checkpoint_path)
    assert checkpoint_path.exists()

    loaded = WorldModel.load(
        checkpoint_path,
        include=["financial.yield_curves", "regime"],
        T=1, H=24, W=24, d_model=32,
        n_layers=2, n_loops=1, device="cpu",
    )
    assert len(loaded.bound.field_names) == len(small_model.bound.field_names)


def test_finetune(small_model):
    """finetune() should complete and return loss history."""
    field_name = small_model.bound.field_names[0]
    spec = DatasetSpec(
        name="test_private",
        input_specs=[InputSpec(key="v", semantic_type="test", field_path=field_name)],
        output_specs=[OutputSpec(key="v", semantic_type="test", field_path=field_name)],
    )
    source = DataSource(spec=spec, data={"v": torch.randn(50)})

    metrics = small_model.finetune(
        datasets=[source],
        n_steps=5,
        batch_size=4,
        log_every=100,
    )

    assert "losses" in metrics
    assert len(metrics["losses"]) == 5
    assert "final_loss" in metrics
    assert metrics["final_loss"] >= 0


def test_finetune_freeze_backbone(small_model):
    """finetune with freeze_backbone should only train encoder/decoder."""
    field_name = small_model.bound.field_names[0]
    spec = DatasetSpec(
        name="test_frozen",
        input_specs=[InputSpec(key="v", semantic_type="test", field_path=field_name)],
        output_specs=[OutputSpec(key="v", semantic_type="test", field_path=field_name)],
    )
    source = DataSource(spec=spec, data={"v": torch.randn(50)})

    # Capture backbone weights before
    bb_before = {n: p.clone() for n, p in small_model.backbone.named_parameters()}

    metrics = small_model.finetune(
        datasets=[source],
        n_steps=3,
        freeze_backbone=True,
        batch_size=4,
        log_every=100,
    )

    assert len(metrics["losses"]) == 3
    # Backbone weights should be unchanged
    for n, p in small_model.backbone.named_parameters():
        assert torch.allclose(p, bb_before[n]), f"Backbone param {n} changed"
    # After finetune, requires_grad should be restored
    assert all(p.requires_grad for p in small_model.backbone.parameters())
