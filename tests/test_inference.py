"""Tests for the inference module."""

import pytest
import torch

from guwm.projection.subset import WorldProjection, project
from guwm.training.backbone import build_world_model
from guwm.training.heterogeneous import FieldEncoder, FieldDecoder
from guwm.inference import WorldModel


@pytest.fixture
def small_model():
    """A small world model for testing inference."""
    proj = WorldProjection(include=["financial.yield_curves", "regime"])
    bound = project(proj, T=1, H=24, W=24, d_model=32)
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

    proj = WorldProjection(include=["financial.yield_curves", "regime"])
    loaded = WorldModel.load(
        checkpoint_path, proj,
        T=1, H=24, W=24, d_model=32,
        n_layers=2, n_loops=1, device="cpu",
    )
    assert len(loaded.bound.field_names) == len(small_model.bound.field_names)
