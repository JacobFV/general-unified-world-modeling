"""Tests for the training infrastructure."""

import pytest
import torch
import torch.nn as nn

from general_unified_world_model.projection.subset import WorldProjection, project
from general_unified_world_model.training.backbone import WorldModelBackbone, build_world_model
from general_unified_world_model.training.heterogeneous import (
    FieldEncoder, FieldDecoder, MaskedCanvasTrainer,
    DatasetSpec, InputSpec, OutputSpec, HeterogeneousDataset,
    build_mixed_dataloader,
)
from general_unified_world_model.training.diffusion import (
    CosineNoiseSchedule, MultiFrequencyNoise, DiffusionWorldModelTrainer,
)


@pytest.fixture
def small_bound():
    """A small projection for testing."""
    proj = WorldProjection(include=["financial.yield_curves", "regime"])
    return project(proj, T=1, H=24, W=24, d_model=32)


def test_backbone_forward(small_bound):
    """Backbone should accept canvas tensor and produce same shape."""
    backbone = build_world_model(small_bound, n_layers=2, n_loops=1)
    batch = torch.randn(2, small_bound.layout.num_positions, 32)
    out = backbone(batch)
    assert out.shape == batch.shape


def test_backbone_looped(small_bound):
    """Looped backbone should still produce correct shape."""
    backbone = build_world_model(small_bound, n_layers=2, n_loops=3)
    batch = torch.randn(2, small_bound.layout.num_positions, 32)
    mask = small_bound.topology.to_additive_mask(small_bound.layout)
    out = backbone(batch, mask=mask)
    assert out.shape == batch.shape


def test_field_encoder(small_bound):
    """FieldEncoder should encode scalar values to d_model."""
    encoder = FieldEncoder(small_bound)
    field_name = small_bound.field_names[0]
    raw = torch.randn(2, 1, 1)
    encoded = encoder(field_name, raw)
    assert encoded.shape == (2, 1, 32)


def test_field_decoder(small_bound):
    """FieldDecoder should decode d_model to raw dimension."""
    decoder = FieldDecoder(small_bound)
    field_name = small_bound.field_names[0]
    latent = torch.randn(2, 1, 32)
    decoded = decoder(field_name, latent)
    assert decoded.shape == (2, 1, 1)


def test_masked_trainer_step(small_bound):
    """MaskedCanvasTrainer should complete a training step."""
    backbone = build_world_model(small_bound, n_layers=2, n_loops=1)
    encoder = FieldEncoder(small_bound)
    decoder = FieldDecoder(small_bound)

    all_params = (
        list(backbone.parameters())
        + list(encoder.parameters())
        + list(decoder.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=1e-3)

    trainer = MaskedCanvasTrainer(
        small_bound, backbone, encoder, decoder, optimizer, device="cpu"
    )

    N = small_bound.layout.num_positions
    batch = {
        "canvas_data": torch.randn(4, N, 32),
        "presence_mask": torch.ones(4, N),
    }

    metrics = trainer.train_step(batch)
    assert "loss" in metrics
    assert metrics["loss"] > 0


def test_heterogeneous_dataset(small_bound):
    """HeterogeneousDataset should handle multiple sources."""
    # Create two data sources with different field coverage
    spec1 = DatasetSpec(
        name="source1",
        input_specs=[
            InputSpec(key="val_a", semantic_type="field path", field_path=small_bound.field_names[0]),
        ],
        output_specs=[
            OutputSpec(key="val_a", semantic_type="field path", field_path=small_bound.field_names[0]),
        ],
    )
    data1 = {"val_a": torch.randn(100)}

    spec2 = DatasetSpec(
        name="source2",
        input_specs=[
            InputSpec(key="val_b", semantic_type="field path", field_path=small_bound.field_names[1]),
        ],
        output_specs=[
            OutputSpec(key="val_b", semantic_type="field path", field_path=small_bound.field_names[1]),
        ],
    )
    data2 = {"val_b": torch.randn(100)}

    dataset = HeterogeneousDataset(
        small_bound,
        sources=[(spec1, data1), (spec2, data2)],
    )

    assert len(dataset) > 0
    sample = dataset[0]
    assert "canvas_data" in sample
    assert "presence_mask" in sample
    assert sample["canvas_data"].shape[0] == small_bound.layout.num_positions


def test_cosine_noise_schedule():
    """Noise schedule should produce valid noise levels."""
    schedule = CosineNoiseSchedule(n_steps=100)

    assert schedule.alphas_cumprod[0] > 0.99  # nearly clean at t=0
    assert schedule.alphas_cumprod[-1] < 0.01  # nearly pure noise at t=T

    x = torch.randn(2, 10, 32)
    t = torch.tensor([0, 50])
    noisy, noise = schedule.q_sample(x, t)
    assert noisy.shape == x.shape
    assert noise.shape == x.shape


def test_multi_frequency_noise(small_bound):
    """MultiFrequencyNoise should scale noise by field period."""
    mfn = MultiFrequencyNoise(small_bound)

    noise = torch.ones(1, small_bound.layout.num_positions, 32)
    scaled = mfn.apply(noise)

    # Scaled noise should have varying magnitudes
    assert scaled.shape == noise.shape
    # Not all positions should have the same noise scale
    unique_scales = torch.unique(scaled[0, :, 0])
    assert len(unique_scales) > 1


def test_diffusion_trainer_step(small_bound):
    """DiffusionWorldModelTrainer should complete a training step."""
    backbone = build_world_model(small_bound, n_layers=2, n_loops=1)
    schedule = CosineNoiseSchedule(n_steps=100)
    trainer = DiffusionWorldModelTrainer(
        small_bound, backbone, schedule, device="cpu"
    )

    N = small_bound.layout.num_positions
    x_clean = torch.randn(2, N, 32)
    presence = torch.ones(2, N)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)

    metrics = trainer.train_step(x_clean, presence, optimizer)
    assert "loss" in metrics
    assert metrics["loss"] >= 0
