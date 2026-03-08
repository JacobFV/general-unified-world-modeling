"""Tests for the projection and subsetting system."""

import pytest
import torch
from canvas_engineering import ConnectivityPolicy

from general_unified_world_model.projection.subset import WorldProjection, project
from general_unified_world_model.projection.temporal import TemporalTopology, TemporalEntity
from general_unified_world_model.schema.business import Business
from general_unified_world_model.schema.individual import Individual


def test_full_projection():
    """include=["*"] should include all fields."""
    proj = WorldProjection(include=["*"])
    bound = project(proj, T=1, H=128, W=128, d_model=64)
    assert len(bound.field_names) > 800


def test_partial_projection():
    """Partial include should produce smaller canvas."""
    full = WorldProjection(include=["*"])
    partial = WorldProjection(include=["financial", "regime"])

    full_bound = project(full, T=1, H=128, W=128, d_model=64)
    partial_bound = project(partial, T=1, H=32, W=32, d_model=64)

    assert len(partial_bound.field_names) < len(full_bound.field_names)


def test_dynamic_firms():
    """Adding firms should create new Business fields."""
    proj = WorldProjection(
        include=["regime"],
        firms=["AAPL", "NVDA", "TSMC"],
    )
    bound = project(proj, T=1, H=64, W=64, d_model=64)

    # Should have fields for each firm
    firm_fields = [f for f in bound.field_names if "firm_AAPL" in f]
    assert len(firm_fields) > 0, "AAPL fields not found"

    firm_fields = [f for f in bound.field_names if "firm_NVDA" in f]
    assert len(firm_fields) > 0, "NVDA fields not found"


def test_dynamic_individuals():
    """Adding individuals should create new Individual fields."""
    proj = WorldProjection(
        include=["regime"],
        individuals=["ceo", "cfo"],
    )
    bound = project(proj, T=1, H=48, W=48, d_model=64)

    person_fields = [f for f in bound.field_names if "person_ceo" in f]
    assert len(person_fields) > 0, "CEO fields not found"


def test_dynamic_countries():
    """Adding countries should create new Country fields."""
    proj = WorldProjection(
        include=["regime"],
        countries=["jp", "kr"],
    )
    bound = project(proj, T=1, H=64, W=64, d_model=64)

    jp_fields = [f for f in bound.field_names if "country_jp" in f]
    assert len(jp_fields) > 0, "Japan fields not found"


def test_connectivity_policy():
    """Custom connectivity should propagate to compiled schema."""
    proj = WorldProjection(
        include=["financial.yield_curves", "regime"],
        connectivity=ConnectivityPolicy(
            intra="isolated",
            parent_child="broadcast",
        ),
    )
    bound = project(proj, T=1, H=24, W=24, d_model=64)
    assert len(bound.topology.connections) > 0


def test_temporal_topology():
    """TemporalTopology should track entity lifetimes."""
    tt = TemporalTopology()
    tt.add("firm_AAPL", Business(), start_tick=100)
    tt.add("firm_MSFT", Business(), start_tick=0)
    tt.add("firm_ENRON", Business(), start_tick=0, end_tick=500)

    # At tick 50: MSFT and ENRON active, AAPL not yet
    active = tt.active_at(50)
    names = {e.name for e in active}
    assert "firm_MSFT" in names
    assert "firm_ENRON" in names
    assert "firm_AAPL" not in names

    # At tick 200: all active
    active = tt.active_at(200)
    names = {e.name for e in active}
    assert "firm_AAPL" in names
    assert "firm_ENRON" in names

    # At tick 600: ENRON gone
    active = tt.active_at(600)
    names = {e.name for e in active}
    assert "firm_AAPL" in names
    assert "firm_ENRON" not in names


def test_projection_forward_pass():
    """A projected schema should support a full forward pass."""
    proj = WorldProjection(include=["financial.yield_curves", "regime"])
    bound = project(proj, T=1, H=24, W=24, d_model=64)

    from general_unified_world_model.training.backbone import build_world_model

    backbone = build_world_model(bound, n_layers=2, n_loops=1)
    batch = torch.randn(2, bound.layout.num_positions, 64)
    mask = bound.topology.to_additive_mask(bound.layout)
    out = backbone(batch, mask=mask)
    assert out.shape == batch.shape
