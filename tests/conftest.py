"""Shared test fixtures."""

import pytest
import torch
from canvas_engineering import ConnectivityPolicy

from guwm.projection.subset import WorldProjection, project


@pytest.fixture
def financial_bound():
    """A financial-only projection."""
    proj = WorldProjection(include=["financial"])
    return project(proj, T=1, H=32, W=32, d_model=32)


@pytest.fixture
def macro_bound():
    """A macro-only projection."""
    proj = WorldProjection(include=["country_us.macro"])
    return project(proj, T=1, H=24, W=24, d_model=32)


@pytest.fixture
def full_bound():
    """Full world model (expensive, use sparingly)."""
    proj = WorldProjection(include=["*"])
    return project(proj, T=1, H=128, W=128, d_model=64)
