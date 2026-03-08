"""Shared test fixtures."""

import os
from pathlib import Path

import pytest
import torch
from canvas_engineering import ConnectivityPolicy

from general_unified_world_model.projection.subset import WorldProjection, project

# Load .env file so API keys are available for live tests
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value


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
    return project(proj, T=1, H=160, W=160, d_model=64)
