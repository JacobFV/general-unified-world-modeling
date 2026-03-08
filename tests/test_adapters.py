"""Tests for data adapters."""

import pytest
import torch

from general_unified_world_model.data.adapters import (
    z_score, minmax, log_return, pct_change, rank_normalize,
    pmi_adapter, earnings_adapter, news_adapter,
)
from general_unified_world_model.training.heterogeneous import DatasetSpec, DataSource, InputSpec, OutputSpec


def test_z_score_transform():
    """z_score should normalize to mean=0, std=1."""
    transform = z_score(10.0, 2.0)
    result = transform(torch.tensor([10.0]))
    assert abs(result.item()) < 1e-5  # should be ~0

    result = transform(torch.tensor([12.0]))
    assert abs(result.item() - 1.0) < 1e-5  # should be ~1


def test_minmax_transform():
    """minmax should map to [0, 1]."""
    transform = minmax(0.0, 100.0)
    assert abs(transform(torch.tensor([0.0])).item()) < 1e-5
    assert abs(transform(torch.tensor([100.0])).item() - 1.0) < 1e-5
    assert abs(transform(torch.tensor([50.0])).item() - 0.5) < 1e-5


def test_log_return_transform():
    """log_return should compute log returns."""
    transform = log_return()
    prices = torch.tensor([100.0, 110.0, 105.0])
    returns = transform(prices)
    assert returns.shape == prices.shape
    assert abs(returns[0].item()) < 1e-5  # first return is 0


def test_pct_change_transform():
    """pct_change should compute percentage changes."""
    transform = pct_change()
    values = torch.tensor([100.0, 110.0])
    changes = transform(values)
    assert abs(changes[1].item() - 0.1) < 1e-5  # 10% change


def test_rank_normalize_transform():
    """rank_normalize should map to [0, 1] based on rank."""
    transform = rank_normalize()
    values = torch.tensor([10.0, 30.0, 20.0])
    ranked = transform(values)
    assert ranked.min().item() >= 0
    assert ranked.max().item() <= 1


def test_pmi_adapter():
    """PMI adapter should create valid DataSource."""
    data = {
        "manufacturing_pmi": torch.tensor([52.1, 51.3, 53.0]),
        "services_pmi": torch.tensor([54.2, 53.8, 55.1]),
    }
    result = pmi_adapter(data, country="us")
    assert isinstance(result, DataSource)
    assert isinstance(result.spec, DatasetSpec)
    assert len(result.spec.input_specs) >= 2
    assert result.spec.base_period == 192


def test_earnings_adapter():
    """Earnings adapter should create valid DataSource for a firm."""
    data = {
        "revenue": torch.tensor([1e9, 1.1e9, 1.2e9]),
        "gross_margin": torch.tensor([0.42, 0.43, 0.41]),
    }
    result = earnings_adapter("AAPL", data)
    assert isinstance(result, DataSource)
    assert len(result.spec.input_specs) >= 2
    assert "firm_AAPL" in result.spec.input_specs[0].field_path


def test_news_adapter():
    """News adapter should handle pre-computed embeddings."""
    embeddings = torch.randn(100, 32)
    result = news_adapter(embeddings)
    assert isinstance(result, DataSource)
    assert len(result.spec.input_specs) == 1
    assert "events.news_embedding" in result.spec.input_specs[0].field_path
