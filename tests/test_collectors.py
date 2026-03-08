"""Tests for data collectors."""

import tempfile
from pathlib import Path

import pytest
import torch

from general_unified_world_model.data.collectors import (
    BaseCollector,
    SyntheticCollector,
    FREDCollector,
    YahooFinanceCollector,
    collect_all,
    _cache_key,
    _generate_correlated_group,
    _zscore_tensor,
    _log_returns_tensor,
    _pct_change_tensor,
)
from general_unified_world_model.training.heterogeneous import DatasetSpec


# ── Helper functions ──────────────────────────────────────────────────────

class TestHelpers:
    def test_zscore(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        normed, mu, sigma = _zscore_tensor(t)
        assert abs(normed.mean().item()) < 1e-5
        assert abs(normed.std().item() - 1.0) < 0.1

    def test_log_returns(self):
        t = torch.tensor([100.0, 110.0, 105.0])
        lr = _log_returns_tensor(t)
        assert lr.shape == t.shape
        assert lr[0].item() == 0.0  # first element is always zero
        assert lr[1].item() > 0  # price went up

    def test_log_returns_single(self):
        t = torch.tensor([100.0])
        lr = _log_returns_tensor(t)
        assert lr.shape == (1,)

    def test_pct_change(self):
        t = torch.tensor([100.0, 110.0, 100.0])
        pc = _pct_change_tensor(t)
        assert pc.shape == t.shape
        assert pc[0].item() == 0.0
        assert abs(pc[1].item() - 0.1) < 1e-5

    def test_cache_key_deterministic(self):
        k1 = _cache_key("test", {"a": 1, "b": 2})
        k2 = _cache_key("test", {"b": 2, "a": 1})
        assert k1 == k2  # order shouldn't matter

    def test_cache_key_differs(self):
        k1 = _cache_key("test", {"a": 1})
        k2 = _cache_key("test", {"a": 2})
        assert k1 != k2


# ── Correlated data generation ────────────────────────────────────────────

class TestCorrelatedGeneration:
    def test_basic_generation(self):
        import numpy as np
        fields = [("field.a", 1), ("field.b", 1), ("field.c", 1)]
        data = _generate_correlated_group(fields, n_timesteps=50, n_latent_factors=2)
        assert len(data) == 3
        for key, tensor in data.items():
            assert tensor.shape == (50,)

    def test_reproducible(self):
        import numpy as np
        fields = [("x", 1)]
        d1 = _generate_correlated_group(fields, 50, rng=np.random.default_rng(123))
        d2 = _generate_correlated_group(fields, 50, rng=np.random.default_rng(123))
        assert torch.allclose(d1["x"], d2["x"])

    def test_smoothing_for_slow_fields(self):
        import numpy as np
        fields = [("fast", 1), ("slow", 192)]
        data = _generate_correlated_group(fields, 200, rng=np.random.default_rng(0))
        # Slow field should be smoother (lower variance of differences)
        fast_var = data["fast"].diff().var().item()
        slow_var = data["slow"].diff().var().item()
        assert slow_var < fast_var

    def test_empty_fields(self):
        data = _generate_correlated_group([], 100)
        assert data == {}


# ── SyntheticCollector ────────────────────────────────────────────────────

class TestSyntheticCollector:
    def test_collect(self):
        cache_dir = tempfile.mkdtemp()
        collector = SyntheticCollector(
            n_timesteps=50, seed=42, cache_dir=cache_dir, force_refresh=True
        )
        spec, data = collector.collect()

        assert isinstance(spec, DatasetSpec)
        assert spec.name == "Synthetic"
        assert len(spec.input_specs) > 0
        assert len(data) > 0

        # Check a few expected fields
        expected_fields = [
            "financial.yield_curves.ten_year",
            "country_us.macro.output.gdp_nowcast",
            "regime.growth_regime",
        ]
        for f in expected_fields:
            assert f in data, f"Missing field: {f}"
            assert data[f].shape == (50,)

    def test_caching(self):
        cache_dir = tempfile.mkdtemp()
        c1 = SyntheticCollector(n_timesteps=30, seed=0, cache_dir=cache_dir)
        spec1, data1 = c1.collect()

        # Second call should load from cache
        c2 = SyntheticCollector(n_timesteps=30, seed=0, cache_dir=cache_dir)
        spec2, data2 = c2.collect()

        assert len(data1) == len(data2)

    def test_different_seeds(self):
        cache_dir = tempfile.mkdtemp()
        c1 = SyntheticCollector(n_timesteps=30, seed=1, cache_dir=cache_dir, force_refresh=True)
        c2 = SyntheticCollector(n_timesteps=30, seed=2, cache_dir=cache_dir, force_refresh=True)
        _, data1 = c1.collect()
        _, data2 = c2.collect()
        # Different seeds should produce different data
        key = list(data1.keys())[0]
        assert not torch.allclose(data1[key], data2[key])

    def test_field_mapping_consistency(self):
        """Every mapping should point to a key in the data dict."""
        collector = SyntheticCollector(n_timesteps=20, force_refresh=True, cache_dir=tempfile.mkdtemp())
        spec, data = collector.collect()
        for mapping in spec.input_specs:
            assert mapping.key in data, f"Mapping {mapping.key} not in data"


# ── FRED Collector (unit tests, no API) ────────────────────────────────────

class TestFREDCollector:
    def test_init_no_key(self):
        collector = FREDCollector(cache_dir=tempfile.mkdtemp())
        assert collector.name == "FRED"

    def test_fetch_no_key_returns_empty(self):
        collector = FREDCollector(api_key=None, cache_dir=tempfile.mkdtemp(), force_refresh=True)
        # Clear any env var for this test
        import os
        old = os.environ.pop("FRED_API_KEY", None)
        try:
            spec, data = collector.collect()
            assert len(data) == 0
        finally:
            if old is not None:
                os.environ["FRED_API_KEY"] = old


# ── Yahoo Finance Collector (unit tests, no API) ──────────────────────────

class TestYahooFinanceCollector:
    def test_init(self):
        collector = YahooFinanceCollector(cache_dir=tempfile.mkdtemp())
        assert collector.name == "YahooFinance"


# ── collect_all ───────────────────────────────────────────────────────────

class TestCollectAll:
    def test_with_synthetic_only(self):
        cache_dir = tempfile.mkdtemp()
        results = collect_all(
            cache_dir=cache_dir,
            force_refresh=True,
            include_synthetic=True,
        )
        # At minimum we should get the synthetic data
        assert len(results) >= 1
        synth = [r for r in results if r[0].name == "Synthetic"]
        assert len(synth) == 1

    def test_without_synthetic(self):
        cache_dir = tempfile.mkdtemp()
        results = collect_all(
            cache_dir=cache_dir,
            force_refresh=True,
            include_synthetic=False,
        )
        synth = [r for r in results if r[0].name == "Synthetic"]
        assert len(synth) == 0
