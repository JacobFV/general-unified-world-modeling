"""Tests for fog region functionality."""

import pytest
from general_unified_world_model.projection.subset import (
    WorldProjection, project,
    _count_leaf_fields, _get_median_period, _apply_fog_to_object,
)
from canvas_engineering import Field


class TestFogHelpers:
    def test_count_leaf_fields(self):
        from general_unified_world_model.schema.financial import YieldCurveState
        yc = YieldCurveState()
        count = _count_leaf_fields(yc)
        assert count == 10  # 10 fields in YieldCurveState

    def test_count_leaf_field_single(self):
        f = Field(1, 1)
        assert _count_leaf_fields(f) == 1

    def test_median_period(self):
        from general_unified_world_model.schema.financial import YieldCurveState
        yc = YieldCurveState()
        mp = _get_median_period(yc)
        assert mp > 0


class TestFogProjection:
    def test_full_include_no_fog(self):
        """Full path include should produce no fog fields."""
        proj = WorldProjection(include=["financial"], fog=True)
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        fog = [n for n in bound.field_names if "_fog_" in n]
        assert len(fog) == 0

    def test_sub_path_creates_fog(self):
        """Sub-path include should create fog for excluded siblings."""
        proj = WorldProjection(include=["financial.yield_curves", "regime"], fog=True)
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        fog = [n for n in bound.field_names if "_fog_" in n]
        assert len(fog) > 0
        # Should have fog for credit, fx, liquidity, equities, crypto, central_banks
        assert len(fog) == 6

    def test_fog_semantic_types(self):
        """Fog fields should have descriptive semantic types."""
        proj = WorldProjection(include=["financial.yield_curves", "regime"], fog=True)
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        for name in bound.field_names:
            if "_fog_" in name:
                spec = bound[name].spec
                assert spec.semantic_type.startswith("fog:")
                assert "unmodeled fields" in spec.semantic_type

    def test_fog_has_period(self):
        """Fog fields should have a valid period from median of excluded fields."""
        proj = WorldProjection(include=["financial.yield_curves", "regime"], fog=True)
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        for name in bound.field_names:
            if "_fog_" in name:
                spec = bound[name].spec
                assert spec.period >= 1

    def test_fog_connectivity(self):
        """Fog fields should have connections to other fields."""
        proj = WorldProjection(include=["financial.yield_curves", "regime"], fog=True)
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        assert bound.topology is not None
        fog_conns = [c for c in bound.topology.connections
                     if "_fog_" in c.src or "_fog_" in c.dst]
        assert len(fog_conns) > 0

    def test_fog_self_connections(self):
        """Fog fields should have self-connections (self-awareness)."""
        proj = WorldProjection(include=["financial.yield_curves", "regime"], fog=True)
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        self_conns = [c for c in bound.topology.connections
                      if "_fog_" in c.src and c.src == c.dst]
        assert len(self_conns) > 0

    def test_fog_disabled(self):
        """fog=False should include everything without fog."""
        proj_fog = WorldProjection(include=["country_us.macro", "regime"], fog=True)
        proj_nofog = WorldProjection(include=["country_us.macro", "regime"], fog=False)
        bound_fog = project(proj_fog, T=1, H=32, W=32, d_model=32)
        bound_nofog = project(proj_nofog, T=1, H=64, W=64, d_model=32)

        fog_fields = [n for n in bound_fog.field_names if "_fog_" in n]
        nofog_fields = [n for n in bound_nofog.field_names if "_fog_" in n]
        assert len(fog_fields) > 0
        assert len(nofog_fields) == 0
        # Without fog, more explicit fields should be present
        assert len(bound_nofog.field_names) > len(bound_fog.field_names)

    def test_country_sub_path_fog(self):
        """Country with sub-path should fog non-included sub-types."""
        proj = WorldProjection(include=["country_us.macro", "regime"], fog=True)
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        fog = [n for n in bound.field_names if "_fog_" in n]
        # Should have fog for politics, demographics (the other sub-types of Country)
        fog_names = set(fog)
        assert "country_us._fog_politics" in fog_names
        assert "country_us._fog_demographics" in fog_names

    def test_wildcard_no_fog(self):
        """Wildcard include should never create fog."""
        proj = WorldProjection(include=["*"], fog=True)
        bound = project(proj, T=1, H=128, W=128, d_model=32)
        fog = [n for n in bound.field_names if "_fog_" in n]
        assert len(fog) == 0

    def test_fog_is_1x1(self):
        """All fog fields should be 1×1 positions."""
        proj = WorldProjection(include=["financial.yield_curves", "regime"], fog=True)
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        for name in bound.field_names:
            if "_fog_" in name:
                bf = bound[name]
                t0, t1, h0, h1, w0, w1 = bf.spec.bounds
                assert (h1 - h0) == 1 and (w1 - w0) == 1, \
                    f"Fog {name} is not 1×1: {h1-h0}×{w1-w0}"

    def test_multiple_sub_paths(self):
        """Multiple sub-paths from same parent should fog the rest."""
        proj = WorldProjection(
            include=["financial.yield_curves", "financial.credit", "regime"],
            fog=True,
        )
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        fog = [n for n in bound.field_names if "_fog_" in n]
        # Should fog fx, liquidity, equities, crypto, central_banks (5 fog fields)
        # yield_curves and credit should NOT be fogged
        assert "financial._fog_fx" in fog
        assert "financial._fog_liquidity" in fog
        # The included ones should not be fogged
        ycurve_fields = [n for n in bound.field_names if "yield_curves" in n and "_fog_" not in n]
        assert len(ycurve_fields) == 10
        credit_fields = [n for n in bound.field_names if ".credit." in n and "_fog_" not in n]
        assert len(credit_fields) == 10
