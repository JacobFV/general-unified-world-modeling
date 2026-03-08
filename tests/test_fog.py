"""Tests for coarse-graining in projections."""

import pytest
from general_unified_world_model.projection.subset import (
    WorldProjection, project,
    _count_leaf_fields, _get_median_period, _coarse_grain,
)
from canvas_engineering import Field


class TestCoarseGrainHelpers:
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


class TestCoarseGrainProjection:
    def test_full_include_no_coarsening(self):
        """Full path include should produce no coarse-grained fields."""
        proj = WorldProjection(include=["financial"])
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        # All 7 sub-types should be fully expanded
        yc_fields = [n for n in bound.field_names if "yield_curves" in n]
        assert len(yc_fields) == 10  # all 10 YieldCurveState fields

    def test_sub_path_coarse_grains_siblings(self):
        """Sub-path include should coarse-grain excluded siblings."""
        proj = WorldProjection(include=["financial.yield_curves", "regime"])
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        # yield_curves should be fully expanded (10 fields)
        yc_fields = [n for n in bound.field_names if "yield_curves" in n]
        assert len(yc_fields) == 10
        # Siblings should be coarse-grained to single fields
        assert "financial.credit" in bound.field_names
        assert "financial.fx" in bound.field_names
        assert "financial.liquidity" in bound.field_names
        assert "financial.equities" in bound.field_names
        assert "financial.crypto" in bound.field_names
        assert "financial.central_banks" in bound.field_names

    def test_coarse_grained_semantic_types(self):
        """Coarse-grained fields should have descriptive semantic types."""
        proj = WorldProjection(include=["financial.yield_curves", "regime"])
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        spec = bound["financial.credit"].spec
        assert "coarse" in spec.semantic_type
        assert "financial.credit" in spec.semantic_type

    def test_coarse_grained_has_period(self):
        """Coarse-grained fields should have a valid period from median."""
        proj = WorldProjection(include=["financial.yield_curves", "regime"])
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        spec = bound["financial.credit"].spec
        assert spec.period >= 1

    def test_coarse_grained_connectivity(self):
        """Coarse-grained fields should have connections."""
        proj = WorldProjection(include=["financial.yield_curves", "regime"])
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        assert bound.topology is not None
        coarse_conns = [c for c in bound.topology.connections
                        if c.src == "financial.credit" or c.dst == "financial.credit"]
        assert len(coarse_conns) > 0

    def test_coarse_grained_self_connections(self):
        """Coarse-grained fields should have self-connections."""
        proj = WorldProjection(include=["financial.yield_curves", "regime"])
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        self_conns = [c for c in bound.topology.connections
                      if c.src == "financial.credit" and c.dst == "financial.credit"]
        assert len(self_conns) > 0

    def test_country_sub_path_coarse_grains(self):
        """Country with sub-path should coarse-grain non-included sub-types."""
        proj = WorldProjection(include=["country_us.macro", "regime"])
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        # politics and demographics should be coarse-grained, not absent
        assert "country_us.politics" in bound.field_names
        assert "country_us.demographics" in bound.field_names

    def test_wildcard_no_coarsening(self):
        """Wildcard include should never coarse-grain."""
        proj = WorldProjection(include=["*"])
        bound = project(proj, T=1, H=128, W=128, d_model=32)
        # Should have fully expanded sub-types, not single fields
        yc_fields = [n for n in bound.field_names if "yield_curves" in n]
        assert len(yc_fields) == 10

    def test_coarse_grained_is_1x1(self):
        """All coarse-grained fields should be 1×1 positions."""
        proj = WorldProjection(include=["financial.yield_curves", "regime"])
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        for name in ["financial.credit", "financial.fx", "financial.liquidity",
                      "financial.equities", "financial.crypto", "financial.central_banks"]:
            bf = bound[name]
            t0, t1, h0, h1, w0, w1 = bf.spec.bounds
            assert (h1 - h0) == 1 and (w1 - w0) == 1, \
                f"{name} is not 1×1: {h1-h0}×{w1-w0}"

    def test_multiple_sub_paths(self):
        """Multiple sub-paths from same parent should coarse-grain the rest."""
        proj = WorldProjection(
            include=["financial.yield_curves", "financial.credit", "regime"],
        )
        bound = project(proj, T=1, H=32, W=32, d_model=32)
        # fx, liquidity, equities, crypto, central_banks should be coarse-grained
        assert "financial.fx" in bound.field_names
        assert "financial.liquidity" in bound.field_names
        # The included ones should be fully expanded
        ycurve_fields = [n for n in bound.field_names
                         if n.startswith("financial.yield_curves.")]
        assert len(ycurve_fields) == 10
        credit_fields = [n for n in bound.field_names
                         if n.startswith("financial.credit.")]
        assert len(credit_fields) == 10

    def test_names_preserved_across_projections(self):
        """The same field should have the same name regardless of resolution."""
        proj_full = WorldProjection(include=["financial"])
        proj_coarse = WorldProjection(include=["financial.yield_curves", "regime"])

        bound_full = project(proj_full, T=1, H=32, W=32, d_model=32)
        bound_coarse = project(proj_coarse, T=1, H=32, W=32, d_model=32)

        # In the full projection, credit sub-fields exist
        credit_sub = [n for n in bound_full.field_names
                      if n.startswith("financial.credit.")]
        assert len(credit_sub) == 10

        # In the coarse projection, "financial.credit" exists as a single field
        assert "financial.credit" in bound_coarse.field_names
        # No sub-fields — it's a leaf now
        credit_sub_coarse = [n for n in bound_coarse.field_names
                             if n.startswith("financial.credit.")]
        assert len(credit_sub_coarse) == 0
