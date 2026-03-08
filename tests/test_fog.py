"""Tests for projection filtering."""

from general_unified_world_model.projection.subset import (
    project, _resolve_path,
)
from general_unified_world_model.schema.world import World


class TestResolvePath:
    def test_top_level(self):
        world = World()
        obj = _resolve_path(world, "financial")
        assert obj is not None

    def test_nested(self):
        world = World()
        obj = _resolve_path(world, "country_us.macro")
        assert obj is not None

    def test_deep(self):
        world = World()
        obj = _resolve_path(world, "country_us.macro.output")
        assert obj is not None

    def test_invalid(self):
        world = World()
        assert _resolve_path(world, "nonexistent") is None
        assert _resolve_path(world, "financial.nonexistent") is None


class TestProjectionFiltering:
    def test_full_include_expands_everything(self):
        """Full path include should expand all sub-types."""
        bound = project(include=["financial"], T=1, H=32, W=32, d_model=32)
        # All YieldCurveState leaf fields should exist
        yc_leaf_fields = [n for n in bound.field_names
                          if "yield_curves" in n and "." in n]
        assert len(yc_leaf_fields) >= 10

    def test_sub_path_is_independent(self):
        """Sub-path include should be a standalone entity."""
        bound = project(include=["financial.yield_curves", "regime"], T=1, H=32, W=32, d_model=32)
        # yield_curves should have its leaf fields
        yc_fields = [n for n in bound.field_names if "ten_year" in n]
        assert len(yc_fields) == 1
        # No siblings — financial.credit, financial.fx, etc. should NOT exist
        credit_fields = [n for n in bound.field_names if "financial" in n and "credit" in n]
        assert len(credit_fields) == 0

    def test_country_sub_path_standalone(self):
        """country_us.macro without country_us is standalone."""
        bound = project(include=["country_us.macro", "regime"], T=1, H=32, W=32, d_model=32)
        # politics and demographics should NOT exist
        politics = [n for n in bound.field_names if "politics" in n]
        assert len(politics) == 0
        # macro fields should exist
        macro_fields = [n for n in bound.field_names if "gdp" in n or "inflation" in n]
        assert len(macro_fields) > 0

    def test_wildcard_includes_everything(self):
        """Wildcard include should expand all sub-types."""
        bound = project(include=["*"], T=1, H=128, W=128, d_model=32)
        yc_leaf_fields = [n for n in bound.field_names
                          if n.startswith("financial.yield_curves.")]
        assert len(yc_leaf_fields) == 10

    def test_multiple_independent_paths(self):
        """Multiple paths are each independently resolved."""
        bound = project(
            include=["financial.yield_curves", "financial.credit", "regime"],
            T=1, H=32, W=32, d_model=32,
        )
        # Both should have their fields
        yc_fields = [n for n in bound.field_names if "ten_year" in n]
        credit_fields = [n for n in bound.field_names if "ig_spread" in n]
        assert len(yc_fields) == 1
        assert len(credit_fields) == 1
        # fx should NOT exist
        fx_fields = [n for n in bound.field_names if "dxy" in n]
        assert len(fx_fields) == 0

    def test_connectivity_exists(self):
        """Included fields should have connections."""
        bound = project(include=["financial.yield_curves", "regime"], T=1, H=32, W=32, d_model=32)
        assert bound.topology is not None
        assert len(bound.topology.connections) > 0

    def test_invalid_path_skipped(self):
        """Invalid paths should be silently skipped."""
        bound = project(include=["nonexistent", "regime"], T=1, H=32, W=32, d_model=32)
        # regime should still work
        regime_fields = [n for n in bound.field_names if "regime" in n]
        assert len(regime_fields) > 0
