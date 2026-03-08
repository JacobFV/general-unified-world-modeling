"""Tests for the rendering system."""

import matplotlib
matplotlib.use("Agg")

import pytest
import torch

from general_unified_world_model import WorldProjection, project
from general_unified_world_model.rendering import (
    CanvasHeatmapRenderer,
    TopologyGraphRenderer,
    CausalGraphRenderer,
    FinancialChartRenderer,
    GeopoliticalMapRenderer,
    RegimeDashboardRenderer,
    SocialGraphRenderer,
    RenderContext,
    render,
    render_ceo_use_case,
    render_government_use_case,
    render_agent_use_case,
)


@pytest.fixture
def financial_ctx():
    proj = WorldProjection(include=["financial", "regime"])
    bound = project(proj, T=1, H=24, W=24, d_model=32)
    return RenderContext(bound_schema=bound)


@pytest.fixture
def full_ctx():
    proj = WorldProjection(include=["*"])
    bound = project(proj, T=1, H=128, W=128, d_model=64)
    return RenderContext(bound_schema=bound)


@pytest.fixture
def ceo_ctx():
    from general_unified_world_model.schema.business import Business
    from general_unified_world_model.schema.individual import Individual
    proj = WorldProjection(
        include=["country_us.macro", "sector_tech", "financial.equities", "regime", "forecasts"],
        entities={
            "firm_ACME": Business(),
            "person_ceo": Individual(),
            "person_cfo": Individual(),
        },
    )
    bound = project(proj, T=1, H=48, W=48, d_model=32)
    return RenderContext(bound_schema=bound)


class TestCanvasHeatmap:
    def test_renders_figure(self, financial_ctx):
        r = CanvasHeatmapRenderer()
        fig = r.render(financial_ctx)
        assert fig is not None
        assert fig.get_size_inches()[0] > 0
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_renders_with_state(self, financial_ctx):
        n = financial_ctx.bound_schema.layout.num_positions
        financial_ctx.state = torch.randn(n, 32)
        r = CanvasHeatmapRenderer()
        fig = r.render(financial_ctx)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_save(self, financial_ctx, tmp_path):
        r = CanvasHeatmapRenderer()
        path = r.save(financial_ctx, tmp_path / "test.png")
        assert path.exists()


class TestTopologyGraph:
    def test_renders_figure(self, financial_ctx):
        r = TopologyGraphRenderer()
        fig = r.render(financial_ctx)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_renders_full_world(self, full_ctx):
        r = TopologyGraphRenderer()
        fig = r.render(full_ctx)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestFinancialCharts:
    def test_renders_demo(self, financial_ctx):
        r = FinancialChartRenderer()
        fig = r.render(financial_ctx)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_renders_with_data(self, financial_ctx):
        financial_ctx.time_series = {
            "financial.yield_curves.ten_year": torch.randn(100),
            "financial.yield_curves.two_year": torch.randn(100),
        }
        r = FinancialChartRenderer()
        fig = r.render(financial_ctx)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestGeopoliticalMap:
    def test_renders_default(self, full_ctx):
        r = GeopoliticalMapRenderer()
        fig = r.render(full_ctx)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestRegimeDashboard:
    def test_renders(self, financial_ctx):
        r = RegimeDashboardRenderer()
        fig = r.render(financial_ctx)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestSocialGraph:
    def test_renders_ceo(self, ceo_ctx):
        r = SocialGraphRenderer()
        fig = r.render(ceo_ctx)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_renders_without_people(self, financial_ctx):
        r = SocialGraphRenderer()
        fig = r.render(financial_ctx)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestCausalGraph:
    def test_renders_ceo(self):
        fig = render_ceo_use_case()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_renders_government(self):
        fig = render_government_use_case()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_renders_agent(self):
        fig = render_agent_use_case()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_renders_from_schema(self, financial_ctx):
        r = CausalGraphRenderer()
        fig = r.render(financial_ctx)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestRenderConvenience:
    def test_render_by_name(self, financial_ctx):
        fig = render(
            financial_ctx.bound_schema,
            "canvas_heatmap",
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_unknown_raises(self, financial_ctx):
        with pytest.raises(ValueError, match="Unknown renderer"):
            render(financial_ctx.bound_schema, "nonexistent")

    def test_render_save(self, financial_ctx, tmp_path):
        render(
            financial_ctx.bound_schema,
            "canvas_heatmap",
            save_path=str(tmp_path / "test.png"),
        )
        assert (tmp_path / "test.png").exists()
