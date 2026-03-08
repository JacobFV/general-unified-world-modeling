"""Generate all README images.

Produces high-quality visualizations for:
1. Canvas heatmaps (full world + macro projection + hedge fund)
2. Topology graphs (macro model + hedge fund model)
3. Causal interaction graphs (CEO + government + agent use cases)
4. Financial charts (demo data)
5. Geopolitical map
6. Regime dashboard
7. Social graph (CEO perspective)

Usage:
    python scripts/generate_assets.py
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from general_unified_world_model import WorldProjection, project, World
from general_unified_world_model.rendering import (
    CanvasHeatmapRenderer,
    TopologyGraphRenderer,
    FinancialChartRenderer,
    GeopoliticalMapRenderer,
    RegimeDashboardRenderer,
    SocialGraphRenderer,
    RenderContext,
    render_ceo_use_case,
    render_government_use_case,
    render_agent_use_case,
)
from canvas_engineering import compile_schema, ConnectivityPolicy

ASSETS = Path(__file__).parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

DPI = 200


def render_canvas_full():
    """Full world model canvas — all 857 fields."""
    print("Rendering: canvas_full_world...")
    world = World()
    bound = compile_schema(
        world, T=1, H=128, W=128, d_model=64,
        connectivity=ConnectivityPolicy(intra="dense", parent_child="hub_spoke"),
    )
    ctx = RenderContext(
        bound_schema=bound,
        title=f"Full World Model — {len(bound.field_names)} fields on 128x128 canvas",
    )
    r = CanvasHeatmapRenderer()
    r.save(ctx, ASSETS / "canvas_full_world.png", dpi=DPI)


def render_canvas_macro():
    """Macro model projection canvas."""
    print("Rendering: canvas_macro_projection...")
    proj = WorldProjection(
        include=["country_us.macro", "financial.yield_curves", "financial.credit",
                 "regime", "forecasts.macro"],
    )
    bound = project(proj, T=1, H=32, W=32, d_model=64)
    ctx = RenderContext(
        bound_schema=bound,
        title=f"Macro Model Projection — {len(bound.field_names)} fields on 32x32",
    )
    r = CanvasHeatmapRenderer()
    r.save(ctx, ASSETS / "canvas_macro_projection.png", dpi=DPI)


def render_topology_macro():
    """Macro model topology graph."""
    print("Rendering: topology_macro...")
    proj = WorldProjection(
        include=["country_us.macro", "financial.yield_curves", "financial.credit",
                 "financial.central_banks", "regime", "forecasts.macro",
                 "narratives.public", "events"],
    )
    bound = project(proj, T=1, H=48, W=48, d_model=64)
    ctx = RenderContext(
        bound_schema=bound,
        title="Macroeconomic Model — Domain Topology",
    )
    r = TopologyGraphRenderer()
    r.save(ctx, ASSETS / "topology_macro.png", dpi=DPI)


def render_topology_hedge_fund():
    """Hedge fund model topology graph."""
    print("Rendering: topology_hedge_fund...")
    proj = WorldProjection(
        include=["financial", "country_us.macro", "country_cn.macro",
                 "regime", "forecasts.macro", "forecasts.financial",
                 "narratives.positioning", "events", "trust"],
        firms=["AAPL", "NVDA"],
    )
    bound = project(proj, T=1, H=64, W=64, d_model=64)
    ctx = RenderContext(
        bound_schema=bound,
        title=f"Hedge Fund Model — {len(bound.field_names)} fields, multi-domain topology",
    )
    r = TopologyGraphRenderer()
    r.save(ctx, ASSETS / "topology_hedge_fund.png", dpi=DPI)


def render_financial_charts():
    """Financial charts with demo data."""
    print("Rendering: financial_charts...")
    proj = WorldProjection(include=["financial"])
    bound = project(proj, T=1, H=32, W=32, d_model=64)
    ctx = RenderContext(
        bound_schema=bound,
        title="Financial Markets — World Model Time Series",
    )
    r = FinancialChartRenderer()
    r.save(ctx, ASSETS / "financial_charts.png", dpi=DPI)


def render_geopolitical_map():
    """Geopolitical state map with vector-to-RGB projection (static + GIF)."""
    print("Rendering: geopolitical_map...")
    proj = WorldProjection(include=["*"])
    bound = project(proj, T=1, H=128, W=128, d_model=64)
    ctx = RenderContext(
        bound_schema=bound,
        title="Geopolitical State — Vector Projection to RGB",
    )
    r = GeopoliticalMapRenderer()
    r.save(ctx, ASSETS / "geopolitical_map.png", dpi=DPI)

    print("Rendering: geopolitical_globe.gif (rotating)...")
    r.render_rotating_gif(
        ctx,
        ASSETS / "geopolitical_globe.gif",
        n_frames=72,
        center_lat=20,
        duration_ms=80,
        dpi=100,
        figsize=(6, 6),
    )


def render_regime_dashboard():
    """Regime state dashboard — minimal bars."""
    print("Rendering: regime_dashboard...")
    proj = WorldProjection(include=["regime", "forecasts"])
    bound = project(proj, T=1, H=24, W=24, d_model=64)
    ctx = RenderContext(
        bound_schema=bound,
        title="regime state",
    )
    r = RegimeDashboardRenderer()
    r.save(ctx, ASSETS / "regime_dashboard.png", dpi=DPI)


def render_social_graph():
    """CEO perspective social graph."""
    print("Rendering: social_graph_ceo...")
    proj = WorldProjection(
        include=["country_us.macro", "sector_tech", "financial.equities",
                 "regime", "forecasts.business", "narratives.elites"],
        firms=["ACME", "RIVAL"],
        individuals=["ceo", "cfo", "board_chair"],
    )
    bound = project(proj, T=1, H=48, W=48, d_model=64)
    ctx = RenderContext(
        bound_schema=bound,
        title="CEO Perspective — Entity Network",
    )
    r = SocialGraphRenderer()
    r.save(ctx, ASSETS / "social_graph_ceo.png", dpi=DPI)


def render_canvas_hedge_fund():
    """Hedge fund canvas for the README example."""
    print("Rendering: canvas_hedge_fund...")
    proj = WorldProjection(
        include=["financial", "country_us.macro", "country_cn.macro",
                 "regime", "forecasts.macro", "forecasts.financial",
                 "narratives.positioning", "events"],
        firms=["AAPL", "NVDA"],
    )
    bound = project(proj, T=1, H=64, W=64, d_model=64)
    ctx = RenderContext(
        bound_schema=bound,
        title=f"Hedge Fund Projection — {len(bound.field_names)} fields on 64x64",
    )
    r = CanvasHeatmapRenderer()
    r.save(ctx, ASSETS / "canvas_hedge_fund.png", dpi=DPI)


# ── Use-case causal interaction graphs ───────────────────────────────


def render_usecase_ceo():
    """CEO use case — causal interaction graph."""
    print("Rendering: usecase_ceo...")
    fig = render_ceo_use_case()
    fig.savefig(ASSETS / "usecase_ceo.png", dpi=DPI, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def render_usecase_government():
    """Government use case — causal interaction graph."""
    print("Rendering: usecase_government...")
    fig = render_government_use_case()
    fig.savefig(ASSETS / "usecase_government.png", dpi=DPI, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def render_usecase_agent():
    """Computer use agent use case — causal interaction graph."""
    print("Rendering: usecase_agent...")
    fig = render_agent_use_case()
    fig.savefig(ASSETS / "usecase_agent.png", dpi=DPI, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    render_canvas_full()
    render_canvas_macro()
    render_canvas_hedge_fund()
    render_topology_macro()
    render_topology_hedge_fund()
    render_usecase_ceo()
    render_usecase_government()
    render_usecase_agent()
    render_financial_charts()
    render_geopolitical_map()
    render_regime_dashboard()
    render_social_graph()
    print(f"\nDone! Generated {len(list(ASSETS.glob('*.png')))} images in {ASSETS}/")
