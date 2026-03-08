"""Rendering: visualize world projections at various planes of abstraction.

The rendering system provides multiple views into the same world model state:
- Canvas heatmaps (raw latent space allocation)
- Topology graphs (causal/attention connectivity)
- Causal interaction graphs (use-case-specific directed acyclic graphs)
- Financial charts (time series line graphs)
- Geopolitical maps (globe with country state → RGB) — requires cartopy
- Social graphs (first-person perspective of actor networks)
- Regime dashboards (compressed state overview)

Each renderer implements the Renderer protocol and can produce
static images (PNG/SVG) or interactive figures (matplotlib/plotly).
"""

from general_unified_world_model.rendering.base import (
    Renderer,
    RenderContext,
    render,
    render_all,
)
from general_unified_world_model.rendering.canvas import CanvasHeatmapRenderer
from general_unified_world_model.rendering.topology import TopologyGraphRenderer
from general_unified_world_model.rendering.causal_graph import (
    CausalGraphRenderer,
    render_ceo_use_case,
    render_government_use_case,
    render_agent_use_case,
)
from general_unified_world_model.rendering.financial import FinancialChartRenderer
from general_unified_world_model.rendering.regime import RegimeDashboardRenderer
from general_unified_world_model.rendering.social import SocialGraphRenderer

# Geopolitical renderer requires cartopy (pip install general-unified-world-model[diagrams])
try:
    from general_unified_world_model.rendering.geopolitical import GeopoliticalMapRenderer
except ImportError:
    pass
