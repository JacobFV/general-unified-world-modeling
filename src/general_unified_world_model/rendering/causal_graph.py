"""Causal interaction graph renderer.

Produces professional LaTeX-style directed acyclic graphs showing
causal relationships between world model domains for specific use cases.
Each use case gets a hand-tuned graph showing the information flow
between entities in the projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from general_unified_world_model.rendering.base import Renderer, RenderContext, _register


# ── Node category styling (pastel fills for paper-quality figures) ────

CATEGORY_STYLE = {
    "macro":        {"fill": "#D6EAF8", "border": "#5DADE2", "text": "#1B4F72"},
    "financial":    {"fill": "#FEF9E7", "border": "#D4AC0D", "text": "#7D6608"},
    "firm":         {"fill": "#FDEBD0", "border": "#E67E22", "text": "#784212"},
    "person":       {"fill": "#FDEDEC", "border": "#E74C3C", "text": "#78281F"},
    "regime":       {"fill": "#F9EBEA", "border": "#CB4335", "text": "#641E16"},
    "forecast":     {"fill": "#D5F5E3", "border": "#27AE60", "text": "#186A3B"},
    "events":       {"fill": "#D1F2EB", "border": "#1ABC9C", "text": "#0E6655"},
    "sector":       {"fill": "#E8DAEF", "border": "#8E44AD", "text": "#4A235A"},
    "intervention": {"fill": "#F5EEF8", "border": "#9B59B6", "text": "#512E5F"},
    "country":      {"fill": "#D4E6F1", "border": "#2E86C1", "text": "#1A5276"},
}


@dataclass
class GNode:
    """A node in a causal interaction graph."""
    name: str
    label: str
    x: float
    y: float
    category: str
    width: float = 2.0
    height: float = 0.7


@dataclass
class GEdge:
    """A directed edge in a causal interaction graph."""
    src: str
    dst: str
    style: str = "solid"
    weight: float = 1.0
    label: str = ""
    rad: float = 0.0


# ── Core rendering function ─────────────────────────────────────────


def _rect_border_point(node: GNode, angle: float) -> tuple[float, float]:
    """Point where a ray from node center at `angle` exits the rectangle."""
    hw = node.width / 2
    hh = node.height / 2
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    if abs(cos_a) < 1e-6:
        return node.x, node.y + hh * (1 if sin_a > 0 else -1)
    if abs(sin_a) < 1e-6:
        return node.x + hw * (1 if cos_a > 0 else -1), node.y

    tx = hw / abs(cos_a)
    ty = hh / abs(sin_a)
    t = min(tx, ty)
    return node.x + t * cos_a, node.y + t * sin_a


def _render_graph(
    nodes: list[GNode],
    edges: list[GEdge],
    title: str,
    figsize: tuple[float, float] = (14, 9),
):
    """Render a causal interaction graph with clean, professional styling."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    bg = "#FAFAFA"
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor(bg)

    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    pad_x, pad_y = 1.8, 1.2
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")

    node_map = {n.name: n for n in nodes}

    # ── edges (behind nodes) ─────────────────────────────────────────
    for edge in edges:
        src = node_map[edge.src]
        dst = node_map[edge.dst]
        dx = dst.x - src.x
        dy = dst.y - src.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            continue

        angle = math.atan2(dy, dx)
        sx, sy = _rect_border_point(src, angle)
        ex, ey = _rect_border_point(dst, angle + math.pi)

        ls = "solid" if edge.style == "solid" else (0, (5, 3))
        conn = f"arc3,rad={edge.rad}" if edge.rad != 0 else "arc3,rad=0.0"

        arrow = FancyArrowPatch(
            (sx, sy), (ex, ey),
            arrowstyle="-|>",
            connectionstyle=conn,
            mutation_scale=13,
            color="#555555",
            linewidth=0.9 + 0.4 * edge.weight,
            linestyle=ls,
            alpha=0.65 if edge.style == "solid" else 0.45,
            zorder=2,
        )
        ax.add_patch(arrow)

        if edge.label:
            mx = (sx + ex) / 2
            my = (sy + ey) / 2
            ax.text(mx, my + 0.15, edge.label, ha="center", va="bottom",
                    fontsize=6.5, fontstyle="italic", color="#777", zorder=3)

    # ── nodes ────────────────────────────────────────────────────────
    for node in nodes:
        st = CATEGORY_STYLE.get(node.category, CATEGORY_STYLE["macro"])

        box = FancyBboxPatch(
            (node.x - node.width / 2, node.y - node.height / 2),
            node.width, node.height,
            boxstyle="round,pad=0.06,rounding_size=0.10",
            facecolor=st["fill"],
            edgecolor=st["border"],
            linewidth=1.3,
            zorder=5,
        )
        ax.add_patch(box)

        ax.text(
            node.x, node.y, node.label,
            ha="center", va="center",
            fontsize=9, fontweight="medium",
            color=st["text"],
            fontfamily="sans-serif",
            zorder=6,
        )

    # ── title ────────────────────────────────────────────────────────
    ax.text(
        (min(xs) + max(xs)) / 2, max(ys) + pad_y * 0.65,
        title,
        ha="center", va="center",
        fontsize=14, fontweight="bold",
        color="#1A1A1A", fontfamily="sans-serif",
    )

    fig.tight_layout(pad=0.5)
    return fig


# ── Use-case graph definitions ───────────────────────────────────────


def render_ceo_use_case():
    """CEO use case: model my company in macroeconomic context.

    Returns a matplotlib Figure with a causal interaction graph.
    """
    nodes = [
        GNode("regime",   "Regime",       6.0, 7.5, "regime",    1.6, 0.65),
        GNode("macro",    "US Macro",     1.5, 5.5, "macro",     2.0, 0.7),
        GNode("sector",   "Tech Sector",  4.5, 5.5, "sector",    2.2, 0.7),
        GNode("yields",   "Yield Curves", 7.5, 5.5, "financial", 2.2, 0.7),
        GNode("equities", "Equities",    10.5, 5.5, "financial", 1.8, 0.7),
        GNode("acme",     "ACME Corp",    4.5, 3.3, "firm",      2.0, 0.7),
        GNode("rival",    "RIVAL Corp",   8.5, 3.3, "firm",      2.0, 0.7),
        GNode("ceo",      "CEO",          2.5, 1.3, "person",    1.2, 0.6),
        GNode("cfo",      "CFO",          4.5, 1.3, "person",    1.2, 0.6),
        GNode("cto",      "CTO",          6.5, 1.3, "person",    1.2, 0.6),
        GNode("forecast", "Forecasts",    6.5, -0.2, "forecast", 2.0, 0.7),
    ]
    edges = [
        # Regime conditions everything
        GEdge("regime", "macro",    rad=0.05),
        GEdge("regime", "sector"),
        GEdge("regime", "yields",   rad=-0.05),
        GEdge("regime", "equities", rad=-0.1),
        # Context → firms
        GEdge("macro",    "acme"),
        GEdge("sector",   "acme"),
        GEdge("sector",   "rival"),
        GEdge("yields",   "acme",   rad=-0.12),
        GEdge("equities", "rival"),
        # Competitive dynamics
        GEdge("rival", "acme", style="dashed", rad=0.18),
        # Executive influence (upward arrows)
        GEdge("ceo", "acme"),
        GEdge("cfo", "acme"),
        GEdge("cto", "acme", rad=-0.08),
        # Outputs
        GEdge("acme",   "forecast"),
        GEdge("regime", "forecast", rad=0.35),
    ]
    return _render_graph(nodes, edges, "CEO: Model My Company in Context")


def render_government_use_case():
    """Government use case: model policy impact across economies.

    Returns a matplotlib Figure with a causal interaction graph.
    """
    nodes = [
        GNode("regime",    "Regime",           6.0, 7.5, "regime",       1.6, 0.65),
        GNode("us",        "United States",    1.5, 5.5, "country",     2.4, 0.7),
        GNode("cn",        "China",            4.0, 5.5, "country",     1.6, 0.7),
        GNode("eu",        "EU",               6.0, 5.5, "country",     1.4, 0.7),
        GNode("jp",        "Japan",            8.0, 5.5, "country",     1.6, 0.7),
        GNode("uk",        "UK",              10.5, 5.5, "country",     1.4, 0.7),
        GNode("financial", "Financial System", 3.0, 3.0, "financial",   2.6, 0.7),
        GNode("interv",    "Interventions",    9.0, 3.0, "intervention", 2.4, 0.7),
        GNode("forecast",  "Forecasts",        6.0, 0.8, "forecast",    2.0, 0.7),
    ]
    edges = [
        # Regime → countries
        GEdge("regime", "us",  rad=0.06),
        GEdge("regime", "cn",  rad=0.04),
        GEdge("regime", "eu"),
        GEdge("regime", "jp",  rad=-0.04),
        GEdge("regime", "uk",  rad=-0.08),
        # Bilateral trade / geopolitical (bidirectional pairs)
        GEdge("us", "cn", rad=0.12),
        GEdge("cn", "us", rad=0.12),
        GEdge("us", "eu", rad=0.12),
        GEdge("eu", "us", rad=0.12),
        GEdge("us", "jp", rad=0.08),
        GEdge("cn", "eu", rad=0.10),
        # Financial system ↔ economies
        GEdge("financial", "us"),
        GEdge("financial", "cn", rad=-0.05),
        GEdge("financial", "eu"),
        GEdge("us", "financial", rad=0.10),
        # Policy intervention
        GEdge("interv", "us",        rad=-0.12),
        GEdge("interv", "financial", rad=0.08),
        # Outputs
        GEdge("us",        "forecast"),
        GEdge("financial", "forecast", rad=0.10),
        GEdge("regime",    "forecast", rad=0.30),
    ]
    return _render_graph(nodes, edges, "Government: Model Policy Impact")


def render_agent_use_case():
    """Computer use agent: model the user's world context.

    Returns a matplotlib Figure with a causal interaction graph.
    """
    nodes = [
        GNode("events",   "Events",             1.5, 5.0, "events",   1.6, 0.7),
        GNode("regime",   "Regime State",        6.0, 5.0, "regime",   2.2, 0.7),
        GNode("user",     "User",                2.5, 2.8, "person",   1.4, 0.7),
        GNode("org",      "User Org",            5.5, 2.8, "firm",     1.8, 0.7),
        GNode("forecast", "Recession Forecast",  4.0, 0.8, "forecast", 2.8, 0.7),
    ]
    edges = [
        GEdge("events", "user"),
        GEdge("events", "org",      rad=0.15),
        GEdge("events", "regime",   rad=0.08),
        GEdge("regime", "forecast"),
        GEdge("user",   "org",      rad=0.12),
        GEdge("org",    "user",     rad=0.12),
        GEdge("user",   "forecast", style="dashed", rad=0.05),
        GEdge("org",    "forecast", style="dashed", rad=-0.05),
    ]
    return _render_graph(
        nodes, edges,
        "Computer Use Agent: Model the User's World",
        figsize=(10, 7),
    )


# ── Registered renderer (generic, from BoundSchema topology) ─────────


@_register
class CausalGraphRenderer(Renderer):
    """Render a causal interaction graph from BoundSchema topology.

    This is the generic version that extracts the graph from the schema.
    For use-case-specific graphs, use the standalone functions above.
    """

    @property
    def name(self) -> str:
        return "causal_graph"

    def render(self, ctx: RenderContext):
        from collections import defaultdict
        from general_unified_world_model.rendering.canvas import _domain_of

        # Group fields by domain
        domain_fields = defaultdict(int)
        for name in ctx.bound_schema.field_names:
            domain_fields[_domain_of(name)] += 1

        domains = sorted(domain_fields.keys())
        if not domains:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No domains", ha="center", va="center")
            return fig

        # Layout domains in a grid
        cols = min(4, len(domains))
        rows = math.ceil(len(domains) / cols)

        nodes = []
        for i, domain in enumerate(domains):
            col = i % cols
            row = i // cols
            x = 2.5 + col * 3.0
            y = (rows - 1 - row) * 2.0 + 1.0
            cat = domain if domain in CATEGORY_STYLE else "macro"
            nodes.append(GNode(domain, domain, x, y, cat, 2.2, 0.7))

        # Extract cross-domain edges from topology
        edges = []
        if ctx.bound_schema.topology is not None:
            edge_counts = defaultdict(int)
            for conn in ctx.bound_schema.topology.connections:
                src_d = _domain_of(conn.src)
                dst_d = _domain_of(conn.dst)
                if src_d != dst_d:
                    edge_counts[(src_d, dst_d)] += 1

            max_count = max(edge_counts.values()) if edge_counts else 1
            seen = set()
            for (src_d, dst_d), count in edge_counts.items():
                key = (src_d, dst_d)
                if key not in seen:
                    w = count / max_count
                    rad = 0.12 if (dst_d, src_d) in edge_counts else 0.0
                    edges.append(GEdge(src_d, dst_d, weight=w, rad=rad))
                    seen.add(key)

        title = ctx.title or f"Causal Graph ({len(domains)} domains)"
        return _render_graph(nodes, edges, title)
