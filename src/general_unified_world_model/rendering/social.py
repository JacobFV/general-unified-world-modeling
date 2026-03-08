"""Social graph renderer: first-person network perspective.

Renders the world model from the perspective of a specific actor —
a CEO, a central banker, a fund manager — showing their network of
connections, influence pathways, and information flows.

Connectivity is derived from:
1. Actual topology connections between entities (cross-domain edges).
2. Structural inference: entities in the same projection are related
   (person-to-firm, firm-to-sector, entity-to-regime, etc.).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from general_unified_world_model.rendering.base import Renderer, RenderContext, _register
from general_unified_world_model.rendering.canvas import DOMAIN_COLORS


NODE_STYLES = {
    "person":    {"shape": "o", "base_size": 600, "zorder": 10},
    "firm":      {"shape": "s", "base_size": 500, "zorder": 8},
    "country":   {"shape": "D", "base_size": 450, "zorder": 7},
    "sector":    {"shape": "^", "base_size": 400, "zorder": 6},
    "regime":    {"shape": "h", "base_size": 350, "zorder": 5},
    "financial": {"shape": "p", "base_size": 350, "zorder": 5},
    "forecasts": {"shape": "v", "base_size": 300, "zorder": 5},
    "narratives": {"shape": "8", "base_size": 300, "zorder": 5},
    "events":    {"shape": "H", "base_size": 300, "zorder": 5},
}


def _entity_type(name: str) -> str:
    """Classify a top-level field into entity type."""
    for prefix in NODE_STYLES:
        if name.startswith(prefix):
            return prefix
    return "financial"


# ── Structural edge inference ────────────────────────────────────────

# Which entity type pairs have structural relationships?
# (type_a, type_b) → base_weight
_STRUCTURAL_EDGES = [
    ("person",   "firm",      0.80),   # executives belong to firms
    ("firm",     "sector",    0.60),   # firms operate in sectors
    ("firm",     "financial", 0.45),   # firms are valued in markets
    ("firm",     "country",   0.40),   # firms operate in countries
    ("person",   "regime",    0.25),   # people respond to regime
    ("firm",     "regime",    0.35),   # firms respond to regime
    ("country",  "regime",    0.50),   # countries constitute regime
    ("country",  "financial", 0.55),   # countries have financial systems
    ("sector",   "financial", 0.40),   # sectors priced in markets
    ("regime",   "forecasts", 0.60),   # regime drives forecasts
    ("firm",     "forecasts", 0.30),   # firms are forecast subjects
    ("person",   "narratives", 0.35),  # people shape narratives
    ("firm",     "narratives", 0.30),  # firms are narrated about
    ("events",   "person",    0.35),   # events affect people
    ("events",   "firm",      0.40),   # events affect firms
    ("events",   "financial", 0.50),   # events move markets
    ("financial","forecasts", 0.45),   # markets inform forecasts
    ("country",  "forecasts", 0.35),   # countries are forecast subjects
]


@_register
class SocialGraphRenderer(Renderer):
    """Render the world model as an entity relationship graph.

    The central node is the focal entity (first person in the projection,
    or the first firm). Surrounding nodes are entities connected by
    topology edges and structural inference.
    """

    @property
    def name(self) -> str:
        return "social_graph"

    def render(self, ctx: RenderContext):
        import matplotlib.pyplot as plt

        entities = self._extract_entities(ctx)
        edges = self._extract_edges(ctx, entities)

        if not entities:
            fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0A0A0A")
            ax.text(0.5, 0.5, "No entities", ha="center", va="center", color="white")
            return fig

        fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor="#060612")
        ax.set_facecolor("#060612")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.axis("off")

        focal = self._find_focal(entities)
        positions = self._layout(focal, entities, edges)

        # ── draw edges ───────────────────────────────────────────────
        for (src, dst), weight in edges.items():
            if src not in positions or dst not in positions:
                continue
            x1, y1 = positions[src]
            x2, y2 = positions[dst]

            alpha = 0.08 + 0.50 * weight
            lw = 0.4 + 2.5 * weight

            # Color by weight: cold (weak) → warm (strong)
            if weight > 0.6:
                edge_color = "#DD8844"
            elif weight > 0.3:
                edge_color = "#5599AA"
            else:
                edge_color = "#334466"

            ax.plot(
                [x1, x2], [y1, y2],
                color=edge_color, alpha=alpha, linewidth=lw,
                solid_capstyle="round", zorder=2,
            )

            # Edge weight label for strong connections
            if weight > 0.55:
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                ax.text(
                    mx, my, f"{weight:.1f}",
                    ha="center", va="center",
                    fontsize=5.5, color="#667788", alpha=0.7,
                    fontweight="bold", zorder=3,
                )

        # ── draw nodes ───────────────────────────────────────────────
        for name, n_fields in entities.items():
            x, y = positions.get(name, (0, 0))
            etype = _entity_type(name)
            style = NODE_STYLES.get(etype, NODE_STYLES["financial"])
            color = DOMAIN_COLORS.get(etype, "#666")

            size = style["base_size"] * (1 + n_fields / 50)
            border_width = 1.0
            if name == focal:
                size *= 2.0
                border_width = 2.5

            # Glow
            ax.scatter(x, y, s=size * 2.5, c=color, alpha=0.06, zorder=style["zorder"] - 1)
            ax.scatter(x, y, s=size * 1.5, c=color, alpha=0.12, zorder=style["zorder"] - 1)

            # Node
            ax.scatter(
                x, y, s=size, c=color, marker=style["shape"],
                edgecolors="white", linewidths=border_width,
                alpha=0.9, zorder=style["zorder"],
            )

            # Label
            label = name.replace("_", " ").title()
            if len(label) > 18:
                label = label[:17] + "\u2026"
            fontsize = 9 if name == focal else 7
            y_offset = -0.08 - (size / 8000)

            ax.text(
                x, y + y_offset, label, ha="center", va="top",
                fontsize=fontsize, color="white",
                fontweight="bold" if name == focal else "normal",
                zorder=style["zorder"] + 1,
            )

            # Field count badge
            ax.text(
                x, y, str(n_fields), ha="center", va="center",
                fontsize=6, color="white", alpha=0.7,
                fontweight="bold", zorder=style["zorder"] + 1,
            )

        # ── legend ────────────────────────────────────────────────────
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        legend_handles = []
        # Node type markers
        for etype, style in NODE_STYLES.items():
            color = DOMAIN_COLORS.get(etype, "#666")
            legend_handles.append(
                mlines.Line2D(
                    [], [], marker=style["shape"], color="none",
                    markerfacecolor=color, markeredgecolor="white",
                    markersize=8, label=etype,
                )
            )
        # Edge weight indicators
        legend_handles.append(
            mlines.Line2D([], [], color="#DD8844", linewidth=2.5, alpha=0.7, label="strong link")
        )
        legend_handles.append(
            mlines.Line2D([], [], color="#5599AA", linewidth=1.5, alpha=0.5, label="moderate link")
        )
        legend_handles.append(
            mlines.Line2D([], [], color="#334466", linewidth=0.8, alpha=0.3, label="weak link")
        )

        legend = ax.legend(
            handles=legend_handles, loc="upper right",
            fontsize=7, facecolor="#101020", edgecolor="#333",
            labelcolor="white", framealpha=0.85,
            handletextpad=0.5, borderpad=0.6,
        )
        legend.set_zorder(20)

        title = ctx.title or f"Entity Network (focal: {focal.replace('_', ' ')})"
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=15)

        fig.tight_layout()
        return fig

    def _find_focal(self, entities: dict[str, int]) -> str:
        """Find the focal entity: first person, then firm, then largest."""
        for name in entities:
            if name.startswith("person"):
                return name
        for name in entities:
            if name.startswith("firm"):
                return name
        return max(entities, key=entities.get)

    def _extract_entities(self, ctx: RenderContext) -> dict[str, int]:
        """Extract top-level entities and their field counts."""
        entities = defaultdict(int)
        for name in ctx.bound_schema.field_names:
            top = name.split(".")[0]
            entities[top] += 1
        return dict(entities)

    def _extract_edges(
        self, ctx: RenderContext, entities: dict,
    ) -> dict[tuple[str, str], float]:
        """Extract edges from topology + structural inference."""
        edge_weights: dict[tuple[str, str], float] = {}

        # 1. Topology-derived edges (actual attention connections)
        if ctx.bound_schema.topology is not None:
            topo_counts: dict[tuple[str, str], int] = defaultdict(int)
            for conn in ctx.bound_schema.topology.connections:
                src_top = conn.src.split(".")[0]
                dst_top = conn.dst.split(".")[0]
                if src_top != dst_top and src_top in entities and dst_top in entities:
                    key = tuple(sorted((src_top, dst_top)))
                    topo_counts[key] += 1

            if topo_counts:
                max_count = max(topo_counts.values())
                for key, count in topo_counts.items():
                    edge_weights[key] = count / max_count

        # 2. Structural inference edges
        entity_types = {name: _entity_type(name) for name in entities}

        for type_a, type_b, base_weight in _STRUCTURAL_EDGES:
            for name_a, t_a in entity_types.items():
                if t_a != type_a:
                    continue
                for name_b, t_b in entity_types.items():
                    if t_b != type_b or name_a == name_b:
                        continue
                    key = tuple(sorted((name_a, name_b)))
                    # Structural edges don't override stronger topology edges
                    current = edge_weights.get(key, 0.0)
                    edge_weights[key] = max(current, base_weight)

        return edge_weights

    def _layout(
        self, focal: str, entities: dict, edges: dict,
    ) -> dict[str, tuple[float, float]]:
        """Layout with focal entity at center, others by connection strength."""
        positions = {focal: (0.0, 0.0)}
        others = [e for e in entities if e != focal]

        if not others:
            return positions

        # Score each entity by max edge weight to focal
        scores = {}
        for name in others:
            key = tuple(sorted((focal, name)))
            scores[name] = edges.get(key, 0.0)

        # Sort by score (closest connections first)
        others_sorted = sorted(others, key=lambda n: -scores.get(n, 0))

        # Adaptive ring sizing
        ring1_count = min(6, len(others_sorted))
        ring2_count = len(others_sorted) - ring1_count

        # Inner ring: closest entities
        for i, name in enumerate(others_sorted[:ring1_count]):
            angle = 2 * math.pi * i / ring1_count - math.pi / 2
            r = 0.65 + 0.1 * (1 - scores.get(name, 0))
            positions[name] = (r * math.cos(angle), r * math.sin(angle))

        # Outer ring: weaker connections
        if ring2_count > 0:
            for i, name in enumerate(others_sorted[ring1_count:]):
                angle = 2 * math.pi * i / ring2_count - math.pi / 4
                r = 1.15 + 0.1 * (1 - scores.get(name, 0))
                positions[name] = (r * math.cos(angle), r * math.sin(angle))

        return positions
