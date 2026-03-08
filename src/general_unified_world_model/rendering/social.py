"""Social graph renderer: first-person network perspective.

Renders the world model from the perspective of a specific actor —
a CEO, a central banker, a fund manager — showing their network of
connections, influence pathways, and information flows.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from general_unified_world_model.rendering.base import Renderer, RenderContext, _register
from general_unified_world_model.rendering.canvas import DOMAIN_COLORS


# Node types and their visual properties
NODE_STYLES = {
    "person":    {"shape": "o", "base_size": 600, "zorder": 10},
    "firm":      {"shape": "s", "base_size": 500, "zorder": 8},
    "country":   {"shape": "D", "base_size": 450, "zorder": 7},
    "sector":    {"shape": "^", "base_size": 400, "zorder": 6},
    "regime":    {"shape": "h", "base_size": 350, "zorder": 5},
    "financial": {"shape": "p", "base_size": 350, "zorder": 5},
}


def _entity_type(name: str) -> str:
    """Classify a top-level field into entity type."""
    for prefix in NODE_STYLES:
        if name.startswith(prefix):
            return prefix
    return "financial"


@_register
class SocialGraphRenderer(Renderer):
    """Render the world model as a social/entity relationship graph.

    The central node is the focal entity (first person in the projection,
    or the first firm). Surrounding nodes are entities connected by
    topology edges, positioned by relationship strength.
    """

    @property
    def name(self) -> str:
        return "social_graph"

    def render(self, ctx: RenderContext):
        import matplotlib.pyplot as plt

        # Extract entities from the schema
        entities = self._extract_entities(ctx)
        edges = self._extract_edges(ctx, entities)

        if not entities:
            fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0A0A0A")
            ax.text(0.5, 0.5, "No entities", ha="center", va="center", color="white")
            return fig

        fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor="#060612")
        ax.set_facecolor("#060612")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")

        # Identify focal entity (first person or firm)
        focal = None
        for name in entities:
            if name.startswith("person"):
                focal = name
                break
        if focal is None:
            for name in entities:
                if name.startswith("firm"):
                    focal = name
                    break
        if focal is None:
            focal = list(entities.keys())[0]

        # Layout: focal in center, others in concentric rings by distance
        positions = self._layout(focal, entities, edges)

        # Draw edges
        for (src, dst), weight in edges.items():
            if src not in positions or dst not in positions:
                continue
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            alpha = 0.1 + 0.4 * weight
            lw = 0.5 + 2.0 * weight

            ax.plot(
                [x1, x2], [y1, y2],
                color="#4488AA", alpha=alpha, linewidth=lw,
                zorder=2,
            )

        # Draw nodes
        for name, n_fields in entities.items():
            x, y = positions.get(name, (0, 0))
            etype = _entity_type(name)
            style = NODE_STYLES.get(etype, NODE_STYLES["financial"])
            color = DOMAIN_COLORS.get(etype, "#666")

            size = style["base_size"] * (1 + n_fields / 50)
            if name == focal:
                size *= 2.0

            # Glow
            ax.scatter(x, y, s=size * 2.5, c=color, alpha=0.08, zorder=style["zorder"] - 1)
            ax.scatter(x, y, s=size * 1.5, c=color, alpha=0.15, zorder=style["zorder"] - 1)
            ax.scatter(
                x, y, s=size, c=color, marker=style["shape"],
                edgecolors="white", linewidths=1.0 if name != focal else 2.5,
                alpha=0.9, zorder=style["zorder"],
            )

            # Label
            label = name.replace("_", " ").title()
            if len(label) > 15:
                label = label[:14] + "..."
            fontsize = 9 if name == focal else 7
            ax.text(
                x, y - 0.10, label, ha="center", va="top",
                fontsize=fontsize, color="white", fontweight="bold" if name == focal else "normal",
                zorder=style["zorder"] + 1,
            )

        title = ctx.title or f"Entity Network (focal: {focal.replace('_', ' ')})"
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=15)

        fig.tight_layout()
        return fig

    def _extract_entities(self, ctx: RenderContext) -> dict[str, int]:
        """Extract top-level entities and their field counts."""
        entities = defaultdict(int)
        for name in ctx.bound_schema.field_names:
            top = name.split(".")[0]
            entities[top] += 1
        return dict(entities)

    def _extract_edges(self, ctx: RenderContext, entities: dict) -> dict[tuple[str, str], float]:
        """Extract inter-entity edges from topology."""
        edges = defaultdict(float)

        if ctx.bound_schema.topology is None:
            return dict(edges)

        total_by_pair = defaultdict(int)
        for conn in ctx.bound_schema.topology.connections:
            src_top = conn.src.split(".")[0]
            dst_top = conn.dst.split(".")[0]
            if src_top != dst_top and src_top in entities and dst_top in entities:
                key = (min(src_top, dst_top), max(src_top, dst_top))
                total_by_pair[key] += 1

        if not total_by_pair:
            return dict(edges)

        max_count = max(total_by_pair.values())
        for key, count in total_by_pair.items():
            edges[key] = count / max_count

        return dict(edges)

    def _layout(self, focal: str, entities: dict, edges: dict) -> dict[str, tuple[float, float]]:
        """Force-directed-ish layout with focal entity at center."""
        positions = {focal: (0.0, 0.0)}
        others = [e for e in entities if e != focal]

        if not others:
            return positions

        # Score each entity by its connection strength to the focal
        scores = {}
        for name in others:
            key = (min(focal, name), max(focal, name))
            scores[name] = edges.get(key, 0.0)

        # Sort by score (closest first)
        others_sorted = sorted(others, key=lambda n: -scores.get(n, 0))

        # Place in concentric rings
        ring1_count = min(6, len(others_sorted))
        ring2_count = len(others_sorted) - ring1_count

        for i, name in enumerate(others_sorted[:ring1_count]):
            angle = 2 * math.pi * i / ring1_count - math.pi / 2
            r = 0.7
            positions[name] = (r * math.cos(angle), r * math.sin(angle))

        for i, name in enumerate(others_sorted[ring1_count:]):
            angle = 2 * math.pi * i / max(ring2_count, 1) - math.pi / 4
            r = 1.2
            positions[name] = (r * math.cos(angle), r * math.sin(angle))

        return positions
