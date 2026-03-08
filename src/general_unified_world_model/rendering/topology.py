"""Topology graph renderer: visualize the attention connectivity structure.

Renders the causal graph — which domains attend to which — as a directed
graph with edge weights. This is the compute graph of the world model.
Two views: domain-level (coarse) and field-level (fine).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from general_unified_world_model.rendering.base import Renderer, RenderContext, _register
from general_unified_world_model.rendering.canvas import DOMAIN_COLORS, _domain_of, _hex_to_rgb


def _group_connections_by_domain(bound_schema) -> dict[tuple[str, str], int]:
    """Group topology connections by domain pair, counting edges."""
    edge_counts = defaultdict(int)

    if bound_schema.topology is None:
        return edge_counts

    for conn in bound_schema.topology.connections:
        src_domain = _domain_of(conn.src)
        dst_domain = _domain_of(conn.dst)
        edge_counts[(src_domain, dst_domain)] += 1

    return dict(edge_counts)


def _get_field_groups(bound_schema) -> dict[str, list[str]]:
    """Group fields by top-level domain."""
    groups = defaultdict(list)
    for name in bound_schema.field_names:
        domain = _domain_of(name)
        groups[domain].append(name)
    return dict(groups)


@_register
class TopologyGraphRenderer(Renderer):
    """Render the topology as a domain-level directed graph.

    Nodes = domains (sized by field count).
    Edges = attention connections (width by edge count).
    """

    @property
    def name(self) -> str:
        return "topology_graph"

    def render(self, ctx: RenderContext):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyArrowPatch

        groups = _get_field_groups(ctx.bound_schema)
        edge_counts = _group_connections_by_domain(ctx.bound_schema)
        domains = sorted(groups.keys())
        n = len(domains)

        if n == 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, "No domains", ha="center", va="center")
            return fig

        fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor="#0A0A0A")
        ax.set_facecolor("#0A0A0A")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Lay out domains in a circle
        positions = {}
        for i, domain in enumerate(domains):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = math.cos(angle)
            y = math.sin(angle)
            positions[domain] = (x, y)

        # Draw edges
        max_count = max(edge_counts.values()) if edge_counts else 1
        for (src, dst), count in edge_counts.items():
            if src not in positions or dst not in positions:
                continue
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            width = 0.5 + 3.0 * (count / max_count)
            alpha = 0.15 + 0.5 * (count / max_count)

            if src == dst:
                # Self-loop: draw arc
                circle = plt.Circle(
                    (x1, y1 + 0.12), 0.1, fill=False,
                    edgecolor=DOMAIN_COLORS.get(src, "#666"), linewidth=width,
                    alpha=alpha,
                )
                ax.add_patch(circle)
            else:
                # Curve the arrow slightly for visual clarity
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                # Offset perpendicular to the line
                dx, dy = x2 - x1, y2 - y1
                length = math.sqrt(dx**2 + dy**2)
                if length > 0:
                    perp_x = -dy / length * 0.08
                    perp_y = dx / length * 0.08
                    mid_x += perp_x
                    mid_y += perp_y

                ax.annotate(
                    "", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        connectionstyle="arc3,rad=0.15",
                        color=DOMAIN_COLORS.get(src, "#666"),
                        lw=width,
                        alpha=alpha,
                    ),
                )

        # Draw nodes
        max_fields = max(len(v) for v in groups.values())
        for domain in domains:
            x, y = positions[domain]
            n_fields = len(groups[domain])
            size = 800 + 2500 * (n_fields / max_fields)
            color = DOMAIN_COLORS.get(domain, "#666")

            ax.scatter(
                x, y, s=size, c=color, zorder=5,
                edgecolors="white", linewidths=1.5, alpha=0.9,
            )
            ax.text(
                x, y - 0.02, domain, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6,
            )
            ax.text(
                x, y - 0.11, f"{n_fields}", ha="center", va="center",
                fontsize=6, color="#AAA", zorder=6,
            )

        title = ctx.title or f"Topology ({len(domains)} domains, {len(edge_counts)} edges)"
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=20)

        fig.tight_layout()
        return fig
