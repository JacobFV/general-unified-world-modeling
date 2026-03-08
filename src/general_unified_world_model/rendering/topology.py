"""Topology graph renderer: visualize the attention connectivity structure.

Renders the causal graph — which domains attend to which — as a directed
graph with edge weights. This is the compute graph of the world model.

Self-loops (intra-domain connections) are shown as a halo around nodes.
Cross-domain connections are shown as directed arrows sized by connection
count, normalized within the cross-domain set so they're always visible.
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
    Edges = cross-domain attention connections (width by edge count).
    Halos = intra-domain connections (shown as glow around nodes).
    """

    @property
    def name(self) -> str:
        return "topology_graph"

    def render(self, ctx: RenderContext):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

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
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")

        # Separate self-loops from cross-domain edges
        self_loops = {}
        cross_edges = {}
        for (src, dst), count in edge_counts.items():
            if src == dst:
                self_loops[src] = count
            else:
                cross_edges[(src, dst)] = count

        # Lay out domains in a circle
        positions = {}
        for i, domain in enumerate(domains):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = math.cos(angle)
            y = math.sin(angle)
            positions[domain] = (x, y)

        # ── Draw cross-domain edges ──────────────────────────────────
        max_cross = max(cross_edges.values()) if cross_edges else 1
        for (src, dst), count in cross_edges.items():
            if src not in positions or dst not in positions:
                continue
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            frac = count / max_cross
            width = 0.8 + 3.5 * frac
            alpha = 0.25 + 0.55 * frac

            # Determine if the reverse edge also exists (bidirectional)
            has_reverse = (dst, src) in cross_edges
            rad = 0.18 if has_reverse else 0.10

            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    connectionstyle=f"arc3,rad={rad}",
                    color=DOMAIN_COLORS.get(src, "#888"),
                    lw=width,
                    alpha=alpha,
                    mutation_scale=14,
                ),
                zorder=3,
            )

        # ── Draw nodes ───────────────────────────────────────────────
        max_fields = max(len(v) for v in groups.values())
        for domain in domains:
            x, y = positions[domain]
            n_fields = len(groups[domain])
            size = 800 + 2500 * (n_fields / max_fields)
            color = DOMAIN_COLORS.get(domain, "#666")

            # Self-loop halo (glow indicating intra-domain density)
            if domain in self_loops:
                halo_size = size * 2.0
                ax.scatter(
                    x, y, s=halo_size, c=color, alpha=0.12,
                    zorder=4, edgecolors="none",
                )

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

        n_cross = len(cross_edges)
        n_self = len(self_loops)
        title = ctx.title or f"Topology ({len(domains)} domains, {n_cross} cross + {n_self} intra)"
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=20)

        fig.tight_layout()
        return fig
