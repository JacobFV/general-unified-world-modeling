"""Regime dashboard renderer: compressed state overview.

A single dense panel showing the world model's latent regime state —
growth, inflation, financial cycle, fragility, geopolitical tension,
technology regime — as a set of gauges and heatmap strips.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from general_unified_world_model.rendering.base import Renderer, RenderContext, _register


REGIME_FIELDS = [
    ("growth_regime", "Growth", "#00FF7F"),
    ("inflation_regime", "Inflation", "#FF4500"),
    ("financial_cycle", "Financial Cycle", "#FFD700"),
    ("credit_cycle", "Credit Cycle", "#FF8C00"),
    ("liquidity_regime", "Liquidity", "#00CED1"),
    ("cooperation_vs_fragmentation", "Cooperation", "#9370DB"),
    ("peace_vs_conflict", "Peace/Conflict", "#DC143C"),
    ("ai_acceleration", "AI Accel.", "#00BFFF"),
    ("fragility", "Fragility", "#FF1493"),
    ("reflexivity_intensity", "Reflexivity", "#FFD700"),
    ("tail_risk_concentration", "Tail Risk", "#FF0000"),
    ("black_swan_proximity", "Black Swan", "#8B0000"),
]


def _draw_gauge(ax, value: float, label: str, color: str, x: float, y: float, radius: float = 0.35):
    """Draw a semi-circular gauge."""
    # Arc background
    theta = np.linspace(math.pi, 0, 100)
    ax.plot(
        x + radius * np.cos(theta),
        y + radius * np.sin(theta),
        color="#333", linewidth=6, solid_capstyle="round",
    )

    # Arc value
    theta_val = np.linspace(math.pi, math.pi - value * math.pi, max(2, int(50 * value)))
    ax.plot(
        x + radius * np.cos(theta_val),
        y + radius * np.sin(theta_val),
        color=color, linewidth=6, solid_capstyle="round", alpha=0.9,
    )

    # Needle
    needle_angle = math.pi - value * math.pi
    nx = x + radius * 0.7 * math.cos(needle_angle)
    ny = y + radius * 0.7 * math.sin(needle_angle)
    ax.plot([x, nx], [y, ny], color="white", linewidth=1.5, alpha=0.8)
    ax.scatter(x, y, s=20, c="white", zorder=5)

    # Label
    ax.text(x, y - radius * 0.35, label, ha="center", va="top",
           fontsize=7, color="white", fontweight="bold")
    ax.text(x, y - radius * 0.55, f"{value:.2f}", ha="center", va="top",
           fontsize=8, color=color, fontweight="bold")


@_register
class RegimeDashboardRenderer(Renderer):
    """Render the regime state as a dashboard of gauges."""

    @property
    def name(self) -> str:
        return "regime_dashboard"

    def render(self, ctx: RenderContext):
        import matplotlib.pyplot as plt

        values = self._extract_regime_values(ctx)

        n = len(REGIME_FIELDS)
        cols = 4
        rows = math.ceil(n / cols)

        fig, ax = plt.subplots(1, 1, figsize=(14, 3.5 * rows), facecolor="#0A0A0A")
        ax.set_facecolor("#0A0A0A")
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-rows + 0.3, 0.8)
        ax.set_aspect("equal")
        ax.axis("off")

        for i, (field_name, label, color) in enumerate(REGIME_FIELDS):
            col = i % cols
            row = i // cols
            x = col
            y = -row
            value = values.get(field_name, 0.5)
            _draw_gauge(ax, value, label, color, x, y, radius=0.35)

        # Bottom bar: compressed world state
        compressed = values.get("compressed_world_state", np.random.rand(32))
        if isinstance(compressed, (float, int)):
            compressed = np.random.rand(32) * compressed
        compressed = np.asarray(compressed).flatten()[:32]

        bar_y = -rows + 0.1
        bar_width = cols / len(compressed)
        for i, v in enumerate(compressed):
            color_val = max(0, min(1, v))
            r = 0.1 + 0.9 * color_val
            g = 0.8 * (1 - color_val)
            b = 0.3 + 0.3 * (1 - color_val)
            ax.add_patch(plt.Rectangle(
                (-0.4 + i * bar_width * 0.95, bar_y - 0.12),
                bar_width * 0.9, 0.08,
                facecolor=(r, g, b), edgecolor="none", alpha=0.85,
            ))
        ax.text(cols / 2 - 0.5, bar_y - 0.28, "Compressed World State",
               ha="center", fontsize=8, color="#666")

        title = ctx.title or "Regime State Dashboard"
        ax.set_title(title, color="white", fontsize=16, fontweight="bold", pad=20)

        fig.tight_layout()
        return fig

    def _extract_regime_values(self, ctx: RenderContext) -> dict[str, float]:
        """Extract regime field values from predictions."""
        values = {}
        preds = ctx.predictions or {}

        for field_name, _, _ in REGIME_FIELDS:
            key = f"regime.{field_name}"
            if key in preds:
                val = preds[key]
                if isinstance(val, torch.Tensor):
                    val = val.float().mean().item()
                values[field_name] = max(0, min(1, abs(val)))
            else:
                # Synthetic defaults
                np.random.seed(hash(field_name) % 2**31)
                values[field_name] = np.random.beta(2, 3)

        return values
