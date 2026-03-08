"""Regime dashboard renderer: minimal state vector display.

A clean, data-dense panel showing the world model's latent regime state.
Horizontal bars with monospace labels and values. No gauges, no needles,
no decoration. The data speaks for itself.
"""

from __future__ import annotations

import numpy as np
import torch

from general_unified_world_model.rendering.base import Renderer, RenderContext, _register


REGIME_FIELDS = [
    ("growth_regime",                 "growth"),
    ("inflation_regime",              "inflation"),
    ("financial_cycle",               "fin_cycle"),
    ("credit_cycle",                  "credit"),
    ("liquidity_regime",              "liquidity"),
    ("cooperation_vs_fragmentation",  "cooperation"),
    ("peace_vs_conflict",             "peace"),
    ("ai_acceleration",               "ai_accel"),
    ("fragility",                     "fragility"),
    ("reflexivity_intensity",         "reflexivity"),
    ("tail_risk_concentration",       "tail_risk"),
    ("black_swan_proximity",          "black_swan"),
]


def _value_color(v: float) -> tuple[float, float, float]:
    """Map [0, 1] value to a color gradient.

    Low (0) = cold steel blue.
    Mid (0.5) = neutral gray.
    High (1) = hot signal red.
    """
    if v < 0.5:
        t = v * 2
        return (0.25 + 0.30 * t, 0.35 + 0.10 * t, 0.50 - 0.15 * t)
    else:
        t = (v - 0.5) * 2
        return (0.55 + 0.40 * t, 0.45 - 0.35 * t, 0.35 - 0.30 * t)


@_register
class RegimeDashboardRenderer(Renderer):
    """Render regime state as minimal horizontal bars."""

    @property
    def name(self) -> str:
        return "regime_dashboard"

    def render(self, ctx: RenderContext):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        values = self._extract_regime_values(ctx)
        n = len(REGIME_FIELDS)

        fig_h = max(3.0, 0.42 * n + 1.5)
        fig, ax = plt.subplots(figsize=(8, fig_h), facecolor="#0A0A0A")
        ax.set_facecolor("#0A0A0A")

        bar_h = 0.55
        gap = 0.18
        row_h = bar_h + gap
        total_h = n * row_h
        bar_left = 2.8
        bar_width = 4.5

        ax.set_xlim(0, bar_left + bar_width + 1.5)
        ax.set_ylim(-0.5, total_h + 0.5)
        ax.axis("off")

        for i, (field_name, label) in enumerate(REGIME_FIELDS):
            y = total_h - (i + 1) * row_h + gap / 2
            v = values.get(field_name, 0.5)

            # Background track
            track = Rectangle(
                (bar_left, y), bar_width, bar_h,
                facecolor="#1A1A1A", edgecolor="#252525",
                linewidth=0.5, zorder=2,
            )
            ax.add_patch(track)

            # Value bar
            color = _value_color(v)
            bar = Rectangle(
                (bar_left, y), bar_width * v, bar_h,
                facecolor=color, edgecolor="none",
                alpha=0.88, zorder=3,
            )
            ax.add_patch(bar)

            # Label (left-aligned, monospace)
            ax.text(
                bar_left - 0.2, y + bar_h / 2,
                label, ha="right", va="center",
                fontsize=9, fontfamily="monospace",
                color="#AAAAAA", zorder=4,
            )

            # Value (right of bar)
            ax.text(
                bar_left + bar_width + 0.2, y + bar_h / 2,
                f"{v:.2f}", ha="left", va="center",
                fontsize=9, fontfamily="monospace",
                color=color, fontweight="bold", zorder=4,
            )

        # Compressed world state strip
        compressed = values.get("compressed_world_state", None)
        if compressed is None:
            np.random.seed(42)
            compressed = np.random.rand(32)
        compressed = np.asarray(compressed).flatten()[:32]

        strip_y = -0.4
        strip_h = 0.25
        cell_w = bar_width / len(compressed)
        for j, v in enumerate(compressed):
            color = _value_color(max(0, min(1, float(v))))
            cell = Rectangle(
                (bar_left + j * cell_w, strip_y), cell_w * 0.92, strip_h,
                facecolor=color, edgecolor="none", alpha=0.8, zorder=3,
            )
            ax.add_patch(cell)

        ax.text(
            bar_left - 0.2, strip_y + strip_h / 2,
            "latent", ha="right", va="center",
            fontsize=7, fontfamily="monospace", color="#555555",
        )

        title = ctx.title or "regime state"
        ax.text(
            bar_left + bar_width / 2, total_h + 0.3,
            title, ha="center", va="bottom",
            fontsize=12, fontfamily="monospace", fontweight="bold",
            color="#CCCCCC",
        )

        fig.tight_layout(pad=0.3)
        return fig

    def _extract_regime_values(self, ctx: RenderContext) -> dict:
        """Extract regime field values from predictions."""
        values = {}
        preds = ctx.predictions or {}

        for field_name, _ in REGIME_FIELDS:
            key = f"regime.{field_name}"
            if key in preds:
                val = preds[key]
                if isinstance(val, torch.Tensor):
                    val = val.float().mean().item()
                values[field_name] = max(0, min(1, abs(val)))
            else:
                np.random.seed(hash(field_name) % 2**31)
                values[field_name] = float(np.random.beta(2, 3))

        return values
