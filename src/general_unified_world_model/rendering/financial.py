"""Financial chart renderer: time series line graphs for market data.

Renders financial fields as professional-grade line charts with
multi-panel layout. Groups by domain (yields, equities, credit, FX).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from general_unified_world_model.rendering.base import Renderer, RenderContext, _register


# Which field paths are "financial chart" material
FINANCIAL_GROUPS = {
    "Yield Curve": [
        "financial.yield_curves.short_rate",
        "financial.yield_curves.two_year",
        "financial.yield_curves.five_year",
        "financial.yield_curves.ten_year",
        "financial.yield_curves.thirty_year",
    ],
    "Credit Spreads": [
        "financial.credit.ig_spread",
        "financial.credit.hy_spread",
    ],
    "Equities": [
        "financial.equities.broad_indices",
        "financial.equities.vix",
    ],
    "FX": [
        "financial.fx.dxy",
        "financial.fx.eurusd",
        "financial.fx.usdjpy",
    ],
    "Energy": [
        "resources.energy.crude_price",
        "resources.energy.natgas_price",
    ],
    "Metals": [
        "resources.metals.gold",
        "resources.metals.copper",
    ],
    "Crypto": [
        "financial.crypto.btc",
        "financial.crypto.eth",
    ],
}

CHART_COLORS = [
    "#FFD700", "#00CED1", "#FF4500", "#00FF7F",
    "#FF1493", "#9370DB", "#00BFFF", "#DC143C",
]


@_register
class FinancialChartRenderer(Renderer):
    """Render financial time series as multi-panel line charts."""

    @property
    def name(self) -> str:
        return "financial_charts"

    def render(self, ctx: RenderContext):
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        ts = ctx.time_series or {}
        preds = ctx.predictions or {}
        all_data = {**ts, **preds}

        # Find which groups have data
        active_groups = {}
        for group_name, field_paths in FINANCIAL_GROUPS.items():
            available = [p for p in field_paths if p in all_data]
            if available:
                active_groups[group_name] = available

        if not active_groups:
            # Generate synthetic demo data if no real data
            active_groups = self._generate_demo_groups()
            all_data = self._generate_demo_data(active_groups)

        n_panels = len(active_groups)
        if n_panels == 0:
            fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0A0A0A")
            ax.text(0.5, 0.5, "No financial data available", ha="center", va="center", color="white")
            return fig

        fig, axes = plt.subplots(
            n_panels, 1, figsize=(14, 3.5 * n_panels),
            facecolor="#0A0A0A", sharex=False,
        )
        if n_panels == 1:
            axes = [axes]

        for ax_idx, (group_name, paths) in enumerate(active_groups.items()):
            ax = axes[ax_idx]
            ax.set_facecolor("#0D0D0D")

            for i, path in enumerate(paths):
                data = all_data.get(path)
                if data is None:
                    continue
                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
                data = np.asarray(data).flatten()

                color = CHART_COLORS[i % len(CHART_COLORS)]
                label = path.split(".")[-1].replace("_", " ")
                ax.plot(data, color=color, linewidth=1.2, alpha=0.9, label=label)

            ax.set_title(group_name, color="white", fontsize=11, fontweight="bold", loc="left")
            ax.tick_params(colors="#666", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#333")
            ax.spines["left"].set_color("#333")
            ax.grid(True, alpha=0.1, color="#444")
            ax.legend(fontsize=7, facecolor="#1A1A1A", edgecolor="#333", labelcolor="white")

        title = ctx.title or "Financial Markets"
        fig.suptitle(title, color="white", fontsize=16, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig

    def _generate_demo_groups(self):
        return {
            "Yield Curve": ["short_rate", "two_year", "ten_year", "thirty_year"],
            "Credit Spreads": ["ig_spread", "hy_spread"],
            "Equities & Vol": ["spx", "vix"],
        }

    def _generate_demo_data(self, groups):
        np.random.seed(42)
        data = {}
        t = np.linspace(0, 4 * np.pi, 500)
        templates = {
            "short_rate": 4.5 + 0.8 * np.sin(t * 0.3) + np.cumsum(np.random.randn(500) * 0.02),
            "two_year": 4.2 + 0.6 * np.sin(t * 0.3 + 0.2) + np.cumsum(np.random.randn(500) * 0.02),
            "ten_year": 3.8 + 0.5 * np.sin(t * 0.25 + 0.5) + np.cumsum(np.random.randn(500) * 0.015),
            "thirty_year": 4.0 + 0.4 * np.sin(t * 0.2 + 0.8) + np.cumsum(np.random.randn(500) * 0.012),
            "ig_spread": 1.2 + 0.3 * np.abs(np.sin(t * 0.4)) + np.random.randn(500) * 0.05,
            "hy_spread": 4.5 + 1.0 * np.abs(np.sin(t * 0.35)) + np.random.randn(500) * 0.15,
            "spx": 4500 * np.exp(np.cumsum(np.random.randn(500) * 0.008)),
            "vix": 18 + 8 * np.abs(np.sin(t * 0.5)) + np.random.randn(500) * 2,
        }
        for group_name, paths in groups.items():
            for path in paths:
                if path in templates:
                    data[path] = templates[path]
        return data
