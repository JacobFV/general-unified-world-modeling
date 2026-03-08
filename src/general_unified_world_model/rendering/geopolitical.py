"""Geopolitical map renderer: globe/map with country states projected to color.

Renders an orthographic projection of the globe where each country's
risk score, regime state, or macro health maps to an RGB color gradient.
Red = high risk / distress. Green = stability. Blue = opportunity.
"""

from __future__ import annotations

import math

import numpy as np

from general_unified_world_model.rendering.base import Renderer, RenderContext, _register


# Simplified country centroids (lon, lat) for the major economies
COUNTRY_CENTROIDS = {
    "us": (-98.5, 39.8),
    "cn": (104.2, 35.9),
    "eu": (10.5, 51.2),
    "jp": (138.3, 36.2),
    "uk": (-1.2, 52.5),
    "in": (78.9, 20.6),
    "kr": (127.8, 36.0),
    "br": (-51.9, -14.2),
    "ru": (105.3, 61.5),
    "au": (133.8, -25.3),
    "ca": (-106.3, 56.1),
    "mx": (-102.6, 23.6),
    "de": (10.4, 51.2),
    "fr": (2.2, 46.2),
    "za": (22.9, -30.6),
    "ng": (8.7, 9.1),
    "sa": (45.1, 23.9),
    "tr": (35.2, 39.0),
    "id": (113.9, -0.8),
    "ar": (-63.6, -38.4),
}

# Approximate country boundaries as circles (radius in degrees)
COUNTRY_SIZES = {
    "us": 12, "cn": 10, "eu": 8, "jp": 3, "uk": 3,
    "in": 7, "kr": 2, "br": 10, "ru": 15, "au": 9,
    "ca": 12, "mx": 6, "de": 3, "fr": 3, "za": 5,
    "ng": 4, "sa": 5, "tr": 4, "id": 6, "ar": 7,
}


def _risk_to_rgb(risk: float) -> tuple[float, float, float]:
    """Map a [0, 1] risk score to an RGB color.

    0.0 = deep teal (low risk)
    0.5 = amber (moderate)
    1.0 = crimson (high risk)
    """
    if risk < 0.5:
        t = risk * 2
        r = 0.0 + t * 0.85
        g = 0.8 - t * 0.4
        b = 0.7 - t * 0.5
    else:
        t = (risk - 0.5) * 2
        r = 0.85 + t * 0.15
        g = 0.4 - t * 0.35
        b = 0.2 - t * 0.15
    return (max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b)))


def _orthographic_project(lon: float, lat: float, center_lon: float = 0, center_lat: float = 20):
    """Orthographic projection."""
    lon_r = math.radians(lon)
    lat_r = math.radians(lat)
    cl = math.radians(center_lon)
    cp = math.radians(center_lat)

    cos_c = math.sin(cp) * math.sin(lat_r) + math.cos(cp) * math.cos(lat_r) * math.cos(lon_r - cl)

    if cos_c < 0:
        return None, None  # Behind the globe

    x = math.cos(lat_r) * math.sin(lon_r - cl)
    y = math.cos(cp) * math.sin(lat_r) - math.sin(cp) * math.cos(lat_r) * math.cos(lon_r - cl)
    return x, y


@_register
class GeopoliticalMapRenderer(Renderer):
    """Render a globe with country states projected to color."""

    @property
    def name(self) -> str:
        return "geopolitical_map"

    def render(self, ctx: RenderContext):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="#050510")

        # Extract country risk scores from predictions or use defaults
        country_risks = self._extract_risks(ctx)

        for ax_idx, center_lon in enumerate([0, 180]):
            ax = axes[ax_idx]
            ax.set_facecolor("#050510")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect("equal")
            ax.axis("off")

            # Draw globe outline
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), color="#1A1A3A", linewidth=1.5)

            # Fill globe with dark blue
            ax.fill(np.cos(theta), np.sin(theta), color="#080820", alpha=0.8)

            # Draw latitude/longitude grid
            for lat in range(-60, 90, 30):
                pts = []
                for lon in range(-180, 181, 2):
                    x, y = _orthographic_project(lon, lat, center_lon, 20)
                    if x is not None:
                        pts.append((x, y))
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, color="#1A1A3A", linewidth=0.3, alpha=0.4)

            for lon in range(-180, 180, 30):
                pts = []
                for lat in range(-90, 91, 2):
                    x, y = _orthographic_project(lon, lat, center_lon, 20)
                    if x is not None:
                        pts.append((x, y))
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, color="#1A1A3A", linewidth=0.3, alpha=0.4)

            # Draw countries
            for code, (lon, lat) in COUNTRY_CENTROIDS.items():
                x, y = _orthographic_project(lon, lat, center_lon, 20)
                if x is None:
                    continue

                risk = country_risks.get(code, 0.5)
                color = _risk_to_rgb(risk)
                size = COUNTRY_SIZES.get(code, 4)
                scatter_size = size * 80

                # Glow effect
                ax.scatter(x, y, s=scatter_size * 3, c=[color], alpha=0.15, zorder=3)
                ax.scatter(x, y, s=scatter_size * 1.5, c=[color], alpha=0.3, zorder=4)
                ax.scatter(x, y, s=scatter_size, c=[color], alpha=0.8, zorder=5,
                          edgecolors="white", linewidths=0.3)

                ax.text(x, y - 0.06, code.upper(), ha="center", va="top",
                       fontsize=6, color="white", alpha=0.7, fontweight="bold", zorder=6)

        title = ctx.title or "Geopolitical Risk Map"
        fig.suptitle(title, color="white", fontsize=16, fontweight="bold", y=0.95)

        # Color bar legend
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        colors_list = [_risk_to_rgb(v) for v in np.linspace(0, 1, 256)]
        cmap = LinearSegmentedColormap.from_list("risk", colors_list)
        sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.08)
        cbar.set_label("Risk Score", color="white", fontsize=10)
        cbar.ax.tick_params(colors="white", labelsize=8)

        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        return fig

    def _extract_risks(self, ctx: RenderContext) -> dict[str, float]:
        """Extract country risk scores from predictions or generate defaults."""
        risks = {}
        preds = ctx.predictions or {}

        for code in COUNTRY_CENTROIDS:
            field = f"country_{code}.national_risk_score"
            if field in preds:
                val = preds[field]
                if hasattr(val, 'item'):
                    risks[code] = max(0, min(1, val.item()))
                else:
                    risks[code] = max(0, min(1, float(val)))
            else:
                # Generate realistic-ish defaults
                defaults = {
                    "us": 0.35, "cn": 0.55, "eu": 0.40, "jp": 0.30,
                    "uk": 0.38, "in": 0.50, "kr": 0.32, "br": 0.58,
                    "ru": 0.82, "au": 0.25, "ca": 0.22, "mx": 0.55,
                    "de": 0.35, "fr": 0.42, "za": 0.60, "ng": 0.70,
                    "sa": 0.48, "tr": 0.65, "id": 0.45, "ar": 0.72,
                }
                risks[code] = defaults.get(code, 0.5)

        return risks
