"""Geopolitical map renderer: globe with country territories filled by state color.

Renders an orthographic globe projection where each country's state vector
is projected to RGB via PCA (3D projection of the latent representation).
Country territories are drawn with real boundaries and filled with the
resulting color.

Requires: cartopy (pip install general-unified-world-model[diagrams])
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from general_unified_world_model.rendering.base import Renderer, RenderContext, _register


# 2-letter code → ISO 3166-1 alpha-3
ISO_MAP = {
    "us": "USA", "cn": "CHN", "jp": "JPN", "uk": "GBR",
    "in": "IND", "kr": "KOR", "br": "BRA", "ru": "RUS",
    "au": "AUS", "ca": "CAN", "mx": "MEX", "de": "DEU",
    "fr": "FRA", "za": "ZAF", "ng": "NGA", "sa": "SAU",
    "tr": "TUR", "id": "IDN", "ar": "ARG",
}

# EU member ISO-3 codes (colored with "eu" state vector when present)
EU_MEMBERS = {
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST",
    "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA",
    "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK",
    "SVN", "ESP", "SWE",
}

# Country centroids for labels
COUNTRY_CENTROIDS = {
    "us": (-98.5, 39.8), "cn": (104.2, 35.9), "eu": (10.5, 51.2),
    "jp": (138.3, 36.2), "uk": (-1.2, 52.5), "in": (78.9, 20.6),
    "kr": (127.8, 36.0), "br": (-51.9, -14.2), "ru": (105.3, 61.5),
    "au": (133.8, -25.3), "ca": (-106.3, 56.1), "mx": (-102.6, 23.6),
    "de": (10.4, 51.2), "fr": (2.2, 46.2), "za": (22.9, -30.6),
    "ng": (8.7, 9.1), "sa": (45.1, 23.9), "tr": (35.2, 39.0),
    "id": (113.9, -0.8), "ar": (-63.6, -38.4),
}


# ── State vector → RGB projection ───────────────────────────────────


def _state_vectors_to_rgb(
    vectors: dict[str, np.ndarray],
) -> dict[str, tuple[float, float, float]]:
    """Project country state vectors to RGB using PCA."""
    if not vectors:
        return {}

    codes = list(vectors.keys())
    matrix = np.stack([np.asarray(vectors[c], dtype=np.float64).flatten() for c in codes])

    if matrix.shape[1] < 3:
        pad = np.zeros((matrix.shape[0], 3 - matrix.shape[1]))
        matrix = np.hstack([matrix, pad])

    centered = matrix - matrix.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ Vt[:3].T
    except np.linalg.LinAlgError:
        projected = centered[:, :3]

    for i in range(3):
        col = projected[:, i]
        mn, mx = col.min(), col.max()
        if mx - mn > 1e-10:
            projected[:, i] = 0.15 + 0.80 * (col - mn) / (mx - mn)
        else:
            projected[:, i] = 0.55

    return {code: tuple(projected[i]) for i, code in enumerate(codes)}


def _generate_default_vectors(codes: list[str], dim: int = 16) -> dict[str, np.ndarray]:
    """Generate reproducible synthetic state vectors for demo rendering."""
    vectors = {}
    for code in codes:
        seed = sum(ord(c) for c in code) * 137
        rng = np.random.RandomState(seed)
        vectors[code] = rng.randn(dim)
    return vectors


# ── Shared globe-frame rendering ─────────────────────────────────────


def _render_globe_frame(
    center_lon: float,
    center_lat: float,
    iso_colors: dict[str, tuple],
    country_labels: dict[str, tuple],
    figsize: tuple[float, float] = (8, 8),
    bg: str = "#050510",
):
    """Render a single globe frame at a given center longitude.

    Returns a matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, facecolor=bg)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(center_lon, center_lat))
    ax.set_global()
    ax.patch.set_facecolor("#080820")
    ax.spines['geo'].set_edgecolor("#1A1A3A")
    ax.spines['geo'].set_linewidth(1.0)

    ax.add_feature(cfeature.OCEAN, facecolor="#080820", edgecolor="none")
    ax.add_feature(cfeature.LAND, facecolor="#141428", edgecolor="#222244", linewidth=0.3)
    ax.gridlines(linewidth=0.3, color="#1A1A3A", alpha=0.4, draw_labels=False)

    reader = shpreader.Reader(
        shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
    )
    for record in reader.records():
        iso3 = record.attributes.get("ADM0_A3", "") or record.attributes.get("ISO_A3", "")
        if iso3 in iso_colors:
            ax.add_geometries(
                [record.geometry], ccrs.PlateCarree(),
                facecolor=iso_colors[iso3], edgecolor="white",
                linewidth=0.4, alpha=0.92, zorder=3,
            )

    for code, (lon, lat) in country_labels.items():
        try:
            x_proj, y_proj = ax.projection.transform_point(lon, lat, ccrs.PlateCarree())
            if not math.isfinite(x_proj) or not math.isfinite(y_proj):
                continue
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            if not (xlims[0] <= x_proj <= xlims[1] and ylims[0] <= y_proj <= ylims[1]):
                continue
            ax.text(
                lon, lat, code.upper(), transform=ccrs.PlateCarree(),
                ha="center", va="center", fontsize=5.5, color="white",
                alpha=0.8, fontweight="bold", zorder=5,
            )
        except Exception:
            pass

    fig.tight_layout(pad=0.3)
    return fig


# ── Renderer ─────────────────────────────────────────────────────────


@_register
class GeopoliticalMapRenderer(Renderer):
    """Render a globe with country territories filled by state-vector color."""

    @property
    def name(self) -> str:
        return "geopolitical_map"

    def render(self, ctx: RenderContext):
        """Render static dual-hemisphere view."""
        import matplotlib.pyplot as plt

        colors = self._extract_colors(ctx)
        iso_colors = self._build_iso_colors(colors)
        labels = {c: COUNTRY_CENTROIDS[c] for c in colors if c in COUNTRY_CENTROIDS}

        fig = plt.figure(figsize=(16, 8), facecolor="#050510")

        for ax_idx, center_lon in enumerate([0, 180]):
            ax = fig.add_subplot(
                1, 2, ax_idx + 1, projection=ccrs.Orthographic(center_lon, 20),
            )
            ax.set_global()
            ax.patch.set_facecolor("#080820")
            ax.spines['geo'].set_edgecolor("#1A1A3A")
            ax.spines['geo'].set_linewidth(1.0)

            ax.add_feature(cfeature.OCEAN, facecolor="#080820", edgecolor="none")
            ax.add_feature(cfeature.LAND, facecolor="#141428", edgecolor="#222244", linewidth=0.3)
            ax.gridlines(linewidth=0.3, color="#1A1A3A", alpha=0.4, draw_labels=False)

            reader = shpreader.Reader(
                shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
            )
            for record in reader.records():
                iso3 = record.attributes.get("ADM0_A3", "") or record.attributes.get("ISO_A3", "")
                if iso3 in iso_colors:
                    ax.add_geometries(
                        [record.geometry], ccrs.PlateCarree(),
                        facecolor=iso_colors[iso3], edgecolor="white",
                        linewidth=0.4, alpha=0.92, zorder=3,
                    )

            for code, (lon, lat) in labels.items():
                try:
                    x_proj, y_proj = ax.projection.transform_point(lon, lat, ccrs.PlateCarree())
                    if not math.isfinite(x_proj) or not math.isfinite(y_proj):
                        continue
                    xlims = ax.get_xlim()
                    ylims = ax.get_ylim()
                    if not (xlims[0] <= x_proj <= xlims[1] and ylims[0] <= y_proj <= ylims[1]):
                        continue
                    ax.text(
                        lon, lat, code.upper(), transform=ccrs.PlateCarree(),
                        ha="center", va="center", fontsize=5.5, color="white",
                        alpha=0.8, fontweight="bold", zorder=5,
                    )
                except Exception:
                    pass

        title = ctx.title or "Geopolitical State — Vector Projection to RGB"
        fig.suptitle(title, color="white", fontsize=15, fontweight="bold", y=0.95)
        fig.tight_layout(rect=[0, 0.02, 1, 0.93])
        return fig

    def render_rotating_gif(
        self,
        ctx: RenderContext,
        path: str | Path,
        n_frames: int = 72,
        center_lat: float = 20,
        duration_ms: int = 80,
        dpi: int = 100,
        figsize: tuple[float, float] = (6, 6),
    ) -> Path:
        """Render a rotating globe GIF.

        Args:
            ctx: RenderContext with schema and optional predictions.
            path: Output file path (.gif).
            n_frames: Number of frames in the rotation (default 72 = 5° per frame).
            center_lat: Latitude of the view center.
            duration_ms: Milliseconds per frame.
            dpi: Resolution per frame.
            figsize: Figure size per frame.

        Returns:
            Path to the saved GIF.
        """
        import matplotlib.pyplot as plt
        from PIL import Image

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        colors = self._extract_colors(ctx)
        iso_colors = self._build_iso_colors(colors)
        labels = {c: COUNTRY_CENTROIDS[c] for c in colors if c in COUNTRY_CENTROIDS}

        frames: list[Image.Image] = []
        lon_step = 360.0 / n_frames

        for i in range(n_frames):
            center_lon = i * lon_step
            fig = _render_globe_frame(
                center_lon, center_lat, iso_colors, labels,
                figsize=figsize, bg="#050510",
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                        facecolor="#050510", pad_inches=0.1)
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).convert("RGB"))

        frames[0].save(
            str(path),
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        return path

    # ── helpers ───────────────────────────────────────────────────────

    def _build_iso_colors(self, colors: dict[str, tuple]) -> dict[str, tuple]:
        """Convert 2-letter codes to ISO-3 lookup."""
        iso_colors = {}
        for code, rgb in colors.items():
            iso3 = ISO_MAP.get(code)
            if iso3:
                iso_colors[iso3] = rgb
            if code == "eu":
                for member in EU_MEMBERS:
                    if member not in iso_colors:
                        iso_colors[member] = rgb
        return iso_colors

    def _extract_colors(self, ctx: RenderContext) -> dict[str, tuple]:
        """Extract country state vectors and project to RGB."""
        preds = ctx.predictions or {}

        country_vectors: dict[str, list[float]] = {}
        for field_name, val in preds.items():
            for code in list(ISO_MAP.keys()) + ["eu"]:
                prefix = f"country_{code}."
                if field_name.startswith(prefix):
                    if code not in country_vectors:
                        country_vectors[code] = []
                    if hasattr(val, "item"):
                        country_vectors[code].append(float(val.item()))
                    elif hasattr(val, "tolist"):
                        flat = np.asarray(val).flatten().tolist()
                        country_vectors[code].extend(flat)
                    else:
                        country_vectors[code].append(float(val))

        if country_vectors:
            max_len = max(len(v) for v in country_vectors.values())
            padded = {}
            for code, vals in country_vectors.items():
                arr = np.array(vals[:max_len], dtype=np.float64)
                if len(arr) < max_len:
                    arr = np.pad(arr, (0, max_len - len(arr)))
                padded[code] = arr
            return _state_vectors_to_rgb(padded)

        active_codes = set()
        for name in ctx.bound_schema.field_names:
            for code in list(ISO_MAP.keys()) + ["eu"]:
                if name.startswith(f"country_{code}"):
                    active_codes.add(code)

        if not active_codes:
            active_codes = set(COUNTRY_CENTROIDS.keys())

        vectors = _generate_default_vectors(sorted(active_codes))
        return _state_vectors_to_rgb(vectors)
