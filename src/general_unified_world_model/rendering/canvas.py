"""Canvas heatmap renderer: visualize the raw spatiotemporal canvas allocation.

Shows which positions on the (T, H, W) grid are assigned to which modalities,
colored by domain layer. This is the most literal view — you're looking at
the latent space layout itself.
"""

from __future__ import annotations

import numpy as np

from general_unified_world_model.rendering.base import Renderer, RenderContext, _register


# Domain → color mapping (high entropy, visually distinct)
DOMAIN_COLORS = {
    "physical":       "#0D1B2A",   # deep navy
    "resources":      "#1B4332",   # forest green
    "financial":      "#D4AF37",   # gold
    "country":        "#8B0000",   # dark red
    "sector":         "#4A0E4E",   # deep purple
    "sc":             "#2F4F4F",   # dark slate
    "firm":           "#FF4500",   # orange red
    "person":         "#DC143C",   # crimson
    "events":         "#00CED1",   # dark turquoise
    "trust":          "#708090",   # slate gray
    "regime":         "#FFD700",   # bright gold
    "interventions":  "#FF1493",   # deep pink
    "forecasts":      "#00FF7F",   # spring green
    "narratives":     "#9370DB",   # medium purple
    "technology":     "#00BFFF",   # deep sky blue
}


def _domain_of(field_name: str) -> str:
    """Extract the top-level domain from a field path."""
    top = field_name.split(".")[0]
    for prefix in DOMAIN_COLORS:
        if top.startswith(prefix):
            return prefix
    return top


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


@_register
class CanvasHeatmapRenderer(Renderer):
    """Render the canvas allocation as a 2D heatmap.

    Each position is colored by its domain. If state data is provided,
    intensity encodes the L2 norm of the latent vector at that position.
    """

    @property
    def name(self) -> str:
        return "canvas_heatmap"

    def render(self, ctx: RenderContext):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap

        layout = ctx.bound_schema.layout
        H, W = layout.H, layout.W
        T = layout.T

        # Build RGB image of canvas allocation
        img = np.ones((H, W, 3)) * 0.08  # near-black background

        # Track which domains appear for legend
        seen_domains = set()

        for field_name in ctx.bound_schema.field_names:
            bf = ctx.bound_schema[field_name]
            domain = _domain_of(field_name)
            color = _hex_to_rgb(DOMAIN_COLORS.get(domain, "#FFFFFF"))
            seen_domains.add(domain)

            indices = bf.indices()
            for idx in indices:
                if idx < H * W:
                    h = idx // W
                    w = idx % W
                    intensity = 0.8
                    if ctx.state is not None and idx < ctx.state.shape[0]:
                        norm = ctx.state[idx].norm().item()
                        intensity = min(1.0, 0.3 + norm * 0.7)
                    img[h, w] = [c * intensity for c in color]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor="#0A0A0A")
        ax.imshow(img, interpolation="nearest", aspect="equal")
        ax.set_facecolor("#0A0A0A")

        title = ctx.title or f"Canvas Allocation ({len(ctx.bound_schema.field_names)} fields, {H}x{W})"
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
        ax.tick_params(colors="white", labelsize=8)

        # Legend
        patches = []
        for domain in sorted(seen_domains):
            color = DOMAIN_COLORS.get(domain, "#FFFFFF")
            patches.append(mpatches.Patch(color=color, label=domain))
        if patches:
            legend = ax.legend(
                handles=patches, loc="upper right", fontsize=7,
                facecolor="#1A1A1A", edgecolor="#333",
                labelcolor="white", framealpha=0.9,
            )

        fig.tight_layout()
        return fig
