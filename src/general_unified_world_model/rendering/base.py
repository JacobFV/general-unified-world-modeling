"""Base rendering abstraction.

Every renderer takes a RenderContext (bound schema + optional state data)
and produces a matplotlib Figure. The abstraction is simple: implement
render() → Figure. Everything else is sugar.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class RenderContext:
    """Everything a renderer needs to produce a visualization.

    Args:
        bound_schema: Compiled BoundSchema from compile_schema / project().
        state: Optional (N, d_model) tensor of canvas values.
            None = render structure only (topology, layout, etc.).
        predictions: Optional dict of field_path → tensor predictions.
        observations: Optional dict of field_path → tensor observations.
        time_series: Optional dict of field_path → (T,) tensor for charts.
        title: Optional title override.
        metadata: Arbitrary rendering metadata.
    """
    bound_schema: Any
    state: Optional[torch.Tensor] = None
    predictions: Optional[dict[str, torch.Tensor]] = None
    observations: Optional[dict[str, torch.Tensor]] = None
    time_series: Optional[dict[str, torch.Tensor]] = None
    title: Optional[str] = None
    metadata: dict[str, Any] = dc_field(default_factory=dict)


class Renderer(abc.ABC):
    """Protocol for world model renderers.

    Each renderer converts a RenderContext into a matplotlib Figure
    at a specific plane of abstraction.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable renderer name."""
        ...

    @abc.abstractmethod
    def render(self, ctx: RenderContext) -> Any:
        """Render the context to a matplotlib Figure.

        Args:
            ctx: RenderContext with schema and optional state data.

        Returns:
            matplotlib.figure.Figure
        """
        ...

    def save(self, ctx: RenderContext, path: str | Path, dpi: int = 150, **kwargs):
        """Render and save to file."""
        fig = self.render(ctx)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight", **kwargs)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return path


# ── Convenience functions ──────────────────────────────────────────────

_REGISTRY: dict[str, type[Renderer]] = {}


def _register(cls: type[Renderer]) -> type[Renderer]:
    """Register a renderer class by name."""
    instance = cls()
    _REGISTRY[instance.name] = cls
    return cls


def render(
    bound_schema,
    renderer: str | Renderer,
    state: torch.Tensor | None = None,
    predictions: dict | None = None,
    time_series: dict | None = None,
    title: str | None = None,
    save_path: str | None = None,
    **kwargs,
):
    """One-shot render.

    Args:
        bound_schema: Compiled BoundSchema.
        renderer: Renderer instance or registered name string.
        state: Optional canvas state tensor.
        predictions: Optional predictions dict.
        time_series: Optional time series dict.
        title: Optional title.
        save_path: If set, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    if isinstance(renderer, str):
        if renderer not in _REGISTRY:
            available = ", ".join(sorted(_REGISTRY.keys()))
            raise ValueError(f"Unknown renderer '{renderer}'. Available: {available}")
        renderer = _REGISTRY[renderer]()

    ctx = RenderContext(
        bound_schema=bound_schema,
        state=state,
        predictions=predictions,
        time_series=time_series,
        title=title,
        metadata=kwargs,
    )

    if save_path:
        renderer.save(ctx, save_path)
    return renderer.render(ctx)


def render_all(
    bound_schema,
    output_dir: str | Path = "renders",
    state: torch.Tensor | None = None,
    predictions: dict | None = None,
    time_series: dict | None = None,
    dpi: int = 150,
) -> dict[str, Path]:
    """Render all registered renderers and save to output_dir.

    Returns:
        Dict mapping renderer name to saved file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    ctx = RenderContext(
        bound_schema=bound_schema,
        state=state,
        predictions=predictions,
        time_series=time_series,
    )

    for name, cls in _REGISTRY.items():
        renderer = cls()
        path = output_dir / f"{name}.png"
        renderer.save(ctx, path, dpi=dpi)
        results[name] = path

    return results
