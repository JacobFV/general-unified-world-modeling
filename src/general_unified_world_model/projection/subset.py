"""Schema projection: select which parts of a schema to compile.

You explicitly list every node you want to model. Only listed paths
(and structural ancestors needed to reach them) are included. Non-listed
siblings are omitted entirely — no implicit coarse-graining.

compile_schema then handles the structural coarse-graining: every nested
type automatically gets a coarse-grained field at its path, providing
hierarchical bottlenecking for cross-level attention.

Usage:
    from general_unified_world_model import World, project

    # Schema root + include
    bound = project(World(), include=["financial", "regime"])

    # With entities
    from general_unified_world_model.schema.business import Business
    bound = project(
        World(),
        include=["financial", "country_us.macro", "regime"],
        entities={"firm_AAPL": Business(), "firm_NVDA": Business()},
    )

    # Default root (World()) when omitted
    bound = project(include=["financial"], T=1, d_model=64)
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from canvas_engineering import ConnectivityPolicy, compile_schema

from general_unified_world_model.schema.world import World


# ── Projection helpers ─────────────────────────────────────────────────

def _resolve_path(root, dotted_path: str):
    """Navigate a dotted path from a root object.

    E.g. _resolve_path(world, "country_us.macro") returns world.country_us.macro.
    Returns None if any step fails.
    """
    obj = root
    for part in dotted_path.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return None
    return obj


def _any_path_matches(field_path: str, patterns: list[str]) -> bool:
    """Check if a field path is exactly one of the patterns."""
    return field_path in patterns


# ── Projection logic ──────────────────────────────────────────────────

def _make_projected_dataclass(
    root,
    include_paths: list[str],
    exclude_paths: list[str],
    entities: dict[str, Any],
):
    """Dynamically construct a projected dataclass from any schema root.

    Since compile_schema only walks dataclass fields, we build a new
    dataclass at runtime with exactly the fields we need.

    Each include path is resolved independently:
    - "financial" → include the full GlobalFinancialLayer
    - "country_us.macro" → navigate to root.country_us.macro,
      include it as a top-level field named "country_us.macro"

    No implicit parent inclusion. If you want country_us as a
    structural wrapper, include "country_us" explicitly.

    Args:
        root: Any dataclass instance to project from.
        include_paths: Dotted paths into root to include. ["*"] means all.
        exclude_paths: Dotted paths to exclude (applied after include).
        entities: Dict mapping entity names (e.g. "firm_AAPL") to
            dataclass instances. Added as named fields on the projection.
    """
    fields_dict = {}

    is_wildcard = include_paths == ["*"]

    if is_wildcard:
        if dataclasses.is_dataclass(root):
            for f in dataclasses.fields(root):
                val = getattr(root, f.name)
                if not _any_path_matches(f.name, exclude_paths):
                    fields_dict[f.name] = val
        else:
            raise ValueError("Wildcard include requires a dataclass root")
    else:
        for path in include_paths:
            if _any_path_matches(path, exclude_paths):
                continue
            obj = _resolve_path(root, path)
            if obj is not None:
                fields_dict[path] = obj

    for name, obj in entities.items():
        fields_dict[name] = obj

    if not fields_dict:
        raise ValueError("Projection resulted in zero fields. Check your include paths.")

    # Build a dynamic dataclass. Dotted paths need sanitized field names
    # for Python identifiers. "country_us.macro" becomes "country_us__macro".
    dc_fields = []
    for path, val in fields_dict.items():
        safe_name = path.replace(".", "__")
        dc_fields.append(
            (safe_name, type(val), dataclasses.field(default_factory=lambda v=val: copy.deepcopy(v)))
        )

    ProjectedType = dataclasses.make_dataclass("ProjectedWorld", dc_fields)
    return ProjectedType()


# ── project() ────────────────────────────────────────────────────────────

def project(
    schema_root=None,
    /,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    entities: dict[str, Any] | None = None,
    T: int = 1,
    H: int | None = None,
    W: int | None = None,
    d_model: int = 64,
    connectivity: ConnectivityPolicy | None = None,
    t_current: int = 0,
) -> "BoundSchema":
    """Compile a schema projection into a BoundSchema.

    Usage:
        # Schema root + include/exclude
        bound = project(World(), include=["financial", "regime"])

        # With entities
        bound = project(World(),
            include=["financial", "country_us.macro"],
            entities={"firm_AAPL": Business()},
        )

        # Auto-sized (H, W computed from field count)
        bound = project(World(), include=["regime"], d_model=64)

        # Default root (World()) when omitted
        bound = project(include=["financial"], T=1, d_model=64)

    Args:
        schema_root: A schema root dataclass, or None (defaults to World()).
        include: Dotted paths to include. Default ["*"] (all).
        exclude: Dotted paths to exclude.
        entities: Dict of entity_name -> dataclass instance.
        T: Temporal extent.
        H, W: Canvas grid dimensions. None = auto-sized.
        d_model: Latent dimensionality per position.
        connectivity: Override connectivity policy.
        t_current: Timestep boundary for output mask.

    Returns:
        BoundSchema with layout, topology, and field accessors.
    """
    if schema_root is None:
        schema_root = World()

    projected = _make_projected_dataclass(
        schema_root,
        include or ["*"],
        exclude or [],
        entities or {},
    )

    conn = connectivity or ConnectivityPolicy(
        intra="dense",
        array_element="ring",
        temporal="dense",
    )

    return compile_schema(
        projected,
        T=T,
        H=H,
        W=W,
        d_model=d_model,
        connectivity=conn,
        t_current=t_current,
    )
