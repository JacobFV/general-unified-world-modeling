"""WorldProjection: declare which parts of the world model you need.

You explicitly list every node you want to model. Only listed paths
(and structural ancestors needed to reach them) are included. Non-listed
siblings are omitted entirely — no implicit coarse-graining.

compile_schema then handles the structural coarse-graining: every nested
type automatically gets a coarse-grained field at its path, providing
hierarchical bottlenecking for cross-level attention.

Usage:
    from general_unified_world_model import World, WorldProjection, project
    from general_unified_world_model.schema.business import Business

    proj = WorldProjection(
        include=[
            "financial",              # all of financial, fully expanded
            "country_us.macro",       # only macro — politics/demographics omitted
            "regime",
        ],
        entities={
            "firm_AAPL": Business(),
            "firm_NVDA": Business(),
        },
    )

    bound = project(proj, T=1, d_model=64)
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field as dc_field
from typing import Any, Optional

from canvas_engineering import ConnectivityPolicy, compile_schema

from general_unified_world_model.schema.world import World


# ── Projection container ────────────────────────────────────────────────
# compile_schema traverses dataclass fields and lists. To add dynamic
# entities (firms, individuals, etc.) we use a container dataclass
# with list fields that hold the dynamic instances.

@dataclass
class ProjectedWorld:
    """A World subset with dynamic entity lists.

    compile_schema traverses:
      - dataclass fields → named regions
      - list fields → array elements with [i] indexing

    This container exposes the same top-level layers as World,
    plus lists for dynamic entities.
    """
    # These will be selectively populated from World
    # Using Optional so they can be None (excluded from compilation)
    pass


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
    world: World,
    include_paths: list[str],
    exclude_paths: list[str],
    entities: dict[str, Any],
):
    """Dynamically construct a projected dataclass from the World.

    Since compile_schema only walks dataclass fields, we build a new
    dataclass at runtime with exactly the fields we need.

    Each include path is resolved independently:
    - "financial" → include the full GlobalFinancialLayer
    - "country_us.macro" → navigate to world.country_us.macro,
      include it as a top-level field named "country_us.macro"

    No implicit parent inclusion. If you want country_us as a
    structural wrapper, include "country_us" explicitly.

    Args:
        world: The full World instance to project from.
        include_paths: Dotted paths into World to include. ["*"] means all.
        exclude_paths: Dotted paths to exclude (applied after include).
        entities: Dict mapping entity names (e.g. "firm_AAPL") to
            dataclass instances. Added as named fields on the projection.
    """
    fields_dict = {}

    is_wildcard = include_paths == ["*"]

    if is_wildcard:
        # Include everything
        for f in dataclasses.fields(World):
            val = getattr(world, f.name)
            if not _any_path_matches(f.name, exclude_paths):
                fields_dict[f.name] = val
    else:
        # Resolve each include path independently
        for path in include_paths:
            if _any_path_matches(path, exclude_paths):
                continue
            # Navigate the dotted path from world
            obj = _resolve_path(world, path)
            if obj is not None:
                # Use a safe field name (dots → underscores for dataclass)
                fields_dict[path] = obj

    # Add dynamic entities as named fields
    for name, obj in entities.items():
        fields_dict[name] = obj

    if not fields_dict:
        raise ValueError("Projection resulted in zero fields. Check your include paths.")

    # Build a dynamic dataclass.
    # Dotted paths need sanitized field names for Python identifiers,
    # but compile_schema uses the field name as path prefix. We use
    # dots→double-underscores so "country_us.macro" becomes
    # "country_us__macro" and compile_schema produces paths like
    # "country_us__macro.gdp". The HeterogeneousDataset handles
    # both naming conventions when routing data.
    dc_fields = []
    for path, val in fields_dict.items():
        safe_name = path.replace(".", "__")
        dc_fields.append(
            (safe_name, type(val), dataclasses.field(default_factory=lambda v=val: copy.deepcopy(v)))
        )

    ProjectedType = dataclasses.make_dataclass("ProjectedWorld", dc_fields)
    return ProjectedType()


# ── Projection ──────────────────────────────────────────────────────────

@dataclass
class WorldProjection:
    """Declares which subset of the World schema to activate.

    Only explicitly listed paths are included. Non-listed siblings are
    omitted entirely. compile_schema handles structural coarse-graining
    (every nested type gets a coarse-grained field automatically).

    Args:
        include: List of dotted paths into the World schema to include.
            E.g. ["financial", "country_us.macro", "regime"].
            Use "*" to include everything.
            Sub-paths keep only the listed children; siblings are dropped.

        exclude: List of dotted paths to exclude (applied after include).

        entities: Dict mapping entity names to dataclass instances.
            Keys are arbitrary names (e.g. "firm_AAPL", "person_fed_chair",
            "sector_tech", "country_jp"). Values must be dataclass instances
            that compile_schema can traverse.

        connectivity: Override the default connectivity policy.

        temporal_start: Dict mapping entity names to their temporal start
            index. E.g. {"firm_AAPL": 100} means Apple's fields only
            appear after timestep 100 in SFT data.
    """
    include: list[str] = dc_field(default_factory=lambda: ["*"])
    exclude: list[str] = dc_field(default_factory=list)

    entities: dict[str, Any] = dc_field(default_factory=dict)

    connectivity: Optional[ConnectivityPolicy] = None
    temporal_start: dict[str, int] = dc_field(default_factory=dict)


def project(
    proj: WorldProjection,
    T: int = 1,
    H: Optional[int] = None,
    W: Optional[int] = None,
    d_model: int = 64,
    connectivity: Optional[ConnectivityPolicy] = None,
    t_current: int = 0,
):
    """Compile a WorldProjection into a BoundSchema.

    This is the main entry point. Declare what you care about,
    get a compiled canvas ready for training.

    Canvas dimensions (H, W) are auto-computed by compile_schema when
    not specified — like a C compiler allocating struct memory.

    Only explicitly included paths are modeled. Non-included siblings
    are omitted, not coarse-grained. compile_schema handles structural
    coarse-graining: every nested type gets a coarse-grained field at
    its path, routing cross-level attention through it.

    Args:
        proj: WorldProjection declaring the subset.
        T: Temporal extent.
        H, W: Canvas grid dimensions. None = auto-sized by compile_schema.
        d_model: Latent dimensionality per position.
        connectivity: Override connectivity policy.
        t_current: Timestep boundary for output mask.

    Returns:
        BoundSchema with layout, topology, and field accessors.

    Example:
        proj = WorldProjection(include=["financial", "regime", "forecasts"])
        bound = project(proj)  # auto-sized canvas
        print(bound.summary())
    """
    world = World()

    projected = _make_projected_dataclass(
        world, proj.include, proj.exclude,
        proj.entities,
    )

    conn = connectivity or proj.connectivity or ConnectivityPolicy(
        intra="dense",
        array_element="ring",
        temporal="dense",
    )

    bound = compile_schema(
        projected,
        T=T,
        H=H,
        W=W,
        d_model=d_model,
        connectivity=conn,
        t_current=t_current,
    )

    return bound
