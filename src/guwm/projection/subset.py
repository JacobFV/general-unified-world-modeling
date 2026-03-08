"""WorldProjection: declare which parts of the world model you need.

A CEO says: "I want to model my company, our employees, and the macro environment."
A government analyst says: "I want to model the macroeconomic impact of this policy."
A computer use agent says: "I need the user's psychology and the world they're embedded in."

For each case, they declare a projection — a subset of the World schema with
a particular connectivity topology — and compile it to a canvas. The full
ontology is always available; the projection selects what's active.

Usage:
    from guwm import World, WorldProjection, project

    # A hedge fund modeling macro + financial + 2 firms
    proj = WorldProjection(
        include=[
            "financial",
            "country_us.macro",
            "country_cn.macro",
            "regime",
            "narratives.positioning",
            "events",
            "forecasts.macro",
            "forecasts.financial",
        ],
        firms=["AAPL", "NVDA"],
        individuals=["fed_chair"],
    )

    bound = project(proj, T=1, H=64, W=64, d_model=64)
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field as dc_field
from typing import Optional, Union

from canvas_engineering import Field, ConnectivityPolicy, compile_schema

from guwm.schema.world import World
from guwm.schema.business import Business
from guwm.schema.individual import Individual
from guwm.schema.sector import Sector
from guwm.schema.supply_chain import SupplyChainNode
from guwm.schema.country import Country


# ── Helpers ──────────────────────────────────────────────────────────────

def _walk_fields(obj, prefix=""):
    """Yield (dotted_path, field_value) for all Fields in a dataclass tree."""
    if not dataclasses.is_dataclass(obj):
        return
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        path = f"{prefix}.{f.name}" if prefix else f.name
        if isinstance(val, Field):
            yield path, val
        elif isinstance(val, list):
            for i, item in enumerate(val):
                yield from _walk_fields(item, f"{path}[{i}]")
        elif dataclasses.is_dataclass(val):
            yield from _walk_fields(val, path)


def _resolve_attr(obj, dotted_path: str):
    """Get nested attribute via dotted path like 'financial.credit.ig_spread'."""
    parts = dotted_path.split(".")
    current = obj
    for part in parts:
        if "[" in part:
            name, idx_str = part.split("[")
            idx = int(idx_str.rstrip("]"))
            current = getattr(current, name)[idx]
        else:
            current = getattr(current, part)
    return current


def _set_attr(obj, dotted_path: str, value):
    """Set nested attribute via dotted path."""
    parts = dotted_path.split(".")
    current = obj
    for part in parts[:-1]:
        if "[" in part:
            name, idx_str = part.split("[")
            idx = int(idx_str.rstrip("]"))
            current = getattr(current, name)[idx]
        else:
            current = getattr(current, part)
    last = parts[-1]
    if "[" in last:
        name, idx_str = last.split("[")
        idx = int(idx_str.rstrip("]"))
        getattr(current, name)[idx] = value
    else:
        object.__setattr__(current, last, value)


def _has_attr(obj, dotted_path: str) -> bool:
    """Check if nested attribute exists."""
    try:
        _resolve_attr(obj, dotted_path)
        return True
    except (AttributeError, IndexError, TypeError):
        return False


def _path_matches(field_path: str, include_path: str) -> bool:
    """Check if a field path is under an include path.

    'financial.credit.ig_spread' matches 'financial' and 'financial.credit'
    but not 'financial.fx'.
    """
    return field_path == include_path or field_path.startswith(include_path + ".")


# ── Projection ──────────────────────────────────────────────────────────

@dataclass
class WorldProjection:
    """Declares which subset of the World schema to activate.

    Args:
        include: List of dotted paths into the World schema to include.
            E.g. ["financial", "country_us.macro", "regime"].
            Use "*" to include everything.

        exclude: List of dotted paths to exclude (applied after include).

        firms: Named firm instances to add. Each becomes a Business block.
            E.g. ["AAPL", "NVDA", "TSMC"].

        individuals: Named individual instances.
            E.g. ["fed_chair", "ceo_nvda"].

        sectors: Named sector instances beyond the defaults.
            E.g. ["healthcare", "industrials"].

        supply_chains: Named supply chain nodes beyond defaults.
            E.g. ["rare_earths", "pharmaceuticals"].

        countries: Named country instances beyond the 3 defaults.
            E.g. ["jp", "in", "kr", "uk"].

        connectivity: Override the default connectivity policy.

        temporal_start: Dict mapping entity names to their temporal start
            index. E.g. {"firm_AAPL": 100} means Apple's fields only
            appear after timestep 100 in SFT data. The compiler will
            set temporal_extent accordingly.
    """
    include: list[str] = dc_field(default_factory=lambda: ["*"])
    exclude: list[str] = dc_field(default_factory=list)

    firms: list[str] = dc_field(default_factory=list)
    individuals: list[str] = dc_field(default_factory=list)
    sectors: list[str] = dc_field(default_factory=list)
    supply_chains: list[str] = dc_field(default_factory=list)
    countries: list[str] = dc_field(default_factory=list)

    connectivity: Optional[ConnectivityPolicy] = None
    temporal_start: dict[str, int] = dc_field(default_factory=dict)


def _build_projected_world(proj: WorldProjection) -> World:
    """Construct a World instance with only the projected fields active."""
    world = World()

    # Add dynamic entities
    for name in proj.firms:
        setattr(world, f"firm_{name}", Business())
    for name in proj.individuals:
        setattr(world, f"person_{name}", Individual())
    for name in proj.sectors:
        setattr(world, f"sector_{name}", Sector())
    for name in proj.supply_chains:
        setattr(world, f"sc_{name}", SupplyChainNode())
    for name in proj.countries:
        setattr(world, f"country_{name}", Country())

    # If include is ["*"], return the full world (minus excludes handled at compile)
    if proj.include == ["*"] and not proj.exclude:
        return world

    # Otherwise, build a selective world
    # We create a new World and only copy over the included subtrees
    selective = World.__new__(World)
    # Initialize all fields to None first
    for f in dataclasses.fields(World):
        object.__setattr__(selective, f.name, None)

    # Copy dynamic entities (always included)
    for name in proj.firms:
        attr = f"firm_{name}"
        if hasattr(world, attr):
            object.__setattr__(selective, attr, getattr(world, attr))
    for name in proj.individuals:
        attr = f"person_{name}"
        if hasattr(world, attr):
            object.__setattr__(selective, attr, getattr(world, attr))
    for name in proj.sectors:
        attr = f"sector_{name}"
        if hasattr(world, attr):
            object.__setattr__(selective, attr, getattr(world, attr))
    for name in proj.supply_chains:
        attr = f"sc_{name}"
        if hasattr(world, attr):
            object.__setattr__(selective, attr, getattr(world, attr))
    for name in proj.countries:
        attr = f"country_{name}"
        if hasattr(world, attr):
            object.__setattr__(selective, attr, getattr(world, attr))

    # Copy included subtrees
    for inc_path in proj.include:
        if inc_path == "*":
            return world  # short-circuit

        # Find the top-level field name
        top_level = inc_path.split(".")[0]
        if hasattr(world, top_level):
            # If the include is the full top-level, copy the whole thing
            if inc_path == top_level:
                object.__setattr__(selective, top_level, getattr(world, top_level))
            else:
                # Partial include — we need the parent structure
                if getattr(selective, top_level) is None:
                    # Deep copy the full subtree (we'll prune later if needed)
                    object.__setattr__(selective, top_level, copy.deepcopy(getattr(world, top_level)))

    # Apply excludes by nulling out excluded subtrees
    # (compile_schema skips None attributes)
    for exc_path in proj.exclude:
        if _has_attr(selective, exc_path):
            top_level = exc_path.split(".")[0]
            if exc_path == top_level:
                object.__setattr__(selective, top_level, None)
            else:
                parent_path = ".".join(exc_path.split(".")[:-1])
                field_name = exc_path.split(".")[-1]
                try:
                    parent = _resolve_attr(selective, parent_path)
                    object.__setattr__(parent, field_name, None)
                except (AttributeError, TypeError):
                    pass

    return selective


def project(
    proj: WorldProjection,
    T: int = 1,
    H: int = 64,
    W: int = 64,
    d_model: int = 64,
    connectivity: Optional[ConnectivityPolicy] = None,
    t_current: int = 0,
):
    """Compile a WorldProjection into a BoundSchema.

    This is the main entry point. Declare what you care about,
    get a compiled canvas ready for training.

    Args:
        proj: WorldProjection declaring the subset.
        T, H, W, d_model: Canvas dimensions.
        connectivity: Override connectivity policy.
        t_current: Timestep boundary for output mask.

    Returns:
        BoundSchema with layout, topology, and field accessors.

    Example:
        proj = WorldProjection(include=["financial", "regime", "forecasts"])
        bound = project(proj, T=1, H=32, W=32, d_model=64)
        print(bound.summary())
    """
    world = _build_projected_world(proj)

    conn = connectivity or proj.connectivity or ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",
        array_element="ring",
        temporal="dense",
    )

    bound = compile_schema(
        world,
        T=T,
        H=H,
        W=W,
        d_model=d_model,
        connectivity=conn,
        t_current=t_current,
    )

    return bound
