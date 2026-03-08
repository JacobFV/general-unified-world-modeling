"""WorldProjection: declare which parts of the world model you need.

A CEO says: "I want to model my company, our employees, and the macro environment."
A government analyst says: "I want to model the macroeconomic impact of this policy."
A computer use agent says: "I need the user's psychology and the world they're embedded in."

For each case, they declare a projection — a subset of the World schema with
a particular connectivity topology — and compile it to a canvas. The full
ontology is always available; the projection selects what's active.

Usage:
    from general_unified_world_model import World, WorldProjection, project

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

from general_unified_world_model.schema.world import World
from general_unified_world_model.schema.business import Business
from general_unified_world_model.schema.individual import Individual
from general_unified_world_model.schema.sector import Sector
from general_unified_world_model.schema.supply_chain import SupplyChainNode
from general_unified_world_model.schema.country import Country


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


def _make_projected_dataclass(
    world: World,
    include_paths: list[str],
    exclude_paths: list[str],
    extra_firms: dict[str, Business],
    extra_individuals: dict[str, Individual],
    extra_sectors: dict[str, Sector],
    extra_supply_chains: dict[str, SupplyChainNode],
    extra_countries: dict[str, Country],
):
    """Dynamically construct a projected dataclass from the World.

    Since compile_schema only walks dataclass fields, we build a new
    dataclass at runtime with exactly the fields we need.
    """
    fields_dict = {}
    defaults = {}

    is_wildcard = include_paths == ["*"]

    # Walk the World's fields and include matching ones
    for f in dataclasses.fields(World):
        path = f.name
        val = getattr(world, f.name)

        if is_wildcard or _any_path_matches(path, include_paths):
            if not _any_path_matches(path, exclude_paths):
                fields_dict[path] = val

    # Add extra dynamic entities as named fields
    for name, obj in extra_firms.items():
        fields_dict[f"firm_{name}"] = obj
    for name, obj in extra_individuals.items():
        fields_dict[f"person_{name}"] = obj
    for name, obj in extra_sectors.items():
        fields_dict[f"sector_{name}"] = obj
    for name, obj in extra_supply_chains.items():
        fields_dict[f"sc_{name}"] = obj
    for name, obj in extra_countries.items():
        fields_dict[f"country_{name}"] = obj

    if not fields_dict:
        raise ValueError("Projection resulted in zero fields. Check your include paths.")

    # Build a dynamic dataclass
    dc_fields = []
    for name, val in fields_dict.items():
        dc_fields.append(
            (name, type(val), dataclasses.field(default_factory=lambda v=val: copy.deepcopy(v)))
        )

    ProjectedType = dataclasses.make_dataclass("ProjectedWorld", dc_fields)
    return ProjectedType()


def _any_path_matches(field_path: str, patterns: list[str]) -> bool:
    """Check if a field path matches any of the include/exclude patterns."""
    for pattern in patterns:
        if field_path == pattern or field_path.startswith(pattern + "."):
            return True
        if pattern.startswith(field_path + "."):
            # Pattern is a sub-path of this field — include the whole parent
            return True
    return False


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
            appear after timestep 100 in SFT data.
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
    world = World()

    extra_firms = {name: Business() for name in proj.firms}
    extra_individuals = {name: Individual() for name in proj.individuals}
    extra_sectors = {name: Sector() for name in proj.sectors}
    extra_supply_chains = {name: SupplyChainNode() for name in proj.supply_chains}
    extra_countries = {name: Country() for name in proj.countries}

    projected = _make_projected_dataclass(
        world, proj.include, proj.exclude,
        extra_firms, extra_individuals, extra_sectors,
        extra_supply_chains, extra_countries,
    )

    conn = connectivity or proj.connectivity or ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",
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
