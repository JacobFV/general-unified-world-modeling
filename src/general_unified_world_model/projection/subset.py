"""WorldProjection: declare which parts of the world model you need.

You explicitly list every node you want to model. Only listed paths
(and structural ancestors needed to reach them) are included. Non-listed
siblings are omitted entirely — no implicit coarse-graining.

compile_schema then handles the structural coarse-graining: every nested
type automatically gets a coarse-grained field at its path, providing
hierarchical bottlenecking for cross-level attention.

Usage:
    from general_unified_world_model import World, WorldProjection, project

    proj = WorldProjection(
        include=[
            "financial",              # all of financial, fully expanded
            "country_us.macro",       # only macro — politics/demographics omitted
            "regime",
        ],
        firms=["AAPL", "NVDA"],
    )

    bound = project(proj, T=1, d_model=64)
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field as dc_field
from typing import Optional

from collections import defaultdict

from canvas_engineering import Field, Connection, ConnectivityPolicy, compile_schema

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




# ── Projection logic ──────────────────────────────────────────────────

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

    Each include path is resolved independently:
    - "financial" → include the full GlobalFinancialLayer
    - "country_us.macro" → navigate to world.country_us.macro,
      include it as a top-level field named "country_us.macro"

    No implicit parent inclusion. If you want country_us as a
    structural wrapper, include "country_us" explicitly.
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


def _any_path_matches(field_path: str, patterns: list[str]) -> bool:
    """Check if a field path is exactly one of the patterns."""
    return field_path in patterns


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

    _add_cross_domain_connections(bound)
    return bound


# ── Cross-domain connections ─────────────────────────────────────────

# Causal relationships between top-level domain prefixes.
# (src_prefix, dst_prefix) → weight.  These are directional.
_CROSS_DOMAIN_PAIRS: list[tuple[str, str, float]] = [
    # Regime conditions everything
    ("regime",        "financial",    0.8),
    ("regime",        "country",      0.8),
    ("regime",        "sector",       0.6),
    ("regime",        "firm",         0.5),
    ("regime",        "forecasts",    0.7),
    # Financial ↔ macro
    ("financial",     "country",      0.7),
    ("country",       "financial",    0.7),
    # Financial → firms, sectors
    ("financial",     "firm",         0.5),
    ("financial",     "sector",       0.5),
    ("financial",     "forecasts",    0.6),
    # Country/macro → forecasts
    ("country",       "forecasts",    0.5),
    ("country",       "narratives",   0.4),
    # Narratives ↔ financial
    ("narratives",    "financial",    0.5),
    ("financial",     "narratives",   0.4),
    # Events → many
    ("events",        "financial",    0.6),
    ("events",        "narratives",   0.5),
    ("events",        "firm",         0.4),
    ("events",        "country",      0.3),
    # Firms / sectors
    ("firm",          "sector",       0.5),
    ("sector",        "firm",         0.4),
    ("firm",          "forecasts",    0.4),
    # Interventions
    ("interventions", "financial",    0.6),
    ("interventions", "country",      0.7),
    # Trust
    ("trust",         "narratives",   0.4),
    ("trust",         "events",       0.3),
    # People
    ("person",        "firm",         0.6),
    ("person",        "narratives",   0.4),
]


def _domain_prefix(field_name: str) -> str:
    """Map a field name to its domain prefix for cross-domain matching."""
    top = field_name.split(".")[0]
    for prefix in ("regime", "financial", "country", "sector", "firm",
                    "person", "events", "narratives", "interventions",
                    "trust", "forecasts", "technology", "resources",
                    "physical", "sc"):
        if top.startswith(prefix):
            return prefix
    return top


def _add_cross_domain_connections(bound) -> None:
    """Append cross-domain attention connections to a compiled BoundSchema.

    compile_schema generates intra-domain connections only.  This function
    adds directed hub connections between domains that have a causal
    relationship (defined in _CROSS_DOMAIN_PAIRS), using representative
    fields from each domain.
    """
    if bound.topology is None:
        return

    # Group fields by domain prefix
    domain_fields: dict[str, list[str]] = defaultdict(list)
    for name in bound.field_names:
        domain_fields[_domain_prefix(name)].append(name)

    present = set(domain_fields.keys())
    new_conns: list[Connection] = []

    for src_prefix, dst_prefix, weight in _CROSS_DOMAIN_PAIRS:
        if src_prefix not in present or dst_prefix not in present:
            continue

        # Pick up to 3 representative fields from each domain
        src_reps = domain_fields[src_prefix][:3]
        dst_reps = domain_fields[dst_prefix][:3]

        for sf in src_reps:
            for df in dst_reps:
                new_conns.append(Connection(src=sf, dst=df, weight=weight))

    bound.topology.connections.extend(new_conns)
