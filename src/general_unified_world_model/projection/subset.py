"""WorldProjection: declare which parts of the world model you need.

A CEO says: "I want to model my company, our employees, and the macro environment."
A government analyst says: "I want to model the macroeconomic impact of this policy."
A computer use agent says: "I need the user's psychology and the world they're embedded in."

For each case, they declare a projection — a subset of the World schema with
a particular connectivity topology — and compile it to a canvas. The full
ontology is always available; the projection selects what's active.

Fog regions: when you project to a subset, any excluded sub-types get
collapsed into 1×1 "fog" vectors that represent the aggregate of the
unmodeled portion. Fog vectors connect to their parent and siblings,
learning simpler dynamics as a stand-in for everything beyond the
modeled level of abstraction.

Usage:
    from general_unified_world_model import World, WorldProjection, project

    # A hedge fund modeling macro + financial + 2 firms
    proj = WorldProjection(
        include=[
            "financial",
            "country_us.macro",       # macro only — politics gets fog
            "regime",
            "narratives.positioning",  # positioning only — media gets fog
        ],
        firms=["AAPL", "NVDA"],
    )

    bound = project(proj, T=1, H=64, W=64, d_model=64)
    # bound now has fog fields like:
    #   country_us._fog_politics (1×1 fog for politics)
    #   narratives._fog_media    (1×1 fog for media)
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field as dc_field
from typing import Optional, Union

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


# ── Fog helpers ────────────────────────────────────────────────────────

def _count_leaf_fields(obj) -> int:
    """Count total Field instances in an object tree."""
    if isinstance(obj, Field):
        return 1
    if not dataclasses.is_dataclass(obj):
        return 0
    count = 0
    for f in dataclasses.fields(type(obj)):
        count += _count_leaf_fields(getattr(obj, f.name))
    return count


def _get_median_period(obj) -> int:
    """Get the median period of all Fields in an object tree."""
    periods = []

    def _collect(o):
        if isinstance(o, Field):
            periods.append(o.period)
        elif dataclasses.is_dataclass(o):
            for f in dataclasses.fields(type(o)):
                _collect(getattr(o, f.name))

    _collect(obj)
    if not periods:
        return 192  # default to monthly
    periods.sort()
    return periods[len(periods) // 2]


def _is_directly_included(path: str, include_paths: list[str]) -> bool:
    """Check if path is explicitly in include or covered by a broader pattern."""
    for p in include_paths:
        if path == p or path.startswith(p + "."):
            return True
    return False


def _has_included_descendant(path: str, include_paths: list[str]) -> bool:
    """Check if any include path targets a descendant of this path."""
    return any(p.startswith(path + ".") for p in include_paths)


def _apply_fog_to_object(obj, parent_path: str, include_paths: list[str]):
    """Replace excluded sub-dataclasses with 1×1 fog Field instances.

    Walks the object's children. For each child sub-dataclass:
    - If directly included → keep as-is
    - If has included descendants → recurse (partial inclusion)
    - Otherwise → replace with a fog field

    Returns a new dataclass instance (or the original if no fog needed).
    """
    if not dataclasses.is_dataclass(obj):
        return obj

    new_fields = []
    modified = False

    for f in dataclasses.fields(type(obj)):
        child_path = f"{parent_path}.{f.name}" if parent_path else f.name
        child_val = getattr(obj, f.name)

        if isinstance(child_val, Field):
            # Leaf Field — always keep (it's part of an included type)
            new_fields.append((f.name, child_val))

        elif dataclasses.is_dataclass(child_val):
            if _is_directly_included(child_path, include_paths):
                # Fully included — keep as-is
                new_fields.append((f.name, child_val))
            elif _has_included_descendant(child_path, include_paths):
                # Partially included — recurse to fog the non-included parts
                filtered = _apply_fog_to_object(child_val, child_path, include_paths)
                new_fields.append((f.name, filtered))
                modified = True
            else:
                # Not included at all — replace with fog field
                n = _count_leaf_fields(child_val)
                mp = _get_median_period(child_val)
                fog = Field(
                    1, 1, period=mp,
                    semantic_type=f"fog: {n} unmodeled fields of {child_path}",
                )
                new_fields.append((f"_fog_{f.name}", fog))
                modified = True

        elif isinstance(child_val, (list, tuple)):
            new_fields.append((f.name, child_val))

    if not modified:
        return obj

    # Build a new dataclass with the filtered fields
    dc_field_specs = []
    for name, val in new_fields:
        dc_field_specs.append(
            (name, type(val), dataclasses.field(
                default_factory=lambda v=val: copy.deepcopy(v)))
        )

    FoggedType = dataclasses.make_dataclass(
        f"Fogged_{type(obj).__name__}", dc_field_specs
    )
    return FoggedType()


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
    fog: bool = True,
):
    """Dynamically construct a projected dataclass from the World.

    Since compile_schema only walks dataclass fields, we build a new
    dataclass at runtime with exactly the fields we need.

    When fog=True, excluded sub-types within included parents are
    replaced with 1×1 fog fields that represent the aggregate of
    the unmodeled portion.
    """
    fields_dict = {}

    is_wildcard = include_paths == ["*"]

    # Walk the World's fields and include matching ones
    for f in dataclasses.fields(World):
        path = f.name
        val = getattr(world, f.name)

        if is_wildcard or _any_path_matches(path, include_paths):
            if not _any_path_matches(path, exclude_paths):
                # Check if this was included via sub-path match (needs fog)
                if fog and not is_wildcard and not _is_directly_included(path, include_paths):
                    # Included because a sub-path matched — apply fog
                    val = _apply_fog_to_object(val, path, include_paths)

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

        fog: Whether to generate fog regions for excluded sub-types.
            When True (default), any excluded sub-type within an included
            parent is replaced with a 1×1 fog field that represents the
            aggregate of the unmodeled portion. Fog fields participate in
            attention, learning simpler dynamics as stand-ins for everything
            beyond the modeled level of abstraction.

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
    fog: bool = True

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

    When fog=True in the projection, excluded sub-types are replaced
    with 1×1 fog regions. For example, include=["country_us.macro"]
    will include the full macro sub-type but create fog fields for
    politics, demographics, etc. Fog fields participate in attention,
    providing a learned summary of unmodeled dynamics.

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
        fog=proj.fog,
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
