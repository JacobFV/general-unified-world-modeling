"""WorldProjection: declare which parts of the world model you need.

A CEO says: "I want to model my company, our employees, and the macro environment."
A government analyst says: "I want to model the macroeconomic impact of this policy."
A computer use agent says: "I need the user's psychology and the world they're embedded in."

For each case, they declare a projection — a subset of the World schema with
a particular connectivity topology — and compile it to a canvas. The full
ontology is always available; the projection selects what's active.

Coarse-graining: when you include a parent path (e.g. "country_us.macro"),
any sibling sub-types not explicitly included (politics, demographics, etc.)
are automatically collapsed to 1×1 coarse-grained fields. These keep
their original name — "country_us.politics" stays "country_us.politics"
whether it's 40 positions (fully expanded) or 1 position (coarse-grained).
This means encoders/decoders keyed by field name transfer across projections
without re-learning.

Usage:
    from general_unified_world_model import World, WorldProjection, project

    # A hedge fund modeling macro + financial + 2 firms
    proj = WorldProjection(
        include=[
            "financial",
            "country_us.macro",       # macro only — politics coarse-grained
            "regime",
            "narratives.positioning",  # positioning only — media coarse-grained
        ],
        firms=["AAPL", "NVDA"],
    )

    bound = project(proj, T=1, d_model=64)
    # bound has coarse-grained fields like:
    #   country_us.politics  (1×1 coarse-grained)
    #   narratives.media     (1×1 coarse-grained)
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


# ── Coarse-graining helpers ────────────────────────────────────────────

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


def _coarse_grain(obj, parent_path: str, include_paths: list[str]):
    """Coarse-grain excluded sub-dataclasses to 1×1 Fields.

    Walks the object's children. For each child sub-dataclass:
    - If directly included → keep as-is
    - If has included descendants → recurse (partial inclusion)
    - Otherwise → replace with a single Field(1, 1) at the same name

    The key property: coarse-grained fields keep their original name.
    "country_us.politics" stays "country_us.politics" whether it's 40
    positions or 1. This means encoders, decoders, and semantic conditioning
    all transfer across projections without re-learning.

    Returns a new dataclass instance (or the original if unchanged).
    """
    if not dataclasses.is_dataclass(obj):
        return obj

    new_fields = []
    modified = False

    for f in dataclasses.fields(type(obj)):
        child_path = f"{parent_path}.{f.name}" if parent_path else f.name
        child_val = getattr(obj, f.name)

        if isinstance(child_val, Field):
            # Leaf Field — always keep
            new_fields.append((f.name, child_val))

        elif dataclasses.is_dataclass(child_val):
            if _is_directly_included(child_path, include_paths):
                # Fully included — keep as-is
                new_fields.append((f.name, child_val))
            elif _has_included_descendant(child_path, include_paths):
                # Partially included — recurse
                filtered = _coarse_grain(child_val, child_path, include_paths)
                new_fields.append((f.name, filtered))
                modified = True
            else:
                # Not included — coarse-grain to a single 1×1 Field.
                # Keep the original field name so weights transfer.
                n = _count_leaf_fields(child_val)
                mp = _get_median_period(child_val)
                coarse = Field(
                    1, 1, period=mp,
                    semantic_type=f"coarse({n}) {child_path}",
                )
                new_fields.append((f.name, coarse))
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

    CoarseType = dataclasses.make_dataclass(
        f"Coarse_{type(obj).__name__}", dc_field_specs
    )
    return CoarseType()


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

    When a parent is included via a sub-path match, excluded children
    are automatically coarse-grained to 1×1 Fields with their original
    names preserved.
    """
    fields_dict = {}

    is_wildcard = include_paths == ["*"]

    # Walk the World's fields and include matching ones
    for f in dataclasses.fields(World):
        path = f.name
        val = getattr(world, f.name)

        if is_wildcard or _any_path_matches(path, include_paths):
            if not _any_path_matches(path, exclude_paths):
                # If included via sub-path, coarse-grain excluded children
                if not is_wildcard and not _is_directly_included(path, include_paths):
                    val = _coarse_grain(val, path, include_paths)

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

    Any sub-type not explicitly included is automatically coarse-grained
    to a single 1×1 position, keeping its original field name. This means
    encoders/decoders transfer across projections without re-learning.

    Args:
        include: List of dotted paths into the World schema to include.
            E.g. ["financial", "country_us.macro", "regime"].
            Use "*" to include everything.
            Sub-paths pull in the parent; siblings are coarse-grained.

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
    Override H, W only when you need a specific grid size.

    Excluded sub-types within included parents are automatically
    coarse-grained to 1×1 positions. For example,
    include=["country_us.macro"] will expand the macro sub-type fully
    but coarse-grain politics, demographics, etc. to single positions.
    Their field names stay the same, so weights transfer across projections.

    Uses bottleneck connectivity by default: each nested sub-type gets a
    1×1 gateway field. Cross-level attention routes through gateways,
    making hierarchical interactions O(1) per entity.

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
