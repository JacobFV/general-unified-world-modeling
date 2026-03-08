"""Tests for the world model schema."""

import dataclasses
import pytest
from canvas_engineering import Field, compile_schema, ConnectivityPolicy

from general_unified_world_model.schema.world import World
from general_unified_world_model.schema.observability import ObservedFast, ObservedDaily, ObservedSlow
from general_unified_world_model.schema.physical import PlanetaryPhysicalLayer, ClimateState
from general_unified_world_model.schema.resources import ResourceLayer, EnergySystem
from general_unified_world_model.schema.financial import GlobalFinancialLayer, YieldCurveState
from general_unified_world_model.schema.macro import MacroEconomy, OutputAndGrowth
from general_unified_world_model.schema.political import PoliticalLayer
from general_unified_world_model.schema.narrative import NarrativeBeliefLayer
from general_unified_world_model.schema.technology import TechnologyLayer
from general_unified_world_model.schema.demographics import DemographicLayer
from general_unified_world_model.schema.sector import Sector
from general_unified_world_model.schema.supply_chain import SupplyChainNode
from general_unified_world_model.schema.business import Business
from general_unified_world_model.schema.individual import Individual
from general_unified_world_model.schema.events import EventTape
from general_unified_world_model.schema.trust import DataChannelTrust
from general_unified_world_model.schema.regime import RegimeState
from general_unified_world_model.schema.intervention import InterventionSpace
from general_unified_world_model.schema.forecast import ForecastBundle
from general_unified_world_model.schema.country import Country


def test_world_instantiation():
    """World() should instantiate with all 19 layers."""
    world = World()
    assert world.physical is not None
    assert world.resources is not None
    assert world.financial is not None
    assert world.narratives is not None
    assert world.technology is not None
    assert world.regime is not None
    assert world.events is not None
    assert world.trust is not None
    assert world.forecasts is not None
    assert world.interventions is not None
    assert world.country_us is not None
    assert world.country_cn is not None
    assert world.country_eu is not None


def test_observability_bundles():
    """Observability bundles should wrap values with epistemic metadata."""
    fast = ObservedFast()
    assert isinstance(fast.value, Field)
    assert fast.value.period == 1
    assert isinstance(fast.confidence, Field)

    daily = ObservedDaily()
    assert daily.value.period == 16
    assert daily.latency.loss_weight == 0.2

    slow = ObservedSlow()
    assert slow.value.period == 192
    assert slow.revision_risk.loss_weight == 0.3


def test_all_fields_are_field_type():
    """Every leaf attribute of every schema type should be a Field."""
    world = World()

    def count_fields(obj, prefix=""):
        count = 0
        if not dataclasses.is_dataclass(obj):
            return 0
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            if isinstance(val, Field):
                count += 1
            elif isinstance(val, list):
                for item in val:
                    count += count_fields(item)
            elif dataclasses.is_dataclass(val):
                count += count_fields(val)
        return count

    n = count_fields(world)
    assert n > 500, f"Expected 500+ fields, got {n}"


def test_compile_full_world():
    """compile_schema should handle the full World with 857 fields."""
    world = World()
    bound = compile_schema(
        world, T=1, H=128, W=128, d_model=64,
        connectivity=ConnectivityPolicy(intra="dense"),
    )
    assert len(bound.field_names) > 800
    assert bound.layout.num_positions > 0
    assert len(bound.topology.connections) > 0


def test_compile_individual_layers():
    """Each layer should compile independently."""
    for obj, min_fields in [
        (ClimateState(), 5),
        (EnergySystem(), 10),
        (YieldCurveState(), 8),
        (MacroEconomy(), 40),
        (PoliticalLayer(), 20),
        (NarrativeBeliefLayer(), 25),
        (TechnologyLayer(), 10),
        (DemographicLayer(), 8),
        (Sector(), 15),
        (SupplyChainNode(), 7),
        (Business(), 40),
        (Individual(), 20),
        (EventTape(), 8),
        (DataChannelTrust(), 14),
        (RegimeState(), 15),
        (InterventionSpace(), 10),
        (ForecastBundle(), 25),
        (Country(), 60),
    ]:
        bound = compile_schema(obj, T=1, H=32, W=32, d_model=32)
        assert len(bound.field_names) >= min_fields, \
            f"{type(obj).__name__}: expected {min_fields}+ fields, got {len(bound.field_names)}"


def test_temporal_periods():
    """Fields should have appropriate temporal periods across timescales."""
    world = World()

    # Fast fields (period=1) — markets, breaking news
    assert world.financial.equities.vix.period == 1
    assert world.events.news_embedding.period == 1

    # Hourly (period=4) — grid load
    assert world.resources.energy.electricity_grid_load.period == 4

    # Daily (period=16) — commodity prices, port congestion
    assert world.physical.infrastructure.port_congestion.period == 16

    # Weekly (period=48) — claims, inventories
    assert world.resources.energy.crude_inventory.period == 48

    # Monthly (period=192) — CPI, PMI
    assert world.country_us.macro.inflation.headline_cpi.period == 192

    # Quarterly (period=576) — earnings, GDP
    assert world.firm_alpha.financials.revenue.period == 576

    # Annual (period=2304) — climate, infrastructure
    assert world.physical.climate.global_temp_anomaly.period == 2304

    # Multi-year (period=4608) — demographics
    assert world.country_us.demographics.population_growth.period == 4608


def test_loss_weights():
    """Critical fields should have higher loss weights."""
    world = World()

    # Regime state has high loss weights
    assert world.regime.growth_regime.loss_weight == 5.0
    assert world.regime.fragility.loss_weight == 5.0
    assert world.regime.black_swan_proximity.loss_weight == 5.0

    # Nuclear risk should be heavily weighted
    assert world.country_us.politics.geopolitics.nuclear_risk.loss_weight == 5.0

    # Firm covenant headroom is critical
    assert world.firm_alpha.financials.covenant_headroom.loss_weight == 3.0

    # Default observability is lower weight
    assert world.trust.trust_census.loss_weight == 1.0


def test_input_only_fields():
    """Some fields should be input-only (is_output=False)."""
    world = World()

    # Identity fields are input-only
    assert world.firm_alpha.identity.is_output is False
    assert world.person_alpha.role.is_output is False
    assert world.country_us.identity.is_output is False

    # Compensation structure is input-only
    assert world.person_alpha.incentives.compensation_structure.is_output is False

    # Output fields should default to True
    assert world.regime.growth_regime.is_output is True
    assert world.financial.equities.vix.is_output is True
