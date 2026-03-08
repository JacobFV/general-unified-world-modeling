"""Layer 2: Resources & Energy (τ1–τ4).

Commodity markets, food systems, water, compute — the physical inputs
to economic production. These bridge geological timescales to market timescales.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    TICK, HOURLY, DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class EnergySystem:
    crude_price:                Field = Field(1, 2, period=TICK)
    crude_inventory:            Field = Field(1, 2, period=WEEKLY)
    crude_production_capacity:  Field = Field(1, 2, period=MONTHLY)
    natgas_price:               Field = Field(1, 2, period=TICK)
    natgas_storage:             Field = Field(1, 2, period=WEEKLY)
    lng_shipping_rates:         Field = Field(1, 2, period=DAILY)
    coal_price:                 Field = Field(1, 1, period=DAILY)
    electricity_grid_load:      Field = Field(1, 4, period=HOURLY)
    renewable_generation:       Field = Field(1, 4, period=HOURLY)
    refinery_utilization:       Field = Field(1, 2, period=WEEKLY)
    strategic_reserves:         Field = Field(1, 2, period=MONTHLY)
    opec_spare_capacity:        Field = Field(1, 2, period=MONTHLY)
    energy_transition_pace:     Field = Field(1, 2, period=ANNUAL)


@dataclass
class MetalsAndMinerals:
    copper:                     Field = Field(1, 2, period=TICK)
    iron_ore:                   Field = Field(1, 2, period=DAILY)
    lithium:                    Field = Field(1, 2, period=WEEKLY)
    rare_earths:                Field = Field(1, 2, period=MONTHLY)
    aluminum:                   Field = Field(1, 1, period=DAILY)
    nickel:                     Field = Field(1, 1, period=DAILY)
    gold:                       Field = Field(1, 2, period=TICK)
    silver:                     Field = Field(1, 1, period=TICK)
    mining_capex_cycle:         Field = Field(1, 2, period=QUARTERLY)


@dataclass
class FoodSystem:
    wheat:                      Field = Field(1, 1, period=DAILY)
    corn:                       Field = Field(1, 1, period=DAILY)
    soybean:                    Field = Field(1, 1, period=DAILY)
    rice:                       Field = Field(1, 1, period=WEEKLY)
    fertilizer_prices:          Field = Field(1, 2, period=WEEKLY)
    food_price_index:           Field = Field(1, 2, period=MONTHLY)
    crop_yield_forecast:        Field = Field(1, 4, period=QUARTERLY)
    food_insecurity:            Field = Field(1, 2, period=MONTHLY)
    arable_land_trend:          Field = Field(1, 1, period=ANNUAL)


@dataclass
class WaterStress:
    aquifer_depletion:          Field = Field(1, 2, period=ANNUAL)
    drought_index:              Field = Field(1, 4, period=MONTHLY)
    desalination_capacity:      Field = Field(1, 1, period=ANNUAL)
    water_conflict_risk:        Field = Field(1, 2, period=QUARTERLY)


@dataclass
class ComputeSupply:
    gpu_spot_price:             Field = Field(1, 2, period=HOURLY)
    datacenter_capacity:        Field = Field(1, 2, period=MONTHLY)
    fab_utilization:            Field = Field(1, 2, period=MONTHLY)
    leading_edge_capacity:      Field = Field(1, 2, period=QUARTERLY)
    chip_inventory_days:        Field = Field(1, 2, period=WEEKLY)
    ai_training_demand:         Field = Field(1, 2, period=WEEKLY)
    semiconductor_capex:        Field = Field(1, 2, period=QUARTERLY)
    export_control_severity:    Field = Field(1, 2, period=MONTHLY)


@dataclass
class ResourceLayer:
    energy:   EnergySystem      = dc_field(default_factory=EnergySystem)
    metals:   MetalsAndMinerals = dc_field(default_factory=MetalsAndMinerals)
    food:     FoodSystem        = dc_field(default_factory=FoodSystem)
    water:    WaterStress       = dc_field(default_factory=WaterStress)
    compute:  ComputeSupply     = dc_field(default_factory=ComputeSupply)
