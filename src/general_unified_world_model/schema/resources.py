"""Layer 2: Resources & Energy (τ1–τ4).

Commodity markets, food systems, water, compute — the physical inputs
to economic production. These bridge geological timescales to market timescales.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field


@dataclass
class EnergySystem:
    crude_price:                Field = Field(1, 2, period=1)
    crude_inventory:            Field = Field(1, 2, period=48)
    crude_production_capacity:  Field = Field(1, 2, period=192)
    natgas_price:               Field = Field(1, 2, period=1)
    natgas_storage:             Field = Field(1, 2, period=48)
    lng_shipping_rates:         Field = Field(1, 2, period=16)
    coal_price:                 Field = Field(1, 1, period=16)
    electricity_grid_load:      Field = Field(1, 4, period=4)
    renewable_generation:       Field = Field(1, 4, period=4)
    refinery_utilization:       Field = Field(1, 2, period=48)
    strategic_reserves:         Field = Field(1, 2, period=192)
    opec_spare_capacity:        Field = Field(1, 2, period=192)
    energy_transition_pace:     Field = Field(1, 2, period=2304)


@dataclass
class MetalsAndMinerals:
    copper:                     Field = Field(1, 2, period=1)
    iron_ore:                   Field = Field(1, 2, period=16)
    lithium:                    Field = Field(1, 2, period=48)
    rare_earths:                Field = Field(1, 2, period=192)
    aluminum:                   Field = Field(1, 1, period=16)
    nickel:                     Field = Field(1, 1, period=16)
    gold:                       Field = Field(1, 2, period=1)
    silver:                     Field = Field(1, 1, period=1)
    mining_capex_cycle:         Field = Field(1, 2, period=576)


@dataclass
class FoodSystem:
    wheat:                      Field = Field(1, 1, period=16)
    corn:                       Field = Field(1, 1, period=16)
    soybean:                    Field = Field(1, 1, period=16)
    rice:                       Field = Field(1, 1, period=48)
    fertilizer_prices:          Field = Field(1, 2, period=48)
    food_price_index:           Field = Field(1, 2, period=192)
    crop_yield_forecast:        Field = Field(1, 4, period=576)
    food_insecurity:            Field = Field(1, 2, period=192)
    arable_land_trend:          Field = Field(1, 1, period=2304)


@dataclass
class WaterStress:
    aquifer_depletion:          Field = Field(1, 2, period=2304)
    drought_index:              Field = Field(1, 4, period=192)
    desalination_capacity:      Field = Field(1, 1, period=2304)
    water_conflict_risk:        Field = Field(1, 2, period=576)


@dataclass
class ComputeSupply:
    gpu_spot_price:             Field = Field(1, 2, period=4)
    datacenter_capacity:        Field = Field(1, 2, period=192)
    fab_utilization:            Field = Field(1, 2, period=192)
    leading_edge_capacity:      Field = Field(1, 2, period=576)
    chip_inventory_days:        Field = Field(1, 2, period=48)
    ai_training_demand:         Field = Field(1, 2, period=48)
    semiconductor_capex:        Field = Field(1, 2, period=576)
    export_control_severity:    Field = Field(1, 2, period=192)


@dataclass
class ResourceLayer:
    energy:   EnergySystem      = dc_field(default_factory=EnergySystem)
    metals:   MetalsAndMinerals = dc_field(default_factory=MetalsAndMinerals)
    food:     FoodSystem        = dc_field(default_factory=FoodSystem)
    water:    WaterStress       = dc_field(default_factory=WaterStress)
    compute:  ComputeSupply     = dc_field(default_factory=ComputeSupply)
