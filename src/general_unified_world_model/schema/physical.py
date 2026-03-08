"""Layer 1: Planetary Physical Substrate (τ6–τ7).

The slowest structural forces: climate, geography, natural disasters.
These constrain everything above them but change on multi-year timescales.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    TICK, DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL, DECADAL,
)


@dataclass
class ClimateState:
    global_temp_anomaly:        Field = Field(1, 2, period=ANNUAL)
    enso_phase:                 Field = Field(1, 2, period=QUARTERLY)
    monsoon_state:              Field = Field(1, 2, period=QUARTERLY)
    polar_vortex_stability:     Field = Field(1, 2, period=QUARTERLY)
    extreme_weather_freq:       Field = Field(1, 2, period=MONTHLY)
    sea_level_trend:            Field = Field(1, 1, period=DECADAL)
    carbon_ppm:                 Field = Field(1, 1, period=ANNUAL)


@dataclass
class GeographicInfrastructure:
    shipping_lane_capacity:     Field = Field(2, 4, period=ANNUAL)
    chokepoint_risk:            Field = Field(1, 4, period=MONTHLY)      # Suez/Hormuz/Malacca/Panama
    undersea_cable_topology:    Field = Field(1, 2, period=ANNUAL)
    rail_freight_network:       Field = Field(1, 2, period=ANNUAL)
    port_congestion:            Field = Field(1, 4, period=DAILY)
    air_freight_utilization:    Field = Field(1, 2, period=DAILY)


@dataclass
class DisasterLayer:
    seismic_risk_structural:    Field = Field(1, 2, period=ANNUAL)
    active_disaster_state:      Field = Field(1, 4, period=TICK)
    pandemic_risk:              Field = Field(1, 2, period=WEEKLY)
    volcanic_risk:              Field = Field(1, 1, period=QUARTERLY)
    wildfire_state:             Field = Field(1, 2, period=DAILY)


@dataclass
class PlanetaryPhysicalLayer:
    climate:        ClimateState             = dc_field(default_factory=ClimateState)
    infrastructure: GeographicInfrastructure = dc_field(default_factory=GeographicInfrastructure)
    disasters:      DisasterLayer            = dc_field(default_factory=DisasterLayer)
