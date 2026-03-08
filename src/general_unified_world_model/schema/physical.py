"""Layer 1: Planetary Physical Substrate (τ6–τ7).

The slowest structural forces: climate, geography, natural disasters.
These constrain everything above them but change on multi-year timescales.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field


@dataclass
class ClimateState:
    global_temp_anomaly:        Field = Field(1, 2, period=2304)
    enso_phase:                 Field = Field(1, 2, period=576)
    monsoon_state:              Field = Field(1, 2, period=576)
    polar_vortex_stability:     Field = Field(1, 2, period=576)
    extreme_weather_freq:       Field = Field(1, 2, period=192)
    sea_level_trend:            Field = Field(1, 1, period=4608)
    carbon_ppm:                 Field = Field(1, 1, period=2304)


@dataclass
class GeographicInfrastructure:
    shipping_lane_capacity:     Field = Field(2, 4, period=2304)
    chokepoint_risk:            Field = Field(1, 4, period=192)      # Suez/Hormuz/Malacca/Panama
    undersea_cable_topology:    Field = Field(1, 2, period=2304)
    rail_freight_network:       Field = Field(1, 2, period=2304)
    port_congestion:            Field = Field(1, 4, period=16)
    air_freight_utilization:    Field = Field(1, 2, period=16)


@dataclass
class DisasterLayer:
    seismic_risk_structural:    Field = Field(1, 2, period=2304)
    active_disaster_state:      Field = Field(1, 4, period=1)
    pandemic_risk:              Field = Field(1, 2, period=48)
    volcanic_risk:              Field = Field(1, 1, period=576)
    wildfire_state:             Field = Field(1, 2, period=16)


@dataclass
class PlanetaryPhysicalLayer:
    climate:        ClimateState             = dc_field(default_factory=ClimateState)
    infrastructure: GeographicInfrastructure = dc_field(default_factory=GeographicInfrastructure)
    disasters:      DisasterLayer            = dc_field(default_factory=DisasterLayer)
