"""Layer: Physical & Digital Infrastructure (τ1–τ6).

Power grids, transport networks, telecoms, urban systems — the built
environment that enables economic activity and shapes spatial constraints.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    HOURLY, DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class PowerGrid:
    generation_capacity:        Field = Field(1, 2, period=MONTHLY)
    grid_reliability:           Field = Field(1, 1, period=DAILY, loss_weight=1.5)
    renewable_penetration:      Field = Field(1, 2, period=QUARTERLY)
    storage_capacity:           Field = Field(1, 1, period=QUARTERLY)
    peak_demand:                Field = Field(1, 2, period=HOURLY)
    transmission_congestion:    Field = Field(1, 1, period=DAILY)
    blackout_risk:              Field = Field(1, 1, period=WEEKLY, loss_weight=2.0)


@dataclass
class TransportNetwork:
    road_congestion:            Field = Field(1, 2, period=HOURLY)
    rail_utilization:           Field = Field(1, 1, period=DAILY)
    aviation_load:              Field = Field(1, 2, period=DAILY)
    port_throughput:            Field = Field(1, 2, period=WEEKLY)
    last_mile_cost:             Field = Field(1, 1, period=MONTHLY)
    ev_adoption:                Field = Field(1, 1, period=QUARTERLY)
    autonomous_vehicle_deployment: Field = Field(1, 1, period=ANNUAL)


@dataclass
class TelecomNetwork:
    bandwidth_demand:           Field = Field(1, 2, period=HOURLY)
    fiber_coverage:             Field = Field(1, 1, period=QUARTERLY)
    satellite_constellation_capacity: Field = Field(1, 1, period=QUARTERLY)
    spectrum_allocation:        Field = Field(1, 2, period=ANNUAL)
    latency_critical_services:  Field = Field(1, 1, period=DAILY)
    fiveg_coverage:             Field = Field(1, 1, period=QUARTERLY)


@dataclass
class UrbanSystems:
    housing_inventory:          Field = Field(1, 2, period=MONTHLY)
    construction_permits:       Field = Field(1, 1, period=MONTHLY)
    commercial_vacancy:         Field = Field(1, 1, period=MONTHLY)
    urban_density_trend:        Field = Field(1, 1, period=ANNUAL)
    smart_city_index:           Field = Field(1, 2, period=ANNUAL)
    public_transit_utilization: Field = Field(1, 1, period=DAILY)


@dataclass
class InfrastructureLayer:
    power:     PowerGrid        = dc_field(default_factory=PowerGrid)
    transport: TransportNetwork = dc_field(default_factory=TransportNetwork)
    telecom:   TelecomNetwork   = dc_field(default_factory=TelecomNetwork)
    urban:     UrbanSystems     = dc_field(default_factory=UrbanSystems)
