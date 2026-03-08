"""Layer: Space & Orbital Systems (τ3–τ6).

The orbital environment, launch economics, and the emerging space economy
— increasingly relevant as LEO becomes a contested commons.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class OrbitalEnvironment:
    active_satellites:          Field = Field(1, 2, period=MONTHLY)
    debris_density:             Field = Field(1, 1, period=QUARTERLY)
    collision_risk:             Field = Field(1, 1, period=WEEKLY, loss_weight=2.0)
    orbit_congestion_leo:       Field = Field(1, 1, period=MONTHLY)


@dataclass
class SpaceEconomy:
    launch_cost:                Field = Field(1, 2, period=QUARTERLY)
    commercial_space_revenue:   Field = Field(1, 2, period=QUARTERLY)
    space_tourism:              Field = Field(1, 1, period=ANNUAL)
    satellite_internet_subscribers: Field = Field(1, 1, period=QUARTERLY)
    space_mining_progress:      Field = Field(1, 1, period=ANNUAL)


@dataclass
class SpaceLayer:
    orbital:  OrbitalEnvironment = dc_field(default_factory=OrbitalEnvironment)
    economy:  SpaceEconomy       = dc_field(default_factory=SpaceEconomy)
