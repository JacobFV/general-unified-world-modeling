"""Layer 8: Demographics (τ7, slowest structural force).

Population, dependency, urbanization, fertility, migration.
These constrain everything on multi-decade timescales.
"""

from dataclasses import dataclass
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    ANNUAL, DECADAL,
)


@dataclass
class DemographicLayer:
    population_growth:          Field = Field(1, 2, period=DECADAL)
    dependency_ratio:           Field = Field(1, 2, period=DECADAL)
    urbanization:               Field = Field(1, 2, period=DECADAL)
    median_age:                 Field = Field(1, 2, period=DECADAL)
    fertility_rate:             Field = Field(1, 2, period=DECADAL)
    life_expectancy:            Field = Field(1, 2, period=DECADAL)
    net_migration:              Field = Field(1, 2, period=ANNUAL)
    education_attainment:       Field = Field(1, 2, period=DECADAL)
    human_capital_index:        Field = Field(1, 2, period=DECADAL)
    working_age_growth:         Field = Field(1, 2, period=DECADAL, loss_weight=2.0)
