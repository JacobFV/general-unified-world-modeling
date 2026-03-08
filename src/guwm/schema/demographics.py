"""Layer 8: Demographics (τ7, slowest structural force).

Population, dependency, urbanization, fertility, migration.
These constrain everything on multi-decade timescales.
"""

from dataclasses import dataclass
from canvas_engineering import Field


@dataclass
class DemographicLayer:
    population_growth:          Field = Field(1, 2, period=4608)
    dependency_ratio:           Field = Field(1, 2, period=4608)
    urbanization:               Field = Field(1, 2, period=4608)
    median_age:                 Field = Field(1, 2, period=4608)
    fertility_rate:             Field = Field(1, 2, period=4608)
    life_expectancy:            Field = Field(1, 2, period=4608)
    net_migration:              Field = Field(1, 2, period=2304)
    education_attainment:       Field = Field(1, 2, period=4608)
    human_capital_index:        Field = Field(1, 2, period=4608)
    working_age_growth:         Field = Field(1, 2, period=4608, loss_weight=2.0)
