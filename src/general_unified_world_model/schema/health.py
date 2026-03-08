"""Layer: Healthcare Systems (τ3–τ6).

Healthcare capacity, pharmaceutical pipeline, population health metrics
— the systems that determine resilience to disease burden and demographic change.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class HealthcareCapacity:
    hospital_beds_per_capita:   Field = Field(1, 1, period=QUARTERLY)
    icu_utilization:            Field = Field(1, 1, period=WEEKLY, loss_weight=1.5)
    healthcare_worker_shortage: Field = Field(1, 1, period=MONTHLY)
    telehealth_adoption:        Field = Field(1, 1, period=QUARTERLY)
    pharma_pipeline:            Field = Field(1, 2, period=QUARTERLY)


@dataclass
class PublicHealth:
    life_expectancy_trend:      Field = Field(1, 1, period=ANNUAL, loss_weight=2.0)
    obesity_rate:               Field = Field(1, 1, period=ANNUAL)
    mental_health_crisis:       Field = Field(1, 1, period=MONTHLY, loss_weight=1.5)
    substance_abuse_trend:      Field = Field(1, 1, period=QUARTERLY)
    health_inequality:          Field = Field(1, 2, period=ANNUAL, loss_weight=1.5)


@dataclass
class HealthLayer:
    capacity:       HealthcareCapacity = dc_field(default_factory=HealthcareCapacity)
    public_health:  PublicHealth       = dc_field(default_factory=PublicHealth)
