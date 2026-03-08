"""Layer: Biological & Ecological Systems (τ3–τ6).

Biodiversity, disease dynamics, agricultural biology — the living substrate
that underpins food systems, public health, and planetary resilience.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class EcosystemState:
    biodiversity_index:         Field = Field(1, 2, period=ANNUAL, loss_weight=2.0)
    deforestation_rate:         Field = Field(1, 1, period=QUARTERLY)
    ocean_acidification:        Field = Field(1, 1, period=ANNUAL)
    coral_reef_health:          Field = Field(1, 1, period=QUARTERLY)
    fish_stock_status:          Field = Field(1, 2, period=QUARTERLY)
    pollinator_decline:         Field = Field(1, 1, period=ANNUAL, loss_weight=1.5)


@dataclass
class DiseaseState:
    pandemic_readiness:         Field = Field(1, 2, period=MONTHLY, loss_weight=2.0)
    novel_pathogen_risk:        Field = Field(1, 1, period=WEEKLY, loss_weight=2.0)
    vaccine_pipeline:           Field = Field(1, 2, period=QUARTERLY)
    antimicrobial_resistance:   Field = Field(1, 1, period=ANNUAL, loss_weight=1.5)
    zoonotic_spillover_risk:    Field = Field(1, 1, period=QUARTERLY, loss_weight=1.5)


@dataclass
class AgriculturalBiology:
    crop_disease_pressure:      Field = Field(1, 1, period=MONTHLY)
    soil_health_index:          Field = Field(1, 2, period=ANNUAL)
    seed_technology:            Field = Field(1, 1, period=QUARTERLY)
    livestock_health:           Field = Field(1, 1, period=MONTHLY)
    aquaculture_output:         Field = Field(1, 1, period=QUARTERLY)


@dataclass
class BiologicalLayer:
    ecosystem:   EcosystemState      = dc_field(default_factory=EcosystemState)
    disease:     DiseaseState        = dc_field(default_factory=DiseaseState)
    agriculture: AgriculturalBiology = dc_field(default_factory=AgriculturalBiology)
