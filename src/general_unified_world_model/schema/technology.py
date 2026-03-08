"""Layer 7: Technology & Innovation (τ5–τ7).

AI capability frontier, biotech, quantum, robotics, fusion, productivity.
The long-run structural drivers of growth and disruption.
"""

from dataclasses import dataclass
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class TechnologyLayer:
    ai_capability_frontier:     Field = Field(1, 4, period=MONTHLY, loss_weight=3.0)
    ai_adoption:                Field = Field(1, 4, period=QUARTERLY)
    ai_safety_governance:       Field = Field(1, 2, period=QUARTERLY)
    ai_compute_scaling:         Field = Field(1, 2, period=MONTHLY)
    biotech_frontier:           Field = Field(1, 2, period=QUARTERLY)
    quantum_progress:           Field = Field(1, 2, period=ANNUAL)
    robotics_deployment:        Field = Field(1, 2, period=QUARTERLY)
    fusion_progress:            Field = Field(1, 1, period=ANNUAL)
    space_commercialization:    Field = Field(1, 2, period=QUARTERLY)
    productivity_growth:        Field = Field(1, 2, period=ANNUAL, loss_weight=2.0)
    automation_displacement:    Field = Field(1, 2, period=QUARTERLY)
    patent_activity:            Field = Field(1, 2, period=QUARTERLY)
    global_r_and_d:             Field = Field(1, 2, period=ANNUAL)
