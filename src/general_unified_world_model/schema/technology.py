"""Layer 7: Technology & Innovation (τ5–τ7).

AI capability frontier, biotech, quantum, robotics, fusion, productivity.
The long-run structural drivers of growth and disruption.
"""

from dataclasses import dataclass
from canvas_engineering import Field


@dataclass
class TechnologyLayer:
    ai_capability_frontier:     Field = Field(1, 4, period=192, loss_weight=3.0)
    ai_adoption:                Field = Field(1, 4, period=576)
    ai_safety_governance:       Field = Field(1, 2, period=576)
    ai_compute_scaling:         Field = Field(1, 2, period=192)
    biotech_frontier:           Field = Field(1, 2, period=576)
    quantum_progress:           Field = Field(1, 2, period=2304)
    robotics_deployment:        Field = Field(1, 2, period=576)
    fusion_progress:            Field = Field(1, 1, period=2304)
    space_commercialization:    Field = Field(1, 2, period=576)
    productivity_growth:        Field = Field(1, 2, period=2304, loss_weight=2.0)
    automation_displacement:    Field = Field(1, 2, period=576)
    patent_activity:            Field = Field(1, 2, period=576)
    global_r_and_d:             Field = Field(1, 2, period=2304)
