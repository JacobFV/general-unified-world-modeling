"""Layer: Education & Human Capital (τ4–τ6).

Education systems, skill pipelines, workforce dynamics — the slow-moving
human capital substrate that determines long-run productivity and adaptability.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class EducationSystem:
    enrollment_rate:            Field = Field(1, 1, period=ANNUAL)
    stem_graduation:            Field = Field(1, 1, period=ANNUAL)
    skill_gap_index:            Field = Field(1, 2, period=QUARTERLY, loss_weight=1.5)
    online_learning_penetration: Field = Field(1, 1, period=QUARTERLY)
    research_output:            Field = Field(1, 2, period=QUARTERLY)
    university_funding:         Field = Field(1, 1, period=ANNUAL)


@dataclass
class WorkforceDevelopment:
    retraining_programs:        Field = Field(1, 1, period=QUARTERLY)
    apprenticeship_rate:        Field = Field(1, 1, period=ANNUAL)
    remote_work_share:          Field = Field(1, 1, period=MONTHLY)
    gig_economy_share:          Field = Field(1, 1, period=QUARTERLY)
    labor_mobility:             Field = Field(1, 1, period=QUARTERLY)


@dataclass
class EducationLayer:
    education: EducationSystem     = dc_field(default_factory=EducationSystem)
    workforce: WorkforceDevelopment = dc_field(default_factory=WorkforceDevelopment)
