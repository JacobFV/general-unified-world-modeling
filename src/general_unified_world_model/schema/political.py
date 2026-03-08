"""Layer 5: Political & Institutional (τ4–τ7).

Executive, legislative, judicial, geopolitical, institutional quality.
These are the governance structures that constrain economic and social outcomes.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class ExecutiveState:
    approval_rating:            Field = Field(1, 2, period=WEEKLY)
    political_capital:          Field = Field(1, 2, period=MONTHLY)
    executive_coherence:        Field = Field(1, 2, period=MONTHLY)
    election_proximity:         Field = Field(1, 1, period=WEEKLY, loss_weight=2.0)
    lame_duck_index:            Field = Field(1, 1, period=MONTHLY)
    cabinet_stability:          Field = Field(1, 2, period=MONTHLY)


@dataclass
class LegislativeState:
    gridlock:                   Field = Field(1, 2, period=MONTHLY)
    majority_margin:            Field = Field(1, 2, period=QUARTERLY)
    bipartisan_capacity:        Field = Field(1, 2, period=QUARTERLY)
    pending_legislation_risk:   Field = Field(1, 4, period=WEEKLY)
    regulatory_pipeline:        Field = Field(1, 4, period=MONTHLY)


@dataclass
class JudicialState:
    independence:               Field = Field(1, 2, period=ANNUAL)
    pending_landmark_cases:     Field = Field(1, 2, period=MONTHLY)
    regulatory_uncertainty:     Field = Field(1, 2, period=MONTHLY, loss_weight=1.5)


@dataclass
class GeopoliticalState:
    conflict_risk:              Field = Field(2, 4, period=DAILY, loss_weight=3.0)
    alliance_cohesion:          Field = Field(1, 4, period=QUARTERLY)
    great_power_tension:        Field = Field(1, 4, period=WEEKLY, loss_weight=2.0)
    nuclear_risk:               Field = Field(1, 2, period=MONTHLY, loss_weight=5.0)
    sanctions_regime:           Field = Field(1, 4, period=WEEKLY)
    arms_trade:                 Field = Field(1, 2, period=QUARTERLY)
    territorial_disputes:       Field = Field(1, 4, period=QUARTERLY)
    cyber_conflict:             Field = Field(1, 2, period=WEEKLY)
    space_competition:          Field = Field(1, 2, period=QUARTERLY)
    economic_coercion:          Field = Field(1, 2, period=WEEKLY)


@dataclass
class InstitutionalQuality:
    rule_of_law:                Field = Field(1, 2, period=ANNUAL)
    corruption:                 Field = Field(1, 2, period=ANNUAL)
    state_capacity:             Field = Field(1, 2, period=ANNUAL)
    property_rights:            Field = Field(1, 2, period=ANNUAL)
    press_freedom:              Field = Field(1, 2, period=ANNUAL)
    democratic_backsliding:     Field = Field(1, 2, period=ANNUAL, loss_weight=2.0)
    social_trust:               Field = Field(1, 2, period=ANNUAL)


@dataclass
class PoliticalLayer:
    executive:    ExecutiveState       = dc_field(default_factory=ExecutiveState)
    legislative:  LegislativeState     = dc_field(default_factory=LegislativeState)
    judicial:     JudicialState        = dc_field(default_factory=JudicialState)
    geopolitics:  GeopoliticalState    = dc_field(default_factory=GeopoliticalState)
    institutions: InstitutionalQuality = dc_field(default_factory=InstitutionalQuality)
