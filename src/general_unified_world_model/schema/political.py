"""Layer 5: Political & Institutional (τ4–τ7).

Executive, legislative, judicial, geopolitical, institutional quality.
These are the governance structures that constrain economic and social outcomes.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field


@dataclass
class ExecutiveState:
    approval_rating:            Field = Field(1, 2, period=48)
    political_capital:          Field = Field(1, 2, period=192)
    executive_coherence:        Field = Field(1, 2, period=192)
    election_proximity:         Field = Field(1, 1, period=48, loss_weight=2.0)
    lame_duck_index:            Field = Field(1, 1, period=192)
    cabinet_stability:          Field = Field(1, 2, period=192)


@dataclass
class LegislativeState:
    gridlock:                   Field = Field(1, 2, period=192)
    majority_margin:            Field = Field(1, 2, period=576)
    bipartisan_capacity:        Field = Field(1, 2, period=576)
    pending_legislation_risk:   Field = Field(1, 4, period=48)
    regulatory_pipeline:        Field = Field(1, 4, period=192)


@dataclass
class JudicialState:
    independence:               Field = Field(1, 2, period=2304)
    pending_landmark_cases:     Field = Field(1, 2, period=192)
    regulatory_uncertainty:     Field = Field(1, 2, period=192, loss_weight=1.5)


@dataclass
class GeopoliticalState:
    conflict_risk:              Field = Field(2, 4, period=16, loss_weight=3.0)
    alliance_cohesion:          Field = Field(1, 4, period=576)
    great_power_tension:        Field = Field(1, 4, period=48, loss_weight=2.0)
    nuclear_risk:               Field = Field(1, 2, period=192, loss_weight=5.0)
    sanctions_regime:           Field = Field(1, 4, period=48)
    arms_trade:                 Field = Field(1, 2, period=576)
    territorial_disputes:       Field = Field(1, 4, period=576)
    cyber_conflict:             Field = Field(1, 2, period=48)
    space_competition:          Field = Field(1, 2, period=576)
    economic_coercion:          Field = Field(1, 2, period=48)


@dataclass
class InstitutionalQuality:
    rule_of_law:                Field = Field(1, 2, period=2304)
    corruption:                 Field = Field(1, 2, period=2304)
    state_capacity:             Field = Field(1, 2, period=2304)
    property_rights:            Field = Field(1, 2, period=2304)
    press_freedom:              Field = Field(1, 2, period=2304)
    democratic_backsliding:     Field = Field(1, 2, period=2304, loss_weight=2.0)
    social_trust:               Field = Field(1, 2, period=2304)


@dataclass
class PoliticalLayer:
    executive:    ExecutiveState       = dc_field(default_factory=ExecutiveState)
    legislative:  LegislativeState     = dc_field(default_factory=LegislativeState)
    judicial:     JudicialState        = dc_field(default_factory=JudicialState)
    geopolitics:  GeopoliticalState    = dc_field(default_factory=GeopoliticalState)
    institutions: InstitutionalQuality = dc_field(default_factory=InstitutionalQuality)
