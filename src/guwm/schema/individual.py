"""Layer 12: Individual (τ2–τ5, very sparse).

Psychological decomposition of strategic decision-makers.
Cognitive style, incentives, network position, current state.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field


@dataclass
class PersonCognitive:
    decision_style:             Field = Field(1, 4, period=576)
    risk_appetite:              Field = Field(1, 2, period=192)
    time_horizon:               Field = Field(1, 2, period=576)
    belief_update_speed:        Field = Field(1, 2, period=192)
    cognitive_load:             Field = Field(1, 2, period=48)
    ideological_priors:         Field = Field(1, 4, period=2304)


@dataclass
class PersonIncentives:
    compensation_structure:     Field = Field(1, 4, period=576, is_output=False)
    career_incentives:          Field = Field(1, 2, period=576)
    reputation_concerns:        Field = Field(1, 2, period=192)
    legal_exposure:             Field = Field(1, 2, period=192)
    legacy_concerns:            Field = Field(1, 2, period=576)
    peer_pressure:              Field = Field(1, 2, period=192)


@dataclass
class PersonNetwork:
    formal_authority:           Field = Field(1, 4, period=576)
    network_centrality:         Field = Field(1, 2, period=576)
    trusted_advisors:           Field = Field(1, 4, period=576)
    board_relationships:        Field = Field(1, 4, period=576)
    political_connections:      Field = Field(1, 2, period=576)
    media_influence:            Field = Field(1, 2, period=192)


@dataclass
class PersonState:
    stress:                     Field = Field(1, 2, period=48)
    health_energy:              Field = Field(1, 2, period=192)
    confidence:                 Field = Field(1, 2, period=48)
    current_focus:              Field = Field(1, 4, period=48)
    public_statements_tone:     Field = Field(1, 2, period=16)
    private_info_proxy:         Field = Field(1, 4, period=48, loss_weight=3.0)


@dataclass
class Individual:
    role:               Field = Field(2, 4, is_output=False)
    org_link:           Field = Field(1, 2, is_output=False)

    cognitive:  PersonCognitive  = dc_field(default_factory=PersonCognitive)
    incentives: PersonIncentives = dc_field(default_factory=PersonIncentives)
    network:    PersonNetwork    = dc_field(default_factory=PersonNetwork)
    state:      PersonState      = dc_field(default_factory=PersonState)

    projected_actions:          Field = Field(2, 4, loss_weight=3.0)
    action_timing:              Field = Field(1, 2, loss_weight=2.0)
    surprise_risk:              Field = Field(1, 2, loss_weight=4.0)
