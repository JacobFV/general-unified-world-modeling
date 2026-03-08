"""Layer 12: Individual (τ2–τ5, very sparse).

Psychological decomposition of strategic decision-makers.
Cognitive style, incentives, network position, current state.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class PersonCognitive:
    decision_style:             Field = Field(1, 4, period=QUARTERLY)
    risk_appetite:              Field = Field(1, 2, period=MONTHLY)
    time_horizon:               Field = Field(1, 2, period=QUARTERLY)
    belief_update_speed:        Field = Field(1, 2, period=MONTHLY)
    cognitive_load:             Field = Field(1, 2, period=WEEKLY)
    ideological_priors:         Field = Field(1, 4, period=ANNUAL)


@dataclass
class PersonIncentives:
    compensation_structure:     Field = Field(1, 4, period=QUARTERLY, is_output=False)
    career_incentives:          Field = Field(1, 2, period=QUARTERLY)
    reputation_concerns:        Field = Field(1, 2, period=MONTHLY)
    legal_exposure:             Field = Field(1, 2, period=MONTHLY)
    legacy_concerns:            Field = Field(1, 2, period=QUARTERLY)
    peer_pressure:              Field = Field(1, 2, period=MONTHLY)


@dataclass
class PersonNetwork:
    formal_authority:           Field = Field(1, 4, period=QUARTERLY)
    network_centrality:         Field = Field(1, 2, period=QUARTERLY)
    trusted_advisors:           Field = Field(1, 4, period=QUARTERLY)
    board_relationships:        Field = Field(1, 4, period=QUARTERLY)
    political_connections:      Field = Field(1, 2, period=QUARTERLY)
    media_influence:            Field = Field(1, 2, period=MONTHLY)


@dataclass
class PersonState:
    stress:                     Field = Field(1, 2, period=WEEKLY)
    health_energy:              Field = Field(1, 2, period=MONTHLY)
    confidence:                 Field = Field(1, 2, period=WEEKLY)
    current_focus:              Field = Field(1, 4, period=WEEKLY)
    public_statements_tone:     Field = Field(1, 2, period=DAILY)
    private_info_proxy:         Field = Field(1, 4, period=WEEKLY, loss_weight=3.0)


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
