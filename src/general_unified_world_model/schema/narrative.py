"""Layer 6: Narrative, Belief & Expectations (τ0–τ4).

Reflexivity layer. Beliefs move markets, markets move beliefs.
Media narratives, elite consensus, public sentiment, investor positioning.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    HOURLY, DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class MediaNarratives:
    crisis_framing:             Field = Field(1, 2, period=HOURLY)
    econ_doom_vs_boom:          Field = Field(1, 2, period=DAILY)
    geopolitical_fear:          Field = Field(1, 2, period=DAILY)
    tech_optimism:              Field = Field(1, 2, period=WEEKLY)
    inequality_salience:        Field = Field(1, 2, period=WEEKLY)
    climate_urgency:            Field = Field(1, 2, period=MONTHLY)
    media_fragmentation:        Field = Field(1, 2, period=ANNUAL)
    info_ecosystem_health:      Field = Field(1, 2, period=ANNUAL)


@dataclass
class EliteConsensus:
    davos_consensus:            Field = Field(1, 2, period=QUARTERLY)
    cb_hawkishness:             Field = Field(1, 4, period=MONTHLY)
    ceo_confidence:             Field = Field(1, 2, period=MONTHLY)
    vc_risk_appetite:           Field = Field(1, 2, period=MONTHLY)
    techno_optimism:            Field = Field(1, 2, period=MONTHLY)
    deglobalization_belief:     Field = Field(1, 2, period=QUARTERLY)
    ai_xrisk_belief:            Field = Field(1, 2, period=QUARTERLY)


@dataclass
class PublicSentiment:
    consumer_confidence:        Field = Field(1, 2, period=MONTHLY)
    economic_anxiety:           Field = Field(1, 2, period=WEEKLY)
    institutional_trust:        Field = Field(1, 4, period=QUARTERLY)
    polarization:               Field = Field(1, 2, period=QUARTERLY)
    social_unrest_risk:         Field = Field(1, 2, period=WEEKLY, loss_weight=2.0)
    migration_pressure:         Field = Field(1, 2, period=MONTHLY)
    birth_rate_sentiment:       Field = Field(1, 2, period=ANNUAL)
    techno_anxiety:             Field = Field(1, 2, period=MONTHLY)


@dataclass
class InvestorPositioning:
    equity_fund_flows:          Field = Field(1, 2, period=WEEKLY)
    bond_fund_flows:            Field = Field(1, 2, period=WEEKLY)
    mm_fund_flows:              Field = Field(1, 2, period=WEEKLY)
    hedge_fund_leverage:        Field = Field(1, 2, period=MONTHLY)
    cta_signal:                 Field = Field(1, 2, period=DAILY)
    retail_sentiment:           Field = Field(1, 2, period=DAILY)
    institutional_rebalancing:  Field = Field(1, 2, period=QUARTERLY)
    crowdedness_risk:           Field = Field(1, 4, period=WEEKLY, loss_weight=1.5)
    short_interest:             Field = Field(1, 2, period=WEEKLY)


@dataclass
class NarrativeBeliefLayer:
    media:       MediaNarratives     = dc_field(default_factory=MediaNarratives)
    elites:      EliteConsensus      = dc_field(default_factory=EliteConsensus)
    public:      PublicSentiment     = dc_field(default_factory=PublicSentiment)
    positioning: InvestorPositioning = dc_field(default_factory=InvestorPositioning)
