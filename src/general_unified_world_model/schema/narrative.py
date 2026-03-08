"""Layer 6: Narrative, Belief & Expectations (τ0–τ4).

Reflexivity layer. Beliefs move markets, markets move beliefs.
Media narratives, elite consensus, public sentiment, investor positioning.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field


@dataclass
class MediaNarratives:
    crisis_framing:             Field = Field(1, 2, period=4)
    econ_doom_vs_boom:          Field = Field(1, 2, period=16)
    geopolitical_fear:          Field = Field(1, 2, period=16)
    tech_optimism:              Field = Field(1, 2, period=48)
    inequality_salience:        Field = Field(1, 2, period=48)
    climate_urgency:            Field = Field(1, 2, period=192)
    media_fragmentation:        Field = Field(1, 2, period=2304)
    info_ecosystem_health:      Field = Field(1, 2, period=2304)


@dataclass
class EliteConsensus:
    davos_consensus:            Field = Field(1, 2, period=576)
    cb_hawkishness:             Field = Field(1, 4, period=192)
    ceo_confidence:             Field = Field(1, 2, period=192)
    vc_risk_appetite:           Field = Field(1, 2, period=192)
    techno_optimism:            Field = Field(1, 2, period=192)
    deglobalization_belief:     Field = Field(1, 2, period=576)
    ai_xrisk_belief:            Field = Field(1, 2, period=576)


@dataclass
class PublicSentiment:
    consumer_confidence:        Field = Field(1, 2, period=192)
    economic_anxiety:           Field = Field(1, 2, period=48)
    institutional_trust:        Field = Field(1, 4, period=576)
    polarization:               Field = Field(1, 2, period=576)
    social_unrest_risk:         Field = Field(1, 2, period=48, loss_weight=2.0)
    migration_pressure:         Field = Field(1, 2, period=192)
    birth_rate_sentiment:       Field = Field(1, 2, period=2304)
    techno_anxiety:             Field = Field(1, 2, period=192)


@dataclass
class InvestorPositioning:
    equity_fund_flows:          Field = Field(1, 2, period=48)
    bond_fund_flows:            Field = Field(1, 2, period=48)
    mm_fund_flows:              Field = Field(1, 2, period=48)
    hedge_fund_leverage:        Field = Field(1, 2, period=192)
    cta_signal:                 Field = Field(1, 2, period=16)
    retail_sentiment:           Field = Field(1, 2, period=16)
    institutional_rebalancing:  Field = Field(1, 2, period=576)
    crowdedness_risk:           Field = Field(1, 4, period=48, loss_weight=1.5)
    short_interest:             Field = Field(1, 2, period=48)


@dataclass
class NarrativeBeliefLayer:
    media:       MediaNarratives     = dc_field(default_factory=MediaNarratives)
    elites:      EliteConsensus      = dc_field(default_factory=EliteConsensus)
    public:      PublicSentiment     = dc_field(default_factory=PublicSentiment)
    positioning: InvestorPositioning = dc_field(default_factory=InvestorPositioning)
