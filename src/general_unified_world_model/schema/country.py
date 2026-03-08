"""Layer 18: Country (composite, one per major economy).

Bundles macro economy, politics, demographics with country-specific
market data, sentiment, and risk scores.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.macro import MacroEconomy
from general_unified_world_model.schema.political import PoliticalLayer
from general_unified_world_model.schema.demographics import DemographicLayer
from general_unified_world_model.schema.temporal_constants import (
    TICK, WEEKLY, MONTHLY,
)


@dataclass
class Country:
    macro:          MacroEconomy     = dc_field(default_factory=MacroEconomy)
    politics:       PoliticalLayer   = dc_field(default_factory=PoliticalLayer)
    demographics:   DemographicLayer = dc_field(default_factory=DemographicLayer)

    domestic_sentiment:         Field = Field(1, 4, period=MONTHLY)
    domestic_media_tone:        Field = Field(1, 2, period=WEEKLY)
    sovereign_yield_curve:      Field = Field(1, 4, period=TICK)
    sovereign_cds:              Field = Field(1, 2, period=TICK)
    local_equity_index:         Field = Field(1, 2, period=TICK)
    banking_system_health:      Field = Field(1, 4, period=MONTHLY)

    identity:                   Field = Field(2, 4, is_output=False)
    trade_partners:             Field = Field(2, 4, is_output=False)
    alliance_membership:        Field = Field(1, 2, is_output=False)

    national_risk_score:        Field = Field(2, 4, loss_weight=3.0)
    contagion_vulnerability:    Field = Field(1, 2, loss_weight=3.0)
