"""Layer 18: Country (composite, one per major economy).

Bundles macro economy, politics, demographics with country-specific
market data, sentiment, and risk scores.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from guwm.schema.macro import MacroEconomy
from guwm.schema.political import PoliticalLayer
from guwm.schema.demographics import DemographicLayer


@dataclass
class Country:
    macro:          MacroEconomy     = dc_field(default_factory=MacroEconomy)
    politics:       PoliticalLayer   = dc_field(default_factory=PoliticalLayer)
    demographics:   DemographicLayer = dc_field(default_factory=DemographicLayer)

    domestic_sentiment:         Field = Field(1, 4, period=192)
    domestic_media_tone:        Field = Field(1, 2, period=48)
    sovereign_yield_curve:      Field = Field(1, 4, period=1)
    sovereign_cds:              Field = Field(1, 2, period=1)
    local_equity_index:         Field = Field(1, 2, period=1)
    banking_system_health:      Field = Field(1, 4, period=192)

    identity:                   Field = Field(2, 4, is_output=False)
    trade_partners:             Field = Field(2, 4, is_output=False)
    alliance_membership:        Field = Field(1, 2, is_output=False)

    national_risk_score:        Field = Field(2, 4, loss_weight=3.0)
    contagion_vulnerability:    Field = Field(1, 2, loss_weight=3.0)
