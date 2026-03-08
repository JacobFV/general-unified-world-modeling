"""Layer: Legal & Regulatory Framework (τ5–τ6).

Regulatory intensity, judicial quality, corruption, intellectual property
— the institutional scaffolding that shapes incentives and transaction costs.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    QUARTERLY, ANNUAL,
)


@dataclass
class RegulatoryEnvironment:
    regulatory_burden_index:    Field = Field(1, 2, period=QUARTERLY)
    antitrust_enforcement:      Field = Field(1, 1, period=QUARTERLY)
    ip_protection_strength:     Field = Field(1, 1, period=ANNUAL)
    environmental_regulation:   Field = Field(1, 1, period=QUARTERLY, loss_weight=1.5)
    financial_regulation_stringency: Field = Field(1, 1, period=QUARTERLY)
    data_privacy_regulation:    Field = Field(1, 1, period=QUARTERLY)


@dataclass
class LegalSystem:
    judicial_independence:      Field = Field(1, 1, period=ANNUAL, loss_weight=2.0)
    contract_enforcement:       Field = Field(1, 1, period=ANNUAL)
    corruption_index:           Field = Field(1, 2, period=ANNUAL, loss_weight=2.0)
    rule_of_law_index:          Field = Field(1, 2, period=ANNUAL, loss_weight=2.0)
    international_arbitration:  Field = Field(1, 1, period=QUARTERLY)


@dataclass
class LegalLayer:
    regulatory: RegulatoryEnvironment = dc_field(default_factory=RegulatoryEnvironment)
    legal:      LegalSystem           = dc_field(default_factory=LegalSystem)
