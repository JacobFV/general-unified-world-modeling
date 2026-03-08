"""Layer 14: Data Channel Trust (meta-epistemic calibration).

How much to trust each data source. Government statistics, market data,
alternative data, corporate disclosures — each has its own reliability profile.
"""

from dataclasses import dataclass
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class DataChannelTrust:
    # Government statistics
    trust_bls:                  Field = Field(1, 2, period=QUARTERLY, loss_weight=1.5)
    trust_census:               Field = Field(1, 1, period=ANNUAL)
    trust_fed:                  Field = Field(1, 2, period=QUARTERLY)
    trust_foreign_gov:          Field = Field(1, 4, period=QUARTERLY)
    # Market data
    trust_exchange:             Field = Field(1, 2, period=MONTHLY)
    trust_otc:                  Field = Field(1, 2, period=MONTHLY)
    trust_credit_rating:        Field = Field(1, 2, period=QUARTERLY)
    # Alternative
    trust_satellite:            Field = Field(1, 2, period=QUARTERLY)
    trust_web_scraping:         Field = Field(1, 2, period=MONTHLY)
    trust_social_sentiment:     Field = Field(1, 2, period=WEEKLY)
    trust_survey:               Field = Field(1, 2, period=MONTHLY)
    # Corporate
    trust_gaap:                 Field = Field(1, 2, period=QUARTERLY)
    trust_mgmt_guidance:        Field = Field(1, 2, period=QUARTERLY, loss_weight=2.0)
    trust_non_gaap:             Field = Field(1, 2, period=QUARTERLY, loss_weight=1.5)
    # Meta
    overall_epistemic_state:    Field = Field(2, 4, period=MONTHLY, loss_weight=2.0)
    information_advantage:      Field = Field(1, 2, period=MONTHLY, loss_weight=1.5)
    adversarial_info_risk:      Field = Field(1, 2, period=WEEKLY, loss_weight=2.0)
