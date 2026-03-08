"""Layer 16: Intervention & Counterfactual.

What-if analysis: declare a policy intervention, observe predicted effects.
Monetary, fiscal, regulatory, sanctions, trade, military, strategic, technology.
"""

from dataclasses import dataclass
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    WEEKLY, MONTHLY, QUARTERLY,
)


@dataclass
class InterventionSpace:
    monetary_policy_change:     Field = Field(2, 4, period=MONTHLY)
    fiscal_policy_change:       Field = Field(2, 4, period=QUARTERLY)
    regulatory_action:          Field = Field(2, 4, period=MONTHLY)
    sanctions_change:           Field = Field(2, 4, period=WEEKLY)
    trade_policy_change:        Field = Field(2, 4, period=MONTHLY)
    military_action:            Field = Field(2, 4, period=WEEKLY, loss_weight=5.0)
    firm_strategic_action:      Field = Field(2, 4, period=QUARTERLY)
    technology_release:         Field = Field(2, 4, period=MONTHLY)
    market_intervention:        Field = Field(2, 4, period=WEEKLY)
    # Counterfactual heads
    effect_3m:                  Field = Field(2, 4, loss_weight=3.0)
    effect_12m:                 Field = Field(2, 4, loss_weight=2.0)
    second_order_effects:       Field = Field(2, 4, loss_weight=2.0)
    unintended_consequences:    Field = Field(2, 4, loss_weight=3.0)
