"""Layer 16: Intervention & Counterfactual.

What-if analysis: declare a policy intervention, observe predicted effects.
Monetary, fiscal, regulatory, sanctions, trade, military, strategic, technology.
"""

from dataclasses import dataclass
from canvas_engineering import Field


@dataclass
class InterventionSpace:
    monetary_policy_change:     Field = Field(2, 4, period=192)
    fiscal_policy_change:       Field = Field(2, 4, period=576)
    regulatory_action:          Field = Field(2, 4, period=192)
    sanctions_change:           Field = Field(2, 4, period=48)
    trade_policy_change:        Field = Field(2, 4, period=192)
    military_action:            Field = Field(2, 4, period=48, loss_weight=5.0)
    firm_strategic_action:      Field = Field(2, 4, period=576)
    technology_release:         Field = Field(2, 4, period=192)
    market_intervention:        Field = Field(2, 4, period=48)
    # Counterfactual heads
    effect_3m:                  Field = Field(2, 4, loss_weight=3.0)
    effect_12m:                 Field = Field(2, 4, loss_weight=2.0)
    second_order_effects:       Field = Field(2, 4, loss_weight=2.0)
    unintended_consequences:    Field = Field(2, 4, loss_weight=3.0)
