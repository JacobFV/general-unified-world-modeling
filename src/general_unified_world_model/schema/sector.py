"""Layer 9: Sector (τ3–τ5, per GICS sector).

Demand, supply, profitability, structural dynamics for industry sectors.
Instantiated per sector (tech, energy, financials, healthcare, etc.).
"""

from dataclasses import dataclass
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class Sector:
    # Demand
    demand_growth:              Field = Field(1, 2, period=MONTHLY)
    pricing_power:              Field = Field(1, 2, period=MONTHLY)
    order_backlog:              Field = Field(1, 2, period=MONTHLY)
    end_market_health:          Field = Field(1, 2, period=MONTHLY)
    # Supply
    capacity_utilization:       Field = Field(1, 2, period=MONTHLY)
    supply_chain_stress:        Field = Field(1, 2, period=WEEKLY)
    inventory_to_sales:         Field = Field(1, 2, period=MONTHLY)
    labor_availability:         Field = Field(1, 2, period=MONTHLY)
    # Profitability
    margins:                    Field = Field(1, 2, period=QUARTERLY)
    input_cost_pressure:        Field = Field(1, 2, period=MONTHLY)
    revenue_growth:             Field = Field(1, 2, period=QUARTERLY)
    # Structural
    capex_cycle:                Field = Field(1, 2, period=QUARTERLY)
    innovation_rate:            Field = Field(1, 2, period=QUARTERLY)
    regulatory_risk:            Field = Field(1, 2, period=MONTHLY, loss_weight=1.5)
    disruption_risk:            Field = Field(1, 2, period=QUARTERLY, loss_weight=2.0)
    esg_pressure:               Field = Field(1, 2, period=QUARTERLY)
    m_and_a_activity:           Field = Field(1, 2, period=QUARTERLY)
    concentration:              Field = Field(1, 2, period=ANNUAL)
    # Observability
    data_quality:               Field = Field(1, 1, period=QUARTERLY, loss_weight=0.3)
