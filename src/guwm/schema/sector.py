"""Layer 9: Sector (τ3–τ5, per GICS sector).

Demand, supply, profitability, structural dynamics for industry sectors.
Instantiated per sector (tech, energy, financials, healthcare, etc.).
"""

from dataclasses import dataclass
from canvas_engineering import Field


@dataclass
class Sector:
    # Demand
    demand_growth:              Field = Field(1, 2, period=192)
    pricing_power:              Field = Field(1, 2, period=192)
    order_backlog:              Field = Field(1, 2, period=192)
    end_market_health:          Field = Field(1, 2, period=192)
    # Supply
    capacity_utilization:       Field = Field(1, 2, period=192)
    supply_chain_stress:        Field = Field(1, 2, period=48)
    inventory_to_sales:         Field = Field(1, 2, period=192)
    labor_availability:         Field = Field(1, 2, period=192)
    # Profitability
    margins:                    Field = Field(1, 2, period=576)
    input_cost_pressure:        Field = Field(1, 2, period=192)
    revenue_growth:             Field = Field(1, 2, period=576)
    # Structural
    capex_cycle:                Field = Field(1, 2, period=576)
    innovation_rate:            Field = Field(1, 2, period=576)
    regulatory_risk:            Field = Field(1, 2, period=192, loss_weight=1.5)
    disruption_risk:            Field = Field(1, 2, period=576, loss_weight=2.0)
    esg_pressure:               Field = Field(1, 2, period=576)
    m_and_a_activity:           Field = Field(1, 2, period=576)
    concentration:              Field = Field(1, 2, period=2304)
    # Observability
    data_quality:               Field = Field(1, 1, period=576, loss_weight=0.3)
