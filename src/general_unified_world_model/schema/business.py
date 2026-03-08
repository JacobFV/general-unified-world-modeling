"""Layer 11: Business (τ2–τ5, sparse instantiation).

Full firm decomposition: financials, operations, strategy, market position, risk.
Each firm also embeds a supply chain node for its position in the value chain.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.supply_chain import SupplyChainNode
from general_unified_world_model.schema.temporal_constants import (
    TICK, DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class FirmFinancials:
    revenue:                    Field = Field(1, 2, period=QUARTERLY)
    revenue_growth:             Field = Field(1, 2, period=QUARTERLY, loss_weight=2.0)
    cogs:                       Field = Field(1, 2, period=QUARTERLY)
    gross_margin:               Field = Field(1, 2, period=QUARTERLY, loss_weight=2.0)
    opex:                       Field = Field(1, 2, period=QUARTERLY)
    operating_margin:           Field = Field(1, 2, period=QUARTERLY, loss_weight=2.0)
    net_income:                 Field = Field(1, 2, period=QUARTERLY)
    fcf:                        Field = Field(1, 2, period=QUARTERLY, loss_weight=2.5)
    cash:                       Field = Field(1, 2, period=QUARTERLY)
    total_debt:                 Field = Field(1, 2, period=QUARTERLY)
    net_debt_to_ebitda:         Field = Field(1, 2, period=QUARTERLY, loss_weight=2.0)
    interest_coverage:          Field = Field(1, 2, period=QUARTERLY, loss_weight=2.0)
    covenant_headroom:          Field = Field(1, 2, period=QUARTERLY, loss_weight=3.0)
    maturity_wall:              Field = Field(1, 2, period=QUARTERLY, loss_weight=2.5)
    working_capital:            Field = Field(1, 2, period=QUARTERLY)
    capex:                      Field = Field(1, 2, period=QUARTERLY)
    share_count:                Field = Field(1, 1, period=QUARTERLY)
    insider_transactions:       Field = Field(1, 2, period=MONTHLY)


@dataclass
class FirmOperations:
    capacity:                   Field = Field(1, 2, period=MONTHLY)
    utilization:                Field = Field(1, 2, period=MONTHLY)
    backlog:                    Field = Field(1, 2, period=MONTHLY)
    pricing_power:              Field = Field(1, 2, period=MONTHLY, loss_weight=1.5)
    customer_concentration:     Field = Field(1, 2, period=QUARTERLY)
    supplier_concentration:     Field = Field(1, 2, period=QUARTERLY)
    quality_incidents:          Field = Field(1, 2, period=MONTHLY)
    headcount:                  Field = Field(1, 2, period=QUARTERLY)
    employee_satisfaction:      Field = Field(1, 2, period=QUARTERLY)
    tech_debt:                  Field = Field(1, 2, period=QUARTERLY)


@dataclass
class FirmStrategy:
    roadmap_clarity:            Field = Field(1, 2, period=QUARTERLY)
    capex_plan:                 Field = Field(1, 2, period=QUARTERLY)
    m_and_a_appetite:           Field = Field(1, 2, period=QUARTERLY)
    geographic_expansion:       Field = Field(1, 2, period=QUARTERLY)
    product_pipeline:           Field = Field(1, 4, period=QUARTERLY)
    moat_durability:            Field = Field(1, 2, period=ANNUAL, loss_weight=2.0)
    management_quality:         Field = Field(1, 2, period=QUARTERLY, loss_weight=2.0)
    capital_allocation:         Field = Field(1, 2, period=QUARTERLY, loss_weight=2.0)
    governance_quality:         Field = Field(1, 2, period=QUARTERLY)
    esg_posture:                Field = Field(1, 2, period=QUARTERLY)


@dataclass
class FirmMarketPosition:
    equity_price:               Field = Field(1, 2, period=TICK)
    implied_vol:                Field = Field(1, 2, period=TICK)
    credit_spread:              Field = Field(1, 2, period=TICK)
    analyst_consensus:          Field = Field(1, 2, period=WEEKLY)
    short_interest:             Field = Field(1, 2, period=WEEKLY)
    institutional_ownership:    Field = Field(1, 2, period=QUARTERLY)
    pe_ratio:                   Field = Field(1, 2, period=DAILY)
    ev_ebitda:                  Field = Field(1, 2, period=DAILY)


@dataclass
class FirmRisk:
    regulatory_exposure:        Field = Field(1, 2, period=MONTHLY)
    litigation_risk:            Field = Field(1, 2, period=MONTHLY)
    cyber_vulnerability:        Field = Field(1, 2, period=MONTHLY)
    key_person_risk:            Field = Field(1, 2, period=QUARTERLY, loss_weight=2.0)
    supply_chain_fragility:     Field = Field(1, 2, period=MONTHLY)
    geopolitical_exposure:      Field = Field(1, 2, period=MONTHLY, loss_weight=1.5)
    climate_transition_risk:    Field = Field(1, 2, period=ANNUAL)
    tech_obsolescence:          Field = Field(1, 2, period=QUARTERLY, loss_weight=2.0)


@dataclass
class Business:
    identity:       Field = Field(2, 4, is_output=False)
    sector_link:    Field = Field(1, 2, is_output=False)
    geography_link: Field = Field(1, 2, is_output=False)

    financials:     FirmFinancials     = dc_field(default_factory=FirmFinancials)
    operations:     FirmOperations     = dc_field(default_factory=FirmOperations)
    strategy:       FirmStrategy       = dc_field(default_factory=FirmStrategy)
    market:         FirmMarketPosition = dc_field(default_factory=FirmMarketPosition)
    risk:           FirmRisk           = dc_field(default_factory=FirmRisk)
    supply_chain:   SupplyChainNode    = dc_field(default_factory=SupplyChainNode)

    latent_health:              Field = Field(2, 4, loss_weight=3.0)
    latent_momentum:            Field = Field(2, 4, loss_weight=2.0)
    latent_tail_risk:           Field = Field(2, 4, loss_weight=4.0)
    recommended_action:         Field = Field(2, 4, loss_weight=3.0)
    fair_value_estimate:        Field = Field(1, 2, loss_weight=2.0)
