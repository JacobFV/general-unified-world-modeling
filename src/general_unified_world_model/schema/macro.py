"""Layer 4: Macroeconomy (τ3–τ5, per country).

GDP, inflation, labor, fiscal, trade, housing — the real economy.
Instantiated per country in the Country composite type.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.observability import ObservedSlow
from general_unified_world_model.schema.temporal_constants import (
    TICK, DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class OutputAndGrowth:
    gdp_nowcast:                Field = Field(1, 2, period=DAILY, loss_weight=2.0)
    gdp_official:               ObservedSlow = dc_field(default_factory=ObservedSlow)
    industrial_production:      Field = Field(1, 2, period=MONTHLY)
    capacity_utilization:       Field = Field(1, 2, period=MONTHLY)
    pmi_manufacturing:          Field = Field(1, 2, period=MONTHLY, loss_weight=1.5)
    pmi_services:               Field = Field(1, 2, period=MONTHLY, loss_weight=1.5)
    retail_sales:               Field = Field(1, 2, period=MONTHLY)
    new_orders:                 Field = Field(1, 2, period=MONTHLY)
    potential_growth:            Field = Field(1, 2, period=ANNUAL)


@dataclass
class InflationState:
    headline_cpi:               Field = Field(1, 2, period=MONTHLY, loss_weight=2.0)
    core_cpi:                   Field = Field(1, 2, period=MONTHLY, loss_weight=2.5)
    pce_deflator:               Field = Field(1, 2, period=MONTHLY)
    ppi:                        Field = Field(1, 2, period=MONTHLY)
    wage_growth:                Field = Field(1, 2, period=MONTHLY)
    rent_inflation:             Field = Field(1, 2, period=MONTHLY)
    expectations_1y:            Field = Field(1, 2, period=MONTHLY)
    expectations_5y:            Field = Field(1, 2, period=MONTHLY, loss_weight=2.0)
    sticky_vs_flexible:         Field = Field(1, 2, period=MONTHLY)
    supply_driven_share:        Field = Field(1, 2, period=MONTHLY)


@dataclass
class LaborMarket:
    unemployment_rate:          Field = Field(1, 2, period=MONTHLY)
    nfp_change:                 Field = Field(1, 2, period=MONTHLY, loss_weight=1.5)
    initial_claims:             Field = Field(1, 2, period=WEEKLY)
    continuing_claims:          Field = Field(1, 2, period=WEEKLY)
    job_openings:               Field = Field(1, 2, period=MONTHLY)
    quits_rate:                 Field = Field(1, 2, period=MONTHLY)
    lfpr:                       Field = Field(1, 2, period=MONTHLY)
    avg_hourly_earnings:        Field = Field(1, 2, period=MONTHLY)
    unit_labor_costs:           Field = Field(1, 2, period=QUARTERLY)
    immigration_flow:           Field = Field(1, 2, period=ANNUAL)


@dataclass
class FiscalState:
    debt_to_gdp:                Field = Field(1, 2, period=QUARTERLY)
    deficit_to_gdp:             Field = Field(1, 2, period=QUARTERLY)
    interest_expense_share:     Field = Field(1, 2, period=QUARTERLY, loss_weight=2.0)
    fiscal_impulse:             Field = Field(1, 2, period=QUARTERLY)
    spending_composition:       Field = Field(1, 4, period=ANNUAL)
    tax_revenue_trend:          Field = Field(1, 2, period=QUARTERLY)
    debt_maturity_profile:      Field = Field(1, 4, period=QUARTERLY)
    sovereign_cds:              Field = Field(1, 2, period=TICK)
    debt_ceiling_proximity:     Field = Field(1, 1, period=WEEKLY, loss_weight=3.0)


@dataclass
class TradeBalance:
    current_account:            Field = Field(1, 2, period=QUARTERLY)
    trade_balance:              Field = Field(1, 2, period=MONTHLY)
    capital_flows_net:          Field = Field(1, 2, period=MONTHLY)
    fdi_flows:                  Field = Field(1, 2, period=QUARTERLY)
    terms_of_trade:             Field = Field(1, 2, period=MONTHLY)
    tariff_effective_rate:      Field = Field(1, 2, period=MONTHLY)
    sanctions_exposure:         Field = Field(1, 2, period=WEEKLY)


@dataclass
class HousingMarket:
    home_price_index:           Field = Field(1, 2, period=MONTHLY)
    housing_starts:             Field = Field(1, 2, period=MONTHLY)
    mortgage_rate:              Field = Field(1, 2, period=DAILY)
    existing_home_sales:        Field = Field(1, 2, period=MONTHLY)
    affordability:              Field = Field(1, 2, period=MONTHLY)
    delinquency_rate:           Field = Field(1, 2, period=QUARTERLY)


@dataclass
class MacroEconomy:
    output:     OutputAndGrowth = dc_field(default_factory=OutputAndGrowth)
    inflation:  InflationState  = dc_field(default_factory=InflationState)
    labor:      LaborMarket     = dc_field(default_factory=LaborMarket)
    fiscal:     FiscalState     = dc_field(default_factory=FiscalState)
    trade:      TradeBalance    = dc_field(default_factory=TradeBalance)
    housing:    HousingMarket   = dc_field(default_factory=HousingMarket)
