"""Layer 4: Macroeconomy (τ3–τ5, per country).

GDP, inflation, labor, fiscal, trade, housing — the real economy.
Instantiated per country in the Country composite type.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.observability import ObservedSlow


@dataclass
class OutputAndGrowth:
    gdp_nowcast:                Field = Field(1, 2, period=16, loss_weight=2.0)
    gdp_official:               ObservedSlow = dc_field(default_factory=ObservedSlow)
    industrial_production:      Field = Field(1, 2, period=192)
    capacity_utilization:       Field = Field(1, 2, period=192)
    pmi_manufacturing:          Field = Field(1, 2, period=192, loss_weight=1.5)
    pmi_services:               Field = Field(1, 2, period=192, loss_weight=1.5)
    retail_sales:               Field = Field(1, 2, period=192)
    new_orders:                 Field = Field(1, 2, period=192)
    potential_growth:            Field = Field(1, 2, period=2304)


@dataclass
class InflationState:
    headline_cpi:               Field = Field(1, 2, period=192, loss_weight=2.0)
    core_cpi:                   Field = Field(1, 2, period=192, loss_weight=2.5)
    pce_deflator:               Field = Field(1, 2, period=192)
    ppi:                        Field = Field(1, 2, period=192)
    wage_growth:                Field = Field(1, 2, period=192)
    rent_inflation:             Field = Field(1, 2, period=192)
    expectations_1y:            Field = Field(1, 2, period=192)
    expectations_5y:            Field = Field(1, 2, period=192, loss_weight=2.0)
    sticky_vs_flexible:         Field = Field(1, 2, period=192)
    supply_driven_share:        Field = Field(1, 2, period=192)


@dataclass
class LaborMarket:
    unemployment_rate:          Field = Field(1, 2, period=192)
    nfp_change:                 Field = Field(1, 2, period=192, loss_weight=1.5)
    initial_claims:             Field = Field(1, 2, period=48)
    continuing_claims:          Field = Field(1, 2, period=48)
    job_openings:               Field = Field(1, 2, period=192)
    quits_rate:                 Field = Field(1, 2, period=192)
    lfpr:                       Field = Field(1, 2, period=192)
    avg_hourly_earnings:        Field = Field(1, 2, period=192)
    unit_labor_costs:           Field = Field(1, 2, period=576)
    immigration_flow:           Field = Field(1, 2, period=2304)


@dataclass
class FiscalState:
    debt_to_gdp:                Field = Field(1, 2, period=576)
    deficit_to_gdp:             Field = Field(1, 2, period=576)
    interest_expense_share:     Field = Field(1, 2, period=576, loss_weight=2.0)
    fiscal_impulse:             Field = Field(1, 2, period=576)
    spending_composition:       Field = Field(1, 4, period=2304)
    tax_revenue_trend:          Field = Field(1, 2, period=576)
    debt_maturity_profile:      Field = Field(1, 4, period=576)
    sovereign_cds:              Field = Field(1, 2, period=1)
    debt_ceiling_proximity:     Field = Field(1, 1, period=48, loss_weight=3.0)


@dataclass
class TradeBalance:
    current_account:            Field = Field(1, 2, period=576)
    trade_balance:              Field = Field(1, 2, period=192)
    capital_flows_net:          Field = Field(1, 2, period=192)
    fdi_flows:                  Field = Field(1, 2, period=576)
    terms_of_trade:             Field = Field(1, 2, period=192)
    tariff_effective_rate:      Field = Field(1, 2, period=192)
    sanctions_exposure:         Field = Field(1, 2, period=48)


@dataclass
class HousingMarket:
    home_price_index:           Field = Field(1, 2, period=192)
    housing_starts:             Field = Field(1, 2, period=192)
    mortgage_rate:              Field = Field(1, 2, period=16)
    existing_home_sales:        Field = Field(1, 2, period=192)
    affordability:              Field = Field(1, 2, period=192)
    delinquency_rate:           Field = Field(1, 2, period=576)


@dataclass
class MacroEconomy:
    output:     OutputAndGrowth = dc_field(default_factory=OutputAndGrowth)
    inflation:  InflationState  = dc_field(default_factory=InflationState)
    labor:      LaborMarket     = dc_field(default_factory=LaborMarket)
    fiscal:     FiscalState     = dc_field(default_factory=FiscalState)
    trade:      TradeBalance    = dc_field(default_factory=TradeBalance)
    housing:    HousingMarket   = dc_field(default_factory=HousingMarket)
