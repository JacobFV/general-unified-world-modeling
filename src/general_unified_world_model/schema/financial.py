"""Layer 3: Global Monetary & Financial (τ0–τ2, high bandwidth).

The fastest-moving layer: markets, yields, credit, FX, liquidity.
This is where reflexivity lives — beliefs move prices, prices move beliefs.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    TICK, DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)


@dataclass
class CentralBankState:
    policy_rate:                Field = Field(1, 2, period=MONTHLY)
    balance_sheet_size:         Field = Field(1, 2, period=WEEKLY)
    forward_guidance_hawkish:   Field = Field(1, 2, period=MONTHLY)
    qe_qt_pace:                Field = Field(1, 2, period=MONTHLY)
    credibility:                Field = Field(1, 2, period=QUARTERLY, loss_weight=2.0)
    dot_plot_median:            Field = Field(1, 2, period=QUARTERLY)


@dataclass
class YieldCurveState:
    short_rate:                 Field = Field(1, 2, period=TICK)
    two_year:                   Field = Field(1, 2, period=TICK)
    five_year:                  Field = Field(1, 2, period=TICK)
    ten_year:                   Field = Field(1, 2, period=TICK)
    thirty_year:                Field = Field(1, 2, period=TICK)
    term_premium:               Field = Field(1, 2, period=DAILY)
    real_rates:                 Field = Field(1, 2, period=DAILY)
    inversion_depth:            Field = Field(1, 1, period=DAILY)
    slope_2s10s:                Field = Field(1, 1, period=TICK)
    breakeven_inflation:        Field = Field(1, 2, period=TICK)


@dataclass
class CreditConditions:
    ig_spread:                  Field = Field(1, 2, period=TICK)
    hy_spread:                  Field = Field(1, 2, period=TICK)
    cds_indices:                Field = Field(1, 4, period=TICK)
    leveraged_loan_spread:      Field = Field(1, 2, period=DAILY)
    bank_lending_standards:     Field = Field(1, 2, period=QUARTERLY)
    credit_impulse:             Field = Field(1, 2, period=MONTHLY)
    private_credit_growth:      Field = Field(1, 2, period=QUARTERLY)
    distress_ratio:             Field = Field(1, 2, period=MONTHLY)
    default_rate:               Field = Field(1, 2, period=MONTHLY)
    covenant_lite_share:        Field = Field(1, 1, period=QUARTERLY)


@dataclass
class FXState:
    dxy:                        Field = Field(1, 2, period=TICK)
    eurusd:                     Field = Field(1, 2, period=TICK)
    usdjpy:                     Field = Field(1, 2, period=TICK)
    usdcny:                     Field = Field(1, 2, period=TICK)
    em_fx_index:                Field = Field(1, 2, period=TICK)
    fx_vol_surface:             Field = Field(2, 4, period=TICK)
    carry_trade_profitability:  Field = Field(1, 2, period=DAILY)
    reserve_currency_shares:    Field = Field(1, 4, period=QUARTERLY)
    sdr_composition:            Field = Field(1, 4, period=ANNUAL)


@dataclass
class LiquidityState:
    fed_reverse_repo:           Field = Field(1, 2, period=DAILY)
    treasury_general_account:   Field = Field(1, 2, period=DAILY)
    bank_reserves:              Field = Field(1, 2, period=WEEKLY)
    money_market_stress:        Field = Field(1, 2, period=TICK)
    repo_rate_spread:           Field = Field(1, 2, period=TICK)
    collateral_scarcity:        Field = Field(1, 2, period=DAILY)
    global_m2_growth:           Field = Field(1, 2, period=MONTHLY)
    dollar_funding_stress:      Field = Field(1, 2, period=TICK)       # FRA-OIS
    cross_currency_basis:       Field = Field(1, 4, period=TICK)


@dataclass
class EquityMarketState:
    broad_indices:              Field = Field(1, 4, period=TICK)
    vix:                        Field = Field(1, 2, period=TICK)
    vol_term_structure:         Field = Field(1, 4, period=TICK)
    sector_rotation:            Field = Field(1, 8, period=DAILY)
    breadth:                    Field = Field(1, 2, period=DAILY)
    earnings_revision_ratio:    Field = Field(1, 2, period=WEEKLY)
    buyback_pace:               Field = Field(1, 2, period=QUARTERLY)
    ipo_issuance:               Field = Field(1, 2, period=MONTHLY)
    margin_debt:                Field = Field(1, 2, period=MONTHLY)
    put_call_ratio:             Field = Field(1, 2, period=TICK)


@dataclass
class CryptoState:
    btc:                        Field = Field(1, 2, period=TICK)
    eth:                        Field = Field(1, 2, period=TICK)
    stablecoin_supply:          Field = Field(1, 2, period=DAILY)
    defi_tvl:                   Field = Field(1, 2, period=DAILY)
    crypto_vol:                 Field = Field(1, 2, period=TICK)
    institutional_flows:        Field = Field(1, 2, period=WEEKLY)


@dataclass
class GlobalFinancialLayer:
    central_banks: CentralBankState  = dc_field(default_factory=CentralBankState)
    yield_curves:  YieldCurveState   = dc_field(default_factory=YieldCurveState)
    credit:        CreditConditions  = dc_field(default_factory=CreditConditions)
    fx:            FXState           = dc_field(default_factory=FXState)
    liquidity:     LiquidityState    = dc_field(default_factory=LiquidityState)
    equities:      EquityMarketState = dc_field(default_factory=EquityMarketState)
    crypto:        CryptoState       = dc_field(default_factory=CryptoState)
