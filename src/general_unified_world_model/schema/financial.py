"""Layer 3: Global Monetary & Financial (τ0–τ2, high bandwidth).

The fastest-moving layer: markets, yields, credit, FX, liquidity.
This is where reflexivity lives — beliefs move prices, prices move beliefs.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field


@dataclass
class CentralBankState:
    policy_rate:                Field = Field(1, 2, period=192)
    balance_sheet_size:         Field = Field(1, 2, period=48)
    forward_guidance_hawkish:   Field = Field(1, 2, period=192)
    qe_qt_pace:                Field = Field(1, 2, period=192)
    credibility:                Field = Field(1, 2, period=576, loss_weight=2.0)
    dot_plot_median:            Field = Field(1, 2, period=576)


@dataclass
class YieldCurveState:
    short_rate:                 Field = Field(1, 2, period=1)
    two_year:                   Field = Field(1, 2, period=1)
    five_year:                  Field = Field(1, 2, period=1)
    ten_year:                   Field = Field(1, 2, period=1)
    thirty_year:                Field = Field(1, 2, period=1)
    term_premium:               Field = Field(1, 2, period=16)
    real_rates:                 Field = Field(1, 2, period=16)
    inversion_depth:            Field = Field(1, 1, period=16)
    slope_2s10s:                Field = Field(1, 1, period=1)
    breakeven_inflation:        Field = Field(1, 2, period=1)


@dataclass
class CreditConditions:
    ig_spread:                  Field = Field(1, 2, period=1)
    hy_spread:                  Field = Field(1, 2, period=1)
    cds_indices:                Field = Field(1, 4, period=1)
    leveraged_loan_spread:      Field = Field(1, 2, period=16)
    bank_lending_standards:     Field = Field(1, 2, period=576)
    credit_impulse:             Field = Field(1, 2, period=192)
    private_credit_growth:      Field = Field(1, 2, period=576)
    distress_ratio:             Field = Field(1, 2, period=192)
    default_rate:               Field = Field(1, 2, period=192)
    covenant_lite_share:        Field = Field(1, 1, period=576)


@dataclass
class FXState:
    dxy:                        Field = Field(1, 2, period=1)
    eurusd:                     Field = Field(1, 2, period=1)
    usdjpy:                     Field = Field(1, 2, period=1)
    usdcny:                     Field = Field(1, 2, period=1)
    em_fx_index:                Field = Field(1, 2, period=1)
    fx_vol_surface:             Field = Field(2, 4, period=1)
    carry_trade_profitability:  Field = Field(1, 2, period=16)
    reserve_currency_shares:    Field = Field(1, 4, period=576)
    sdr_composition:            Field = Field(1, 4, period=2304)


@dataclass
class LiquidityState:
    fed_reverse_repo:           Field = Field(1, 2, period=16)
    treasury_general_account:   Field = Field(1, 2, period=16)
    bank_reserves:              Field = Field(1, 2, period=48)
    money_market_stress:        Field = Field(1, 2, period=1)
    repo_rate_spread:           Field = Field(1, 2, period=1)
    collateral_scarcity:        Field = Field(1, 2, period=16)
    global_m2_growth:           Field = Field(1, 2, period=192)
    dollar_funding_stress:      Field = Field(1, 2, period=1)       # FRA-OIS
    cross_currency_basis:       Field = Field(1, 4, period=1)


@dataclass
class EquityMarketState:
    broad_indices:              Field = Field(1, 4, period=1)
    vix:                        Field = Field(1, 2, period=1)
    vol_term_structure:         Field = Field(1, 4, period=1)
    sector_rotation:            Field = Field(1, 8, period=16)
    breadth:                    Field = Field(1, 2, period=16)
    earnings_revision_ratio:    Field = Field(1, 2, period=48)
    buyback_pace:               Field = Field(1, 2, period=576)
    ipo_issuance:               Field = Field(1, 2, period=192)
    margin_debt:                Field = Field(1, 2, period=192)
    put_call_ratio:             Field = Field(1, 2, period=1)


@dataclass
class CryptoState:
    btc:                        Field = Field(1, 2, period=1)
    eth:                        Field = Field(1, 2, period=1)
    stablecoin_supply:          Field = Field(1, 2, period=16)
    defi_tvl:                   Field = Field(1, 2, period=16)
    crypto_vol:                 Field = Field(1, 2, period=1)
    institutional_flows:        Field = Field(1, 2, period=48)


@dataclass
class GlobalFinancialLayer:
    central_banks: CentralBankState  = dc_field(default_factory=CentralBankState)
    yield_curves:  YieldCurveState   = dc_field(default_factory=YieldCurveState)
    credit:        CreditConditions  = dc_field(default_factory=CreditConditions)
    fx:            FXState           = dc_field(default_factory=FXState)
    liquidity:     LiquidityState    = dc_field(default_factory=LiquidityState)
    equities:      EquityMarketState = dc_field(default_factory=EquityMarketState)
    crypto:        CryptoState       = dc_field(default_factory=CryptoState)
