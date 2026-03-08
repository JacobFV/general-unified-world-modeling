"""Layer 17: Forecast Bundle (structured output heads).

The model's predictions: macro forecasts, financial outlook, geopolitical risk,
business trajectory, actor behavior, and decomposed uncertainty.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field


@dataclass
class MacroForecast:
    recession_prob_3m:          Field = Field(1, 2, loss_weight=5.0)
    recession_prob_12m:         Field = Field(1, 2, loss_weight=4.0)
    gdp_growth_3m:              Field = Field(1, 2, loss_weight=3.0)
    gdp_growth_12m:             Field = Field(1, 2, loss_weight=2.0)
    inflation_path_12m:         Field = Field(1, 4, loss_weight=3.0)
    rates_path_12m:             Field = Field(1, 4, loss_weight=3.0)
    unemployment_path_12m:      Field = Field(1, 4, loss_weight=2.0)


@dataclass
class FinancialForecast:
    credit_stress_3m:           Field = Field(1, 2, loss_weight=4.0)
    equity_regime_3m:           Field = Field(1, 4, loss_weight=3.0)
    vol_regime_3m:              Field = Field(1, 2, loss_weight=3.0)
    sector_rotation_3m:         Field = Field(1, 8, loss_weight=2.0)
    curve_shape_3m:             Field = Field(1, 4, loss_weight=2.0)
    fx_regime_3m:               Field = Field(1, 4, loss_weight=2.0)
    liquidity_crisis_prob:      Field = Field(1, 2, loss_weight=5.0)


@dataclass
class GeopoliticalForecast:
    conflict_escalation_3m:     Field = Field(1, 4, loss_weight=5.0)
    sanctions_change_3m:        Field = Field(1, 4, loss_weight=3.0)
    alliance_shift_12m:         Field = Field(1, 4, loss_weight=2.0)
    regime_change_prob:         Field = Field(1, 4, loss_weight=4.0)
    election_outcome:           Field = Field(1, 4, loss_weight=3.0)


@dataclass
class BusinessForecast:
    revenue_surprise:           Field = Field(1, 2, loss_weight=3.0)
    margin_trajectory:          Field = Field(1, 2, loss_weight=3.0)
    default_prob_12m:           Field = Field(1, 2, loss_weight=5.0)
    strategic_pivot_prob:       Field = Field(1, 2, loss_weight=3.0)
    m_and_a_prob:               Field = Field(1, 2, loss_weight=2.0)
    mgmt_change_prob:           Field = Field(1, 2, loss_weight=3.0)


@dataclass
class ActorForecast:
    likely_next_action:         Field = Field(2, 4, loss_weight=3.0)
    action_timing:              Field = Field(1, 2, loss_weight=2.0)
    surprise_probability:       Field = Field(1, 2, loss_weight=4.0)
    influence_trajectory:       Field = Field(1, 2, loss_weight=2.0)


@dataclass
class UncertaintyDecomposition:
    aleatoric_macro:            Field = Field(1, 2, loss_weight=2.0)
    aleatoric_geopolitical:     Field = Field(1, 2, loss_weight=2.0)
    epistemic_data_gaps:        Field = Field(1, 2, loss_weight=2.0)
    epistemic_model_limits:     Field = Field(1, 2, loss_weight=1.0)
    scenario_divergence:        Field = Field(1, 4, loss_weight=3.0)
    calibration_score:          Field = Field(1, 2, loss_weight=3.0)


@dataclass
class ForecastBundle:
    macro:        MacroForecast            = dc_field(default_factory=MacroForecast)
    financial:    FinancialForecast        = dc_field(default_factory=FinancialForecast)
    geopolitical: GeopoliticalForecast     = dc_field(default_factory=GeopoliticalForecast)
    business:     BusinessForecast         = dc_field(default_factory=BusinessForecast)
    actor:        ActorForecast            = dc_field(default_factory=ActorForecast)
    uncertainty:  UncertaintyDecomposition = dc_field(default_factory=UncertaintyDecomposition)
