"""Layer 15: Regime State (privileged global latent, τ5–τ7).

The most compressed representation of the world's macro state.
Growth regime, inflation regime, financial cycle, geopolitical order,
technology phase, systemic risk. These are the regime variables that
determine which causal channels are active.
"""

from dataclasses import dataclass
from canvas_engineering import Field


@dataclass
class RegimeState:
    # Economic regime
    growth_regime:              Field = Field(2, 4, period=576, loss_weight=5.0)
    inflation_regime:           Field = Field(2, 4, period=576, loss_weight=5.0)
    financial_cycle:            Field = Field(2, 4, period=576, loss_weight=4.0)
    credit_cycle:               Field = Field(2, 4, period=576, loss_weight=4.0)
    liquidity_regime:           Field = Field(2, 4, period=192, loss_weight=4.0)
    # Geopolitical regime
    cooperation_vs_fragmentation: Field = Field(2, 4, period=2304, loss_weight=4.0)
    peace_vs_conflict:          Field = Field(2, 4, period=576, loss_weight=5.0)
    hegemonic_stability:        Field = Field(2, 4, period=4608, loss_weight=3.0)
    globalization_vs_autarky:   Field = Field(2, 4, period=2304, loss_weight=3.0)
    # Technology regime
    ai_acceleration:            Field = Field(2, 4, period=576, loss_weight=4.0)
    energy_transition_phase:    Field = Field(2, 4, period=2304, loss_weight=3.0)
    productivity_regime:        Field = Field(2, 4, period=2304, loss_weight=3.0)
    # Systemic risk
    fragility:                  Field = Field(2, 4, period=192, loss_weight=5.0)
    reflexivity_intensity:      Field = Field(2, 4, period=192, loss_weight=4.0)
    tail_risk_concentration:    Field = Field(2, 4, period=192, loss_weight=5.0)
    black_swan_proximity:       Field = Field(2, 4, period=48, loss_weight=5.0)
    # Compressed summary
    compressed_world_state:     Field = Field(4, 8, period=192, loss_weight=3.0)
