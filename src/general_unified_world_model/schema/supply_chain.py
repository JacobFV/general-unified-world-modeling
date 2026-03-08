"""Layer 10: Supply Chain Node (τ2–τ4, graph structure).

Critical bottleneck nodes in global supply chains.
Each node measures concentration, inventory, lead time, fragility.
"""

from dataclasses import dataclass
from canvas_engineering import Field


@dataclass
class SupplyChainNode:
    upstream_concentration:     Field = Field(1, 2, period=192)
    downstream_concentration:   Field = Field(1, 2, period=192)
    inventory:                  Field = Field(1, 2, period=48)
    lead_time:                  Field = Field(1, 2, period=48)
    logistics_friction:         Field = Field(1, 2, period=16)
    bottleneck_severity:        Field = Field(1, 2, period=16, loss_weight=2.0)
    substitutability:           Field = Field(1, 2, period=576)
    geographic_risk:            Field = Field(1, 2, period=192)
    single_point_of_failure:    Field = Field(1, 1, period=576, loss_weight=3.0)
