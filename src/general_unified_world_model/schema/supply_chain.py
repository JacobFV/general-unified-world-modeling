"""Layer 10: Supply Chain Node (τ2–τ4, graph structure).

Critical bottleneck nodes in global supply chains.
Each node measures concentration, inventory, lead time, fragility.
"""

from dataclasses import dataclass
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    DAILY, WEEKLY, MONTHLY, QUARTERLY,
)


@dataclass
class SupplyChainNode:
    upstream_concentration:     Field = Field(1, 2, period=MONTHLY)
    downstream_concentration:   Field = Field(1, 2, period=MONTHLY)
    inventory:                  Field = Field(1, 2, period=WEEKLY)
    lead_time:                  Field = Field(1, 2, period=WEEKLY)
    logistics_friction:         Field = Field(1, 2, period=DAILY)
    bottleneck_severity:        Field = Field(1, 2, period=DAILY, loss_weight=2.0)
    substitutability:           Field = Field(1, 2, period=QUARTERLY)
    geographic_risk:            Field = Field(1, 2, period=MONTHLY)
    single_point_of_failure:    Field = Field(1, 1, period=QUARTERLY, loss_weight=3.0)
