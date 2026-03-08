"""Observability bundles — epistemic wrappers for measurements.

Every measurement comes with uncertainty. These are reusable companion
types that wrap any value with epistemic metadata.
"""

from dataclasses import dataclass
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    TICK, DAILY, MONTHLY,
)


@dataclass
class ObservedFast:
    """High-frequency observable with real-time confidence."""
    value:      Field = Field(2, 2, period=TICK)
    confidence: Field = Field(1, 1, period=TICK, loss_weight=0.3)


@dataclass
class ObservedDaily:
    """Daily observable with latency metadata."""
    value:      Field = Field(2, 2, period=DAILY)
    confidence: Field = Field(1, 1, period=DAILY, loss_weight=0.3)
    latency:    Field = Field(1, 1, period=DAILY, loss_weight=0.2)


@dataclass
class ObservedSlow:
    """Low-frequency observable with revision risk."""
    value:         Field = Field(2, 2, period=MONTHLY)
    confidence:    Field = Field(1, 1, period=MONTHLY, loss_weight=0.5)
    revision_risk: Field = Field(1, 1, period=MONTHLY, loss_weight=0.3)
