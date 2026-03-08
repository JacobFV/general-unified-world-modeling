"""Observability bundles — epistemic wrappers for measurements.

Every measurement comes through fog. These are reusable companion
types that wrap any value with epistemic metadata.
"""

from dataclasses import dataclass
from canvas_engineering import Field


@dataclass
class ObservedFast:
    """High-frequency observable with real-time confidence."""
    value:      Field = Field(2, 2, period=1)
    confidence: Field = Field(1, 1, period=1, loss_weight=0.3)


@dataclass
class ObservedDaily:
    """Daily observable with latency metadata."""
    value:      Field = Field(2, 2, period=16)
    confidence: Field = Field(1, 1, period=16, loss_weight=0.3)
    latency:    Field = Field(1, 1, period=16, loss_weight=0.2)


@dataclass
class ObservedSlow:
    """Low-frequency observable with revision risk."""
    value:         Field = Field(2, 2, period=192)
    confidence:    Field = Field(1, 1, period=192, loss_weight=0.5)
    revision_risk: Field = Field(1, 1, period=192, loss_weight=0.3)
