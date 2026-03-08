"""Layer 13: Event Tape (τ0–τ1, dense in time, compressed spatially).

Real-time event stream: news, social signals, filings, policy announcements,
conflict events, disasters. The fastest-updating modality in the world model.
"""

from dataclasses import dataclass
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    TICK, DAILY, WEEKLY, MONTHLY, QUARTERLY,
)


@dataclass
class EventTape:
    news_embedding:             Field = Field(4, 8, period=TICK, loss_weight=0.5)
    social_signal:              Field = Field(2, 4, period=TICK, loss_weight=0.3)
    filing_events:              Field = Field(2, 4, period=DAILY)
    earnings_call_signal:       Field = Field(2, 4, period=QUARTERLY)
    policy_announcement:        Field = Field(2, 4, period=TICK, loss_weight=2.0)
    conflict_event:             Field = Field(2, 4, period=TICK, loss_weight=3.0)
    disaster_event:             Field = Field(2, 4, period=TICK, loss_weight=2.0)
    trade_data_release:         Field = Field(1, 2, period=MONTHLY)
    central_bank_comms:         Field = Field(2, 4, period=MONTHLY, loss_weight=2.0)
    election_event:             Field = Field(2, 4, period=WEEKLY, loss_weight=2.0)
