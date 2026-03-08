"""Layer 13: Event Tape (τ0–τ1, dense in time, compressed spatially).

Real-time event stream: news, social signals, filings, policy announcements,
conflict events, disasters. The fastest-updating modality in the world model.
"""

from dataclasses import dataclass
from canvas_engineering import Field


@dataclass
class EventTape:
    news_embedding:             Field = Field(4, 8, period=1, loss_weight=0.5)
    social_signal:              Field = Field(2, 4, period=1, loss_weight=0.3)
    filing_events:              Field = Field(2, 4, period=16)
    earnings_call_signal:       Field = Field(2, 4, period=576)
    policy_announcement:        Field = Field(2, 4, period=1, loss_weight=2.0)
    conflict_event:             Field = Field(2, 4, period=1, loss_weight=3.0)
    disaster_event:             Field = Field(2, 4, period=1, loss_weight=2.0)
    trade_data_release:         Field = Field(1, 2, period=192)
    central_bank_comms:         Field = Field(2, 4, period=192, loss_weight=2.0)
    election_event:             Field = Field(2, 4, period=48, loss_weight=2.0)
