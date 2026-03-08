"""Temporal frequency constants for the General Unified World Model.

The world model uses 8 temporal frequency classes (τ0–τ7) representing
different update cadences. Period = "this field updates every N canvas frames."
The actual real-world time per tick depends on training data resolution.
"""

# τ0: Sub-minute / real-time / every tick
TICK = 1

# τ1: Hourly / fast intraday
HOURLY = 4

# τ2: Daily
DAILY = 16

# τ3: Weekly
WEEKLY = 48

# τ4: Monthly
MONTHLY = 192

# τ5: Quarterly
QUARTERLY = 576

# τ6: Annual / multi-year
ANNUAL = 2304

# τ7: Decadal / geological
DECADAL = 4608


# Mapping for human-readable period names
PERIOD_NAMES = {
    TICK: "τ0 (tick)",
    HOURLY: "τ1 (hourly)",
    DAILY: "τ2 (daily)",
    WEEKLY: "τ3 (weekly)",
    MONTHLY: "τ4 (monthly)",
    QUARTERLY: "τ5 (quarterly)",
    ANNUAL: "τ6 (annual)",
    DECADAL: "τ7 (decadal)",
}
