"""Layer: Cybersecurity & Digital Threats (τ2–τ5).

Attack surfaces, threat actors, digital ecosystem concentration, and
information integrity — the adversarial landscape of connected systems.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field
from general_unified_world_model.schema.temporal_constants import (
    DAILY, WEEKLY, MONTHLY, QUARTERLY,
)


@dataclass
class CyberThreatLandscape:
    attack_surface:             Field = Field(1, 2, period=WEEKLY)
    ransomware_frequency:       Field = Field(1, 1, period=WEEKLY, loss_weight=1.5)
    nation_state_activity:      Field = Field(1, 2, period=MONTHLY, loss_weight=2.0)
    zero_day_inventory:         Field = Field(1, 1, period=WEEKLY, loss_weight=2.0)
    critical_infrastructure_targeting: Field = Field(1, 1, period=DAILY, loss_weight=2.0)
    supply_chain_compromise_risk: Field = Field(1, 1, period=MONTHLY, loss_weight=1.5)


@dataclass
class DigitalEcosystem:
    platform_concentration:     Field = Field(1, 2, period=QUARTERLY)
    data_sovereignty_regulation: Field = Field(1, 1, period=QUARTERLY)
    ai_generated_content_share: Field = Field(1, 1, period=MONTHLY)
    digital_identity_adoption:  Field = Field(1, 1, period=QUARTERLY)
    open_source_health:         Field = Field(1, 2, period=QUARTERLY)


@dataclass
class CyberLayer:
    threats:   CyberThreatLandscape = dc_field(default_factory=CyberThreatLandscape)
    ecosystem: DigitalEcosystem     = dc_field(default_factory=DigitalEcosystem)
