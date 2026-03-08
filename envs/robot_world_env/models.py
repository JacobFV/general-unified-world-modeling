"""Data models for the Robot World Environment.

Three robot morphologies operate in a shared physical dynamics world model:
- logistics: freight optimization in shipping/rail networks
- disaster: hazard mitigation across seismic/wildfire/pandemic
- climate: tracking anomalies and maintaining sensor coverage
"""

from typing import Dict, List
from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class RobotWorldAction(Action):
    """Action for the Robot World environment."""

    morphology: str = Field(
        ...,
        description="Robot morphology: 'logistics', 'disaster', or 'climate'",
    )
    values: List[float] = Field(
        ...,
        description="Continuous actuator values mapped to the morphology's act_fields",
    )


class RobotWorldObservation(Observation):
    """Observation from the Robot World environment."""

    morphology: str = Field(default="", description="Active robot morphology")
    obs_dict: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Sensor readings keyed by field path",
    )
    step_count: int = Field(default=0, description="Current step in episode")
    obs_fields: List[str] = Field(
        default_factory=list,
        description="Available sensor field paths for this morphology",
    )
    act_fields: List[str] = Field(
        default_factory=list,
        description="Available actuator field paths for this morphology",
    )
    act_dims: int = Field(default=0, description="Total actuator dimensionality")
