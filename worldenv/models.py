"""WorldEnv data models — action and observation for world model environments."""

import math
import random
from typing import Any, Dict, List, Optional
from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class WorldAction(Action):
    """Agent action in the world environment.

    Supports four action types:
    - observe: Examine a specific field value
    - predict: Predict a field value (rewarded for accuracy)
    - intervene: Set an action field (drives dynamics)
    - step: Advance time without acting
    """

    action_type: str = Field(
        default="step",
        description="One of: observe, predict, intervene, step",
    )
    target_field: str = Field(
        default="",
        description="Schema field path to observe/predict/intervene on",
    )
    value: float = Field(
        default=0.0,
        description="Value for predict/intervene actions",
    )
    message: str = Field(
        default="",
        description="Free-form text action for LLM agents (JSON-formatted action)",
    )


class WorldObservation(Observation):
    """Observation from the world environment.

    Returns visible field values for the current scenario projection,
    plus episode metadata.
    """

    visible_fields: Dict[str, float] = Field(
        default_factory=dict,
        description="Field path -> current value for observable fields",
    )
    step_count: int = Field(default=0, description="Current step in episode")
    scenario_name: str = Field(default="", description="Active scenario name")
    info: str = Field(default="", description="Human-readable episode status")
    available_scenarios: List[str] = Field(
        default_factory=list,
        description="All available scenario names (present on reset)",
    )
    obs_field_names: List[str] = Field(
        default_factory=list,
        description="Observable field paths for this scenario",
    )
    act_field_names: List[str] = Field(
        default_factory=list,
        description="Actionable field paths for this scenario",
    )
