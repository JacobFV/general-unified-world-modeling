"""Data models for the Corporate World Environment.

Three agent roles operate in a corporate world model:
- Employee: navigates career growth, manages stress and focus
- CEO: maximizes shareholder value through strategy and allocation
- HR: optimizes retention and workforce productivity

Each role observes and acts on different fields of the same world model.
"""

from typing import Dict, List, Optional
from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class CorporateWorldAction(Action):
    """Action for the Corporate World environment.

    The agent sends a role identifier and a list of continuous action values.
    Actions are mapped to the corresponding role's act_fields on the canvas.
    """

    role: str = Field(
        ...,
        description="Agent role: 'employee', 'ceo', or 'hr'",
    )
    values: List[float] = Field(
        ...,
        description="Continuous action values mapped to the role's act_fields",
    )


class CorporateWorldObservation(Observation):
    """Observation from the Corporate World environment.

    Returns the named observation values for the active role,
    plus reward and episode metadata.
    """

    role: str = Field(default="", description="Active agent role")
    obs_dict: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Observation values keyed by field path",
    )
    step_count: int = Field(default=0, description="Current step in episode")
    obs_fields: List[str] = Field(
        default_factory=list,
        description="Available observation field paths for this role",
    )
    act_fields: List[str] = Field(
        default_factory=list,
        description="Available action field paths for this role",
    )
    act_dims: int = Field(default=0, description="Total action dimensionality")
