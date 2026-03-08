"""WorldEnv client — connect to a deployed WorldEnv server."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import WorldAction, WorldObservation


class WorldEnv(EnvClient[WorldAction, WorldObservation]):
    """Client for the WorldEnv environment.

    Sample any RL scenario from the General Unified World Model by
    connecting to a deployed WorldEnv server.

    Example:
        >>> with WorldEnv(base_url="https://jacob-valdez-worldenv.hf.space") as env:
        ...     result = env.reset()
        ...     print(result.observation.scenario_name)
        ...     print(result.observation.obs_field_names)
        ...
        ...     action = WorldAction(
        ...         action_type="intervene",
        ...         target_field="firm.strategy.rd_intensity",
        ...         value=0.8,
        ...     )
        ...     result = env.step(action)
        ...     print(result.reward)
    """

    def _step_payload(self, action: WorldAction) -> Dict:
        return {
            "action_type": action.action_type,
            "target_field": action.target_field,
            "value": action.value,
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[WorldObservation]:
        obs_data = payload.get("observation", {})
        observation = WorldObservation(
            visible_fields=obs_data.get("visible_fields", {}),
            step_count=obs_data.get("step_count", 0),
            scenario_name=obs_data.get("scenario_name", ""),
            info=obs_data.get("info", ""),
            available_scenarios=obs_data.get("available_scenarios", []),
            obs_field_names=obs_data.get("obs_field_names", []),
            act_field_names=obs_data.get("act_field_names", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
