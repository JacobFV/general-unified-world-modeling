"""Robot World Environment Client."""

from typing import Dict
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient
from .models import RobotWorldAction, RobotWorldObservation


class RobotWorldEnv(EnvClient[RobotWorldAction, RobotWorldObservation]):
    """Client for the Robot World Environment.

    Example:
        >>> with RobotWorldEnv(base_url="https://jacob-valdez-robot-world-env.hf.space") as env:
        ...     result = env.reset()
        ...     print(result.observation.morphology, result.observation.obs_fields)
    """

    def _step_payload(self, action: RobotWorldAction) -> Dict:
        return {"morphology": action.morphology, "values": action.values}

    def _parse_result(self, payload: Dict) -> StepResult[RobotWorldObservation]:
        obs_data = payload.get("observation", {})
        observation = RobotWorldObservation(
            morphology=obs_data.get("morphology", ""),
            obs_dict=obs_data.get("obs_dict", {}),
            step_count=obs_data.get("step_count", 0),
            obs_fields=obs_data.get("obs_fields", []),
            act_fields=obs_data.get("act_fields", []),
            act_dims=obs_data.get("act_dims", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict) -> State:
        return State(episode_id=payload.get("episode_id"), step_count=payload.get("step_count", 0))
