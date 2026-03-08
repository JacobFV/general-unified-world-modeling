"""Corporate World Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import CorporateWorldAction, CorporateWorldObservation


class CorporateWorldEnv(
    EnvClient[CorporateWorldAction, CorporateWorldObservation]
):
    """Client for the Corporate World Environment.

    Connect to a running Corporate World env server and interact
    as one of three roles: employee, ceo, or hr.

    Example:
        >>> with CorporateWorldEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.role, result.observation.obs_fields)
        ...
        ...     action = CorporateWorldAction(role="ceo", values=[0.5, -0.2, 0.1, 0.0, 0.3])
        ...     result = client.step(action)
        ...     print(result.observation.obs_dict)
    """

    def _step_payload(self, action: CorporateWorldAction) -> Dict:
        return {
            "role": action.role,
            "values": action.values,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CorporateWorldObservation]:
        obs_data = payload.get("observation", {})
        observation = CorporateWorldObservation(
            role=obs_data.get("role", ""),
            obs_dict=obs_data.get("obs_dict", {}),
            step_count=obs_data.get("step_count", 0),
            obs_fields=obs_data.get("obs_fields", []),
            act_fields=obs_data.get("act_fields", []),
            act_dims=obs_data.get("act_dims", 0),
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
