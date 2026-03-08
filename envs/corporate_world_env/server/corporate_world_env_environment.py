"""Corporate World Environment Implementation.

A multi-role RL environment backed by a GeneralUnifiedWorldModel.
Three agent perspectives share the same learned corporate dynamics:

- Employee: obs=satisfaction/stress/confidence, act=focus/risk_appetite
- CEO: obs=revenue/margins/equity/health, act=allocation/capex/M&A
- HR: obs=satisfaction/headcount/risk, act=headcount/compensation

The world model predicts the next state; reward is computed per-role.
"""

from uuid import uuid4
from typing import Optional

import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import CorporateWorldAction, CorporateWorldObservation

# Role definitions: obs_fields, act_fields per role
ROLE_SPECS = {
    "employee": {
        "obs_fields": [
            "firm_acme.operations.employee_satisfaction",
            "firm_acme.operations.utilization",
            "person_employee.state.stress",
            "person_employee.state.confidence",
            "person_employee.state.current_focus",
            "person_employee.incentives.career_incentives",
            "person_employee.incentives.reputation_concerns",
            "person_employee.network.network_centrality",
        ],
        "act_fields": [
            "person_employee.state.current_focus",
            "person_employee.cognitive.risk_appetite",
            "person_employee.incentives.peer_pressure",
        ],
    },
    "ceo": {
        "obs_fields": [
            "firm_acme.financials.revenue",
            "firm_acme.financials.revenue_growth",
            "firm_acme.financials.operating_margin",
            "firm_acme.financials.fcf",
            "firm_acme.financials.net_debt_to_ebitda",
            "firm_acme.market.equity_price",
            "firm_acme.market.analyst_consensus",
            "firm_acme.operations.utilization",
            "firm_acme.operations.headcount",
            "firm_acme.latent_health",
            "firm_acme.latent_momentum",
            "firm_acme.latent_tail_risk",
            "regime.growth_regime",
            "regime.volatility",
        ],
        "act_fields": [
            "firm_acme.strategy.capital_allocation",
            "firm_acme.strategy.capex_plan",
            "firm_acme.strategy.m_and_a_appetite",
            "firm_acme.strategy.geographic_expansion",
            "firm_acme.operations.capacity",
        ],
    },
    "hr": {
        "obs_fields": [
            "firm_acme.operations.employee_satisfaction",
            "firm_acme.operations.headcount",
            "firm_acme.operations.quality_incidents",
            "firm_acme.operations.tech_debt",
            "firm_acme.risk.key_person_risk",
            "firm_acme.financials.opex",
            "person_employee.state.stress",
            "person_employee.state.confidence",
        ],
        "act_fields": [
            "firm_acme.operations.headcount",
            "firm_acme.operations.employee_satisfaction",
            "person_employee.incentives.compensation_structure",
        ],
    },
}


def _compute_reward(role: str, obs_dict: dict) -> float:
    """Compute reward based on role."""
    def _get(key, default=0.0):
        vals = obs_dict.get(key, [default])
        return float(np.mean(vals)) if len(vals) > 0 else default

    if role == "employee":
        return (
            _get("firm_acme.operations.employee_satisfaction")
            + _get("person_employee.state.confidence")
            - 0.5 * _get("person_employee.state.stress")
            + 0.3 * _get("person_employee.incentives.reputation_concerns")
        )
    elif role == "ceo":
        return (
            _get("firm_acme.financials.revenue_growth")
            + 0.5 * _get("firm_acme.financials.operating_margin")
            + 0.3 * _get("firm_acme.market.equity_price")
            + 0.2 * _get("firm_acme.latent_health")
        )
    elif role == "hr":
        return (
            _get("firm_acme.operations.employee_satisfaction")
            + 0.3 * _get("firm_acme.operations.headcount")
            - 0.5 * _get("firm_acme.operations.quality_incidents")
            - 0.4 * _get("firm_acme.risk.key_person_risk")
        )
    return 0.0


class CorporateWorldEnvironment(Environment):
    """Corporate world environment backed by a GeneralUnifiedWorldModel.

    On first use, builds a world model with firm + individual entities
    and extracts per-role Gymnasium envs. The OpenEnv step/reset API
    delegates to the appropriate role's env.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._model = None
        self._envs = {}
        self._current_role = "employee"
        self._max_steps = 200
        self._obs_cache = {}

    def _ensure_model(self):
        """Lazy-build the world model and per-role envs."""
        if self._model is not None:
            return

        from general_unified_world_model import GeneralUnifiedWorldModel
        from general_unified_world_model.schema.business import Business
        from general_unified_world_model.schema.individual import Individual

        self._model = GeneralUnifiedWorldModel(
            include=["regime", "financial.equities", "financial.credit", "sector_tech"],
            entities={
                "firm_acme": Business(),
                "person_ceo": Individual(),
                "person_employee": Individual(),
                "person_hr_director": Individual(),
            },
            d_model=32,
            n_layers=2,
            n_heads=2,
            n_loops=1,
        )

        for role, spec in ROLE_SPECS.items():
            self._envs[role] = self._model.to_openenv(
                obs_fields=spec["obs_fields"],
                act_fields=spec["act_fields"],
                reward_fn=lambda obs, act, info, r=role: _compute_reward(r, obs),
                max_steps=self._max_steps,
                n_denoise_steps=3,
            )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> CorporateWorldObservation:
        self._ensure_model()
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        role = kwargs.get("role", "employee")
        self._current_role = role if role in ROLE_SPECS else "employee"

        env = self._envs[self._current_role]
        obs, info = env.reset(seed=seed)

        obs_dict = info.get("obs_dict", {})
        self._obs_cache = {k: v.tolist() for k, v in obs_dict.items()}

        spec = ROLE_SPECS[self._current_role]
        act_dim = sum(
            env._act_sizes.get(fp, 1) for fp in spec["act_fields"]
        )

        return CorporateWorldObservation(
            role=self._current_role,
            obs_dict=self._obs_cache,
            step_count=0,
            obs_fields=spec["obs_fields"],
            act_fields=spec["act_fields"],
            act_dims=act_dim,
            done=False,
            reward=0.0,
        )

    def step(self, action: CorporateWorldAction, **kwargs) -> CorporateWorldObservation:
        self._ensure_model()
        role = action.role if action.role in ROLE_SPECS else self._current_role
        self._current_role = role
        self._state.step_count += 1

        env = self._envs[role]
        spec = ROLE_SPECS[role]

        # Pad/trim action values to match act_dim
        act_dim = sum(env._act_sizes.get(fp, 1) for fp in spec["act_fields"])
        values = list(action.values)
        if len(values) < act_dim:
            values.extend([0.0] * (act_dim - len(values)))
        action_array = np.array(values[:act_dim], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action_array)

        obs_dict = info.get("obs_dict", {})
        self._obs_cache = {k: v.tolist() for k, v in obs_dict.items()}

        return CorporateWorldObservation(
            role=role,
            obs_dict=self._obs_cache,
            step_count=self._state.step_count,
            obs_fields=spec["obs_fields"],
            act_fields=spec["act_fields"],
            act_dims=act_dim,
            done=terminated or truncated,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state
