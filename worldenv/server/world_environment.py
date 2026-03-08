"""WorldEnv — Environment server implementation.

Demonstrates sampling RL environments from a General Unified World Model
by projecting to scenario-specific field subsets.

Uses synthetic correlated dynamics for rapid demo deployment. Plug in
the full GUWM backbone (general-unified-world-model on PyPI) to get
genuine learned dynamics.
"""

import math
import random
import uuid
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import WorldAction, WorldObservation

# ── Scenario definitions ─────────────────────────────────────────
# Each scenario is a projection of the world model:
# obs_fields = what the agent observes
# act_fields = what the agent can control
# reward_field = what is being optimized

SCENARIOS = {
    "corporate_employee": {
        "description": (
            "Navigate corporate dynamics as an employee. "
            "Observe company metrics and peer dynamics. "
            "Make career and project decisions to grow."
        ),
        "obs_fields": [
            "firm.financials.revenue",
            "firm.financials.profit_margin",
            "firm.operations.headcount",
            "firm.operations.utilization",
            "firm.strategy.market_position",
            "sector.demand.growth_rate",
            "macro.gdp_growth",
            "macro.cpi_inflation",
            "person.cognitive.expertise_depth",
            "person.incentives.compensation",
            "person.state.stress",
            "person.state.confidence",
        ],
        "act_fields": [
            "person.cognitive.skill_investment",
            "person.network.team_contribution",
            "firm.operations.project_allocation",
        ],
        "reward_field": "person.incentives.career_growth",
    },
    "corporate_executive": {
        "description": (
            "Lead a company as CEO. Observe markets, competitors, macro. "
            "Make strategic decisions to maximize shareholder value."
        ),
        "obs_fields": [
            "firm.financials.revenue",
            "firm.financials.ebitda_margin",
            "firm.financials.cash_flow",
            "firm.market.market_share",
            "firm.risk.competitive_threat",
            "sector.demand.growth_rate",
            "sector.supply.capacity_utilization",
            "financial.equities.sector_pe",
            "macro.gdp_growth",
            "regime.growth_regime",
            "regime.financial_cycle",
            "events.earnings_surprise",
        ],
        "act_fields": [
            "firm.strategy.rd_intensity",
            "firm.strategy.pricing_power",
            "firm.operations.headcount_delta",
            "firm.operations.capex_allocation",
            "firm.strategy.market_expansion",
        ],
        "reward_field": "firm.financials.total_shareholder_return",
    },
    "macro_policy": {
        "description": (
            "Set economic policy. Observe macro indicators and financial conditions. "
            "Adjust monetary and fiscal levers to stabilize the economy."
        ),
        "obs_fields": [
            "macro.gdp_growth",
            "macro.cpi_inflation",
            "macro.unemployment_rate",
            "financial.yield_curves.ten_year",
            "financial.credit.ig_spread",
            "financial.equities.vix",
            "regime.growth_regime",
            "regime.inflation_regime",
            "narrative.consumer_confidence",
        ],
        "act_fields": [
            "interventions.monetary.rate_decision",
            "interventions.fiscal.spending_delta",
            "interventions.regulatory.tightening_index",
        ],
        "reward_field": "forecasts.macro.stability_index",
    },
    "logistics_robot": {
        "description": (
            "Optimize freight logistics. Observe shipping capacity, port congestion, "
            "and rail networks. Route cargo to maximize throughput."
        ),
        "obs_fields": [
            "physical.infrastructure.shipping_lane_capacity",
            "physical.infrastructure.port_congestion",
            "physical.infrastructure.air_freight_utilization",
            "physical.infrastructure.rail_freight_network",
            "physical.infrastructure.chokepoint_risk",
            "physical.climate.extreme_weather_freq",
        ],
        "act_fields": [
            "physical.infrastructure.shipping_lane_capacity",
            "physical.infrastructure.rail_freight_network",
        ],
        "reward_field": "logistics.throughput_index",
    },
    "disaster_robot": {
        "description": (
            "Coordinate disaster response. Observe seismic, wildfire, and pandemic risk. "
            "Deploy resources to mitigate active hazards."
        ),
        "obs_fields": [
            "physical.disasters.active_disaster_state",
            "physical.disasters.seismic_risk_structural",
            "physical.disasters.pandemic_risk",
            "physical.disasters.wildfire_state",
            "physical.disasters.volcanic_risk",
            "physical.climate.extreme_weather_freq",
        ],
        "act_fields": [
            "physical.disasters.active_disaster_state",
            "physical.disasters.wildfire_state",
        ],
        "reward_field": "disaster.mitigation_index",
    },
    "climate_monitor": {
        "description": (
            "Track climate dynamics. Observe temperature anomalies, carbon levels, "
            "and ENSO phase. Maintain monitoring infrastructure."
        ),
        "obs_fields": [
            "physical.climate.global_temp_anomaly",
            "physical.climate.carbon_ppm",
            "physical.climate.enso_phase",
            "physical.climate.sea_level_trend",
            "physical.climate.polar_vortex_stability",
            "physical.infrastructure.undersea_cable_topology",
        ],
        "act_fields": [
            "physical.infrastructure.undersea_cable_topology",
        ],
        "reward_field": "climate.monitoring_coverage",
    },
}


def _correlated_dynamics(field_values: dict, act_field: str, act_value: float, noise: float = 0.02) -> dict:
    """Apply correlated updates — actions ripple through related fields."""
    updated = dict(field_values)
    updated[act_field] = act_value
    for f in list(updated.keys()):
        # Small correlated drift based on the action
        coupling = math.sin(hash(f + act_field) % 7) * 0.1
        updated[f] = round(updated[f] + coupling * act_value * 0.05 + random.gauss(0, noise), 4)
    return updated


class WorldEnvironment(Environment):
    """WorldEnv server — sample any RL environment from a world model projection.

    On reset(), select a scenario (or random). The environment returns
    correlated field dynamics for that projection. Intervene on action
    fields to drive the world state.

    To upgrade from synthetic to learned dynamics:
        pip install general-unified-world-model
        # Then swap _init_field_values() and _step_field_values() to use
        # GeneralUnifiedWorldModel.predict() via the to_openenv() API.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._field_values: dict = {}
        self._scenario_config: Optional[dict] = None
        self._scenario_name: str = ""
        self._score: float = 0.0
        self._max_steps: int = 100

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario: Optional[str] = None,
        **kwargs,
    ) -> WorldObservation:
        if seed is not None:
            random.seed(seed)

        name = scenario if scenario in SCENARIOS else random.choice(list(SCENARIOS.keys()))
        self._scenario_name = name
        self._scenario_config = SCENARIOS[name]
        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )
        self._score = 0.0

        # Initialize field values with correlated synthetic dynamics
        base_phase = random.uniform(0, 2 * math.pi)
        all_fields = (
            self._scenario_config["obs_fields"]
            + self._scenario_config["act_fields"]
            + [self._scenario_config["reward_field"]]
        )
        self._field_values = {}
        for f in all_fields:
            cycle = 0.3 * math.sin(base_phase + hash(f) % 10 * 0.5)
            self._field_values[f] = round(cycle + random.gauss(0, 0.1), 4)

        return WorldObservation(
            visible_fields={
                f: self._field_values[f]
                for f in self._scenario_config["obs_fields"]
            },
            step_count=0,
            scenario_name=name,
            info=self._scenario_config["description"],
            available_scenarios=list(SCENARIOS.keys()),
            obs_field_names=self._scenario_config["obs_fields"],
            act_field_names=self._scenario_config["act_fields"],
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: WorldAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> WorldObservation:
        if self._scenario_config is None:
            return self.reset()

        self._state.step_count += 1
        reward = 0.0

        if action.action_type == "predict":
            if action.target_field in self._field_values:
                error = abs(action.value - self._field_values[action.target_field])
                reward = max(0.0, 1.0 - error) * 0.5

        elif action.action_type == "intervene":
            if action.target_field in self._scenario_config["act_fields"]:
                self._field_values = _correlated_dynamics(
                    self._field_values, action.target_field, action.value,
                )
                reward = 0.1

        elif action.action_type == "observe":
            reward = 0.01

        # Natural drift
        for f in list(self._field_values.keys()):
            self._field_values[f] = round(
                self._field_values[f] + random.gauss(0, 0.005), 4
            )

        # Reward field
        rf = self._scenario_config["reward_field"]
        act_sum = sum(
            self._field_values.get(f, 0)
            for f in self._scenario_config["act_fields"]
        )
        self._field_values[rf] = round(act_sum * 0.1 + reward, 4)
        self._score += reward

        done = self._state.step_count >= self._max_steps

        return WorldObservation(
            visible_fields={
                f: self._field_values[f]
                for f in self._scenario_config["obs_fields"]
            },
            step_count=self._state.step_count,
            scenario_name=self._scenario_name,
            info=(
                f"Step {self._state.step_count}/{self._max_steps} | "
                f"Score: {self._score:.3f} | "
                f"Scenario: {self._scenario_name}"
            ),
            obs_field_names=self._scenario_config["obs_fields"],
            act_field_names=self._scenario_config["act_fields"],
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state
