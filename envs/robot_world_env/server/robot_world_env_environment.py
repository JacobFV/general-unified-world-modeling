"""Robot World Environment — three morphologies, one physical dynamics model."""

import math
import random
from typing import Optional
from uuid import uuid4

import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import RobotWorldAction, RobotWorldObservation

MORPHOLOGY_SPECS = {
    "logistics": {
        "obs_fields": [
            "physical.infrastructure.shipping_lane_capacity",
            "physical.infrastructure.port_congestion",
            "physical.infrastructure.air_freight_utilization",
            "physical.infrastructure.rail_freight_network",
            "physical.infrastructure.chokepoint_risk",
        ],
        "act_fields": [
            "physical.infrastructure.shipping_lane_capacity",
            "physical.infrastructure.rail_freight_network",
        ],
    },
    "disaster": {
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
    },
    "climate": {
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
    },
}


def _logistics_reward(obs):
    cap = obs.get("physical.infrastructure.shipping_lane_capacity", [0])[0]
    cong = obs.get("physical.infrastructure.port_congestion", [0])[0]
    return float(cap - 0.8 * cong)


def _disaster_reward(obs):
    risk = sum(abs(obs.get(k, [0])[0]) for k in [
        "physical.disasters.active_disaster_state",
        "physical.disasters.wildfire_state",
        "physical.disasters.pandemic_risk",
    ])
    return float(-risk + 1.0)


def _climate_reward(obs):
    cable = obs.get("physical.infrastructure.undersea_cable_topology", [0])[0]
    anomaly = abs(obs.get("physical.climate.global_temp_anomaly", [0])[0])
    return float(cable - 0.5 * anomaly)


REWARD_FNS = {"logistics": _logistics_reward, "disaster": _disaster_reward, "climate": _climate_reward}


class RobotWorldEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._model = None
        self._envs = {}
        self._current_morphology = "logistics"
        self._max_steps = 200
        self._obs_cache = {}

    def _ensure_model(self):
        if self._model is not None:
            return
        from general_unified_world_model import GeneralUnifiedWorldModel
        self._model = GeneralUnifiedWorldModel(
            include=["physical.climate", "physical.infrastructure", "physical.disasters",
                     "resources", "infrastructure", "technology", "regime"],
            d_model=32, n_layers=2, n_heads=2, n_loops=1,
        )
        for morphology, spec in MORPHOLOGY_SPECS.items():
            rf = REWARD_FNS[morphology]
            self._envs[morphology] = self._model.to_openenv(
                obs_fields=spec["obs_fields"],
                act_fields=spec["act_fields"],
                reward_fn=lambda obs, act, info, r=rf: r(obs),
                max_steps=self._max_steps,
                n_denoise_steps=3,
            )

    def reset(self, seed=None, episode_id=None, **kwargs):
        self._ensure_model()
        morphology = kwargs.get("morphology", "logistics")
        self._current_morphology = morphology if morphology in MORPHOLOGY_SPECS else "logistics"
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        env = self._envs[self._current_morphology]
        obs, info = env.reset(seed=seed)
        obs_dict = info.get("obs_dict", {})
        self._obs_cache = {k: v.tolist() for k, v in obs_dict.items()}
        spec = MORPHOLOGY_SPECS[self._current_morphology]
        act_dim = sum(env._act_sizes.get(fp, 1) for fp in spec["act_fields"])
        return RobotWorldObservation(
            morphology=self._current_morphology,
            obs_dict=self._obs_cache, step_count=0,
            obs_fields=spec["obs_fields"], act_fields=spec["act_fields"], act_dims=act_dim,
            done=False, reward=0.0,
        )

    def step(self, action, timeout_s=None, **kwargs):
        self._ensure_model()
        morphology = action.morphology if action.morphology in MORPHOLOGY_SPECS else self._current_morphology
        self._current_morphology = morphology
        self._state.step_count += 1
        env = self._envs[morphology]
        spec = MORPHOLOGY_SPECS[morphology]
        act_dim = sum(env._act_sizes.get(fp, 1) for fp in spec["act_fields"])
        values = list(action.values) + [0.0] * act_dim
        action_array = np.array(values[:act_dim], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action_array)
        obs_dict = info.get("obs_dict", {})
        self._obs_cache = {k: v.tolist() for k, v in obs_dict.items()}
        return RobotWorldObservation(
            morphology=morphology, obs_dict=self._obs_cache, step_count=self._state.step_count,
            obs_fields=spec["obs_fields"], act_fields=spec["act_fields"], act_dims=act_dim,
            done=terminated or truncated, reward=reward,
        )

    @property
    def state(self):
        return self._state
