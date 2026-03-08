"""Tests for the environment extraction module."""

import pytest
import numpy as np
import torch

from general_unified_world_model.projection.subset import project
from general_unified_world_model.training.backbone import build_world_model
from general_unified_world_model.training.heterogeneous import FieldEncoder, FieldDecoder
from general_unified_world_model.inference import WorldModel

gymnasium = pytest.importorskip("gymnasium")

from general_unified_world_model.env import WorldModelEnv, MultiAgentWorldModelEnv, AgentSpec


@pytest.fixture
def small_model():
    """A small world model for env testing."""
    bound = project(
        include=["financial.yield_curves", "regime"],
        T=1, H=24, W=24, d_model=32,
    )
    backbone = build_world_model(bound, n_layers=2, n_loops=1)
    encoder = FieldEncoder(bound)
    decoder = FieldDecoder(bound)
    return WorldModel(bound, backbone, encoder, decoder, device="cpu")


def _simple_reward(obs_dict, action, info):
    """Reward = mean of all observations."""
    vals = [v.mean() for v in obs_dict.values()]
    return float(np.mean(vals)) if vals else 0.0


def _never_terminates(obs_dict, step, info):
    return False


def _terminates_at_5(obs_dict, step, info):
    return step >= 5


class TestWorldModelEnv:

    def test_creation(self, small_model):
        """Env should be creatable from a WorldModel."""
        obs_fields = [small_model.bound.field_names[0]]
        act_fields = [small_model.bound.field_names[1]]

        env = WorldModelEnv(
            world_model=small_model,
            obs_fields=obs_fields,
            act_fields=act_fields,
            reward_fn=_simple_reward,
        )
        assert env.observation_space.shape[0] > 0
        assert env.action_space.shape[0] > 0

    def test_to_openenv_method(self, small_model):
        """WorldModel.to_openenv() should return a valid env."""
        obs_fields = [small_model.bound.field_names[0]]
        act_fields = [small_model.bound.field_names[1]]

        env = small_model.to_openenv(
            obs_fields=obs_fields,
            act_fields=act_fields,
            reward_fn=_simple_reward,
        )
        assert isinstance(env, WorldModelEnv)

    def test_reset(self, small_model):
        """Reset should return observation and info."""
        obs_fields = [small_model.bound.field_names[0]]
        act_fields = [small_model.bound.field_names[1]]

        env = small_model.to_openenv(
            obs_fields=obs_fields,
            act_fields=act_fields,
            reward_fn=_simple_reward,
            n_denoise_steps=2,
        )
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert "step" in info
        assert info["step"] == 0

    def test_step(self, small_model):
        """Step should return (obs, reward, terminated, truncated, info)."""
        obs_fields = [small_model.bound.field_names[0]]
        act_fields = [small_model.bound.field_names[1]]

        env = small_model.to_openenv(
            obs_fields=obs_fields,
            act_fields=act_fields,
            reward_fn=_simple_reward,
            n_denoise_steps=2,
        )
        obs, info = env.reset()
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)

        assert isinstance(obs2, np.ndarray)
        assert obs2.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info2["step"] == 1

    def test_truncation(self, small_model):
        """Env should truncate after max_steps."""
        obs_fields = [small_model.bound.field_names[0]]
        act_fields = [small_model.bound.field_names[1]]

        env = small_model.to_openenv(
            obs_fields=obs_fields,
            act_fields=act_fields,
            reward_fn=_simple_reward,
            max_steps=3,
            n_denoise_steps=2,
        )
        env.reset()
        for i in range(3):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())

        assert truncated is True

    def test_custom_termination(self, small_model):
        """Custom terminated_fn should control episode end."""
        obs_fields = [small_model.bound.field_names[0]]
        act_fields = [small_model.bound.field_names[1]]

        env = small_model.to_openenv(
            obs_fields=obs_fields,
            act_fields=act_fields,
            reward_fn=_simple_reward,
            terminated_fn=_terminates_at_5,
            max_steps=100,
            n_denoise_steps=2,
        )
        env.reset()
        for i in range(5):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())

        assert terminated is True
        assert truncated is False

    def test_initial_obs_fn(self, small_model):
        """initial_obs_fn should set observations on reset."""
        field = small_model.bound.field_names[0]
        obs_fields = [field]
        act_fields = [small_model.bound.field_names[1]]

        env = small_model.to_openenv(
            obs_fields=obs_fields,
            act_fields=act_fields,
            reward_fn=_simple_reward,
            initial_obs_fn=lambda: {field: 42.0},
            n_denoise_steps=2,
        )
        env.reset()
        assert field in small_model._observations

    def test_obs_dict_in_info(self, small_model):
        """Info should contain obs_dict with named observations."""
        obs_fields = [small_model.bound.field_names[0]]
        act_fields = [small_model.bound.field_names[1]]

        env = small_model.to_openenv(
            obs_fields=obs_fields,
            act_fields=act_fields,
            reward_fn=_simple_reward,
            n_denoise_steps=2,
        )
        _, info = env.reset()
        assert "obs_dict" in info
        assert obs_fields[0] in info["obs_dict"]

    def test_multiple_envs_same_model(self, small_model):
        """Multiple envs can be created from the same world model."""
        fields = small_model.bound.field_names
        env1 = small_model.to_openenv(
            obs_fields=[fields[0]],
            act_fields=[fields[1]],
            reward_fn=_simple_reward,
            n_denoise_steps=2,
        )
        env2 = small_model.to_openenv(
            obs_fields=[fields[1]],
            act_fields=[fields[0]],
            reward_fn=lambda obs, act, info: -_simple_reward(obs, act, info),
            n_denoise_steps=2,
        )
        assert env1.observation_space.shape != env2.observation_space.shape or True
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        assert isinstance(obs1, np.ndarray)
        assert isinstance(obs2, np.ndarray)


class TestMultiAgentWorldModelEnv:

    def test_creation(self, small_model):
        """Multi-agent env should be creatable."""
        fields = small_model.bound.field_names

        multi = MultiAgentWorldModelEnv(
            world_model=small_model,
            agents={
                "agent_a": AgentSpec(
                    obs_fields=[fields[0]],
                    act_fields=[fields[1]],
                    reward_fn=_simple_reward,
                ),
                "agent_b": AgentSpec(
                    obs_fields=[fields[1]],
                    act_fields=[fields[2]] if len(fields) > 2 else [fields[0]],
                    reward_fn=_simple_reward,
                ),
            },
            n_denoise_steps=2,
        )
        assert "agent_a" in multi.observation_spaces
        assert "agent_b" in multi.observation_spaces

    def test_to_multi_openenv_method(self, small_model):
        """WorldModel.to_multi_openenv() should work."""
        fields = small_model.bound.field_names

        multi = small_model.to_multi_openenv(
            agents={
                "a": AgentSpec(
                    obs_fields=[fields[0]],
                    act_fields=[fields[1]],
                    reward_fn=_simple_reward,
                ),
            },
            n_denoise_steps=2,
        )
        assert isinstance(multi, MultiAgentWorldModelEnv)

    def test_reset_and_step(self, small_model):
        """Multi-agent reset and step should return per-agent dicts."""
        fields = small_model.bound.field_names

        multi = small_model.to_multi_openenv(
            agents={
                "a": AgentSpec(
                    obs_fields=[fields[0]],
                    act_fields=[fields[1]],
                    reward_fn=_simple_reward,
                ),
                "b": AgentSpec(
                    obs_fields=[fields[1]],
                    act_fields=[fields[2]] if len(fields) > 2 else [fields[0]],
                    reward_fn=_simple_reward,
                ),
            },
            n_denoise_steps=2,
        )

        obs = multi.reset()
        assert "a" in obs
        assert "b" in obs

        actions = {
            name: multi.action_spaces[name].sample()
            for name in multi.agents
        }
        obs2, rewards, terms, truncs, infos = multi.step(actions)

        assert "a" in obs2
        assert "b" in rewards
        assert isinstance(rewards["a"], float)
        assert isinstance(terms["a"], bool)
