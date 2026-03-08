"""Gymnasium environments extracted from WorldModel instances.

A trained WorldModel captures dynamics over a canvas of fields. This module
lets you carve out RL environments from that learned dynamics by selecting
which fields the agent observes, which it acts on, and how reward is computed.

The same world model can yield many different environments — an employee
navigation env, a CEO strategy env, a robot control env — all sharing the
same underlying latent dynamics.

Usage:
    model = GeneralUnifiedWorldModel(include=[...])
    model.finetune(data)

    env = model.to_openenv(
        obs_fields=["firm.operations.employee_satisfaction", "firm.financials.revenue"],
        act_fields=["firm.strategy.capital_allocation", "firm.operations.capacity"],
        reward_fn=lambda obs, act, info: obs["firm.financials.revenue"].mean(),
    )

    obs, info = env.reset()
    for _ in range(100):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYMNASIUM = True
    except ImportError:
        HAS_GYMNASIUM = False


def _require_gymnasium():
    if not HAS_GYMNASIUM:
        raise ImportError(
            "gymnasium is required for WorldModelEnv. "
            "Install it with: pip install gymnasium"
        )


class WorldModelEnv(gym.Env if HAS_GYMNASIUM else object):
    """Gymnasium environment backed by a WorldModel's learned dynamics.

    The world model acts as the simulator. At each step:
    1. Agent actions are written to act_fields on the canvas
    2. The world model predicts the next state (forward pass)
    3. Observations are read from obs_fields
    4. Reward is computed from the observation dict

    This lets you extract arbitrarily many agent perspectives from
    a single trained world model.

    Args:
        world_model: A trained WorldModel instance (shared, not copied).
        obs_fields: Field paths the agent can observe.
        act_fields: Field paths the agent can act on.
        reward_fn: (obs_dict, action_array, info_dict) -> float.
        terminated_fn: Optional (obs_dict, step, info_dict) -> bool.
            Defaults to never terminating (episode ends by truncation).
        max_steps: Maximum episode length before truncation.
        n_denoise_steps: Diffusion denoising steps per predict() call.
            Lower = faster but noisier. 10 is a good default for RL.
        act_low: Lower bound for continuous action space.
        act_high: Upper bound for continuous action space.
        initial_obs_fn: Optional callable returning dict of initial
            observations to set on reset. If None, canvas starts at zero.
        render_mode: Gymnasium render mode (None, "human", "rgb_array").
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        world_model,
        obs_fields: list[str],
        act_fields: list[str],
        reward_fn: Callable[[dict[str, np.ndarray], np.ndarray, dict], float],
        terminated_fn: Callable[[dict[str, np.ndarray], int, dict], bool] | None = None,
        max_steps: int = 200,
        n_denoise_steps: int = 10,
        act_low: float = -1.0,
        act_high: float = 1.0,
        initial_obs_fn: Callable[[], dict[str, Any]] | None = None,
        render_mode: str | None = None,
    ):
        _require_gymnasium()
        super().__init__()

        self.world_model = world_model
        self.obs_fields = obs_fields
        self.act_fields = act_fields
        self.reward_fn = reward_fn
        self.terminated_fn = terminated_fn
        self.max_steps = max_steps
        self.n_denoise_steps = n_denoise_steps
        self.initial_obs_fn = initial_obs_fn
        self.render_mode = render_mode

        # Build field → position count maps
        self._obs_sizes = {}
        self._obs_dim = 0
        for fp in obs_fields:
            try:
                bf = world_model.bound[fp]
                n = bf.num_positions
            except (KeyError, AttributeError):
                n = 1
            self._obs_sizes[fp] = n
            self._obs_dim += n

        self._act_sizes = {}
        self._act_dim = 0
        for fp in act_fields:
            try:
                bf = world_model.bound[fp]
                n = bf.num_positions
            except (KeyError, AttributeError):
                n = 1
            self._act_sizes[fp] = n
            self._act_dim += n

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=float(act_low), high=float(act_high),
            shape=(self._act_dim,), dtype=np.float32,
        )

        self._step_count = 0
        self._last_obs_dict: dict[str, np.ndarray] = {}
        self._last_predictions: dict[str, torch.Tensor] = {}

    # ── Core Gymnasium API ───────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment. Clears canvas and returns initial obs."""
        super().reset(seed=seed)
        self._step_count = 0
        self.world_model.clear_observations()

        # Apply initial observations if provided
        if self.initial_obs_fn is not None:
            init_obs = self.initial_obs_fn()
            for fp, val in init_obs.items():
                self.world_model.observe(fp, val)

        # Run one prediction to populate the canvas
        self._last_predictions = self.world_model.predict(
            n_steps=self.n_denoise_steps,
        )

        obs_vec, obs_dict = self._read_obs()
        self._last_obs_dict = obs_dict

        info = {
            "step": 0,
            "predictions": {k: v.numpy() for k, v in self._last_predictions.items()},
            "obs_dict": obs_dict,
        }
        return obs_vec, info

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step.

        1. Write action values to act_fields on the canvas
        2. Predict next state via world model
        3. Read observations from obs_fields
        4. Compute reward and check termination
        """
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32)

        # 1. Write actions to canvas (per-position scalars)
        offset = 0
        for fp in self.act_fields:
            n = self._act_sizes[fp]
            act_slice = action[offset:offset + n]
            try:
                bf = self.world_model.bound[fp]
                indices = bf.indices()
                for i, idx in enumerate(indices):
                    if i < len(act_slice):
                        val_t = torch.tensor([[[float(act_slice[i])]]],
                                             dtype=torch.float32,
                                             device=self.world_model.device)
                        enc = self.world_model.encoder(fp, val_t)
                        self.world_model._canvas[0, idx] = enc[0, 0]
                # Store in observations dict for conditioning
                self.world_model._observations[fp] = torch.tensor(
                    act_slice, dtype=torch.float32,
                )
            except (KeyError, AttributeError):
                # Fallback: observe as scalar (first value)
                self.world_model.observe(fp, float(act_slice[0]))
            offset += n

        # 2. Predict next state
        self._last_predictions = self.world_model.predict(
            n_steps=self.n_denoise_steps,
        )

        # Write predictions back as observations for next step
        for fp, val in self._last_predictions.items():
            if fp not in self.act_fields:
                self.world_model.observe(fp, val)

        # 3. Read observations
        obs_vec, obs_dict = self._read_obs()
        self._last_obs_dict = obs_dict

        info = {
            "step": self._step_count,
            "predictions": {k: v.numpy() for k, v in self._last_predictions.items()},
            "obs_dict": obs_dict,
        }

        # 4. Reward and termination
        reward = float(self.reward_fn(obs_dict, action, info))

        terminated = False
        if self.terminated_fn is not None:
            terminated = bool(self.terminated_fn(obs_dict, self._step_count, info))

        truncated = self._step_count >= self.max_steps

        return obs_vec, reward, terminated, truncated, info

    def render(self):
        """Render the current canvas state."""
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        elif self.render_mode == "human":
            self._render_human()
        return None

    def close(self):
        """Clean up."""
        pass

    # ── Internals ────────────────────────────────────────────────────

    def _read_obs(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Read observation fields from the latest predictions/canvas."""
        obs_parts = []
        obs_dict = {}

        for fp in self.obs_fields:
            n = self._obs_sizes[fp]
            if fp in self._last_predictions:
                val = self._last_predictions[fp].detach().cpu().numpy().flatten()
            else:
                # Fall back to canvas read
                try:
                    bf = self.world_model.bound[fp]
                    indices = bf.indices()
                    canvas = self.world_model.get_canvas()
                    val = canvas[0, indices, 0].detach().cpu().numpy()
                except (KeyError, AttributeError):
                    val = np.zeros(n, dtype=np.float32)

            # Pad or trim to expected size
            if len(val) < n:
                val = np.pad(val, (0, n - len(val)))
            elif len(val) > n:
                val = val[:n]

            obs_parts.append(val.astype(np.float32))
            obs_dict[fp] = val.astype(np.float32)

        return np.concatenate(obs_parts), obs_dict

    def _render_rgb(self) -> np.ndarray:
        """Render canvas as an RGB image for visualization."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Left: observation fields as bar chart
            obs_labels = []
            obs_vals = []
            for fp in self.obs_fields:
                if fp in self._last_obs_dict:
                    v = self._last_obs_dict[fp]
                    for i, val in enumerate(v):
                        obs_labels.append(f"{fp.split('.')[-1]}[{i}]" if len(v) > 1
                                         else fp.split(".")[-1])
                        obs_vals.append(float(val))

            if obs_vals:
                colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in obs_vals]
                axes[0].barh(range(len(obs_vals)), obs_vals, color=colors)
                axes[0].set_yticks(range(len(obs_labels)))
                axes[0].set_yticklabels(obs_labels, fontsize=7)
            axes[0].set_title(f"Observations (step {self._step_count})")

            # Right: canvas heatmap
            canvas = self.world_model.get_canvas()[0].detach().cpu().numpy()
            H = self.world_model.bound.layout.H
            W = self.world_model.bound.layout.W
            # Show first channel as heatmap
            img = canvas[:H * W, 0].reshape(H, W)
            axes[1].imshow(img, cmap="viridis", aspect="auto")
            axes[1].set_title("Canvas (channel 0)")

            fig.tight_layout()
            canvas_agg = FigureCanvasAgg(fig)
            canvas_agg.draw()
            buf = np.asarray(canvas_agg.buffer_rgba())
            plt.close(fig)
            return buf[:, :, :3]
        except ImportError:
            return np.zeros((100, 200, 3), dtype=np.uint8)

    def _render_human(self):
        """Print a text summary of current state."""
        print(f"--- Step {self._step_count} ---")
        for fp in self.obs_fields:
            if fp in self._last_obs_dict:
                vals = self._last_obs_dict[fp]
                print(f"  {fp}: {vals}")
        print()


class MultiAgentWorldModelEnv:
    """Multiple Gymnasium envs sharing a single WorldModel.

    Each agent has its own observation and action fields carved from the
    same underlying dynamics model. After all agents act, a single
    predict() call advances the shared world state.

    Usage:
        multi = MultiAgentWorldModelEnv(
            world_model=model,
            agents={
                "employee": AgentSpec(
                    obs_fields=["firm.operations.employee_satisfaction"],
                    act_fields=["person_alice.state.current_focus"],
                    reward_fn=employee_reward,
                ),
                "ceo": AgentSpec(
                    obs_fields=["firm.financials.revenue", "firm.market.equity_price"],
                    act_fields=["firm.strategy.capital_allocation"],
                    reward_fn=ceo_reward,
                ),
            },
        )
        obs = multi.reset()  # {"employee": ..., "ceo": ...}
        obs, rewards, terms, truncs, infos = multi.step({
            "employee": employee_action,
            "ceo": ceo_action,
        })
    """

    def __init__(
        self,
        world_model,
        agents: dict[str, "AgentSpec"],
        max_steps: int = 200,
        n_denoise_steps: int = 10,
        initial_obs_fn: Callable[[], dict[str, Any]] | None = None,
    ):
        _require_gymnasium()
        self.world_model = world_model
        self.agents = agents
        self.max_steps = max_steps
        self.n_denoise_steps = n_denoise_steps
        self.initial_obs_fn = initial_obs_fn
        self._step_count = 0
        self._last_predictions: dict[str, torch.Tensor] = {}

        # Build per-agent spaces
        self.observation_spaces = {}
        self.action_spaces = {}
        self._agent_obs_sizes: dict[str, dict[str, int]] = {}
        self._agent_act_sizes: dict[str, dict[str, int]] = {}

        for name, spec in agents.items():
            obs_sizes = {}
            obs_dim = 0
            for fp in spec.obs_fields:
                try:
                    n = world_model.bound[fp].num_positions
                except (KeyError, AttributeError):
                    n = 1
                obs_sizes[fp] = n
                obs_dim += n
            self._agent_obs_sizes[name] = obs_sizes

            act_sizes = {}
            act_dim = 0
            for fp in spec.act_fields:
                try:
                    n = world_model.bound[fp].num_positions
                except (KeyError, AttributeError):
                    n = 1
                act_sizes[fp] = n
                act_dim += n
            self._agent_act_sizes[name] = act_sizes

            self.observation_spaces[name] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            )
            self.action_spaces[name] = spaces.Box(
                low=spec.act_low, high=spec.act_high,
                shape=(act_dim,), dtype=np.float32,
            )

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset all agents. Returns dict of agent_name -> obs."""
        self._step_count = 0
        self.world_model.clear_observations()

        if self.initial_obs_fn is not None:
            for fp, val in self.initial_obs_fn().items():
                self.world_model.observe(fp, val)

        self._last_predictions = self.world_model.predict(
            n_steps=self.n_denoise_steps,
        )

        obs = {}
        for name in self.agents:
            obs[name] = self._read_agent_obs(name)
        return obs

    def step(
        self, actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """All agents act simultaneously, then world advances."""
        self._step_count += 1

        # Write all agent actions to canvas (per-position scalars)
        for name, action in actions.items():
            spec = self.agents[name]
            action = np.asarray(action, dtype=np.float32)
            offset = 0
            for fp in spec.act_fields:
                n = self._agent_act_sizes[name][fp]
                act_slice = action[offset:offset + n]
                try:
                    bf = self.world_model.bound[fp]
                    indices = bf.indices()
                    for i, idx in enumerate(indices):
                        if i < len(act_slice):
                            val_t = torch.tensor([[[float(act_slice[i])]]],
                                                 dtype=torch.float32,
                                                 device=self.world_model.device)
                            enc = self.world_model.encoder(fp, val_t)
                            self.world_model._canvas[0, idx] = enc[0, 0]
                    self.world_model._observations[fp] = torch.tensor(
                        act_slice, dtype=torch.float32,
                    )
                except (KeyError, AttributeError):
                    self.world_model.observe(fp, float(act_slice[0]))
                offset += n

        # Single shared prediction
        self._last_predictions = self.world_model.predict(
            n_steps=self.n_denoise_steps,
        )

        # Persist predictions for next step
        for fp, val in self._last_predictions.items():
            all_act_fields = set()
            for spec in self.agents.values():
                all_act_fields.update(spec.act_fields)
            if fp not in all_act_fields:
                self.world_model.observe(fp, val)

        truncated = self._step_count >= self.max_steps

        obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}
        for name, spec in self.agents.items():
            obs_vec = self._read_agent_obs(name)
            obs_dict = self._read_agent_obs_dict(name)
            action = actions.get(name, np.zeros(self.action_spaces[name].shape))

            info = {
                "step": self._step_count,
                "obs_dict": obs_dict,
            }

            obs[name] = obs_vec
            rewards[name] = float(spec.reward_fn(obs_dict, action, info))
            terminateds[name] = (
                bool(spec.terminated_fn(obs_dict, self._step_count, info))
                if spec.terminated_fn else False
            )
            truncateds[name] = truncated
            infos[name] = info

        return obs, rewards, terminateds, truncateds, infos

    def _read_agent_obs(self, agent_name: str) -> np.ndarray:
        parts = []
        spec = self.agents[agent_name]
        for fp in spec.obs_fields:
            n = self._agent_obs_sizes[agent_name][fp]
            if fp in self._last_predictions:
                val = self._last_predictions[fp].detach().cpu().numpy().flatten()
            else:
                try:
                    bf = self.world_model.bound[fp]
                    canvas = self.world_model.get_canvas()
                    val = canvas[0, bf.indices(), 0].detach().cpu().numpy()
                except (KeyError, AttributeError):
                    val = np.zeros(n, dtype=np.float32)
            if len(val) < n:
                val = np.pad(val, (0, n - len(val)))
            elif len(val) > n:
                val = val[:n]
            parts.append(val.astype(np.float32))
        return np.concatenate(parts)

    def _read_agent_obs_dict(self, agent_name: str) -> dict[str, np.ndarray]:
        result = {}
        spec = self.agents[agent_name]
        for fp in spec.obs_fields:
            n = self._agent_obs_sizes[agent_name][fp]
            if fp in self._last_predictions:
                val = self._last_predictions[fp].detach().cpu().numpy().flatten()
            else:
                val = np.zeros(n, dtype=np.float32)
            if len(val) < n:
                val = np.pad(val, (0, n - len(val)))
            elif len(val) > n:
                val = val[:n]
            result[fp] = val.astype(np.float32)
        return result


class AgentSpec:
    """Specification for one agent in a multi-agent WorldModelEnv.

    Args:
        obs_fields: Field paths this agent observes.
        act_fields: Field paths this agent controls.
        reward_fn: (obs_dict, action, info) -> float.
        terminated_fn: Optional (obs_dict, step, info) -> bool.
        act_low: Action space lower bound.
        act_high: Action space upper bound.
    """

    def __init__(
        self,
        obs_fields: list[str],
        act_fields: list[str],
        reward_fn: Callable[[dict[str, np.ndarray], np.ndarray, dict], float],
        terminated_fn: Callable[[dict[str, np.ndarray], int, dict], bool] | None = None,
        act_low: float = -1.0,
        act_high: float = 1.0,
    ):
        self.obs_fields = obs_fields
        self.act_fields = act_fields
        self.reward_fn = reward_fn
        self.terminated_fn = terminated_fn
        self.act_low = act_low
        self.act_high = act_high
