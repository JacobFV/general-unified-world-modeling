"""Robot morphology environments from a shared physical dynamics model.

Demonstrates using a world model trained on physical and infrastructure
dynamics to extract different robot control environments. The same
physical world model captures forces, energy, materials, and logistics.
We carve out:

1. Logistics Robot Env — a warehouse robot that observes freight capacity,
   port congestion, and acts on logistics throughput
2. Disaster Response Env — a robot that observes disaster state, seismic
   risk, and acts to mitigate wildfire/pandemic/volcanic risk
3. Climate Monitor Env — a sensor platform that observes climate state
   and acts to adjust monitoring infrastructure

Each robot has a different "morphology" (obs/act decomposition) but they
all operate in the same physical world with shared causal dynamics.

Run: python examples/09_robot_envs.py
"""

import numpy as np

from general_unified_world_model import (
    GeneralUnifiedWorldModel, WorldModelEnv, AgentSpec,
)

# ── 1. Build physical dynamics world model ──────────────────────────

print("=" * 70)
print("Robot Morphology Environments from Physical Dynamics")
print("=" * 70)
print()
print("Building physical dynamics world model...")

model = GeneralUnifiedWorldModel(
    include=[
        # Physical substrate
        "physical.climate",
        "physical.infrastructure",
        "physical.disasters",

        # Resource dynamics
        "resources",

        # Infrastructure layer
        "infrastructure",

        # Technology (for robot capabilities)
        "technology",

        # Regime (compressed state)
        "regime",
    ],
    d_model=32,
    n_layers=2,
    n_heads=2,
    n_loops=1,
)

print(f"  Fields:     {len(model.bound.field_names)}")
print(f"  Positions:  {model.n_positions}")
print(f"  Connections: {len(model.bound.topology.connections)}")
print()

# ── 2. Define robot reward functions ────────────────────────────────


def logistics_reward(obs: dict, action: np.ndarray, info: dict) -> float:
    """Logistics robot: maximize throughput, minimize congestion."""
    throughput = 0.0
    congestion = 0.0
    for k, v in obs.items():
        if "freight" in k or "capacity" in k or "utilization" in k:
            throughput += v.mean()
        if "congestion" in k:
            congestion += v.mean()
    return float(throughput - 0.8 * congestion)


def disaster_reward(obs: dict, action: np.ndarray, info: dict) -> float:
    """Disaster robot: minimize active disaster impact."""
    risk_sum = 0.0
    for k, v in obs.items():
        if "risk" in k or "disaster" in k or "wildfire" in k or "pandemic" in k:
            risk_sum += abs(v.mean())
    return float(-risk_sum + 1.0)  # positive baseline for no disasters


def climate_reward(obs: dict, action: np.ndarray, info: dict) -> float:
    """Climate monitor: track anomalies, maintain coverage."""
    anomaly = 0.0
    coverage = 0.0
    for k, v in obs.items():
        if "anomaly" in k or "carbon" in k or "temp" in k:
            anomaly += abs(v.mean())
        if "cable" in k or "network" in k:
            coverage += v.mean()
    return float(coverage - 0.5 * anomaly)


# ── 3. Extract robot environments ──────────────────────────────────

print("Extracting robot environments...")
print()

# Logistics Robot
logistics_env = model.to_openenv(
    obs_fields=[
        "physical.infrastructure.shipping_lane_capacity",
        "physical.infrastructure.port_congestion",
        "physical.infrastructure.air_freight_utilization",
        "physical.infrastructure.rail_freight_network",
        "physical.infrastructure.chokepoint_risk",
    ],
    act_fields=[
        "physical.infrastructure.shipping_lane_capacity",
        "physical.infrastructure.rail_freight_network",
    ],
    reward_fn=logistics_reward,
    max_steps=100,
    n_denoise_steps=3,
)
print(f"  Logistics Robot: obs={logistics_env.observation_space.shape[0]}"
      f"  act={logistics_env.action_space.shape[0]}")

# Disaster Response Robot
disaster_env = model.to_openenv(
    obs_fields=[
        "physical.disasters.active_disaster_state",
        "physical.disasters.seismic_risk_structural",
        "physical.disasters.pandemic_risk",
        "physical.disasters.wildfire_state",
        "physical.disasters.volcanic_risk",
        "physical.climate.extreme_weather_freq",
    ],
    act_fields=[
        "physical.disasters.active_disaster_state",
        "physical.disasters.wildfire_state",
    ],
    reward_fn=disaster_reward,
    max_steps=100,
    n_denoise_steps=3,
)
print(f"  Disaster Robot:  obs={disaster_env.observation_space.shape[0]}"
      f"  act={disaster_env.action_space.shape[0]}")

# Climate Monitor
climate_env = model.to_openenv(
    obs_fields=[
        "physical.climate.global_temp_anomaly",
        "physical.climate.carbon_ppm",
        "physical.climate.enso_phase",
        "physical.climate.sea_level_trend",
        "physical.climate.polar_vortex_stability",
        "physical.infrastructure.undersea_cable_topology",
    ],
    act_fields=[
        "physical.infrastructure.undersea_cable_topology",
    ],
    reward_fn=climate_reward,
    max_steps=100,
    n_denoise_steps=3,
)
print(f"  Climate Monitor: obs={climate_env.observation_space.shape[0]}"
      f"  act={climate_env.action_space.shape[0]}")
print()

# ── 4. Run episodes ────────────────────────────────────────────────

print("Running episodes (random policies, 30 steps each)...")
print()

episode_data = {}
for name, env in [("Logistics", logistics_env),
                   ("Disaster", disaster_env),
                   ("Climate", climate_env)]:
    obs, _ = env.reset()
    rewards = []
    observations = []
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        observations.append(obs.copy())
        if terminated or truncated:
            break

    episode_data[name] = {
        "rewards": rewards,
        "observations": np.array(observations),
    }
    print(f"  {name:12s} | "
          f"total_reward={sum(rewards):+.3f} | "
          f"avg_reward={np.mean(rewards):+.3f} | "
          f"obs_range=[{np.min(observations):.2f}, {np.max(observations):.2f}]")

# ── 5. Multi-robot coordination ────────────────────────────────────

print()
print("Multi-robot coordination (shared dynamics)...")

multi_env = model.to_multi_openenv(
    agents={
        "logistics": AgentSpec(
            obs_fields=[
                "physical.infrastructure.shipping_lane_capacity",
                "physical.infrastructure.port_congestion",
            ],
            act_fields=[
                "physical.infrastructure.shipping_lane_capacity",
            ],
            reward_fn=logistics_reward,
        ),
        "disaster": AgentSpec(
            obs_fields=[
                "physical.disasters.active_disaster_state",
                "physical.disasters.wildfire_state",
            ],
            act_fields=[
                "physical.disasters.active_disaster_state",
            ],
            reward_fn=disaster_reward,
        ),
        "climate": AgentSpec(
            obs_fields=[
                "physical.climate.global_temp_anomaly",
                "physical.climate.carbon_ppm",
            ],
            act_fields=[
                "physical.infrastructure.undersea_cable_topology",
            ],
            reward_fn=climate_reward,
        ),
    },
    n_denoise_steps=3,
)

obs = multi_env.reset()
total_rewards = {name: 0.0 for name in multi_env.agents}

for step in range(10):
    actions = {
        name: multi_env.action_spaces[name].sample()
        for name in multi_env.agents
    }
    obs, rewards, terminateds, truncateds, infos = multi_env.step(actions)
    for name in multi_env.agents:
        total_rewards[name] += rewards[name]

print(f"  Agents: {list(multi_env.agents.keys())}")
for name, r in total_rewards.items():
    print(f"    {name:12s} total_reward: {r:+.3f}")

# ── 6. Visualization ────────────────────────────────────────────────

print()
print("Generating visualization...")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle("Robot Morphology Environments from Physical Dynamics World Model",
                 fontsize=14, fontweight="bold")

    # ── Panel 1: Causal graph showing world model → robots ──────────
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)

    # Central world model
    wm = mpatches.FancyBboxPatch((5.5, 3.5), 5, 2, boxstyle="round,pad=0.3",
                                  facecolor="#3498db", alpha=0.3,
                                  edgecolor="#2c3e50", linewidth=2)
    ax.add_patch(wm)
    ax.text(8, 4.5, f"Physical Dynamics World Model\n"
            f"{len(model.bound.field_names)} fields | "
            f"{model.n_positions} positions",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # Causal layer labels
    layers = [
        ("Climate", "#e67e22", 1.5),
        ("Infrastructure", "#27ae60", 5.5),
        ("Disasters", "#c0392b", 9.5),
        ("Resources", "#8e44ad", 13.5),
    ]
    for label, color, x in layers:
        box = mpatches.FancyBboxPatch((x - 1.2, 2), 2.4, 1,
                                       boxstyle="round,pad=0.15",
                                       facecolor=color, alpha=0.2,
                                       edgecolor=color, linewidth=1)
        ax.add_patch(box)
        ax.text(x, 2.5, label, ha="center", va="center", fontsize=8,
                color=color, fontweight="bold")
        ax.annotate("", xy=(x, 3.5), xytext=(x, 3.0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    # Robot boxes below
    robots = [
        ("Logistics\nRobot", "#2ecc71", 2.5, logistics_env),
        ("Disaster\nRobot", "#e74c3c", 8, disaster_env),
        ("Climate\nMonitor", "#3498db", 13.5, climate_env),
    ]
    for label, color, x, env in robots:
        box = mpatches.FancyBboxPatch((x - 1.3, 0.2), 2.6, 1.3,
                                       boxstyle="round,pad=0.2",
                                       facecolor=color, alpha=0.2,
                                       edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, 0.85, f"{label}\nobs={env.observation_space.shape[0]} "
                f"act={env.action_space.shape[0]}",
                ha="center", va="center", fontsize=7)
        ax.annotate("", xy=(x, 2.0), xytext=(x, 1.5),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.5,
                                   connectionstyle="arc3,rad=0"))

    ax.axis("off")
    ax.set_title("Causal Architecture: Same Dynamics, Different Morphologies", fontsize=11)

    # ── Panels 2-4: Reward trajectories per robot ──────────────────
    colors_map = {"Logistics": "#2ecc71", "Disaster": "#e74c3c", "Climate": "#3498db"}

    for i, (name, data) in enumerate(episode_data.items()):
        ax = fig.add_subplot(gs[1, i])
        rewards = data["rewards"]
        ax.plot(rewards, color=colors_map[name], linewidth=1.5)
        ax.fill_between(range(len(rewards)), rewards, alpha=0.2,
                        color=colors_map[name])
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        cumulative = np.cumsum(rewards)
        ax2 = ax.twinx()
        ax2.plot(cumulative, color=colors_map[name], linewidth=1,
                 linestyle="--", alpha=0.5)
        ax2.set_ylabel("Cumulative", fontsize=7, alpha=0.5)
        ax.set_title(f"{name} Robot Rewards", fontsize=10)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Reward", fontsize=8)
        ax.grid(True, alpha=0.2)

    # ── Panels 5-7: Observation trajectories per robot ──────────────
    for i, (name, data) in enumerate(episode_data.items()):
        ax = fig.add_subplot(gs[2, i])
        obs_arr = data["observations"]
        n_dims = min(obs_arr.shape[1], 5)
        for d in range(n_dims):
            ax.plot(obs_arr[:, d], alpha=0.7, linewidth=1,
                    label=f"dim {d}")
        ax.set_title(f"{name} Observations", fontsize=10)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.legend(fontsize=6, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.2)

    out_path = "docs/assets/robot_envs.png"
    import os
    os.makedirs("docs/assets", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

except ImportError:
    print("  (matplotlib not available — skipping visualization)")

print()
print("Done! Three robot morphologies extracted from one physical dynamics model:")
print("  - Logistics Robot: freight optimization in shipping/rail networks")
print("  - Disaster Robot: hazard mitigation across seismic/wildfire/pandemic")
print("  - Climate Monitor: tracking anomalies and maintaining sensor coverage")
print("  All share the same learned physical dynamics.")
