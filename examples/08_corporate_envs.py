"""Corporate multi-agent environments from a single world model.

Demonstrates extracting multiple RL environments from the same underlying
dynamics model. A corporate world model captures firm financials, operations,
strategy, and key people. We carve out three different agent perspectives:

1. Employee Navigation Env — an employee maximizes career growth subject
   to corporate constraints (stress, focus, reputation)
2. CEO Strategy Env — the CEO maximizes shareholder value through
   capital allocation, strategy, and hiring decisions
3. HR Policy Env — HR maximizes retention and productivity through
   policy levers

All three share the SAME learned dynamics — actions in one env affect
the world state observed by others.

Run: python examples/08_corporate_envs.py
"""

import numpy as np

from general_unified_world_model import (
    GeneralUnifiedWorldModel, AgentSpec,
)
from general_unified_world_model.schema.business import Business
from general_unified_world_model.schema.individual import Individual

# ── 1. Build a corporate world model ────────────────────────────────

print("=" * 70)
print("Corporate Multi-Agent Environments")
print("=" * 70)
print()
print("Building corporate world model...")

model = GeneralUnifiedWorldModel(
    include=[
        # Macro context
        "regime",
        "financial.equities",
        "financial.credit",

        # Sector dynamics
        "sector_tech",
    ],
    entities={
        # Our company
        "firm_acme": Business(),
        # Key people
        "person_ceo": Individual(),
        "person_employee": Individual(),
        "person_hr_director": Individual(),
    },
    d_model=32,
    n_layers=2,
    n_heads=2,
    n_loops=1,
)

print(f"  Fields:     {len(model.bound.field_names)}")
print(f"  Positions:  {model.n_positions}")
print(f"  Connections: {len(model.bound.topology.connections)}")
print()

# ── 2. Define reward functions for each agent role ──────────────────


def employee_reward(obs: dict, action: np.ndarray, info: dict) -> float:
    """Employee wants high satisfaction, confidence, low stress."""
    satisfaction = obs.get("firm_acme.operations.employee_satisfaction", np.array([0]))[0]
    confidence = obs.get("person_employee.state.confidence", np.array([0]))[0]
    stress = obs.get("person_employee.state.stress", np.array([0]))[0]
    reputation = obs.get("person_employee.incentives.reputation_concerns", np.array([0]))[0]
    return float(satisfaction + confidence - 0.5 * stress + 0.3 * reputation)


def ceo_reward(obs: dict, action: np.ndarray, info: dict) -> float:
    """CEO wants revenue growth, high margins, rising equity price."""
    revenue_growth = obs.get("firm_acme.financials.revenue_growth", np.array([0]))[0]
    margin = obs.get("firm_acme.financials.operating_margin", np.array([0]))[0]
    equity = obs.get("firm_acme.market.equity_price", np.array([0]))[0]
    health = obs.get("firm_acme.latent_health", np.array([0]))[0]
    return float(revenue_growth + 0.5 * margin + 0.3 * equity + 0.2 * health)


def hr_reward(obs: dict, action: np.ndarray, info: dict) -> float:
    """HR wants high satisfaction, low turnover risk, productive workforce."""
    satisfaction = obs.get("firm_acme.operations.employee_satisfaction", np.array([0]))[0]
    headcount = obs.get("firm_acme.operations.headcount", np.array([0]))[0]
    quality = obs.get("firm_acme.operations.quality_incidents", np.array([0]))[0]
    key_person_risk = obs.get("firm_acme.risk.key_person_risk", np.array([0]))[0]
    return float(satisfaction + 0.3 * headcount - 0.5 * quality - 0.4 * key_person_risk)


# ── 3. Extract single-agent environments ────────────────────────────

print("Extracting single-agent environments...")
print()

# Employee Navigation Environment
employee_env = model.to_openenv(
    obs_fields=[
        # What the employee can see
        "firm_acme.operations.employee_satisfaction",
        "firm_acme.operations.utilization",
        "person_employee.state.stress",
        "person_employee.state.confidence",
        "person_employee.state.current_focus",
        "person_employee.incentives.career_incentives",
        "person_employee.incentives.reputation_concerns",
        "person_employee.network.network_centrality",
    ],
    act_fields=[
        # What the employee can control
        "person_employee.state.current_focus",
        "person_employee.cognitive.risk_appetite",
        "person_employee.incentives.peer_pressure",
    ],
    reward_fn=employee_reward,
    max_steps=50,
    n_denoise_steps=3,
)

print(f"  Employee Env:")
print(f"    Observation dim: {employee_env.observation_space.shape[0]}")
print(f"    Action dim:      {employee_env.action_space.shape[0]}")

# CEO Strategy Environment
ceo_env = model.to_openenv(
    obs_fields=[
        # CEO's view of the company
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
        # Market context
        "regime.growth_regime",
        "regime.volatility",
    ],
    act_fields=[
        # CEO's levers
        "firm_acme.strategy.capital_allocation",
        "firm_acme.strategy.capex_plan",
        "firm_acme.strategy.m_and_a_appetite",
        "firm_acme.strategy.geographic_expansion",
        "firm_acme.operations.capacity",
    ],
    reward_fn=ceo_reward,
    max_steps=50,
    n_denoise_steps=3,
)

print(f"  CEO Env:")
print(f"    Observation dim: {ceo_env.observation_space.shape[0]}")
print(f"    Action dim:      {ceo_env.action_space.shape[0]}")

# HR Policy Environment
hr_env = model.to_openenv(
    obs_fields=[
        "firm_acme.operations.employee_satisfaction",
        "firm_acme.operations.headcount",
        "firm_acme.operations.quality_incidents",
        "firm_acme.operations.tech_debt",
        "firm_acme.risk.key_person_risk",
        "firm_acme.financials.opex",
        "person_employee.state.stress",
        "person_employee.state.confidence",
    ],
    act_fields=[
        "firm_acme.operations.headcount",
        "firm_acme.operations.employee_satisfaction",
        "person_employee.incentives.compensation_structure",
    ],
    reward_fn=hr_reward,
    max_steps=50,
    n_denoise_steps=3,
)

print(f"  HR Env:")
print(f"    Observation dim: {hr_env.observation_space.shape[0]}")
print(f"    Action dim:      {hr_env.action_space.shape[0]}")
print()

# ── 4. Run each single-agent env ────────────────────────────────────

print("Running single-agent episodes (random policy)...")
print()

for name, env in [("Employee", employee_env), ("CEO", ceo_env), ("HR", hr_env)]:
    obs, info = env.reset()
    total_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"  {name:10s} | 10 steps | total reward: {total_reward:+.3f}")

print()

# ── 5. Multi-agent environment (all agents act simultaneously) ──────

print("Extracting multi-agent environment...")

multi_env = model.to_multi_openenv(
    agents={
        "employee": AgentSpec(
            obs_fields=[
                "firm_acme.operations.employee_satisfaction",
                "person_employee.state.stress",
                "person_employee.state.confidence",
                "person_employee.incentives.career_incentives",
            ],
            act_fields=[
                "person_employee.state.current_focus",
                "person_employee.cognitive.risk_appetite",
            ],
            reward_fn=employee_reward,
        ),
        "ceo": AgentSpec(
            obs_fields=[
                "firm_acme.financials.revenue_growth",
                "firm_acme.financials.operating_margin",
                "firm_acme.market.equity_price",
                "firm_acme.latent_health",
                "regime.growth_regime",
            ],
            act_fields=[
                "firm_acme.strategy.capital_allocation",
                "firm_acme.strategy.capex_plan",
            ],
            reward_fn=ceo_reward,
        ),
        "hr": AgentSpec(
            obs_fields=[
                "firm_acme.operations.employee_satisfaction",
                "firm_acme.operations.headcount",
                "firm_acme.risk.key_person_risk",
            ],
            act_fields=[
                "firm_acme.operations.headcount",
            ],
            reward_fn=hr_reward,
        ),
    },
    n_denoise_steps=3,
)

print(f"  Agents: {list(multi_env.agents.keys())}")
for name in multi_env.agents:
    print(f"    {name:10s} obs_dim={multi_env.observation_spaces[name].shape[0]}"
          f"  act_dim={multi_env.action_spaces[name].shape[0]}")
print()

# Run multi-agent episode
print("Running multi-agent episode (random policies)...")
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

print()
for name, r in total_rewards.items():
    print(f"  {name:10s} | total reward: {r:+.3f}")

# ── 6. Visualization ────────────────────────────────────────────────

print()
print("Generating visualization...")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Corporate Multi-Agent Environments from One World Model",
                 fontsize=14, fontweight="bold")

    # Panel 1: Environment architecture diagram
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")

    # World model box
    wm_box = mpatches.FancyBboxPatch((2.5, 4), 5, 2, boxstyle="round,pad=0.3",
                                      facecolor="#3498db", alpha=0.3, edgecolor="#2c3e50", linewidth=2)
    ax.add_patch(wm_box)
    ax.text(5, 5, f"World Model\n{model.n_positions} positions\n{len(model.bound.field_names)} fields",
            ha="center", va="center", fontsize=9, fontweight="bold")

    # Agent boxes
    agent_info = [
        ("Employee", "#2ecc71", 1, 1.5, employee_env),
        ("CEO", "#e74c3c", 4, 1.5, ceo_env),
        ("HR", "#9b59b6", 7, 1.5, hr_env),
    ]
    for label, color, x, y, env in agent_info:
        box = mpatches.FancyBboxPatch((x - 0.8, y - 0.6), 2.2, 1.2,
                                       boxstyle="round,pad=0.2",
                                       facecolor=color, alpha=0.3,
                                       edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.3, y, f"{label}\nobs={env.observation_space.shape[0]}"
                f" act={env.action_space.shape[0]}",
                ha="center", va="center", fontsize=7)
        ax.annotate("", xy=(x + 0.3, 4), xytext=(x + 0.3, y + 0.6),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.5))

    # Title arrow from world model
    ax.text(5, 7, "Shared Latent Dynamics", ha="center", fontsize=8,
            style="italic", color="#7f8c8d")
    ax.annotate("", xy=(5, 6.2), xytext=(5, 6.8),
                arrowprops=dict(arrowstyle="->", color="#7f8c8d"))
    ax.axis("off")
    ax.set_title("Architecture", fontsize=11)

    # Panel 2: Observation field overlap Venn-ish diagram
    ax = axes[0, 1]
    obs_sets = {
        "Employee": set(employee_env.obs_fields),
        "CEO": set(ceo_env.obs_fields),
        "HR": set(hr_env.obs_fields),
    }
    colors = ["#2ecc71", "#e74c3c", "#9b59b6"]
    angles = [0, 120, 240]
    radius = 2.0
    for i, (label, fields) in enumerate(obs_sets.items()):
        angle_rad = np.radians(angles[i] + 90)
        cx = 5 + 0.8 * np.cos(angle_rad)
        cy = 5 + 0.8 * np.sin(angle_rad)
        circle = plt.Circle((cx, cy), radius, alpha=0.15, color=colors[i],
                            edgecolor=colors[i], linewidth=1.5)
        ax.add_patch(circle)
        tx = 5 + 2.5 * np.cos(angle_rad)
        ty = 5 + 2.5 * np.sin(angle_rad)
        ax.text(tx, ty, f"{label}\n({len(fields)} obs)", ha="center",
                va="center", fontsize=8, color=colors[i], fontweight="bold")

    # Shared fields
    shared = obs_sets["Employee"] & obs_sets["HR"]
    ax.text(5, 5, f"{len(shared)} shared", ha="center", va="center",
            fontsize=8, color="#7f8c8d")
    ax.set_xlim(1, 9)
    ax.set_ylim(1, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Observation Field Overlap", fontsize=11)

    # Panel 3: Reward trajectories
    ax = axes[1, 0]
    for name, env_obj, color in [("Employee", employee_env, "#2ecc71"),
                                  ("CEO", ceo_env, "#e74c3c"),
                                  ("HR", hr_env, "#9b59b6")]:
        obs, _ = env_obj.reset()
        rewards = []
        for _ in range(20):
            action = env_obj.action_space.sample()
            obs, reward, _, _, _ = env_obj.step(action)
            rewards.append(reward)
        ax.plot(rewards, label=name, color=color, linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Trajectories (Random Policy)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Action and observation dimensions comparison
    ax = axes[1, 1]
    agents_list = ["Employee", "CEO", "HR"]
    obs_dims = [employee_env.observation_space.shape[0],
                ceo_env.observation_space.shape[0],
                hr_env.observation_space.shape[0]]
    act_dims = [employee_env.action_space.shape[0],
                ceo_env.action_space.shape[0],
                hr_env.action_space.shape[0]]

    x_pos = np.arange(len(agents_list))
    width = 0.35
    bars1 = ax.bar(x_pos - width / 2, obs_dims, width, label="Obs dim",
                   color=["#2ecc71", "#e74c3c", "#9b59b6"], alpha=0.5)
    bars2 = ax.bar(x_pos + width / 2, act_dims, width, label="Act dim",
                   color=["#2ecc71", "#e74c3c", "#9b59b6"], alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents_list)
    ax.set_ylabel("Dimensionality")
    ax.set_title("Space Dimensions per Agent", fontsize=11)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = "docs/assets/corporate_envs.png"
    import os
    os.makedirs("docs/assets", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

except ImportError:
    print("  (matplotlib not available — skipping visualization)")

print()
print("Done! Three environments extracted from one corporate world model:")
print("  - Employee: navigate career in corporate dynamics")
print("  - CEO: maximize shareholder value via strategy")
print("  - HR: optimize retention and productivity")
print("  All share the same learned dynamics.")
