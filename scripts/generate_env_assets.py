"""Generate beautiful visualization assets for the environment extraction docs."""

import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe

os.makedirs("docs/assets", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 1. Hero diagram: World Model → Multiple Environments
# ═══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect("equal")
ax.axis("off")

# Background gradient effect
for i in range(100):
    y = i / 100 * 9
    ax.axhspan(y, y + 0.09, color=plt.cm.Blues(0.03 + 0.03 * (i / 100)), zorder=0)

# Central world model — large box
wm_box = FancyBboxPatch((4, 5.2), 8, 2.8, boxstyle="round,pad=0.4",
                          facecolor="#1a1a2e", edgecolor="#e94560", linewidth=3,
                          alpha=0.95, zorder=5)
ax.add_patch(wm_box)

ax.text(8, 7.2, "WORLD MODEL", ha="center", va="center",
        fontsize=22, fontweight="bold", color="white", zorder=6,
        path_effects=[pe.withStroke(linewidth=3, foreground="#e94560")])
ax.text(8, 6.4, "Learned Dynamics  |  857 Fields  |  19 Layers",
        ha="center", va="center", fontsize=11, color="#a8a8b3", zorder=6)
ax.text(8, 5.8, "canvas-engineering structured latent space",
        ha="center", va="center", fontsize=9, color="#6c6c80", style="italic", zorder=6)

# Canvas visualization inside world model
for cx in range(20):
    for cy in range(3):
        x = 5.5 + cx * 0.25
        y = 5.4 + cy * 0.12
        color = plt.cm.viridis(np.random.random() * 0.6 + 0.2)
        rect = plt.Rectangle((x, y), 0.22, 0.1, facecolor=color, alpha=0.4, zorder=6)
        ax.add_patch(rect)

# Title
ax.text(8, 8.5, "Environment Extraction from World Models",
        ha="center", va="center", fontsize=18, fontweight="bold",
        color="#1a1a2e", zorder=6)
ax.text(8, 8.1, "Same dynamics, different agent perspectives",
        ha="center", va="center", fontsize=12, color="#555", style="italic", zorder=6)

# Environments as colored perspective cones
envs = [
    {"name": "Employee\nNavigation", "color": "#2ecc71", "x": 2, "obs": 18, "act": 8,
     "desc": "career growth\nstress, focus, peers"},
    {"name": "CEO\nStrategy", "color": "#e74c3c", "x": 5.5, "obs": 51, "act": 10,
     "desc": "shareholder value\nallocation, M&A"},
    {"name": "HR\nPolicy", "color": "#9b59b6", "x": 9, "obs": 16, "act": 8,
     "desc": "retention\nheadcount, comp"},
    {"name": "Logistics\nRobot", "color": "#f39c12", "x": 12.5, "obs": 5, "act": 2,
     "desc": "throughput\nfreight, routing"},
]

for env in envs:
    x, color = env["x"], env["color"]

    # Perspective cone from world model down to env
    cone = plt.Polygon(
        [(x + 0.8, 5.2), (x - 0.5, 2.8), (x + 2.1, 2.8)],
        facecolor=color, alpha=0.12, edgecolor=color,
        linewidth=1.5, linestyle="--", zorder=3,
    )
    ax.add_patch(cone)

    # Environment box
    env_box = FancyBboxPatch((x - 0.5, 1.0), 2.6, 1.8, boxstyle="round,pad=0.3",
                              facecolor=color, alpha=0.2, edgecolor=color,
                              linewidth=2, zorder=5)
    ax.add_patch(env_box)

    # Gym icon
    ax.text(x + 0.8, 2.4, env["name"], ha="center", va="center",
            fontsize=10, fontweight="bold", color=color, zorder=6)
    ax.text(x + 0.8, 1.7, f"obs={env['obs']}  act={env['act']}",
            ha="center", va="center", fontsize=8, color="#555", zorder=6)
    ax.text(x + 0.8, 1.3, env["desc"], ha="center", va="center",
            fontsize=7, color="#777", style="italic", zorder=6)

    # Arrow from world model
    ax.annotate("", xy=(x + 0.8, 2.8), xytext=(x + 0.8, 5.2),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                               connectionstyle="arc3,rad=0"),
                zorder=7)

# Labels
ax.text(1, 0.5, "gymnasium.Env", ha="left", fontsize=10, color="#888",
        style="italic", zorder=6)
ax.text(15, 0.5, "to_openenv()", ha="right", fontsize=10, color="#888",
        style="italic", zorder=6)

fig.tight_layout()
fig.savefig("docs/assets/env_hero.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print("Saved: docs/assets/env_hero.png")


# ═══════════════════════════════════════════════════════════════════
# 2. Flow chart: to_openenv() pipeline
# ═══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.set_aspect("equal")
ax.axis("off")

# Background
for i in range(100):
    y = i / 100 * 6
    ax.axhspan(y, y + 0.06, color=plt.cm.Greys(0.02 + 0.01 * (i / 100)), zorder=0)

# Step boxes
steps = [
    {"x": 1, "y": 3, "w": 2.2, "h": 2, "color": "#3498db",
     "title": "1. Select Fields",
     "body": "obs_fields=[...]\nact_fields=[...]"},
    {"x": 4, "y": 3, "w": 2.2, "h": 2, "color": "#2ecc71",
     "title": "2. Define Reward",
     "body": "reward_fn(\n  obs, act, info\n) -> float"},
    {"x": 7, "y": 3, "w": 2.2, "h": 2, "color": "#e67e22",
     "title": "3. Build Spaces",
     "body": "obs_space: Box(N)\nact_space: Box(M)"},
    {"x": 10, "y": 3, "w": 2.2, "h": 2, "color": "#e74c3c",
     "title": "4. Step Loop",
     "body": "act → canvas\npredict()\nread obs"},
]

for step in steps:
    box = FancyBboxPatch((step["x"], step["y"]), step["w"], step["h"],
                          boxstyle="round,pad=0.2",
                          facecolor=step["color"], alpha=0.15,
                          edgecolor=step["color"], linewidth=2, zorder=5)
    ax.add_patch(box)
    ax.text(step["x"] + step["w"] / 2, step["y"] + step["h"] - 0.35,
            step["title"], ha="center", va="center",
            fontsize=11, fontweight="bold", color=step["color"], zorder=6)
    ax.text(step["x"] + step["w"] / 2, step["y"] + step["h"] / 2 - 0.2,
            step["body"], ha="center", va="center",
            fontsize=8, color="#444", family="monospace", zorder=6)

# Arrows between steps
for i in range(len(steps) - 1):
    x1 = steps[i]["x"] + steps[i]["w"]
    x2 = steps[i + 1]["x"]
    y = steps[i]["y"] + steps[i]["h"] / 2
    ax.annotate("", xy=(x2, y + 1), xytext=(x1, y + 1),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2), zorder=7)

# Loop arrow from step 4 back
ax.annotate("", xy=(11.1, 3), xytext=(11.1, 2.2),
            arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2,
                           connectionstyle="arc3,rad=-0.5"), zorder=7)
ax.text(12.5, 2.5, "env.step()", ha="center", fontsize=9, color="#e74c3c",
        fontweight="bold", zorder=6)

# Bottom: data flow detail
ax.text(7, 1.2, "agent.act(obs) → action → canvas → backbone.predict() → decode → obs, reward",
        ha="center", va="center", fontsize=10, color="#555",
        family="monospace", zorder=6,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#ccc"))

ax.text(7, 5.6, "WorldModel.to_openenv() Pipeline",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#1a1a2e")

fig.tight_layout()
fig.savefig("docs/assets/env_pipeline.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print("Saved: docs/assets/env_pipeline.png")


# ═══════════════════════════════════════════════════════════════════
# 3. Reward landscape visualization
# ═══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 5))

# Three different "reward landscapes" from same dynamics
np.random.seed(42)
x_grid = np.linspace(-3, 3, 200)
y_grid = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x_grid, y_grid)

landscapes = [
    {"title": "Employee: Career Reward", "color": "Greens",
     "fn": lambda x, y: np.exp(-(x - 1)**2 - (y - 0.5)**2) * 2 - 0.5 * np.exp(-(x + 1)**2 - (y + 1)**2)
            + 0.3 * np.sin(2 * x) * np.cos(2 * y)},
    {"title": "CEO: Shareholder Value", "color": "Reds",
     "fn": lambda x, y: 0.8 * np.exp(-(x**2 + y**2) / 4) + 0.5 * np.exp(-(x - 2)**2 - (y - 2)**2)
            - 0.3 * np.exp(-(x + 2)**2 - (y - 1)**2) + 0.2 * np.sin(3 * x)},
    {"title": "HR: Retention Score", "color": "Purples",
     "fn": lambda x, y: np.exp(-(x + 0.5)**2 / 3 - (y - 1)**2 / 2) * 1.5
            + 0.4 * np.exp(-(x - 1.5)**2 - (y + 0.5)**2) - 0.6 * np.exp(-((x + 2)**2 + (y + 2)**2) / 3)},
]

for i, ls in enumerate(landscapes):
    ax = fig.add_subplot(1, 3, i + 1)
    Z = ls["fn"](X, Y)
    im = ax.contourf(X, Y, Z, levels=30, cmap=ls["color"], alpha=0.8)
    ax.contour(X, Y, Z, levels=10, colors="white", alpha=0.3, linewidths=0.5)

    # Optimal path (gradient ascent)
    path_x, path_y = [-2], [-2]
    lr = 0.1
    for _ in range(40):
        cx, cy = path_x[-1], path_y[-1]
        dx = (ls["fn"](cx + 0.01, cy) - ls["fn"](cx - 0.01, cy)) / 0.02
        dy = (ls["fn"](cx, cy + 0.01) - ls["fn"](cx, cy - 0.01)) / 0.02
        path_x.append(cx + lr * dx)
        path_y.append(cy + lr * dy)

    ax.plot(path_x, path_y, 'w-', linewidth=2, alpha=0.8)
    ax.plot(path_x[0], path_y[0], 'wo', markersize=8, markeredgecolor='black')
    ax.plot(path_x[-1], path_y[-1], 'w*', markersize=15, markeredgecolor='black')

    ax.set_title(ls["title"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Action dim 1", fontsize=9)
    ax.set_ylabel("Action dim 2", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Reward")

fig.suptitle("Same World Dynamics → Different Reward Landscapes",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("docs/assets/env_reward_landscapes.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print("Saved: docs/assets/env_reward_landscapes.png")


# ═══════════════════════════════════════════════════════════════════
# 4. Multi-agent causal graph
# ═══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.set_aspect("equal")
ax.axis("off")

# Background
for i in range(100):
    y = i / 100 * 8
    ax.axhspan(y, y + 0.08, color=plt.cm.Blues(0.01 + 0.02 * (i / 100)), zorder=0)

ax.text(7, 7.5, "Multi-Agent Environment: Shared World, Independent Rewards",
        ha="center", fontsize=14, fontweight="bold", color="#1a1a2e")

# World state in center
world_circ = plt.Circle((7, 4.5), 1.5, facecolor="#1a1a2e", alpha=0.1,
                          edgecolor="#1a1a2e", linewidth=2, zorder=3)
ax.add_patch(world_circ)
ax.text(7, 4.9, "Shared Canvas", ha="center", fontsize=11,
        fontweight="bold", color="#1a1a2e")
ax.text(7, 4.3, "predict()", ha="center", fontsize=10, color="#555",
        family="monospace")
ax.text(7, 3.8, "one forward pass", ha="center", fontsize=8, color="#888",
        style="italic")

# Agents around the circle
agents = [
    {"name": "Agent A\n(Employee)", "color": "#2ecc71", "angle": 150,
     "obs_label": "satisfaction\nstress, focus", "act_label": "effort\nskills"},
    {"name": "Agent B\n(CEO)", "color": "#e74c3c", "angle": 30,
     "obs_label": "revenue\nmargins, equity", "act_label": "allocation\nstrategy"},
    {"name": "Agent C\n(HR)", "color": "#9b59b6", "angle": 270,
     "obs_label": "headcount\nturnover", "act_label": "hiring\ncomp"},
]

for agent in agents:
    angle_rad = np.radians(agent["angle"])
    radius = 3.2
    cx = 7 + radius * np.cos(angle_rad)
    cy = 4.5 + radius * np.sin(angle_rad)

    # Agent box
    box = FancyBboxPatch((cx - 1.2, cy - 0.8), 2.4, 1.6,
                          boxstyle="round,pad=0.2",
                          facecolor=agent["color"], alpha=0.2,
                          edgecolor=agent["color"], linewidth=2, zorder=5)
    ax.add_patch(box)
    ax.text(cx, cy + 0.35, agent["name"], ha="center", va="center",
            fontsize=10, fontweight="bold", color=agent["color"], zorder=6)

    # Obs arrow (world → agent)
    obs_x = 7 + 1.6 * np.cos(angle_rad)
    obs_y = 4.5 + 1.6 * np.sin(angle_rad)
    ax.annotate("", xy=(cx - 0.8 * np.cos(angle_rad), cy - 0.6 * np.sin(angle_rad)),
                xytext=(obs_x, obs_y),
                arrowprops=dict(arrowstyle="-|>", color=agent["color"], lw=2,
                               linestyle="-"), zorder=7)

    # Act arrow (agent → world, dashed)
    act_angle = angle_rad + 0.3
    ax.annotate("",
                xy=(7 + 1.6 * np.cos(act_angle), 4.5 + 1.6 * np.sin(act_angle)),
                xytext=(cx - 0.6 * np.cos(act_angle), cy - 0.4 * np.sin(act_angle)),
                arrowprops=dict(arrowstyle="-|>", color=agent["color"], lw=2,
                               linestyle="--"), zorder=7)

    # Labels
    label_r = 4.8
    lx = 7 + label_r * np.cos(angle_rad)
    ly = 4.5 + label_r * np.sin(angle_rad)
    ax.text(lx, ly + 0.3, f"obs: {agent['obs_label']}", ha="center",
            fontsize=7, color=agent["color"], style="italic", zorder=6)
    ax.text(lx, ly - 0.3, f"act: {agent['act_label']}", ha="center",
            fontsize=7, color=agent["color"], style="italic", zorder=6)

# Legend
ax.plot([1.5, 2.5], [1.2, 1.2], '-', color="#555", lw=2)
ax.annotate("", xy=(2.5, 1.2), xytext=(2.3, 1.2),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=2))
ax.text(3, 1.2, "observe (obs_fields)", va="center", fontsize=8, color="#555")

ax.plot([1.5, 2.5], [0.7, 0.7], '--', color="#555", lw=2)
ax.annotate("", xy=(2.5, 0.7), xytext=(2.3, 0.7),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=2, linestyle="--"))
ax.text(3, 0.7, "act (act_fields)", va="center", fontsize=8, color="#555")

fig.tight_layout()
fig.savefig("docs/assets/env_multiagent.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print("Saved: docs/assets/env_multiagent.png")


# ═══════════════════════════════════════════════════════════════════
# 5. Robot morphology decomposition
# ═══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.set_aspect("equal")
ax.axis("off")

# Background
for i in range(100):
    y = i / 100 * 7
    ax.axhspan(y, y + 0.07, color=plt.cm.Oranges(0.01 + 0.015 * (i / 100)), zorder=0)

ax.text(7, 6.5, "Physical Dynamics → Robot Morphologies",
        ha="center", fontsize=15, fontweight="bold", color="#1a1a2e")

# Central physics engine
physics_box = FancyBboxPatch((4.5, 3.5), 5, 2.2, boxstyle="round,pad=0.35",
                              facecolor="#2c3e50", alpha=0.9,
                              edgecolor="#e67e22", linewidth=3, zorder=5)
ax.add_patch(physics_box)
ax.text(7, 5.1, "Physical Dynamics Engine", ha="center",
        fontsize=14, fontweight="bold", color="white", zorder=6)
ax.text(7, 4.5, "Climate | Infrastructure | Disasters | Resources",
        ha="center", fontsize=9, color="#bdc3c7", zorder=6)
ax.text(7, 4.0, "133 fields | 525 positions | 1399 connections",
        ha="center", fontsize=8, color="#95a5a6", style="italic", zorder=6)

# Robot morphologies
robots = [
    {"name": "Logistics Robot", "icon": "truck", "color": "#27ae60", "x": 1.5,
     "sensors": "shipping capacity\nport congestion\nrail freight",
     "actuators": "route planning\ncapacity mgmt"},
    {"name": "Disaster Robot", "icon": "shield", "color": "#c0392b", "x": 5.5,
     "sensors": "seismic risk\nwildfire state\npandemic level",
     "actuators": "response deploy\nhazard mitigation"},
    {"name": "Climate Monitor", "icon": "thermometer", "color": "#2980b9", "x": 9.5,
     "sensors": "temp anomaly\ncarbon ppm\nsea level",
     "actuators": "sensor placement\ninfra tuning"},
]

for robot in robots:
    x, color = robot["x"], robot["color"]

    # Morphology cone
    cone = plt.Polygon(
        [(x + 1.5, 3.5), (x + 0.2, 2.3), (x + 2.8, 2.3)],
        facecolor=color, alpha=0.1, edgecolor=color,
        linewidth=1.5, linestyle="--", zorder=3,
    )
    ax.add_patch(cone)

    # Robot body (circle)
    body = plt.Circle((x + 1.5, 1.5), 0.9, facecolor=color, alpha=0.15,
                       edgecolor=color, linewidth=2, zorder=5)
    ax.add_patch(body)
    ax.text(x + 1.5, 1.85, robot["name"], ha="center", fontsize=9,
            fontweight="bold", color=color, zorder=6)
    ax.text(x + 1.5, 1.15, robot["sensors"], ha="center", fontsize=6.5,
            color="#555", zorder=6)

    # Arrows
    ax.annotate("", xy=(x + 1.5, 2.4), xytext=(x + 1.5, 3.5),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2), zorder=7)

    # Actuator label
    ax.text(x + 1.5, 0.3, robot["actuators"], ha="center", fontsize=6.5,
            color=color, style="italic", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.08, edgecolor=color))

fig.tight_layout()
fig.savefig("docs/assets/env_robots.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print("Saved: docs/assets/env_robots.png")

print("\nAll assets generated!")
