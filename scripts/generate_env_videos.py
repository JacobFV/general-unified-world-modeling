"""Generate animated videos for corporate interaction graph and robot trajectories."""

import os
import math
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, PillowWriter

os.makedirs("docs/assets", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 1. Corporate Interaction Graph Animation
# ═══════════════════════════════════════════════════════════════════

print("Generating corporate interaction graph animation...")

fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0d1117")
ax.set_xlim(-5, 5)
ax.set_ylim(-4, 4)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("#0d1117")

# Node definitions: (x, y, label, color, size)
NODES = {
    "CEO":          (0.0,  2.5,  "#e74c3c", 400),
    "CFO":          (-2.0, 1.2,  "#e74c3c", 280),
    "CTO":          (2.0,  1.2,  "#e74c3c", 280),
    "HR Director":  (-3.5, 0.0,  "#9b59b6", 240),
    "Employee A":   (-2.5,-1.5,  "#2ecc71", 200),
    "Employee B":   (-1.0,-1.5,  "#2ecc71", 200),
    "Employee C":   (1.0, -1.5,  "#2ecc71", 200),
    "Employee D":   (2.5, -1.5,  "#2ecc71", 200),
    "Revenue":      (3.5,  2.0,  "#f39c12", 220),
    "Market":       (4.0,  0.5,  "#f39c12", 200),
    "Macro":        (3.5, -1.0,  "#3498db", 200),
}

# Edge definitions: (src, dst, color, weight)
BASE_EDGES = [
    ("CEO", "CFO", "#e74c3c", 2.5),
    ("CEO", "CTO", "#e74c3c", 2.5),
    ("CEO", "Revenue", "#f39c12", 1.5),
    ("CFO", "HR Director", "#9b59b6", 1.5),
    ("CFO", "Revenue", "#f39c12", 2.0),
    ("CTO", "Employee C", "#2ecc71", 1.5),
    ("CTO", "Employee D", "#2ecc71", 1.5),
    ("HR Director", "Employee A", "#9b59b6", 2.0),
    ("HR Director", "Employee B", "#9b59b6", 2.0),
    ("Revenue", "Market", "#f39c12", 1.5),
    ("Market", "Macro", "#3498db", 1.0),
    ("Macro", "CEO", "#3498db", 1.0),
]

node_coords = {n: (NODES[n][0], NODES[n][1]) for n in NODES}

# Build animation frames
N_FRAMES = 90
random.seed(42)
np.random.seed(42)

def draw_frame(t):
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#0d1117")

    phase = 2 * math.pi * t / N_FRAMES

    # Draw edges with animated flow pulses
    for src, dst, ecolor, weight in BASE_EDGES:
        sx, sy = node_coords[src]
        dx, dy = node_coords[dst]

        # Pulsing alpha
        pulse = 0.4 + 0.4 * math.sin(phase + hash(src + dst) % 7)
        ax.annotate("",
                    xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=ecolor,
                        lw=weight * 0.8,
                        alpha=pulse,
                        connectionstyle="arc3,rad=0.1",
                    ),
                    zorder=2)

        # Animated "signal" dot traveling along edge
        frac = (t / N_FRAMES * 3 + hash(src) * 0.3) % 1.0
        sig_x = sx + frac * (dx - sx)
        sig_y = sy + frac * (dy - sy)
        ax.plot(sig_x, sig_y, 'o', color=ecolor, markersize=4,
                alpha=0.8 * math.sin(math.pi * frac), zorder=4)

    # Draw nodes
    for name, (nx, ny, ncolor, nsize) in NODES.items():
        # Pulsing node size
        pulse_size = nsize * (1.0 + 0.1 * math.sin(phase + hash(name) % 5))
        ax.scatter(nx, ny, s=pulse_size, c=ncolor, alpha=0.9, zorder=5,
                   edgecolors="white", linewidths=1.5)

        # Label
        offset_y = -0.45 if ny < 0 else 0.45
        ax.text(nx, ny + offset_y, name, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold", zorder=6)

    # Title
    ax.text(0, 3.6, "Corporate World Model — Agent Interaction Graph",
            ha="center", fontsize=13, fontweight="bold", color="white")
    ax.text(0, 3.15, f"Step {t} | Shared dynamics, independent reward landscapes",
            ha="center", fontsize=8.5, color="#888", style="italic")

    # Legend
    legend_items = [
        mpatches.Patch(color="#e74c3c", label="Executive (CEO/CFO/CTO)"),
        mpatches.Patch(color="#9b59b6", label="HR"),
        mpatches.Patch(color="#2ecc71", label="Employees"),
        mpatches.Patch(color="#f39c12", label="Financial"),
        mpatches.Patch(color="#3498db", label="Macro"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=7,
              facecolor="#1a1a2e", labelcolor="white", edgecolor="#333",
              framealpha=0.8)

ani = animation.FuncAnimation(fig, draw_frame, frames=N_FRAMES, interval=60)
writer = PillowWriter(fps=20)
ani.save("docs/assets/corporate_interaction_graph.gif", writer=writer, dpi=120)
plt.close(fig)
print("Saved: docs/assets/corporate_interaction_graph.gif")


# ═══════════════════════════════════════════════════════════════════
# 2. Robot World Animation
# ═══════════════════════════════════════════════════════════════════

print("Generating robot trajectories animation...")

fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="#0d1117")
fig.patch.set_facecolor("#0d1117")

N_FRAMES = 80

# Pre-compute robot trajectories
def make_trajectory(seed, kind):
    rng = np.random.RandomState(seed)
    T = N_FRAMES
    if kind == "logistics":
        # Ship routes on a map-like grid
        xs = np.cumsum(rng.randn(T) * 0.08) * 2
        ys = np.cumsum(rng.randn(T) * 0.05)
        return xs - xs.mean(), ys - ys.mean()
    elif kind == "disaster":
        # Response robot: patrol pattern
        t = np.linspace(0, 4 * math.pi, T)
        xs = 2 * np.sin(t * 0.7) + rng.randn(T) * 0.1
        ys = 1.5 * np.cos(t) + rng.randn(T) * 0.1
        return xs, ys
    else:  # climate
        # Sensor sweep
        t = np.linspace(0, 2 * math.pi, T)
        xs = 3 * np.cos(t) + rng.randn(T) * 0.05
        ys = 2 * np.sin(2 * t) * 0.8 + rng.randn(T) * 0.05
        return xs, ys

ROBOT_DATA = [
    ("Logistics Robot", "#27ae60", "logistics",
     "Freight throughput optimization\nact: routing & capacity"),
    ("Disaster Robot", "#c0392b", "disaster",
     "Hazard mitigation\nact: response deployment"),
    ("Climate Monitor", "#2980b9", "climate",
     "Sensor coverage\nact: infrastructure placement"),
]

trajs = [(make_trajectory(i + 7, kind), color, name, desc)
         for i, (name, color, kind, desc) in enumerate(ROBOT_DATA)]

# Build background environments
BG_COLORS = ["#001a0d", "#1a0000", "#00001a"]
FIELD_LABELS = [
    ["Shipping Capacity", "Port Congestion", "Rail Network"],
    ["Seismic Risk", "Wildfire State", "Pandemic Level"],
    ["Temp Anomaly", "Carbon ppm", "ENSO Phase"],
]

field_series = {}
for i, (name, color, kind, desc) in enumerate(ROBOT_DATA):
    rng = np.random.RandomState(i * 42)
    for label in FIELD_LABELS[i]:
        base = rng.randn()
        noise = rng.randn(N_FRAMES) * 0.05
        drift = np.cumsum(rng.randn(N_FRAMES) * 0.02)
        field_series[(i, label)] = base + drift + noise


def draw_robot_frame(t):
    for col_idx, (ax, (traj, color, name, desc)) in enumerate(zip(axes, trajs)):
        ax.clear()
        ax.set_facecolor(BG_COLORS[col_idx])

        xs, ys = traj
        # Background field heatmap
        xlim = (xs.min() - 1, xs.max() + 1)
        ylim = (ys.min() - 1, ys.max() + 1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Draw field values as background scatter
        rng = np.random.RandomState(col_idx * 100 + t)
        n_pts = 40
        bg_x = np.linspace(xlim[0], xlim[1], n_pts)
        bg_y = np.linspace(ylim[0], ylim[1], n_pts)
        BX, BY = np.meshgrid(bg_x, bg_y)
        phase = 2 * math.pi * t / N_FRAMES
        Z = np.sin(BX * 0.8 + phase) * np.cos(BY * 0.6 - phase * 0.7)
        ax.contourf(BX, BY, Z, levels=15, cmap="viridis", alpha=0.3)

        # Draw trajectory trail (fade out)
        trail_len = min(t + 1, 25)
        for j in range(max(0, t - trail_len), t):
            alpha = 0.15 + 0.65 * (j - (t - trail_len)) / trail_len
            ax.plot(xs[j:j+2], ys[j:j+2], '-', color=color, alpha=alpha, linewidth=1.5)

        # Draw robot (current position)
        if t < len(xs):
            ax.plot(xs[t], ys[t], 'o', color=color, markersize=14,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=10)
            # Direction arrow
            if t > 0:
                dx = xs[t] - xs[t-1]
                dy = ys[t] - ys[t-1]
                norm = math.sqrt(dx**2 + dy**2 + 1e-6)
                ax.annotate("", xy=(xs[t] + dx/norm*0.4, ys[t] + dy/norm*0.4),
                            xytext=(xs[t], ys[t]),
                            arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5),
                            zorder=11)

        # Sensor readings bars
        bar_x = xlim[0] + 0.1 * (xlim[1] - xlim[0])
        for fi, label in enumerate(FIELD_LABELS[col_idx]):
            val = field_series[(col_idx, label)][min(t, N_FRAMES - 1)]
            bar_w = 0.6 * (xlim[1] - xlim[0]) * min(1.0, max(0.0, (val + 2) / 4))
            bar_y = ylim[0] + (ylim[1] - ylim[0]) * (0.1 + fi * 0.08)
            ax.barh(bar_y, bar_w, left=bar_x, height=0.06*(ylim[1]-ylim[0]),
                    color=color, alpha=0.7)
            ax.text(bar_x - 0.05, bar_y, label.split()[0], ha="right", va="center",
                    fontsize=6, color="white", alpha=0.8)

        ax.set_title(f"{name}\n{desc}", fontsize=8.5, color="white",
                     fontweight="bold", pad=6)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    fig.suptitle(
        f"Robot World Model Environments — Step {t}/{N_FRAMES}\n"
        "Three morphologies, one physical dynamics model",
        color="white", fontsize=11, fontweight="bold", y=1.0,
    )
    fig.tight_layout()


ani2 = animation.FuncAnimation(fig, draw_robot_frame, frames=N_FRAMES, interval=70)
writer2 = PillowWriter(fps=18)
ani2.save("docs/assets/robot_trajectories.gif", writer=writer2, dpi=110)
plt.close(fig)
print("Saved: docs/assets/robot_trajectories.gif")

print("\nAll videos generated!")
