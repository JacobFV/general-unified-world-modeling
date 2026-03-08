---
title: WorldEnv
emoji: 🌍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: true
app_port: 7860
base_path: /web
tags:
  - openenv
  - world-model
  - rl
  - multi-agent
  - canvas-engineering
---

# WorldEnv — Sample Any RL Environment from a World Model

[![PyPI](https://img.shields.io/pypi/v/general-unified-world-model.svg)](https://pypi.org/project/general-unified-world-model/)
[![canvas-engineering](https://img.shields.io/pypi/v/canvas-engineering.svg?label=canvas-engineering)](https://pypi.org/project/canvas-engineering/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.1-blue)](https://github.com/meta-pytorch/OpenEnv)

**Instead of building one fixed RL environment, WorldEnv lets you *sample* any environment from a General Unified World Model by projecting to different field subsets.**

## Quick Start

```bash
pip install openenv-core
```

```python
from openenv.core import EnvClient
from openenv.core.env_server.types import Action, Observation

# Connect to the deployed Space
with WorldEnv(base_url="https://jacob-valdez-worldenv.hf.space") as env:
    # Reset with any scenario
    result = env.reset()
    print(f"Scenario: {result.observation.scenario_name}")
    print(f"Observable fields: {result.observation.obs_field_names}")

    # Take an action
    result = env.step(WorldAction(
        action_type="intervene",
        target_field="firm.strategy.rd_intensity",
        value=0.8,
    ))
    print(f"Reward: {result.reward}")
```

## Available Scenarios (6 projections, one world model)

| Scenario | Fields | Domain |
|----------|--------|--------|
| `corporate_employee` | satisfaction, stress, career | Corporate dynamics |
| `corporate_executive` | revenue, margins, market share | CEO strategy |
| `macro_policy` | GDP, inflation, rates | Economic policy |
| `logistics_robot` | shipping, port congestion, freight | Freight optimization |
| `disaster_robot` | seismic, wildfire, pandemic | Hazard mitigation |
| `climate_monitor` | temp anomaly, carbon, ENSO | Climate tracking |

## How It Works

```
General Unified World Model (857 fields, 19 layers)
         │
    project() ──── select subset of fields
         │
    WorldEnv ──── obs_fields → agent sees
                  act_fields → agent controls
                  reward_fn  → what to optimize
```

The world model is built on [canvas-engineering](https://pypi.org/project/canvas-engineering/) — a type system for multimodal latent computation. `project()` compiles any field subset into a structured canvas with automatic attention topology.

## Full World Model Integration

```bash
pip install "general-unified-world-model[env]"
```

```python
from general_unified_world_model import GeneralUnifiedWorldModel
from general_unified_world_model.schema.business import Business

# 857-field world model
model = GeneralUnifiedWorldModel(
    include=["financial", "regime", "sector_tech"],
    entities={"firm_acme": Business()},
)

# Sample a CEO environment
env = model.to_openenv(
    obs_fields=["firm_acme.financials.revenue_growth", "regime.growth_regime"],
    act_fields=["firm_acme.strategy.capital_allocation"],
    reward_fn=lambda obs, act, info: obs["firm_acme.financials.revenue_growth"].mean(),
)
obs, info = env.reset()
```

## Training with TRL GRPO

```python
from trl import GRPOConfig, GRPOTrainer

# WorldEnv integrates directly with TRL's GRPO
# See: https://jacobfv.github.io/general-unified-world-modeling/environments/
```

## Links

- 📦 [general-unified-world-model on PyPI](https://pypi.org/project/general-unified-world-model/)
- 📦 [canvas-engineering on PyPI](https://pypi.org/project/canvas-engineering/)
- 📖 [Environment Extraction Docs](https://jacobfv.github.io/general-unified-world-modeling/environments/)
- 💻 [GitHub Source](https://github.com/JacobFV/general-unified-world-modeling/tree/develop/worldenv)
- 🌐 [Corporate World Env Space](https://huggingface.co/spaces/jacob-valdez/corporate-world-env)
- 🤖 [Robot World Env Space](https://huggingface.co/spaces/jacob-valdez/robot-world-env)

## OpenEnv Hackathon

Built for the **OpenEnv Hackathon — March 2026**.

**Problem Statements:** 3.1 (World Modeling — Professional Tasks) + 5 (Wild Card)

**Partner Tracks:**
- Scaler AI Labs: Multi-App RL Environment for Enterprise Workflows
- Halluminate: Multi-Actor Environments
