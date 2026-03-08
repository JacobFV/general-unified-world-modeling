---
title: Corporate World Environment
emoji: 🏢
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - world-model
  - multi-agent
  - rl
---

# Corporate World Environment

A multi-role RL environment backed by a [General Unified World Model](https://github.com/JacobFV/general-unified-world-modeling). Three agent perspectives share the same learned corporate dynamics:

| Role | Observes | Controls | Reward |
|------|----------|----------|--------|
| **Employee** | satisfaction, stress, confidence, career incentives | focus, risk appetite | career growth - stress |
| **CEO** | revenue, margins, equity price, latent health | capital allocation, capex, M&A | shareholder value |
| **HR** | satisfaction, headcount, quality incidents, risk | headcount, compensation | retention + productivity |

## Quick Start

```bash
pip install git+https://github.com/JacobFV/general-unified-world-modeling.git#subdirectory=envs/corporate_world_env
```

```python
from corporate_world_env import CorporateWorldEnv, CorporateWorldAction

with CorporateWorldEnv(base_url="https://JacobFV-corporate-world-env.hf.space") as client:
    result = client.reset()
    print(f"Role: {result.observation.role}")
    print(f"Obs fields: {result.observation.obs_fields}")
    print(f"Act dims: {result.observation.act_dims}")

    # Play as CEO
    action = CorporateWorldAction(role="ceo", values=[0.5, -0.2, 0.1, 0.0, 0.3])
    result = client.step(action)
    print(f"Reward: {result.reward}")
    print(f"Observations: {result.observation.obs_dict}")
```

## Architecture

The environment builds a `GeneralUnifiedWorldModel` with **240 fields** across:

- Firm financials, operations, strategy, market position, risk
- 3 individual entities (CEO, employee, HR director) with psychological decomposition
- Sector and regime context for macro dynamics

Each role's `step()`:
1. Writes actions to its `act_fields` on the canvas
2. Runs `predict()` through the world model's diffusion backbone
3. Reads observations from its `obs_fields`
4. Computes role-specific reward

## Roles

### Employee (obs=18, act=8)
Navigate career growth in corporate dynamics. Observe satisfaction, stress, confidence, and career incentives. Control focus areas and risk appetite.

### CEO (obs=51, act=10)
Maximize shareholder value. Observe revenue, margins, equity price, latent health, and macro regime. Control capital allocation, capex plans, and M&A strategy.

### HR (obs=16, act=8)
Optimize retention and workforce productivity. Observe satisfaction, headcount, quality incidents, and key person risk. Control hiring and compensation.

## Links

- [General Unified World Model](https://github.com/JacobFV/general-unified-world-modeling)
- [Environment Extraction Docs](https://jacobfv.github.io/general-unified-world-modeling/environments/)
- [Example Script](https://github.com/JacobFV/general-unified-world-modeling/blob/develop/examples/08_corporate_envs.py)
