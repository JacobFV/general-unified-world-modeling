---
title: Robot World Environment
emoji: 🤖
colorFrom: red
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - world-model
  - robotics
  - rl
---

# Robot World Environment

Three robot morphologies extracted from the same physical dynamics world model.

| Morphology | Sensors | Actuators | Task |
|------------|---------|-----------|------|
| **Logistics** | shipping capacity, port congestion | routing, capacity mgmt | freight throughput |
| **Disaster** | seismic, wildfire, pandemic | response deployment | hazard mitigation |
| **Climate** | temp anomaly, carbon ppm | sensor placement | monitoring coverage |

## Quick Start

```python
from robot_world_env import RobotWorldEnv, RobotWorldAction

with RobotWorldEnv(base_url="https://jacob-valdez-robot-world-env.hf.space") as env:
    result = env.reset()
    print(result.observation.morphology, result.observation.obs_fields)

    action = RobotWorldAction(morphology="logistics", values=[0.7, -0.3])
    result = env.step(action)
    print(result.reward)
```

## Links

- [General Unified World Model](https://github.com/JacobFV/general-unified-world-modeling)
- [Docs](https://jacobfv.github.io/general-unified-world-modeling/environments/)
- [Example Script](https://github.com/JacobFV/general-unified-world-modeling/blob/develop/examples/09_robot_envs.py)
