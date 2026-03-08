# General Unified World Model

**A typed causal ontology of civilization, built on [canvas-engineering](https://github.com/JacobFV/canvas-engineering) structured latent spaces.**

[![PyPI](https://img.shields.io/pypi/v/general-unified-world-model.svg)](https://pypi.org/project/general-unified-world-model/)
[![Tests](https://github.com/JacobFV/general-unified-world-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/JacobFV/general-unified-world-modeling/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

857 fields across 19 semantic layers. Train on heterogeneous data with masked loss -- no field needs to be present in every dataset. The model learns the joint distribution from partial observations.

## Install

```bash
pip install general-unified-world-model
```

## 30-second overview

```python
from canvas_engineering import compile_schema, ConnectivityPolicy
from general_unified_world_model import World, WorldProjection, project

# Full world: 857 fields on 128x128 canvas
world = World()
bound = compile_schema(
    world, T=1, H=128, W=128, d_model=64,
    connectivity=ConnectivityPolicy(intra="dense", parent_child="hub_spoke"),
)

# Or project to what you need
proj = WorldProjection(
    include=["financial", "country_us.macro", "regime", "forecasts"],
    firms=["AAPL", "NVDA"],
)
bound = project(proj, T=1, H=64, W=64, d_model=64)
```

## Key concepts

- **Schema** -- 19 layers from planetary physics to individual psychology, each field typed with temporal frequency and loss weight
- **Projection** -- declare which fields you care about, compile to a smaller canvas
- **Heterogeneous training** -- mask missing fields in loss, train on what you have
- **Canvas engineering** -- topology is the compute graph; field positions on `(T, H, W)` grid determine attention connectivity

For full details, see the [README on GitHub](https://github.com/JacobFV/general-unified-world-modeling).

## Documentation

| Page | Contents |
|------|----------|
| [Architecture](architecture.md) | Schema layers, projection system, canvas-engineering integration |
| [Training](training.md) | Training pipeline, DAG curriculum, data adapters |
| [API Reference](api.md) | `World`, `WorldProjection`, `project()`, `WorldModel` |
| [Examples](examples.md) | Usage snippets, links to example scripts |
| [Training Curriculum](training_curriculum.md) | Fork-join DAG curriculum research design |
