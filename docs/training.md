# Training

## Overview

Training the world model requires handling heterogeneous data sources where each dataset covers different fields. The key design: **mask missing fields in the loss, train on what you have**.

```
Dataset A (FRED):     GDP ok  CPI ok  VIX --  Yields --
Dataset B (Yahoo):    GDP --  CPI --  VIX ok  Yields ok
Dataset C (News):     GDP --  CPI --  VIX --  Yields --  News ok

Canvas loss:  L = sum((prediction - target)^2 * presence * loss_weight)
```

All three datasets train the shared regime latent, even with zero field overlap.

## Data adapters

Built-in adapters map real-world data sources to schema field paths:

```python
from general_unified_world_model.data.adapters import (
    fred_adapter,           # 50+ FRED macro series
    yahoo_finance_adapter,  # equities, FX, commodities, crypto
    pmi_adapter,            # ISM PMI surveys
    earnings_adapter,       # quarterly earnings data
    news_adapter,           # news embedding streams
    tabular_adapter,        # generic CSV/Parquet
)

# Each returns (DatasetSpec, data)
fred_spec, fred_data = fred_adapter(api_key="...", start_date="2010-01-01")
yahoo_spec, yahoo_data = yahoo_finance_adapter(include_equity=True, include_fx=True)
```

### DatasetSpec and FieldMapping

```python
spec = DatasetSpec(
    name="FRED",
    mappings=[
        FieldMapping("gdp", "country_us.macro.output.gdp_nowcast"),
        FieldMapping("cpi", "country_us.macro.inflation.headline_cpi"),
    ],
)
```

Each `FieldMapping` maps a source column name to a dotted field path in the schema.

### Mixed dataloader

```python
loader = build_mixed_dataloader(
    bound,
    sources=[(fred_spec, fred_data), (yahoo_spec, yahoo_data)],
    batch_size=32,
)
```

`HeterogeneousDataset` places each dataset's values onto the canvas and generates a `presence_mask` indicating which positions have ground truth.

## Training components

### FieldEncoder / FieldDecoder

Per-field linear layers that project scalar values to/from `d_model`-dimensional canvas positions. Incorporates semantic embeddings so the backbone receives field identity as input.

### MaskedCanvasTrainer

Core training loop. Computes loss only where `presence_mask == 1`:

```python
trainer = MaskedCanvasTrainer(
    bound_schema=bound,
    backbone=backbone,
    encoder=encoder,
    decoder=decoder,
    optimizer=optimizer,
)
trainer.train(dataloader, n_steps=10000)
```

### WorldModelBackbone

Transformer backbone operating on canvas positions. Uses canvas-engineering's topology for the attention mask -- only connected positions attend to each other.

```python
backbone = build_world_model(
    bound, n_layers=8, n_heads=4, d_ff=256, n_loops=3,
)
```

### DiffusionWorldModelTrainer

Wraps the backbone in a diffusion objective. Uses `CosineNoiseSchedule` with frequency-aware noise scaling -- slow fields (high period) get less noise.

## Sequential curriculum (CurriculumTrainer)

Three-phase sequential training:

1. **Phase 1: Independent domains** -- train each domain on small canvases in parallel
2. **Phase 2: Domain coupling** -- merge causally adjacent domains, train on medium canvases
3. **Phase 3: Full integration** -- all domains on one canvas, regime latent learns cross-domain structure

```python
config = CurriculumConfig(
    phases=[
        PhaseConfig(domains=[...], canvas_size=32, n_steps=10000),
        PhaseConfig(domains=[...], canvas_size=64, n_steps=8000),
        PhaseConfig(domains=[...], canvas_size=128, n_steps=20000),
    ],
)
trainer = CurriculumTrainer(config)
trainer.run()
```

## DAG curriculum (DAGCurriculumTrainer)

The fork-join DAG curriculum extends the sequential approach with parallel branches and weight merging. See the [Training Curriculum](training_curriculum.md) research document for full details.

Key ideas:

- **Fork nodes** copy weights to parallel branches, each training a different projection
- **Join nodes** merge branches via weight averaging (simple, TIES, or DARE)
- **Language-conditioned projections** -- each DAG node is described in natural language, passed to `llm_project()` to generate the `WorldProjection`
- **Semantic embedding conditioning** -- field identity comes from frozen text embeddings, enabling compatible representations across branches

### DAG structure

```
Tier 0 (6 parallel foundation nodes, 32x32 canvas)
  --> merge
Tier 1 (3 parallel cross-domain nodes, 48x48 canvas)
  --> merge
Tier 2 (3 parallel complex interaction nodes, 64x64 canvas)
  --> merge
Tier 3 (1 full integration node, 128x128 canvas)
```

```python
trainer = DAGCurriculumTrainer(dag_config)
trainer.run()
```

## Compute requirements

| Tier | Nodes | Canvas | Fields/node | A100-hrs/node |
|------|-------|--------|-------------|---------------|
| 0 | 6 | 32x32 | 25-60 | 2-4 |
| 1 | 3 | 48x48 | 80-105 | 6-10 |
| 2 | 3 | 64x64 | 140-180 | 12-18 |
| 3 | 1 | 128x128 | 857 | 80-120 |

Total: ~250 A100-hours. With 6-way parallelism, wall-clock time drops to ~130 hours.
