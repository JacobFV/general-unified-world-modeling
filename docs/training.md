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

### InputSpec / OutputSpec

`InputSpec` declares an input modality — a column your dataset provides to the canvas. `OutputSpec` declares an output modality — where the model predicts and receives gradient. Together they replace the old `FieldMapping` with richer semantics:

```python
from general_unified_world_model import DatasetSpec, InputSpec, OutputSpec

spec = DatasetSpec(
    name="FRED",
    input_specs=[
        InputSpec(
            key="gdp",
            semantic_type="US real GDP quarterly growth rate",
            field_path="country_us.macro.output.gdp_nowcast",
        ),
        InputSpec(
            key="cpi",
            semantic_type="US CPI year-over-year percent change",
            field_path="country_us.macro.inflation.headline_cpi",
        ),
    ],
    output_specs=[
        OutputSpec(
            key="gdp",
            semantic_type="US real GDP quarterly growth rate",
            field_path="country_us.macro.output.gdp_nowcast",
            loss_weight=2.0,
        ),
        OutputSpec(
            key="cpi",
            semantic_type="US CPI year-over-year percent change",
            field_path="country_us.macro.inflation.headline_cpi",
            loss_weight=2.5,
        ),
    ],
)
```

`InputSpec` supports custom encoders, transforms, and frequency hints. `OutputSpec` supports custom decoders, loss functions, and per-output loss weights.

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

### Backbone options

=== "CogVideoX (default)"

    Graft onto a pretrained CogVideoX video diffusion transformer. Only ~0.1% of parameters are trainable — the rest provide frozen spatiotemporal priors from video pretraining.

    ```python
    from general_unified_world_model import DAGCurriculumTrainer

    trainer = DAGCurriculumTrainer(
        nodes=dag,
        data_sources=data_sources,
        backbone="cogvideox",
        pretrained_model_id="THUDM/CogVideoX-2b",
        device="cuda",
    )
    ```

    See [CogVideoX Backbone](cogvideox.md) for architecture details.

=== "From scratch"

    Train a small custom transformer from random initialization. Useful for testing, CPU-only environments, or when CogVideoX is unavailable.

    ```python
    backbone = build_world_model(
        bound, n_layers=8, n_heads=4, d_ff=256, n_loops=3,
    )
    ```

### DiffusionWorldModelTrainer

Wraps the backbone in a diffusion objective. Uses `CosineNoiseSchedule` with frequency-aware noise scaling — slow fields (high period) get less noise.

## Sequential curriculum (CurriculumTrainer)

Three-phase sequential training:

1. **Phase 1: Independent domains** — train each domain on small canvases in parallel
2. **Phase 2: Domain coupling** — merge causally adjacent domains, train on medium canvases
3. **Phase 3: Full integration** — all domains on one canvas, regime latent learns cross-domain structure

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

The fork-join DAG curriculum extends the sequential approach with parallel branches and weight merging. See [DAG Curriculum](dag_curriculum.md) for full details.

Key ideas:

- **Fork nodes** copy weights to parallel branches, each training a different projection
- **Join nodes** merge branches via weight averaging
- **12 nodes across 4 tiers**: foundation (6 parallel) → cross-domain (3) → complex (2) → integration (1)
- **CogVideoX grafting**: frozen backbone shared across all 12 nodes, only trainable loop params are merged

```python
trainer = DAGCurriculumTrainer(
    nodes=dag,
    data_sources=data_sources,
    backbone="cogvideox",
    device="cuda",
)
trainer.run()
```

### LLM-driven curriculum design

The `build_curriculum()` function uses an LLM to examine available datasets and design an optimal training schedule:

```python
from general_unified_world_model import build_curriculum, DatasetProfile

curriculum = build_curriculum(
    goal="Fine-tune to learn cardiovascular patient health",
    datasets=[
        DatasetProfile(name="Hospital EHR", ...),
        DatasetProfile(name="Insurance Claims", ...),
    ],
)
```

## Compute requirements

| Tier | Nodes | Fields/node | H100-hrs/node |
|------|-------|-------------|---------------|
| 0 | 6 | 25-60 | 2-4 |
| 1 | 3 | 80-105 | 6-10 |
| 2 | 2 | 140-180 | 12-18 |
| 3 | 1 | 857 | 80-120 |

Total: ~175,000 training steps across 12 nodes. With 6-way parallelism at tier 0, wall-clock time drops significantly.
