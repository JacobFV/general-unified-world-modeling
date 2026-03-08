# API Reference

## Schema

### `World`

Top-level schema dataclass. Contains all 19 layers as fields. Instantiate with `World()` to get the full 857-field schema.

```python
from general_unified_world_model import World

world = World()
```

Compile directly with canvas-engineering:

```python
from canvas_engineering import compile_schema, ConnectivityPolicy

bound = compile_schema(
    world, T=1, H=128, W=128, d_model=64,
    connectivity=ConnectivityPolicy(intra="dense", parent_child="hub_spoke"),
)
```

See [Schema Reference](schema.md) for a full breakdown of all 19 layers and 857 fields.

---

## Projection

### `WorldProjection`

Declares which parts of the world schema to include.

```python
from general_unified_world_model import WorldProjection

proj = WorldProjection(
    include=["financial", "country_us.macro", "regime", "forecasts"],
    exclude=[],                 # optional: paths to exclude
    firms=["AAPL", "NVDA"],     # dynamic firm entities
    individuals=["ceo"],        # dynamic individual entities
    countries=["jp", "uk"],     # additional countries beyond US/CN/EU
    sectors=["tech"],           # additional GICS sectors
    supply_chains=[],           # supply chain node IDs
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `include` | `list[str]` | Dotted paths into the World schema to include |
| `exclude` | `list[str]` | Paths to exclude (applied after include) |
| `firms` | `list[str]` | Ticker symbols for dynamic `Business` entities |
| `individuals` | `list[str]` | IDs for dynamic `Individual` entities |
| `countries` | `list[str]` | ISO codes for additional `Country` entities |
| `sectors` | `list[str]` | Names for additional `Sector` entities |
| `supply_chains` | `list[str]` | IDs for `SupplyChainNode` entities |

### `project()`

Compiles a `WorldProjection` into a `BoundSchema`.

```python
from general_unified_world_model import project

bound = project(proj, T=1, d_model=64)  # H, W auto-sized
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `projection` | `WorldProjection` | The projection to compile |
| `T` | `int` | Temporal dimension (typically 1) |
| `H` | `int \| None` | Canvas height (None = auto-size) |
| `W` | `int \| None` | Canvas width (None = auto-size) |
| `d_model` | `int` | Latent dimension per position |
| `connectivity` | `ConnectivityPolicy` | Optional connectivity override |

**Returns:** `BoundSchema` with `.field_names`, `.layout`, `.topology`, `.attention_mask`.

---

## Inference

### `WorldModel`

High-level inference interface. Load a trained checkpoint, observe known fields, predict the rest.

```python
from general_unified_world_model import WorldModel

model = WorldModel.load("checkpoint.pt", projection)
```

#### `WorldModel.load(path, projection, device="cuda")`

Class method. Loads backbone, encoder, and decoder weights from a checkpoint.

#### `WorldModel.observe(field_path, value)`

Set an observed value for a field.

```python
model.observe("financial.yield_curves.ten_year", 4.25)
model.observe("country_us.macro.inflation.headline_cpi", 3.1)
```

#### `WorldModel.predict(n_steps=50)`

Run diffusion inference conditioned on observations. Returns a dict mapping field paths to predicted values.

```python
predictions = model.predict(n_steps=50)
recession_prob = predictions["forecasts.macro.recession_prob_3m"]
regime = predictions["regime.growth_regime"]
```

#### `WorldModel.reset()`

Clear all observations.

---

## Training

### `build_world_model(bound, n_layers, n_heads, d_ff, n_loops)`

Build a `WorldModelBackbone` transformer from scratch for a given `BoundSchema`.

```python
from general_unified_world_model import build_world_model

backbone = build_world_model(bound, n_layers=8, n_heads=4, d_ff=256, n_loops=3)
```

### `CogVideoXBackbone` / `build_cogvideox_world_model()`

CogVideoX-grafted backbone. Loads a pretrained CogVideoX transformer, freezes its blocks, and adds per-block loop embeddings + projection layers as the only trainable parameters (~0.1% of total).

```python
from general_unified_world_model.training.backbone import (
    CogVideoXBackbone,
    build_cogvideox_world_model,
)

backbone = build_cogvideox_world_model(
    transformer=cogvideox_transformer,
    bound_schema=bound,
    n_loops=3,
)
```

See [CogVideoX Backbone](cogvideox.md) for architecture details.

### `FieldEncoder` / `FieldDecoder`

Per-field linear projections between scalar values and `d_model`-dimensional canvas positions.

```python
from general_unified_world_model import FieldEncoder, FieldDecoder

encoder = FieldEncoder(bound)
decoder = FieldDecoder(bound)
```

### `DatasetSpec` / `InputSpec` / `OutputSpec`

Declare how a data source maps to schema fields. `InputSpec` defines input modalities, `OutputSpec` defines output modalities with loss configuration.

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
    ],
    output_specs=[
        OutputSpec(
            key="gdp",
            semantic_type="US real GDP quarterly growth rate",
            field_path="country_us.macro.output.gdp_nowcast",
            loss_weight=2.0,
        ),
    ],
)
```

**InputSpec parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | Column/key name in the raw data |
| `semantic_type` | `str` | Natural language description for conditioning |
| `field_path` | `str` | Dotted path in the world schema |
| `dtype` | `str` | Data type hint (default `"float32"`) |
| `encoder` | `Any` | Custom encoder module (optional) |
| `region_size` | `int \| None` | Override canvas region size |
| `transform` | `Callable` | Pre-processing transform (optional) |
| `frequency` | `int \| None` | Update frequency (None = every tick) |

**OutputSpec parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | Column/key name for ground truth |
| `semantic_type` | `str` | Natural language description |
| `field_path` | `str` | Dotted path in the world schema |
| `dtype` | `str` | Data type hint (default `"float32"`) |
| `decoder` | `Any` | Custom decoder module (optional) |
| `loss_fn` | `str \| Callable` | Loss function (default `"mse"`) |
| `loss_weight` | `float` | Relative loss weight (default `1.0`) |
| `transform` | `Callable` | Inverse transform on predictions (optional) |
| `frequency` | `int \| None` | Update frequency |

### `build_mixed_dataloader(bound, sources, batch_size)`

Create a dataloader that interleaves multiple heterogeneous data sources.

```python
from general_unified_world_model import build_mixed_dataloader

loader = build_mixed_dataloader(
    bound,
    sources=[(fred_spec, fred_data), (yahoo_spec, yahoo_data)],
    batch_size=32,
)
```

### `MaskedCanvasTrainer`

Training loop with masked loss over canvas positions.

```python
from general_unified_world_model import MaskedCanvasTrainer

trainer = MaskedCanvasTrainer(
    bound_schema=bound,
    backbone=backbone,
    encoder=encoder,
    decoder=decoder,
    optimizer=optimizer,
)
trainer.train(dataloader, n_steps=10000)
```

### `DAGCurriculumTrainer`

Fork-join DAG curriculum with weight merging. Supports both CogVideoX and scratch backbones.

```python
from general_unified_world_model import DAGCurriculumTrainer

trainer = DAGCurriculumTrainer(
    nodes=dag,
    data_sources=data_sources,
    backbone="cogvideox",                      # or "scratch"
    pretrained_model_id="THUDM/CogVideoX-2b",  # for cogvideox
    device="cuda",
)
trainer.run()
```

### `build_curriculum()` / `DatasetProfile`

LLM-driven curriculum design. Examines datasets and generates an optimal training schedule.

```python
from general_unified_world_model import build_curriculum, DatasetProfile

curriculum = build_curriculum(
    goal="Learn financial risk dynamics",
    datasets=[DatasetProfile(name="FRED", ...)],
)
nodes = curriculum.to_training_nodes()
```

### `CurriculumTrainer` / `CurriculumConfig`

Sequential multi-phase curriculum training.

---

## Temporal entities

### `TemporalTopology`

Manages entities that appear and disappear over time.

```python
from general_unified_world_model import TemporalTopology
from general_unified_world_model.schema.business import Business

tt = TemporalTopology()
tt.add("firm_AAPL", Business(), start_tick=100)
tt.add("firm_ENRON", Business(), start_tick=0, end_tick=500)

active = tt.active_at(50)
mask = tt.generate_temporal_attention_mask((0, 1000), bound_schema)
```

---

## Rendering

### `render(bound_schema, renderer_name, **kwargs)`

Convenience function for visualization. Requires `pip install general-unified-world-model[viz]`.

```python
from general_unified_world_model import render

fig = render(bound, "canvas_heatmap")
fig = render(bound, "topology_graph")
render(bound, "canvas_heatmap", save_path="output.png")
```

Available renderers: `canvas_heatmap`, `topology_graph`, `financial_chart`, `geopolitical_map`, `regime_dashboard`, `social_graph`.

---

## LLM projection builder

### `llm_project(description, provider="anthropic", api_key=None)`

Use an LLM to design a `WorldProjection` from a natural language description.

```python
from general_unified_world_model import llm_project

result = llm_project(
    "Hedge fund PM modeling US macro, rates, credit, Apple and NVIDIA",
    provider="anthropic",
)
bound = result.compile(T=1, d_model=64)
```

**Returns:** object with `.projection`, `.reasoning`, `.compile()` method.

---

## Transfer distance

### `TransferDistanceEstimator`

Estimates generalization distance between projections using semantic embedding similarity. Used by the curriculum to decide domain coupling order.

```python
from general_unified_world_model import TransferDistanceEstimator

estimator = TransferDistanceEstimator()
distance = estimator.estimate(projection_a, projection_b)
```
