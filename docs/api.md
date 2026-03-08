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

Declares which parts of a schema to include. Uses a unified `entities` dict for dynamic entities (firms, individuals, countries, sectors, supply chain nodes).

```python
from general_unified_world_model import WorldProjection
from general_unified_world_model.schema.business import Business
from general_unified_world_model.schema.individual import Individual

proj = WorldProjection(
    include=["financial", "country_us.macro", "regime", "forecasts"],
    exclude=[],
    entities={
        "firm_AAPL": Business(),
        "firm_NVDA": Business(),
        "person_ceo": Individual(),
    },
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `include` | `list[str]` | Dotted paths into the schema to include. `["*"]` = all. |
| `exclude` | `list[str]` | Paths to exclude (applied after include) |
| `entities` | `dict[str, Any]` | Dynamic entities: name -> dataclass instance |
| `connectivity` | `ConnectivityPolicy \| None` | Override connectivity policy |
| `temporal_start` | `dict[str, int]` | Entity temporal start indices |

### `project()`

Compiles a schema projection into a `BoundSchema`. Accepts either a schema root directly or a `WorldProjection`.

```python
from general_unified_world_model import World, project
from general_unified_world_model.schema.business import Business

# Clean API: schema root + include/exclude
bound = project(World(), include=["financial", "regime"], d_model=64)

# With entities
bound = project(
    World(),
    include=["financial", "country_us.macro"],
    entities={"firm_AAPL": Business()},
    d_model=64,
)

# Default root (World())
bound = project(include=["regime"], d_model=64)

# Legacy: WorldProjection
proj = WorldProjection(include=["financial"])
bound = project(proj, T=1, d_model=64)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `proj_or_root` | `WorldProjection \| dataclass \| None` | Schema root or projection. None defaults to `World()`. |
| `include` | `list[str] \| None` | Dotted paths to include. Default `["*"]`. |
| `exclude` | `list[str] \| None` | Paths to exclude. |
| `entities` | `dict[str, Any] \| None` | Dynamic entity dict. |
| `T` | `int` | Temporal dimension (typically 1) |
| `H` | `int \| None` | Canvas height (None = auto-size) |
| `W` | `int \| None` | Canvas width (None = auto-size) |
| `d_model` | `int` | Latent dimension per position |
| `connectivity` | `ConnectivityPolicy \| None` | Connectivity policy override |
| `t_current` | `int` | Timestep boundary for output mask |

**Returns:** `BoundSchema` with `.field_names`, `.layout`, `.topology`, `.fields`.

---

## Inference

### `WorldModel`

General-purpose world model base class. Works with any canvas schema. Holds runtime canvas state and supports dynamic layout/topology changes.

```python
from general_unified_world_model import WorldModel

# From a pre-compiled schema (auto-builds backbone/encoder/decoder)
model = WorldModel(bound_schema, device="cpu")

# From any schema root
model = WorldModel.from_schema(my_schema, include=["sensor"], d_model=32)

# From checkpoint
model = WorldModel.load("checkpoint.pt", bound_schema=bound)
```

#### `WorldModel(bound_schema, backbone=None, encoder=None, decoder=None, device, ...)`

Constructor. When backbone/encoder/decoder are None, they are auto-built from the schema.

#### `WorldModel.observe(field_path, value, t=None)`

Set an observed value. Writes to both the observations dict and the canvas tensor.

```python
model.observe("financial.yield_curves.ten_year", 4.25)
```

#### `WorldModel.predict(n_steps=50)`

Run diffusion inference conditioned on observations. Returns a dict mapping field paths to predicted values.

```python
predictions = model.predict(n_steps=50)
recession_prob = predictions["forecasts.macro.recession_prob_3m"]
```

#### `WorldModel.get_canvas(t=None)`

Get the canvas state tensor. Returns `(1, N, d_model)`.

```python
canvas = model.get_canvas()       # full canvas
frame = model.get_canvas(t=0)     # positions at timestep 0
```

#### `WorldModel.clear_observations()`

Clear all observations and reset canvas to zeros.

#### `WorldModel.resize_layout(H=None, W=None, T=None, d_model=None)`

Change canvas dimensions. Recompiles schema, transfers data by field name, zero-initializes new positions.

```python
model.resize_layout(H=64, W=64)  # expand canvas
```

#### `WorldModel.set_topology(topology)`

Change the attention topology. Rebuilds dispatchers in backbone blocks.

```python
from canvas_engineering import CanvasTopology
model.set_topology(CanvasTopology.dense(regions))
```

#### `WorldModel.add_region(name, spec)` / `WorldModel.remove_region(name)`

Dynamic region management. New regions are zero-initialized.

#### `WorldModel.ingest(data, spec)`

Populate canvas from a data dict using a DatasetSpec.

```python
model.ingest({"gdp": 2.1, "cpi": 3.4}, spec=fred_spec)
```

#### `WorldModel.from_schema(schema_root, include, exclude, entities, ...)`

Class method. Create a WorldModel from any dataclass schema root.

#### `WorldModel.save(path)` / `WorldModel.load(path, ...)`

Save/load checkpoints (backbone, encoder, decoder, canvas state, observations).

---

### `GeneralUnifiedWorldModel`

Convenience subclass of `WorldModel` with the built-in 857-field `World()` schema.

```python
from general_unified_world_model import GeneralUnifiedWorldModel
from general_unified_world_model.schema.business import Business

model = GeneralUnifiedWorldModel(
    include=["financial", "country_us.macro", "regime", "forecasts"],
    entities={"firm_AAPL": Business()},
    d_model=64,
)

model.observe("financial.yield_curves.ten_year", 4.25)
predictions = model.predict()
```

#### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `include` | `list[str] \| None` | Schema paths to include. Default: all 857 fields. |
| `exclude` | `list[str] \| None` | Paths to exclude. |
| `entities` | `dict[str, Any] \| None` | Dynamic entities. |
| `T` | `int` | Temporal extent. |
| `H`, `W` | `int \| None` | Canvas size. None = auto-sized. |
| `d_model` | `int` | Latent dimension. |
| `device` | `str` | Device. |
| `n_layers` | `int` | Backbone depth (default 6). |
| `n_loops` | `int` | Looped attention iterations (default 3). |
| `dataset_specs` | `list[DatasetSpec] \| None` | Pre-registered dataset specs. |

#### `GeneralUnifiedWorldModel.project_subset(include, exclude, entities)`

Create a new `GeneralUnifiedWorldModel` from a subset of the current model's fields.

#### `GeneralUnifiedWorldModel.load(path, include, exclude, ...)`

Load from checkpoint with simplified signature (no WorldProjection needed).

---

## Training

### `build_world_model(bound, n_layers, n_heads, d_ff, n_loops, use_dispatch)`

Build a `WorldModelBackbone` transformer for a given `BoundSchema`. When `use_dispatch=True` (default), uses per-connection attention dispatch — each connection uses its declared attention function type.

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

### `MaskedCanvasTrainer`

Training loop with masked loss over canvas positions.

### `DAGCurriculumTrainer`

Fork-join DAG curriculum with weight merging. Supports both CogVideoX and scratch backbones.

```python
from general_unified_world_model import DAGCurriculumTrainer

trainer = DAGCurriculumTrainer(
    nodes=dag,
    data_sources=data_sources,
    backbone="cogvideox",
    device="cuda",
)
trainer.run()
```

### `build_curriculum()` / `DatasetProfile`

LLM-driven curriculum design. Examines datasets and generates an optimal training schedule.

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

---

## Transfer distance

### `TransferDistanceEstimator`

Estimates generalization distance between projections using semantic embedding similarity.

```python
from general_unified_world_model import TransferDistanceEstimator

estimator = TransferDistanceEstimator()
distance = estimator.estimate(projection_a, projection_b)
```
