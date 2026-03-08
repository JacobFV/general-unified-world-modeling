# Examples

Example scripts are in the [`examples/`](https://github.com/JacobFV/general-unified-world-modeling/tree/develop/examples) directory.

| Script | Description |
|--------|-------------|
| `01_quickstart.py` | Compile full world model, inspect fields and topology |
| `02_ceo_company_model.py` | CEO use case: model a company in macro/market context |
| `03_government_policy.py` | Government use case: policy impact analysis |
| `04_computer_use_agent.py` | Agent use case: user psychology + world context |
| `05_train_financial.py` | Train on real FRED + Yahoo Finance data |
| `06_curriculum_training.py` | Full DAG curriculum training |

## Quick examples

### Compile and inspect the full world

```python
from canvas_engineering import compile_schema, ConnectivityPolicy
from general_unified_world_model import World

world = World()
bound = compile_schema(
    world, T=1, H=128, W=128, d_model=64,
    connectivity=ConnectivityPolicy(intra="dense", parent_child="hub_spoke"),
)

print(f"{len(bound.field_names)} fields")
print(f"{bound.layout.num_positions} positions")
print(f"{len(bound.topology.connections)} connections")
```

### Project for a hedge fund

```python
from general_unified_world_model import WorldProjection, project

proj = WorldProjection(
    include=[
        "financial",
        "country_us.macro",
        "regime",
        "forecasts.macro",
        "forecasts.financial",
    ],
    firms=["AAPL", "NVDA"],
)

bound = project(proj, T=1, d_model=64)  # H, W auto-sized
```

### Train on heterogeneous data

```python
from general_unified_world_model import (
    project, WorldProjection, build_world_model,
    FieldEncoder, FieldDecoder, MaskedCanvasTrainer,
    DatasetSpec, InputSpec, OutputSpec, build_mixed_dataloader,
)

proj = WorldProjection(include=["financial", "country_us.macro", "regime"])
bound = project(proj, T=1, d_model=64)

backbone = build_world_model(bound, n_layers=4, n_heads=4, d_ff=256, n_loops=3)
encoder = FieldEncoder(bound)
decoder = FieldDecoder(bound)

macro_spec = DatasetSpec(
    name="FRED",
    input_specs=[
        InputSpec("gdp", "US GDP growth rate", "country_us.macro.output.gdp_nowcast"),
        InputSpec("cpi", "US CPI YoY", "country_us.macro.inflation.headline_cpi"),
    ],
    output_specs=[
        OutputSpec("gdp", "US GDP growth rate", "country_us.macro.output.gdp_nowcast"),
        OutputSpec("cpi", "US CPI YoY", "country_us.macro.inflation.headline_cpi"),
    ],
)
market_spec = DatasetSpec(
    name="Yahoo",
    input_specs=[
        InputSpec("vix", "CBOE VIX index", "financial.equities.vix"),
        InputSpec("ust10y", "10-year Treasury yield", "financial.yield_curves.ten_year"),
    ],
    output_specs=[
        OutputSpec("vix", "CBOE VIX index", "financial.equities.vix"),
        OutputSpec("ust10y", "10-year Treasury yield", "financial.yield_curves.ten_year"),
    ],
)

loader = build_mixed_dataloader(
    bound,
    sources=[(macro_spec, macro_data), (market_spec, market_data)],
    batch_size=32,
)
```

### DAG curriculum with CogVideoX

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

### Inference

```python
from general_unified_world_model import WorldModel

model = WorldModel.load("checkpoint.pt", projection)
model.observe("financial.yield_curves.ten_year", 4.25)
model.observe("country_us.macro.inflation.headline_cpi", 3.1)

predictions = model.predict(n_steps=50)
print(predictions["forecasts.macro.recession_prob_3m"])
```

### LLM-powered projection

```python
from general_unified_world_model import llm_project

result = llm_project(
    "I'm a hedge fund PM. Model US macro, rates, credit, "
    "Apple and NVIDIA. I care about recession risk.",
    provider="anthropic",
)
bound = result.compile(T=1, d_model=64)
```

### LLM-driven curriculum design

```python
from general_unified_world_model import build_curriculum, DatasetProfile

curriculum = build_curriculum(
    goal="Fine-tune to learn cardiovascular patient health",
    datasets=[
        DatasetProfile(name="Hospital EHR", description="Electronic health records"),
        DatasetProfile(name="Insurance Claims", description="Claims and diagnoses"),
    ],
)

nodes = curriculum.to_training_nodes()
```
