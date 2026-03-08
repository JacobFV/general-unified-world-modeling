# Examples

Example scripts are in the [`examples/`](https://github.com/JacobFV/general-unified-world-modeling/tree/develop/examples) directory. Each is self-contained and runnable.

## Example gallery

| Script | Description | Source |
|--------|-------------|--------|
| `01_quickstart.py` | Compile full world model, inspect fields and topology | [:octicons-code-16:](https://github.com/JacobFV/general-unified-world-modeling/blob/develop/examples/01_quickstart.py) |
| `02_ceo_company_model.py` | CEO use case: model a company in macro/market context | [:octicons-code-16:](https://github.com/JacobFV/general-unified-world-modeling/blob/develop/examples/02_ceo_company_model.py) |
| `03_government_policy.py` | Government use case: policy impact analysis | [:octicons-code-16:](https://github.com/JacobFV/general-unified-world-modeling/blob/develop/examples/03_government_policy.py) |
| `04_computer_use_agent.py` | Agent use case: user psychology + world context | [:octicons-code-16:](https://github.com/JacobFV/general-unified-world-modeling/blob/develop/examples/04_computer_use_agent.py) |
| `05_train_financial.py` | Train on real FRED + Yahoo Finance data | [:octicons-code-16:](https://github.com/JacobFV/general-unified-world-modeling/blob/develop/examples/05_train_financial.py) |
| `06_curriculum_training.py` | Full DAG curriculum training | [:octicons-code-16:](https://github.com/JacobFV/general-unified-world-modeling/blob/develop/examples/06_curriculum_training.py) |

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

<figure markdown>
  ![Full world canvas](assets/canvas_full_world.png){ loading=lazy }
  <figcaption>Output: 857 fields on a 128x128 canvas with 11,735 attention connections.</figcaption>
</figure>

[:octicons-code-16: Source: examples/01_quickstart.py](https://github.com/JacobFV/general-unified-world-modeling/blob/develop/examples/01_quickstart.py){ .md-button }

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

<figure markdown>
  ![Hedge fund canvas](assets/canvas_hedge_fund.png){ loading=lazy }
  <figcaption>Hedge fund projection: financial, macro, regime, forecasts, and two firms on a 64x64 canvas.</figcaption>
</figure>

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

[:octicons-code-16: Source: examples/05_train_financial.py](https://github.com/JacobFV/general-unified-world-modeling/blob/develop/examples/05_train_financial.py){ .md-button }

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

[:octicons-code-16: Source: examples/06_curriculum_training.py](https://github.com/JacobFV/general-unified-world-modeling/blob/develop/examples/06_curriculum_training.py){ .md-button }

### Inference

```python
from general_unified_world_model import WorldModel

model = WorldModel.load("checkpoint.pt", projection)
model.observe("financial.yield_curves.ten_year", 4.25)
model.observe("country_us.macro.inflation.headline_cpi", 3.1)

predictions = model.predict(n_steps=50)
print(predictions["forecasts.macro.recession_prob_3m"])
```

!!! warning "Coming soon"
    Pretrained checkpoints for inference are not yet released. Training is in progress on H100 GPUs. Check back for checkpoint download links.

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

## Visualizations

All visualizations are generated by [`scripts/generate_assets.py`](https://github.com/JacobFV/general-unified-world-modeling/blob/develop/scripts/generate_assets.py). To regenerate:

```bash
python scripts/generate_assets.py
```

<div class="grid" markdown>

<figure markdown>
  ![Regime dashboard](assets/regime_dashboard.png){ loading=lazy }
  <figcaption>Regime state dashboard</figcaption>
</figure>

<figure markdown>
  ![Social graph](assets/social_graph_ceo.png){ loading=lazy }
  <figcaption>CEO entity network</figcaption>
</figure>

</div>
