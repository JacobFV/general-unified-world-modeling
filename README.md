# general-unified-world-model

### A typed causal ontology of civilization, built on [canvas-engineering](https://github.com/JacobFV/canvas-engineering) structured latent spaces.

[![PyPI](https://img.shields.io/pypi/v/general-unified-world-model.svg)](https://pypi.org/project/general-unified-world-model/)
[![Tests](https://github.com/JacobFV/general-unified-world-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/JacobFV/general-unified-world-modeling/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-144%2F144-brightgreen.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> Canvas engineering structures what a diffusion model *thinks in*. This repo declares a **857-field typed schema** spanning planetary physics through individual psychology, compiles it onto a structured latent canvas, and trains it on heterogeneous real-world data — without throwing out samples that are missing fields.

---

<p align="center">
<img src="https://raw.githubusercontent.com/JacobFV/general-unified-world-modeling/develop/assets/canvas_full_world.png" width="100%" alt="Full World Model — 857 fields on 128x128 canvas" />
</p>
<p align="center"><em>Full World Model — 857 fields allocated on a 128×128 canvas. Each colored region is a semantic domain.</em></p>

---

## The idea

Every dataset in the world describes a slice of the same underlying reality. GDP data captures macroeconomic output. Market data captures prices. News captures narratives. Earnings calls capture firm strategy. But no single dataset captures everything.

Traditional approaches either:
- **(a)** restrict to the intersection — throw out data missing any field
- **(b)** impute missing values — introduce noise

**General Unified World Model** takes option **(c)**: mask missing fields in the loss, train on what you have. Each dataset declares which fields it populates. The model learns the joint distribution across all modalities, even though no single dataset contains everything.

The key enabler is **canvas-engineering** — a type system for multimodal latent computation. Each field in the world model occupies specific positions on a 3D `(T, H, W)` canvas grid, with declared temporal frequency, loss weight, and connectivity. The topology is the compute graph.

## Quick start

```bash
pip install general-unified-world-model
```

### Compile the full world model

```python
from canvas_engineering import compile_schema, ConnectivityPolicy
from general_unified_world_model import World

world = World()
bound = compile_schema(
    world,
    T=1, H=128, W=128, d_model=64,
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",
    ),
)

print(f"{len(bound.field_names)} fields, "
      f"{bound.layout.num_positions} positions, "
      f"{len(bound.topology.connections)} connections")
# 857 fields, 16384 positions, 11735 connections
```

### Project to a subset

You don't need the full 857-field model. Declare what you care about:

```python
from general_unified_world_model import WorldProjection, project

# Hedge fund: macro + financial + two firms
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

bound = project(proj, T=1, H=64, W=64, d_model=64)
# ~200 fields, focused on what matters
```

### Or just describe what you need

You don't have to construct projections by hand. Describe your modeling needs in plain English and let `general-unified-world-model` build the projection for you:

```python
from general_unified_world_model import llm_project

result = llm_project(
    "I'm a hedge fund PM. I need to model US macro, rates, credit, "
    "and two firms: Apple and NVIDIA. I care about recession risk "
    "and the Fed's next move.",
    provider="anthropic",  # or "openai"
)

# The LLM selects the right fields automatically
bound = result.compile(T=1, H=64, W=64, d_model=64)
print(result.reasoning)
# "Hedge fund needs financial markets (yield curves, credit spreads,
#  equities), US macro (GDP, inflation, labor), regime indicators
#  for recession detection, and firm-level nodes for AAPL and NVDA..."
```

### Train on heterogeneous data

```python
from general_unified_world_model import (
    WorldProjection, project, build_world_model,
    FieldEncoder, FieldDecoder, MaskedCanvasTrainer,
    DatasetSpec, FieldMapping, build_mixed_dataloader,
)

# Two data sources with different field coverage
macro_spec = DatasetSpec(
    name="FRED",
    mappings=[
        FieldMapping("gdp", "country_us.macro.output.gdp_nowcast"),
        FieldMapping("cpi", "country_us.macro.inflation.headline_cpi"),
    ],
)
market_spec = DatasetSpec(
    name="Yahoo",
    mappings=[
        FieldMapping("vix", "financial.equities.vix"),
        FieldMapping("ust10y", "financial.yield_curves.ten_year"),
    ],
)

# Both train the same canvas — missing fields are masked, not imputed
loader = build_mixed_dataloader(
    bound,
    sources=[(macro_spec, macro_data), (market_spec, market_data)],
    batch_size=32,
)
```

## The schema

19 layers, 857 fields, 8 temporal frequency classes:

| Layer | Fields | Frequency | What it captures |
|-------|--------|-----------|------------------|
| **Planetary Physical** | Climate, infrastructure, disasters | τ6–τ7 (annual–multi-year) | Slow structural constraints |
| **Resources & Energy** | Crude, metals, food, water, compute | τ1–τ4 (hourly–monthly) | Physical inputs to production |
| **Global Financial** | Yields, credit, FX, equities, crypto | τ0–τ2 (sub-minute–daily) | High-bandwidth reflexive core |
| **Macroeconomy** | GDP, inflation, labor, fiscal, trade, housing | τ3–τ5 (weekly–quarterly) | Real economy per country |
| **Political** | Executive, legislative, judicial, geopolitical | τ4–τ7 (monthly–multi-year) | Governance structures |
| **Narrative & Belief** | Media, elite consensus, public sentiment | τ0–τ4 (sub-minute–monthly) | Reflexivity layer |
| **Technology** | AI, biotech, quantum, robotics, productivity | τ5–τ7 (quarterly–multi-year) | Long-run structural drivers |
| **Demographics** | Population, dependency, urbanization | τ7 (multi-year) | Slowest structural force |
| **Sector** | Demand, supply, margins, disruption risk | τ3–τ5 (weekly–quarterly) | Per GICS sector |
| **Supply Chain** | Concentration, lead time, bottleneck severity | τ2–τ4 (daily–monthly) | Graph-structured nodes |
| **Business** | Financials, operations, strategy, market, risk | τ2–τ5 (daily–quarterly) | Per firm (sparse) |
| **Individual** | Cognitive, incentives, network, state | τ2–τ5 (daily–quarterly) | Key decision-makers (very sparse) |
| **Event Tape** | News, social, filings, policy, conflict | τ0–τ1 (sub-minute–hourly) | Real-time event stream |
| **Data Channel Trust** | Government, market, alternative, corporate | τ3–τ7 | Meta-epistemic calibration |
| **Regime State** | Growth, inflation, financial cycle, fragility | τ5–τ7 | Compressed global latent |
| **Intervention** | Monetary, fiscal, regulatory, military + effects | τ2–τ5 | Counterfactual analysis |
| **Forecast Bundle** | Recession prob, credit stress, conflict risk | output | Structured prediction heads |
| **Country** | Macro + politics + demographics per country | composite | Per major economy |

### Temporal frequency classes

```
τ0 = sub-minute   (period=1)      markets, breaking news
τ1 = hourly        (period=4)      grid load, commodities
τ2 = daily         (period=16)     commodity prices, port congestion
τ3 = weekly        (period=48)     claims, inventories, payroll
τ4 = monthly       (period=192)    CPI, PMI, company closes
τ5 = quarterly     (period=576)    earnings, GDP, capex
τ6 = annual        (period=2304)   demographics, infrastructure
τ7 = multi-year    (period=4608)   regime changes, tech diffusion
```

## Use cases

### CEO: "Model my company in context"

```python
proj = WorldProjection(
    include=[
        "country_us.macro",
        "sector_tech",
        "financial.yield_curves",
        "financial.equities",
        "regime",
        "forecasts",
    ],
    firms=["ACME", "RIVAL"],
    individuals=["ceo", "cfo", "cto"],
)
```

<p align="center">
<img src="https://raw.githubusercontent.com/JacobFV/general-unified-world-modeling/develop/assets/usecase_ceo.png" width="90%" alt="CEO use case — causal interaction graph" />
</p>
<p align="center"><em>Causal interaction graph: regime conditions macro and financial context, which flows into firm dynamics. Executive decisions influence the firm. Competitive dynamics (dashed) between ACME and RIVAL. Forecasts are the structured output.</em></p>

### Government: "Model policy impact"

```python
proj = WorldProjection(
    include=[
        "country_us",
        "country_cn.macro",
        "country_eu.macro",
        "financial",
        "interventions",
        "forecasts",
        "regime",
    ],
    countries=["jp", "uk"],
)
```

<p align="center">
<img src="https://raw.githubusercontent.com/JacobFV/general-unified-world-modeling/develop/assets/usecase_government.png" width="90%" alt="Government use case — causal interaction graph" />
</p>
<p align="center"><em>Policy transmission graph: regime state conditions all economies. Bilateral trade and financial linkages connect countries (bidirectional arrows). Interventions propagate through the financial system and the domestic economy to produce structured forecasts.</em></p>

### Computer use agent: "Model the user's world"

```python
proj = WorldProjection(
    include=[
        "events",
        "regime.compressed_world_state",
        "forecasts.macro.recession_prob_3m",
    ],
    individuals=["user"],
    firms=["user_org"],
)
```

<p align="center">
<img src="https://raw.githubusercontent.com/JacobFV/general-unified-world-modeling/develop/assets/usecase_agent.png" width="70%" alt="Agent use case — causal interaction graph" />
</p>
<p align="center"><em>Minimal world context for an agent: real-time events feed into the user and organization models. Regime state drives recession forecasts. User and organization are bidirectionally linked.</em></p>

## Training architecture

### Phase 1: Independent domains (parallelizable)

Train each domain separately on small canvases. Financial markets, US macro, narratives, etc. each get their own backbone. This is fast because canvases are small.

### Phase 2: Domain coupling

Merge causally adjacent domains (financial + macro, narratives + financial). Pretrained encoders/decoders transfer via matching field names. The shared regime latent begins learning cross-domain structure.

### Phase 3: Full integration

All domains on one canvas. The regime state gets gradient from everything. This is the most expensive phase but leverages all pretrained structure.

### Phase 4: Task-specific fine-tuning

Freeze backbone. Train projection-specific heads (recession prediction, equity regime, conflict escalation).

### Why this works

The semantic type system lets us proxy **generalization distance** between any two modalities by their **semantic embedding distance**. GDP growth and industrial production are semantically close — their latent dynamics will be correlated. GDP growth and seismic risk are semantically far — nearly independent. This guides curriculum design: couple close domains first, distant later.

## Heterogeneous data training

The key innovation: **masked loss on structured canvas**.

```
Dataset A (FRED):     GDP ✓  CPI ✓  VIX ✗  Yields ✗
Dataset B (Yahoo):    GDP ✗  CPI ✗  VIX ✓  Yields ✓
Dataset C (News):     GDP ✗  CPI ✗  VIX ✗  Yields ✗  News ✓

Canvas loss:  L = Σ (prediction - target)² × presence × loss_weight
                      ↑ model predicts all    ↑ only active  ↑ from schema
```

Both A and B train the **shared regime latent**, even though their field coverage doesn't overlap. The regime latent learns to compress the joint distribution from partial observations.

## Data adapters

Built-in adapters for common data sources:

```python
from general_unified_world_model.data.adapters import fred_adapter, yahoo_finance_adapter

# FRED: 50+ macro series mapped to world model fields
fred_spec, fred_data = fred_adapter(api_key="...", start_date="2010-01-01")

# Yahoo Finance: equities, FX, commodities, crypto
yahoo_spec, yahoo_data = yahoo_finance_adapter(
    include_equity=True, include_fx=True,
    firm_tickers={"AAPL": "firm_AAPL"},
)

# Generic CSV/Parquet
from general_unified_world_model.data.adapters import tabular_adapter
spec, data = tabular_adapter(
    "My Dataset", "data.csv",
    column_mappings={"gdp_growth": "country_us.macro.output.gdp_nowcast"},
    transforms={"gdp_growth": "z_score"},
)
```

## Temporal entities

Entities can appear and disappear over time:

```python
from general_unified_world_model import TemporalTopology
from general_unified_world_model.schema.business import Business

tt = TemporalTopology()
tt.add("firm_AAPL", Business(), start_tick=100)    # founded
tt.add("firm_ENRON", Business(), start_tick=0, end_tick=500)  # dissolved

# At tick 50: ENRON exists, AAPL doesn't yet
active = tt.active_at(50)

# Generate attention mask that blocks inactive entities
mask = tt.generate_temporal_attention_mask((0, 1000), bound_schema)
```

## Inference

```python
from general_unified_world_model import WorldModel

model = WorldModel.load("checkpoint.pt", projection)

# Observe what you know
model.observe("financial.yield_curves.ten_year", 4.25)
model.observe("country_us.macro.inflation.headline_cpi", 3.1)
model.observe("financial.equities.vix", 18.5)

# Predict everything else
predictions = model.predict(n_steps=50)

recession_prob = predictions["forecasts.macro.recession_prob_3m"]
regime = predictions["regime.growth_regime"]
credit_stress = predictions["forecasts.financial.credit_stress_3m"]
```

## Visualizations

The rendering system provides multiple views into the same world model state. Install the `viz` extra for rendering support: `pip install general-unified-world-model[viz]`

### Canvas heatmaps

Each field occupies a contiguous region on the (H, W) canvas. Colors indicate semantic domain; intensity shows state magnitude.

<p align="center">
<img src="https://raw.githubusercontent.com/JacobFV/general-unified-world-modeling/develop/assets/canvas_macro_projection.png" width="48%" alt="Macro Model Projection" />
<img src="https://raw.githubusercontent.com/JacobFV/general-unified-world-modeling/develop/assets/canvas_hedge_fund.png" width="48%" alt="Hedge Fund Projection" />
</p>
<p align="center"><em>Left: Macro model projection (~40 fields on 32×32). Right: Hedge fund projection with AAPL+NVDA (~200 fields on 64×64).</em></p>

### Domain topology graphs

Nodes are semantic domains, edges show attention connectivity between them. Node size ∝ field count, edge width ∝ connection density.

<p align="center">
<img src="https://raw.githubusercontent.com/JacobFV/general-unified-world-modeling/develop/assets/topology_macro.png" width="48%" alt="Macro Model Topology" />
<img src="https://raw.githubusercontent.com/JacobFV/general-unified-world-modeling/develop/assets/topology_hedge_fund.png" width="48%" alt="Hedge Fund Model Topology" />
</p>
<p align="center"><em>Left: A macroeconomic model's domain graph — macro, rates, credit, and regime are tightly coupled. Right: A hedge fund model adds firm-level nodes and cross-domain positioning.</em></p>

These topology graphs show how different projections create different compute graphs. The macro model has a tight cluster around rates/credit/macro. The hedge fund model fans out to include firm-level nodes (AAPL, NVDA) with edges to financial and macro domains.

### Financial charts

Time series views of world model fields, auto-generated or from real observations.

<p align="center">
<img src="https://raw.githubusercontent.com/JacobFV/general-unified-world-modeling/develop/assets/financial_charts.png" width="80%" alt="Financial Charts" />
</p>

### Geopolitical state map

Each country's latent state vector is projected to RGB via PCA — the color is a 3D projection of the full state representation, not a scalar risk score. Real country boundaries rendered on orthographic globes with cartopy.

<p align="center">
<img src="https://raw.githubusercontent.com/JacobFV/general-unified-world-modeling/develop/assets/geopolitical_globe.gif" width="50%" alt="Rotating Geopolitical Globe" />
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/JacobFV/general-unified-world-modeling/develop/assets/geopolitical_map.png" width="80%" alt="Geopolitical Dual-Hemisphere Map" />
</p>

### Regime dashboard

Horizontal bars for the 12 regime state fields — value magnitude, no decoration. The compressed world state latent strip at the bottom.

<p align="center">
<img src="https://raw.githubusercontent.com/JacobFV/general-unified-world-modeling/develop/assets/regime_dashboard.png" width="80%" alt="Regime Dashboard" />
</p>

### Social graph (CEO perspective)

First-person entity network. Focal entity centered, others positioned by connection strength. Edge weight and color encode relationship intensity (topology-derived + structurally inferred). Field count shown inside each node.

<p align="center">
<img src="https://raw.githubusercontent.com/JacobFV/general-unified-world-modeling/develop/assets/social_graph_ceo.png" width="80%" alt="CEO Social Graph" />
</p>

### Rendering API

```python
from general_unified_world_model import render

# By renderer name
fig = render(bound, "canvas_heatmap")
fig = render(bound, "topology_graph")
fig = render(bound, "financial_chart")

# Save directly
render(bound, "canvas_heatmap", save_path="output.png")

# Or use renderer classes directly
from general_unified_world_model.rendering import (
    CanvasHeatmapRenderer, TopologyGraphRenderer, CausalGraphRenderer,
    FinancialChartRenderer, GeopoliticalMapRenderer,
    RegimeDashboardRenderer, SocialGraphRenderer,
    RenderContext,
    render_ceo_use_case, render_government_use_case, render_agent_use_case,
)

ctx = RenderContext(bound_schema=bound, title="My Model")
renderer = CanvasHeatmapRenderer()
fig = renderer.render(ctx)
renderer.save(ctx, "output.png", dpi=200)
```

## LLM-powered projection builder

Don't want to manually specify field paths? Describe your modeling needs in plain English and let an LLM design the projection for you.

```python
from general_unified_world_model import llm_project

result = llm_project(
    "I'm a hedge fund PM. I need to model US macro, rates, credit, "
    "and two firms: Apple and NVIDIA. I care about recession risk "
    "and the Fed's next move.",
    provider="anthropic",  # or "openai"
    api_key="sk-ant-...",  # or set ANTHROPIC_API_KEY env var
)

# Result contains the designed projection + reasoning
print(result.reasoning)
# "Hedge fund needs financial markets, US macro, regime indicators..."

# Compile to a BoundSchema
bound = result.compile(T=1, H=64, W=64, d_model=64)
print(f"{len(bound.field_names)} fields selected")
```

Uses raw HTTP calls — no SDK dependencies. Supports both Anthropic and OpenAI providers.

## Installation

```bash
# Core
pip install general-unified-world-model

# With real data adapters
pip install general-unified-world-model[data]

# With training infrastructure
pip install general-unified-world-model[train]

# Everything
pip install general-unified-world-model[all]
```

Requires Python 3.10+ and PyTorch 2.0+.

## Examples

```
examples/
├── 01_quickstart.py           # Compile full world model, inspect fields
├── 02_ceo_company_model.py    # CEO use case: company + context
├── 03_government_policy.py    # Government: policy impact analysis
├── 04_computer_use_agent.py   # Agent: user psychology + world context
├── 05_train_financial.py      # Train on real FRED + Yahoo data
└── 06_curriculum_training.py  # Full 3-phase curriculum training
```

## Development

```bash
git clone https://github.com/JacobFV/general-unified-world-modeling.git
cd general-unified-world-modeling
pip install -e ".[dev]"
pytest
```

### Branch structure

- `develop` — active development, PRs target here
- `release` — stable releases, tagged commits trigger PyPI publish

### Running tests

```bash
# Full suite (144 tests)
pytest

# With coverage
pytest --cov=general_unified_world_model --cov-report=term-missing

# Specific module
pytest tests/test_schema.py -v
```

### Project layout

```
src/general_unified_world_model/
├── schema/           # 19 schema modules (physical → forecast)
│   ├── world.py      # Top-level World composition (857 fields)
│   ├── physical.py   # Planetary physical substrate
│   ├── resources.py  # Energy, metals, food, water, compute
│   ├── financial.py  # Global monetary & financial
│   ├── macro.py      # Macroeconomy (per country)
│   ├── political.py  # Political & institutional
│   ├── narrative.py  # Narrative, belief & expectations
│   ├── technology.py # Technology & innovation
│   ├── demographics.py
│   ├── sector.py     # Per GICS sector
│   ├── supply_chain.py
│   ├── business.py   # Per firm (sparse)
│   ├── individual.py # Key decision-makers (very sparse)
│   ├── events.py     # Real-time event tape
│   ├── trust.py      # Data channel trust (meta-epistemic)
│   ├── regime.py     # Privileged regime latent
│   ├── intervention.py
│   ├── forecast.py   # Structured output heads
│   ├── country.py    # Composite per country
│   └── observability.py  # Reusable epistemic bundles
├── projection/       # Subsetting & connectivity
│   ├── subset.py     # WorldProjection, project()
│   ├── temporal.py   # Temporal entity management
│   └── transfer.py   # Semantic transfer distance
├── training/         # Training infrastructure
│   ├── backbone.py   # Transformer backbone
│   ├── heterogeneous.py  # Masked canvas trainer
│   ├── diffusion.py  # Diffusion objective
│   └── curriculum.py # Multi-phase curriculum
├── data/             # Data adapters
│   └── adapters.py   # FRED, Yahoo, PMI, earnings, news, CSV
├── rendering/        # Visualization system
│   ├── base.py       # Renderer protocol, RenderContext, registry
│   ├── canvas.py     # Canvas heatmap (field allocation view)
│   ├── topology.py   # Domain topology graph
│   ├── financial.py  # Financial time series charts
│   ├── geopolitical.py  # Globe map + rotating GIF
│   ├── regime.py     # Regime state dashboard
│   └── social.py     # Social/entity network graph
├── llm/              # LLM-powered projection builder
│   └── projection_builder.py  # Natural language → WorldProjection
└── inference.py      # Observe/predict API
```

## License

Apache 2.0
