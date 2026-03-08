# general-unified-world-model

### A typed causal ontology of civilization, built on [canvas-engineering](https://github.com/JacobFV/canvas-engineering) structured latent spaces.

[![Tests](https://img.shields.io/badge/tests-39%2F39-brightgreen.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> Canvas engineering structures what a diffusion model *thinks in*. This repo declares a **857-field typed schema** spanning planetary physics through individual psychology, compiles it onto a structured latent canvas, and trains it on heterogeneous real-world data — without throwing out samples that are missing fields.

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
from guwm import World

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
from guwm import WorldProjection, project

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

### Train on heterogeneous data

```python
from guwm import (
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
from guwm.data.adapters import fred_adapter, yahoo_finance_adapter

# FRED: 50+ macro series mapped to world model fields
fred_spec, fred_data = fred_adapter(api_key="...", start_date="2010-01-01")

# Yahoo Finance: equities, FX, commodities, crypto
yahoo_spec, yahoo_data = yahoo_finance_adapter(
    include_equity=True, include_fx=True,
    firm_tickers={"AAPL": "firm_AAPL"},
)

# Generic CSV/Parquet
from guwm.data.adapters import tabular_adapter
spec, data = tabular_adapter(
    "My Dataset", "data.csv",
    column_mappings={"gdp_growth": "country_us.macro.output.gdp_nowcast"},
    transforms={"gdp_growth": "z_score"},
)
```

## Temporal entities

Entities can appear and disappear over time:

```python
from guwm import TemporalTopology
from guwm.schema.business import Business

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
from guwm import WorldModel

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

## License

Apache 2.0
