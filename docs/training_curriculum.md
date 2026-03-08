# Fork-Join DAG Training Curriculum for the General Unified World Model

**Status:** Research Design Document
**Last updated:** 2026-03-07
**Repository:** JacobFV/general-unified-world-modeling

---

## 1. Motivation

The General Unified World Model (GUWM) spans 857 fields across 19 semantic
domains, compiled onto structured latent canvases via `canvas-engineering`.
Training a single monolithic model on all domains simultaneously is
intractable for three reasons:

1. **Data heterogeneity.** Financial markets produce tick-level data (period 1),
   macro indicators arrive monthly (period 192), and demographics change over
   years (period 4608). A single training loop cannot sample efficiently
   across these timescales.

2. **Capacity interference.** Early in training, high-bandwidth financial
   signals dominate the shared latent space, starving slower domains of
   gradient. Fields like geopolitical stability and demographic trends get
   crowded out.

3. **Compute cost.** The full world canvas (T=1, H=128, W=128) has 16,384
   positions and 11,735 topology connections. Training a 12-layer, 3-loop
   transformer on this from scratch at full resolution is prohibitively
   expensive before the model has learned basic single-domain dynamics.

The fork-join DAG curriculum solves all three problems by decomposing training
into small parallel tasks that are later merged via weight averaging.

---

## 2. Fork-Join DAG: Core Concept

### 2.1 Overview

The curriculum is a directed acyclic graph (DAG) where each node is a
training run on a specific schema projection (a subset of the full schema).
There are two node types:

- **Fork nodes** duplicate the current model weights and dispatch copies to
  parallel training branches, each operating on a different domain-specific
  projection.
- **Join nodes** merge the trained branch copies back into a single model via
  weight averaging (model souping).

```
                         ┌──────────────────────────────────────────────┐
                         │            CURRICULUM DAG                    │
                         └──────────────────────────────────────────────┘

  TIER 0 (Foundation)          TIER 1 (Cross-domain)        TIER 2 (Complex)        TIER 3
  ════════════════════          ═════════════════════        ════════════════        ═══════

  ┌─────────────────┐
  │  basic_finance   │───┐
  └─────────────────┘   │
  ┌─────────────────┐   │     ┌───────────────────────┐
  │ basic_economics  │───┼────▶│ economics_drives_     │───┐
  └─────────────────┘   │     │ finance                │   │
  ┌─────────────────┐   │     └───────────────────────┘   │
  │ basic_politics   │───┤                                 │   ┌──────────────────┐
  └─────────────────┘   │     ┌───────────────────────┐   ├──▶│ corporate_       │
  ┌─────────────────┐   ├────▶│ geopolitics_drives_   │───┤   │ strategy_in_     │
  │ basic_resources  │───┤     │ commodities           │   │   │ context          │───┐
  └─────────────────┘   │     └───────────────────────┘   │   └──────────────────┘   │
  ┌─────────────────┐   │                                 │                           │
  │ basic_technology │───┤     ┌───────────────────────┐   │   ┌──────────────────┐   │  ┌───────────┐
  └─────────────────┘   ├────▶│ narratives_drive_     │───┼──▶│ policy_impact_   │───┼─▶│ FULL      │
  ┌─────────────────┐   │     │ markets               │   │   │ analysis         │   │  │ WORLD     │
  │ basic_narratives │───┘     └───────────────────────┘   │   └──────────────────┘   │  │ MODEL     │
  └─────────────────┘                                     │                           │  └───────────┘
                                                          │   ┌──────────────────┐   │
                                                          └──▶│ supply_chain_    │───┘
                                                              │ risk             │
                                                              └──────────────────┘

  ─── fork (copy weights) ───▶  join (merge weights)  ═══ tier boundary (sync point)
```

### 2.2 Why Fork-Join?

The naive alternative is sequential phased training (the current
`CurriculumTrainer` in `training/curriculum.py`): Phase 1 trains domains
independently, Phase 2 couples them, Phase 3 integrates. But this has two
deficiencies:

1. **Weight transfer is lossy.** Copying per-field encoder/decoder weights
   between projections of different sizes discards backbone representations.
   The backbone must relearn spatial relationships from scratch at each phase.

2. **No parallelism within a phase.** Phase 2 couplings run sequentially even
   though they are independent.

Fork-join fixes both. The backbone weights are shared across all branches of
a fork: every branch starts from the same checkpoint and trains on its own
projection. At the join, weight averaging preserves representations learned
by each branch, because semantic embeddings decouple field identity from
weight function (see Section 7).

---

## 3. Language-Conditioned Projections

### 3.1 From Text to Canvas

Each DAG node is described by a natural language string. This description is
passed to `llm_project()` (in `llm/projection_builder.py`), which calls an
LLM to select the appropriate include paths, entities, and exclusions:

```python
from general_unified_world_model.llm import llm_project

result = llm_project(
    "basic finance: yield curves, credit spreads, equity indices, FX rates"
)
bound = result.compile(T=1, H=32, W=32, d_model=64)
```

The LLM reads the full World schema (857 fields across 19 domains) and
returns a JSON with `include` paths, dynamic entities, and reasoning. This is
compiled to a `BoundSchema` with layout, topology, and attention masks.

### 3.2 Node Descriptions

Each training node in the DAG has a linguistic description that determines
its projection. This makes the curriculum human-readable and auditable:

**Tier 0 (Foundation):**

| Node | Description |
|------|-------------|
| `basic_finance` | "basic finance: yield curves, credit spreads, equity indices, VIX, FX rates, central bank policy rates, liquidity indicators" |
| `basic_economics` | "basic economics: GDP, inflation (CPI, PCE, PPI), unemployment, labor market, housing, PMI surveys, fiscal position" |
| `basic_politics` | "basic politics: government approval, legislative activity, election polling, geopolitical tension indices, sanctions" |
| `basic_resources` | "basic resources: crude oil, natural gas, metals (gold, silver, copper), agricultural commodities, energy grid stability" |
| `basic_technology` | "basic technology: semiconductor supply, AI compute capacity, digital infrastructure, R&D investment, patent activity" |
| `basic_narratives` | "basic narratives: news embeddings, social media sentiment, positioning and flows, public confidence, institutional trust" |

**Tier 1 (Cross-domain):**

| Node | Description |
|------|-------------|
| `economics_drives_finance` | "how economics affects financial markets: macro indicators driving rates, credit spreads, equity risk premium, and the yield curve" |
| `geopolitics_drives_commodities` | "how geopolitics and conflict affect commodity prices: sanctions, trade policy, energy security, strategic reserve dynamics" |
| `narratives_drive_markets` | "how narratives and sentiment drive financial markets: news impact on equity volatility, sentiment-driven credit cycles, positioning feedback loops" |

**Tier 2 (Complex):**

| Node | Description |
|------|-------------|
| `corporate_strategy_in_context` | "CEO decision making in context of macro and market conditions: revenue forecasting, capital allocation, M&A timing, competitive positioning" |
| `policy_impact_analysis` | "government policy impact on macro outcomes: fiscal stimulus multipliers, monetary transmission mechanism, regulatory burden on sectors" |
| `supply_chain_risk` | "supply chain disruption risk: semiconductor shortages, energy supply shocks, food security, critical mineral dependencies, substitution dynamics" |

**Tier 3 (Integration):**

| Node | Description |
|------|-------------|
| `full_world_model` | "full world model: all domains, all interactions, all timescales — the complete causal web from physical substrate through financial markets to individual decisions" |

### 3.3 Why Language Conditioning?

Language-conditioned projections provide three advantages:

1. **Reproducibility.** The projection is fully determined by the description
   string plus the LLM. Different researchers can inspect and modify the
   curriculum by editing plain text.

2. **Flexibility.** New training nodes can be added by writing a sentence. No
   code changes needed — the LLM maps the intent to schema paths.

3. **Auditability.** The LLM returns a `reasoning` field explaining why each
   domain was included. This creates a provenance trail for the curriculum.

---

## 4. Concrete DAG Structure

### 4.1 Tier 0: Foundation (Parallel)

All six foundation nodes run in parallel from a randomly initialized model.
Each operates on a small canvas (H=32, W=32) with a 4-layer, 3-loop
backbone.

```
             ┌────────────────┐   Canvas: 32x32   Fields: ~45
             │ basic_finance   │   Sources: FRED (rates), Yahoo (equities, FX)
             └────────────────┘   Steps: 10,000   LR: 1e-4

             ┌────────────────┐   Canvas: 32x32   Fields: ~60
             │ basic_economics │   Sources: FRED (macro), PMI surveys
             └────────────────┘   Steps: 10,000   LR: 1e-4

             ┌────────────────┐   Canvas: 32x32   Fields: ~35
             │ basic_politics  │   Sources: Institutional surveys, election data
             └────────────────┘   Steps: 10,000   LR: 1e-4

             ┌────────────────┐   Canvas: 32x32   Fields: ~30
             │ basic_resources │   Sources: Yahoo (commodities), EIA
             └────────────────┘   Steps: 10,000   LR: 1e-4

             ┌────────────────┐   Canvas: 32x32   Fields: ~25
             │ basic_technology│   Sources: SIA semiconductor data, patent DBs
             └────────────────┘   Steps: 10,000   LR: 1e-4

             ┌────────────────┐   Canvas: 32x32   Fields: ~40
             │ basic_narratives│   Sources: News embeddings, social sentiment
             └────────────────┘   Steps: 10,000   LR: 1e-4
```

After all six complete, a **join node** averages their backbone weights into
a single merged checkpoint. Encoder/decoder weights are concatenated (not
averaged) since each branch has different field sets.

### 4.2 Tier 1: Cross-Domain Coupling (Parallel)

Three cross-domain nodes run in parallel, each starting from the Tier 0
merged checkpoint. Each uses a larger canvas (H=48, W=48) and a 6-layer
backbone.

```
             ┌──────────────────────────┐   Canvas: 48x48   Fields: ~105
             │ economics_drives_finance  │   Includes: financial, country_us.macro,
             └──────────────────────────┘   regime, forecasts.macro, forecasts.financial
                                            Steps: 8,000   LR: 5e-5

             ┌──────────────────────────┐   Canvas: 48x48   Fields: ~80
             │ geopolitics_drives_       │   Includes: resources, country_us/cn/eu.macro,
             │ commodities               │   political, regime, supply chains
             └──────────────────────────┘   Steps: 8,000   LR: 5e-5

             ┌──────────────────────────┐   Canvas: 48x48   Fields: ~85
             │ narratives_drive_markets  │   Includes: narratives, financial, events,
             └──────────────────────────┘   regime, trust
                                            Steps: 8,000   LR: 5e-5
```

After all three complete, another **join** merges their backbone weights.
The regime latent is particularly important here: each branch trained it
from different causal angles, and the merge integrates these perspectives.

### 4.3 Tier 2: Complex Interactions (Parallel)

Three complex-interaction nodes, each starting from the Tier 1 merged
checkpoint. Canvas grows to H=64, W=64 with an 8-layer backbone.

```
             ┌──────────────────────────┐   Canvas: 64x64   Fields: ~180
             │ corporate_strategy_in_   │   Includes: financial, country_us.macro,
             │ context                   │   firm_alpha, firm_beta, sector_tech,
             └──────────────────────────┘   regime, forecasts, narratives
                                            Steps: 8,000   LR: 3e-5

             ┌──────────────────────────┐   Canvas: 64x64   Fields: ~150
             │ policy_impact_analysis    │   Includes: country_us/cn/eu (full),
             └──────────────────────────┘   financial, interventions, regime,
                                            forecasts, political
                                            Steps: 8,000   LR: 3e-5

             ┌──────────────────────────┐   Canvas: 64x64   Fields: ~140
             │ supply_chain_risk         │   Includes: sc_semiconductors, sc_energy,
             └──────────────────────────┘   sc_food, resources, technology, financial,
                                            regime, country_us/cn
                                            Steps: 8,000   LR: 3e-5
```

Join merges all three into the pre-integration checkpoint.

### 4.4 Tier 3: Full Integration

A single node trains the complete world model on the full canvas (H=128,
W=128) with a 12-layer, 3-loop backbone. This starts from the Tier 2
merged checkpoint.

```
             ┌──────────────────────────┐   Canvas: 128x128   Fields: 857
             │ full_world_model          │   Includes: ["*"] (all domains)
             └──────────────────────────┘   All data sources, all timescales
                                            Steps: 20,000   LR: 1e-5
                                            16,384 positions, 11,735 connections
```

### 4.5 DAG Summary Table

| Tier | Nodes | Canvas | Layers | Fields/node | Steps/node | Parallelism |
|------|-------|--------|--------|-------------|------------|-------------|
| 0    | 6     | 32x32  | 4      | 25-60       | 10,000     | 6-way       |
| 1    | 3     | 48x48  | 6      | 80-105      | 8,000      | 3-way       |
| 2    | 3     | 64x64  | 8      | 140-180     | 8,000      | 3-way       |
| 3    | 1     | 128x128| 12     | 857         | 20,000     | 1-way       |

---

## 5. Weight Merging Strategy

### 5.1 Simple Averaging (Baseline)

At each join node, the backbone weights are merged by arithmetic mean:

```
theta_merged = (1/K) * sum(theta_k for k in branches)
```

where `theta_k` is the state dict of branch `k` and `K` is the number of
branches. This is applied element-wise to all backbone parameters
(attention projections, feedforward weights, layer norms).

**Encoder/decoder handling.** Per-field encoders and decoders are NOT
averaged — they are accumulated. If branch A trained fields {f1, f2} and
branch B trained fields {f3, f4}, the merged model has encoders for
{f1, f2, f3, f4}. For fields that appear in multiple branches (e.g.,
`regime.*` is always included), the encoder weights are averaged.

### 5.2 Advanced Merging: TIES and DARE

Simple averaging works when branches diverge modestly from the fork point.
For branches that train longer or on very different data distributions,
more sophisticated merging strategies may be needed:

**TIES Merging** (Trim, Elect Sign, Merge):
1. Compute task vectors: `delta_k = theta_k - theta_init` for each branch.
2. Trim: zero out the smallest magnitude deltas (bottom 80%).
3. Elect sign: for each parameter, use the majority sign across branches.
4. Merge: average only the deltas that agree in sign.

```
delta_k = theta_k - theta_fork
delta_k_trimmed = trim(delta_k, keep_ratio=0.2)
sign = majority_sign(delta_k_trimmed for k in branches)
theta_merged = theta_fork + mean(delta_k_trimmed * (sign_k == sign) for k)
```

**DARE** (Drop And REscale):
1. Randomly drop a fraction `p` of each task vector's elements.
2. Rescale remaining elements by `1/(1-p)`.
3. Average the rescaled deltas.

This acts as a form of dropout-regularized merging that reduces interference
between branches.

**Task Arithmetic:**
Weighted sum of task vectors with hyperparameter search over coefficients:

```
theta_merged = theta_fork + sum(lambda_k * delta_k for k in branches)
```

The coefficients `lambda_k` can be tuned on a held-out validation set that
spans all domains.

### 5.3 When Merging Works

Weight averaging succeeds when the branches learn **compatible
representations** in the shared latent space. Three properties enable this:

1. **Shared initialization.** All branches fork from the same checkpoint, so
   the representation space is aligned at the start.

2. **Semantic embeddings.** Field identity is injected via frozen text
   embeddings (see Section 7), not via weight specialization. The backbone
   weights learn general dynamics, not field-specific patterns.

3. **Short branch length.** Each branch trains for 8,000-10,000 steps with
   low learning rate (3e-5 to 1e-4), preventing excessive drift from the
   fork point. The branches stay in the same loss basin.

### 5.4 Merging Algorithm (Pseudocode)

```python
def fork_join_merge(
    fork_checkpoint: dict,
    branch_checkpoints: list[dict],
    strategy: str = "simple",
    trim_ratio: float = 0.2,
) -> dict:
    """Merge multiple branch checkpoints at a join node."""

    if strategy == "simple":
        merged = {}
        for key in fork_checkpoint:
            tensors = [ckpt[key] for ckpt in branch_checkpoints]
            merged[key] = torch.stack(tensors).mean(dim=0)
        return merged

    elif strategy == "ties":
        merged = {}
        for key in fork_checkpoint:
            deltas = [ckpt[key] - fork_checkpoint[key]
                      for ckpt in branch_checkpoints]
            # Trim small magnitudes
            for i, d in enumerate(deltas):
                threshold = d.abs().quantile(1 - trim_ratio)
                deltas[i] = d * (d.abs() >= threshold)
            # Elect sign
            sign_votes = torch.stack([d.sign() for d in deltas])
            elected = sign_votes.sum(dim=0).sign()
            # Merge agreeing deltas
            for i, d in enumerate(deltas):
                deltas[i] = d * (d.sign() == elected)
            merged[key] = fork_checkpoint[key] + torch.stack(deltas).mean(dim=0)
        return merged

    elif strategy == "task_arithmetic":
        # Requires tuned lambda coefficients
        ...
```

---

## 6. Checkpointing

### 6.1 Checkpoint Policy

Every node in the DAG produces a checkpoint on completion. Additionally,
intermediate checkpoints are saved every `save_every` steps within a node's
training run. All checkpoints include full metadata for reproducibility.

### 6.2 Checkpoint Metadata Schema

```json
{
    "node_id": "tier1/economics_drives_finance",
    "tier": 1,
    "node_index": 0,
    "description": "how economics affects financial markets: macro indicators driving rates, credit spreads, equity risk premium, and the yield curve",

    "dag_parents": ["tier0/basic_finance", "tier0/basic_economics"],
    "fork_checkpoint": "checkpoints/tier0_merged.pt",
    "merge_strategy": "ties",
    "merge_trim_ratio": 0.2,

    "projection": {
        "include": ["financial", "country_us.macro", "regime", "forecasts.macro", "forecasts.financial"],
        "exclude": [],
        "firms": [],
        "individuals": ["fed_chair"],
        "countries": [],
        "sectors": []
    },

    "canvas": {
        "T": 1,
        "H": 48,
        "W": 48,
        "d_model": 64,
        "num_positions": 2304,
        "num_connections": 3847,
        "num_fields": 105
    },

    "architecture": {
        "n_layers": 6,
        "n_heads": 4,
        "d_ff": 256,
        "n_loops": 3,
        "dropout": 0.1,
        "total_params": 2847616
    },

    "training": {
        "steps_completed": 8000,
        "learning_rate": 5e-5,
        "batch_size": 32,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "final_loss": 0.0234,
        "final_coverage": 0.73,
        "wall_time_seconds": 14400,
        "gpu_type": "A100-80GB",
        "n_gpus": 1
    },

    "data_sources": [
        {"name": "FRED", "n_series": 42, "n_rows": 6000, "weight": 1.0},
        {"name": "Yahoo Finance", "n_tickers": 15, "n_rows": 6000, "weight": 1.0}
    ],

    "files": {
        "backbone": "tier1_economics_drives_finance_backbone.pt",
        "encoder": "tier1_economics_drives_finance_encoder.pt",
        "decoder": "tier1_economics_drives_finance_decoder.pt"
    },

    "created_at": "2026-03-07T14:30:00Z",
    "software_version": "0.3.0",
    "canvas_engineering_version": "0.1.1"
}
```

### 6.3 Checkpoint Directory Layout

```
checkpoints/
├── tier0/
│   ├── basic_finance/
│   │   ├── step_5000.pt
│   │   ├── step_10000.pt
│   │   └── metadata.json
│   ├── basic_economics/
│   │   └── ...
│   ├── basic_politics/
│   ├── basic_resources/
│   ├── basic_technology/
│   ├── basic_narratives/
│   └── merged.pt                    ← join output
├── tier1/
│   ├── economics_drives_finance/
│   ├── geopolitics_drives_commodities/
│   ├── narratives_drive_markets/
│   └── merged.pt
├── tier2/
│   ├── corporate_strategy_in_context/
│   ├── policy_impact_analysis/
│   ├── supply_chain_risk/
│   └── merged.pt
└── tier3/
    ├── full_world_model/
    │   ├── step_5000.pt
    │   ├── step_10000.pt
    │   ├── step_15000.pt
    │   ├── step_20000.pt
    │   └── metadata.json
    └── final.pt                     ← production checkpoint
```

---

## 7. Semantic Embedding Conditioning

### 7.1 How It Works

Each canvas position holds a latent vector in R^d_model. But the same
backbone weights must learn dynamics for GDP growth, equity volatility,
and geopolitical tension indices — quantities with fundamentally different
statistical properties.

The solution: **semantic embedding conditioning.** Each field has a frozen
text embedding (from a pretrained language model) that is projected into
the canvas latent space:

```
e_field = Linear_proj(frozen_text_embed(field_name)) + learnable_residual
```

This embedding is added to the canvas position at initialization, giving the
backbone a semantic signal about what each position represents. The
`FieldEncoder` (in `training/heterogeneous.py`) performs this projection:
raw scalar values are encoded through field-specific linear layers that also
incorporate the semantic identity.

### 7.2 Why This Enables Fork-Join

Without semantic conditioning, the backbone would need to learn different
weight patterns for different field types. Financial dynamics would be
encoded in certain attention heads, macro dynamics in others. Merging
two such specialized models would create interference — the heads would
fight over what to compute.

With semantic conditioning, the backbone learns **conditional dynamics**:
"given that this position represents a yield curve point (as signaled by
its embedding), apply yield-curve-appropriate temporal smoothing." The same
attention weights handle GDP when the position's embedding says "GDP."

This means:
1. **Branches learn compatible representations.** A branch trained on
   finance and a branch trained on macro use the same backbone weights to
   implement different dynamics, selected by the semantic embeddings.

2. **Weight averaging preserves both.** The merged model can handle any
   field, because the field identity is in the embedding (data), not in the
   weights (parameters).

3. **Zero-shot generalization.** A field not seen during any branch's
   training can still get reasonable predictions if its semantic embedding
   is close to a trained field's embedding.

### 7.3 Embedding Source

The frozen embeddings come from a pretrained text encoder. The field name
is expanded to a descriptive string:

```
"financial.yield_curves.ten_year"
  → "10-year US Treasury yield, a long-term risk-free interest rate
     benchmark updated daily"
```

This expansion is cached at schema construction time. The text encoder
produces a 768-dim embedding that is linearly projected to d_model (64)
with a learned residual added during training.

---

## 8. Data Sources

### 8.1 Tier 0 Data Mapping

Each foundation node maps to specific real-world data sources:

**basic_finance:**

| Source | API | Fields | Frequency |
|--------|-----|--------|-----------|
| FRED: DFF, DGS2, DGS5, DGS10, DGS30, T10Y2Y, T10YIE | fredapi | yield_curves.* | Daily |
| FRED: BAMLC0A4CBBB, BAMLH0A0HYM2 | fredapi | credit.ig_spread, credit.hy_spread | Daily |
| FRED: DFEDTARU, WALCL | fredapi | central_banks.* | Monthly/Weekly |
| FRED: RRPONTSYD, WTREGEN, TOTRESNS | fredapi | liquidity.* | Daily/Weekly |
| Yahoo: ^GSPC, ^DJI, ^IXIC, ^RUT, ^VIX | yfinance | equities.* | Daily |
| Yahoo: DX-Y.NYB, EURUSD=X, JPY=X, CNY=X | yfinance | fx.* | Daily |
| Yahoo: BTC-USD, ETH-USD | yfinance | crypto.* | Daily |

**basic_economics:**

| Source | API | Fields | Frequency |
|--------|-----|--------|-----------|
| FRED: GDP, GDPC1, INDPRO, TCU, RSAFS | fredapi | country_us.macro.output.* | Monthly/Quarterly |
| FRED: CPIAUCSL, CPILFESL, PCEPI, PPIFIS | fredapi | country_us.macro.inflation.* | Monthly |
| FRED: UNRATE, PAYEMS, ICSA, CCSA, JTSJOL | fredapi | country_us.macro.labor.* | Weekly/Monthly |
| FRED: CSUSHPINSA, HOUST, MORTGAGE30US | fredapi | country_us.macro.housing.* | Monthly/Weekly |
| FRED: GFDEGDQ188S, FYFSGDA188S | fredapi | country_us.macro.fiscal.* | Quarterly |
| ISM PMI Manufacturing/Services | Custom | country_us.macro.output.pmi_* | Monthly |

**basic_politics:**

| Source | Fields | Frequency |
|--------|--------|-----------|
| Gallup / Pew presidential approval | political.approval_rating | Monthly |
| GovTrack legislative activity | political.legislative_activity | Weekly |
| PredictIt / Polymarket election data | political.election_polling | Daily |
| Global Peace Index components | political.geopolitical_tension | Annual (interpolated) |
| OFAC sanctions data | political.sanctions_regime | Event-driven |

**basic_resources:**

| Source | API | Fields | Frequency |
|--------|-----|--------|-----------|
| Yahoo: CL=F, NG=F | yfinance | resources.energy.* | Daily |
| Yahoo: GC=F, SI=F, HG=F | yfinance | resources.metals.* | Daily |
| EIA petroleum status | Custom | resources.energy.inventories | Weekly |
| USDA crop reports | Custom | resources.agriculture.* | Monthly |
| LME base metals | Custom | resources.metals.industrial_* | Daily |

**basic_technology:**

| Source | Fields | Frequency |
|--------|--------|-----------|
| SIA semiconductor sales | technology.semiconductor_supply | Monthly |
| TOP500 compute benchmarks | technology.ai_compute_capacity | Semi-annual |
| USPTO patent filings | technology.patent_activity | Monthly |
| OECD R&D spending | technology.rd_investment | Annual (interpolated) |

**basic_narratives:**

| Source | Fields | Frequency |
|--------|--------|-----------|
| Pre-computed news article embeddings | events.news_embedding | Hourly |
| Twitter/X financial sentiment | narratives.public.social_sentiment | Hourly |
| AAII investor sentiment survey | narratives.positioning.retail_sentiment | Weekly |
| CFTC Commitments of Traders | narratives.positioning.cot_* | Weekly |
| University of Michigan Consumer Sentiment | country_us.domestic_sentiment | Monthly |

### 8.2 Data Pipeline

Raw data flows through the adapter pipeline:

```
Raw API → Adapter (fred_adapter, yahoo_finance_adapter, etc.)
       → FieldMapping (source_key → target_field + transform)
       → DatasetSpec (name, mappings, base_period, weight)
       → HeterogeneousDataset (place on canvas, compute presence_mask)
       → MaskedCanvasTrainer (masked loss, gradient only where data exists)
```

The `HeterogeneousDataset` handles the critical property: each dataset
only populates a subset of canvas positions, and the loss is masked to only
backpropagate through fields with ground truth. The shared regime latent
receives gradient from ALL data sources.

---

## 9. Compute Requirements

### 9.1 Per-Node Estimates

Estimates assume A100-80GB GPUs with mixed-precision training:

| Node | Positions | Params | Steps | Batch | A100-hrs | Notes |
|------|-----------|--------|-------|-------|----------|-------|
| **Tier 0** (each) | 1,024 | ~800K | 10,000 | 64 | 2-4 | Small canvas, fast |
| **Tier 0** (all 6) | — | — | — | — | 4 (parallel) | 6 GPUs, wall-clock |
| Tier 0 merge | — | — | — | — | <0.1 | CPU, seconds |
| **Tier 1** (each) | 2,304 | ~2.8M | 8,000 | 32 | 6-10 | |
| **Tier 1** (all 3) | — | — | — | — | 10 (parallel) | 3 GPUs |
| Tier 1 merge | — | — | — | — | <0.1 | |
| **Tier 2** (each) | 4,096 | ~8M | 8,000 | 32 | 12-18 | Quadratic attention |
| **Tier 2** (all 3) | — | — | — | — | 18 (parallel) | 3 GPUs |
| Tier 2 merge | — | — | — | — | <0.1 | |
| **Tier 3** | 16,384 | ~45M | 20,000 | 16 | 80-120 | Full canvas |

### 9.2 Total Compute Budget

| Scenario | GPUs | Wall-clock hrs | Total A100-hrs |
|----------|------|---------------|----------------|
| Fully sequential | 1 | ~250 | ~250 |
| Maximum parallelism | 6 | ~130 | ~250 |
| Practical (reuse GPUs) | 3 | ~150 | ~250 |

The DAG structure reduces wall-clock time by nearly 2x compared to sequential
training while consuming the same total compute. The real win is that each
node trains on a smaller, more focused dataset, which converges faster than
training the full model from scratch.

### 9.3 Memory Requirements

| Tier | Canvas positions | Attention matrix size | Peak memory (fp16) |
|------|------------------|-----------------------|--------------------|
| 0    | 1,024            | 1M elements           | ~4 GB              |
| 1    | 2,304            | 5.3M elements         | ~10 GB             |
| 2    | 4,096            | 16.8M elements        | ~24 GB             |
| 3    | 16,384           | 268M elements         | ~72 GB             |

Tier 3 requires an A100-80GB or equivalent. Tiers 0-2 can run on smaller
GPUs (A10, L4, or even consumer GPUs for Tier 0).

---

## 10. Training Dynamics

### 10.1 Loss Masking

The `MaskedCanvasTrainer` computes loss only on positions where ground truth
data exists. This is essential for heterogeneous training:

```
loss = sum(per_pos_loss * presence_mask * loss_weight_mask) / n_active
```

where `presence_mask` comes from the dataset (1 where data exists) and
`loss_weight_mask` comes from the schema (higher weight for output/forecast
positions).

### 10.2 Multi-Frequency Diffusion

Fields update at different temporal frequencies (8 classes, tau_0 through
tau_7). The `MultiFrequencyNoise` module scales diffusion noise inversely
with log-period:

```
noise_scale = base_noise / (1 + log(period))
```

This prevents the diffusion process from wasting capacity on slow-changing
fields (like demographics) that should hold nearly constant.

### 10.3 Learning Rate Schedule

Each tier uses a cosine decay schedule with linear warmup:

```
Tier 0: warmup 200 steps → peak 1e-4 → cosine decay to 1e-6
Tier 1: warmup 100 steps → peak 5e-5 → cosine decay to 5e-7
Tier 2: warmup 100 steps → peak 3e-5 → cosine decay to 3e-7
Tier 3: warmup 500 steps → peak 1e-5 → cosine decay to 1e-7
```

The decreasing peak LR across tiers prevents catastrophic forgetting of
earlier representations. Each tier makes smaller adjustments to an
increasingly capable base model.

---

## 11. Extending the DAG

### 11.1 Adding a New Node

To add a new training node to the DAG:

1. **Write a description.** E.g., "high-frequency trading: order flow,
   market microstructure, tick-level dynamics, LOB state, execution costs."

2. **Choose parents.** Which existing nodes should this depend on?
   Probably `basic_finance` at minimum.

3. **Choose tier.** Based on complexity and number of interacting domains.

4. **Register in the DAG config:**

```python
DAG_NODES = {
    ...
    "hft_microstructure": {
        "tier": 1,
        "parents": ["basic_finance"],
        "description": "high-frequency trading: order flow, ...",
        "canvas": {"H": 48, "W": 48},
        "architecture": {"n_layers": 6, "n_loops": 3},
        "training": {"steps": 8000, "lr": 5e-5},
    },
}
```

### 11.2 Country-Specific Branches

The DAG naturally supports geographic specialization:

```
basic_economics ──┬── us_macro_deep (FRED full series)
                  ├── cn_macro_deep (NBS, PBoC data)
                  ├── eu_macro_deep (Eurostat, ECB data)
                  └── em_macro_survey (World Bank, IMF)
```

Each country branch trains on country-specific data, then merges with the
others to produce a model that understands global macro interactions.

### 11.3 Firm-Specific Fine-Tuning

After the full DAG completes, task-specific fine-tuning adds firm
projections:

```python
result = llm_project(
    "Apple's business: revenue by segment, iPhone/Mac/Services mix, "
    "supply chain through TSMC, competitive position vs Samsung/Google, "
    "in context of macro, rates, and consumer sentiment"
)
bound = result.compile(T=1, H=64, W=64, d_model=64)
```

This creates a projection with `firm_AAPL`, relevant sectors, and macro
context. The fine-tuning phase freezes the backbone and trains only the
firm-specific encoder/decoder heads.

---

## 12. Open Questions

### 12.1 Merge Topology

**Q1: Is a flat merge optimal, or should we use hierarchical merging?**

The current design merges all branches at each tier boundary. An alternative
would merge subsets first (e.g., merge `basic_finance` + `basic_economics`
before merging with `basic_resources`), creating a binary tree of merges.
This might preserve more structure but adds complexity.

**Q2: How many branches can be merged before quality degrades?**

Model souping literature suggests diminishing returns beyond 5-8 branches.
Our Tier 0 has 6 branches — is this near the limit? Ablation needed.

### 12.2 Branch Length

**Q3: What is the optimal number of steps per branch?**

Too few steps: branches don't learn enough to contribute. Too many: branches
drift apart and merging fails. The sweet spot likely depends on domain
complexity and dataset size. Adaptive stopping criteria (e.g., stop when
validation loss plateaus) may outperform fixed step counts.

**Q4: Should branch length vary by domain?**

Financial data has orders of magnitude more samples than political or
technology data. Should finance branches train longer? Or would this cause
them to dominate the merged representation?

### 12.3 Merge Strategy Selection

**Q5: Can we learn the optimal merge coefficients?**

Instead of uniform averaging or hand-tuned task arithmetic coefficients,
could we use a validation set spanning all domains to learn
`lambda_k` per branch? This would be a form of meta-learning over the
curriculum.

**Q6: Does the optimal merge strategy change across tiers?**

Tier 0 branches are very different (finance vs. politics) — TIES might be
better. Tier 2 branches share more structure — simple averaging might
suffice.

### 12.4 Semantic Embeddings

**Q7: Which text encoder produces the best field embeddings?**

Options include sentence-transformers, OpenAI embeddings, or a fine-tuned
encoder trained specifically on financial/economic ontology text. The
embedding quality directly affects how well the fork-join merge works.

**Q8: Should the semantic embeddings be unfrozen in Tier 3?**

During full integration, allowing the text embeddings to fine-tune might
improve field discrimination. But this risks breaking the alignment that
made the merge work in earlier tiers.

### 12.5 Scaling

**Q9: How does the curriculum scale to 20+ countries and 50+ firms?**

The current design has 3 countries and 2 firms. A production model might
need 20 countries and 50 firms. This changes the canvas from 857 fields to
5,000+. Does the DAG structure still help, or does the Tier 3 full
integration become infeasible?

**Q10: Can we avoid Tier 3 entirely?**

If the merged Tier 2 checkpoint already performs well across all domains,
perhaps the expensive full-canvas training is unnecessary. The model could
run inference on different projections without ever training on the full
16,384-position canvas.

### 12.6 Evaluation

**Q11: How do we evaluate a world model?**

Standard ML metrics (MSE, log-likelihood) apply to individual fields, but
the value of a world model is in capturing cross-domain dependencies. We
need evaluation tasks that test:
- Conditional prediction: given A, does the model predict B correctly?
- Counterfactual reasoning: if we intervene on A, does B shift plausibly?
- Temporal coherence: are multi-step rollouts stable and calibrated?
- Cross-frequency consistency: do slow and fast fields agree?

---

## 13. Relationship to Existing Code

The fork-join DAG curriculum described here extends the existing
`CurriculumTrainer` in `training/curriculum.py`, which implements a simpler
sequential 3-phase curriculum (independent, coupling, integration). The
key additions are:

1. **DAG structure** instead of linear phases.
2. **Weight merging** instead of per-field weight transfer.
3. **Language-conditioned projections** via `llm_project()` instead of
   hand-coded `DomainSpec` include paths.
4. **Semantic embedding conditioning** to enable merge compatibility.
5. **Checkpointing with full metadata** for reproducibility.

The existing `project()`, `WorldModelBackbone`,
`FieldEncoder`, `FieldDecoder`, `MaskedCanvasTrainer`, and
`HeterogeneousDataset` are all reused without modification. The DAG
curriculum is an orchestration layer on top of these components.

---

## References

- Wortsman et al., "Model soups: averaging weights of multiple fine-tuned
  models improves accuracy without increasing inference time" (ICML 2022)
- Yadav et al., "TIES-Merging: Resolving Interference When Merging Models"
  (NeurIPS 2023)
- Yu et al., "Language Models are Super Mario: Absorbing Abilities from
  Homologous Models as a Free Lunch" (DARE, 2024)
- Ilharco et al., "Editing Models with Task Arithmetic" (ICLR 2023)
- Bengio et al., "Curriculum Learning" (ICML 2009)
