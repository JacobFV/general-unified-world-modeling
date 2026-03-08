# Architecture

## Schema overview

The world model schema is a nested dataclass hierarchy with 19 layers and 857 fields. Each field is a `canvas_engineering.Field` with declared temporal frequency, loss weight, and semantic type.

```
World
  physical          -- climate, infrastructure, disasters (tau6-tau7)
  resources         -- crude, metals, food, water, compute (tau1-tau4)
  financial         -- yields, credit, FX, equities, crypto (tau0-tau2)
  country_{code}    -- per-country macro + politics + demographics
  political         -- executive, legislative, judicial, geopolitical (tau4-tau7)
  narratives        -- media, sentiment, positioning (tau0-tau4)
  technology        -- AI, biotech, quantum, robotics (tau5-tau7)
  demographics      -- population, dependency, urbanization (tau7)
  sector_{name}     -- per GICS sector demand/supply/margins (tau3-tau5)
  supply_chain_{id} -- concentration, lead time, bottleneck (tau2-tau4)
  firm_{ticker}     -- financials, operations, strategy (tau2-tau5)
  individual_{id}   -- cognitive, incentives, network (tau2-tau5)
  events            -- news, social, filings, policy (tau0-tau1)
  trust             -- data channel meta-epistemic calibration (tau3-tau7)
  regime            -- compressed global latent (tau5-tau7)
  interventions     -- monetary, fiscal, regulatory + effects (tau2-tau5)
  forecasts         -- recession prob, credit stress, conflict risk (output)
```

### Temporal frequency classes

Eight frequency classes span sub-minute to multi-year:

| Class | Period | Examples |
|-------|--------|----------|
| tau0 | 1 (sub-minute) | markets, breaking news |
| tau1 | 4 (hourly) | grid load, commodities |
| tau2 | 16 (daily) | commodity prices, port congestion |
| tau3 | 48 (weekly) | claims, inventories, payroll |
| tau4 | 192 (monthly) | CPI, PMI, company closes |
| tau5 | 576 (quarterly) | earnings, GDP, capex |
| tau6 | 2304 (annual) | demographics, infrastructure |
| tau7 | 4608 (multi-year) | regime changes, tech diffusion |

### Dynamic entities

Firms, individuals, countries, sectors, and supply chain nodes are created at runtime using `dataclasses.make_dataclass`. This allows projections to include arbitrary named entities (e.g., `firm_AAPL`, `individual_ceo`) without hardcoding them in the schema.

## Projection system

`WorldProjection` declares which parts of the schema to include:

```python
@dataclass
class WorldProjection:
    include: list[str]          # dotted paths into schema
    exclude: list[str] = ...    # paths to exclude
    firms: list[str] = ...      # dynamic firm entities
    individuals: list[str] = ...
    countries: list[str] = ...
    sectors: list[str] = ...
    supply_chains: list[str] = ...
```

`project()` takes a `WorldProjection`, creates a `ProjectedWorld` dataclass with only the requested fields, and calls `canvas_engineering.compile_schema()` to produce a `BoundSchema` with layout, topology, and attention masks.

### How projection works

1. Start with full `World()` instance (857 fields).
2. Walk the dataclass tree, keeping only fields matching `include` paths.
3. Create dynamic entity instances for requested firms/individuals/etc.
4. Build a `ProjectedWorld` dataclass containing only the active subtree.
5. Call `compile_schema(projected_world, T, H, W, d_model)` to allocate fields on the canvas grid.

The result is a `BoundSchema` with:

- `field_names` -- flat list of dotted field paths
- `layout` -- grid positions for each field
- `topology` -- attention connections between fields
- `attention_mask` -- precomputed mask for the transformer

## Canvas-engineering integration

This library depends on [canvas-engineering](https://github.com/JacobFV/canvas-engineering) for:

- **`Field`** -- the typed unit. Each field has `period`, `loss_weight`, `semantic_tag`.
- **`compile_schema`** -- packs all Fields onto `(T, H, W)` grid, computes layout and topology.
- **`ConnectivityPolicy`** -- controls intra-domain vs. cross-domain attention patterns.
- **`BoundSchema`** -- the compiled output containing layout, topology, and masks.

The schema defines *what* the world model knows. Canvas-engineering defines *how* it computes over that knowledge. The topology IS the compute graph -- fields that are connected attend to each other in the transformer backbone.

### Connectivity policies

```python
ConnectivityPolicy(
    intra="dense",          # fields within same domain attend to each other
    parent_child="hub_spoke",  # parent domains connect to child domains
)
```

Full world: T=1, H=128, W=128 produces 16,384 positions and 11,735 connections. Smaller projections use H=24-64 depending on field count.
