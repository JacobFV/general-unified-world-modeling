"""Build a world model from a plain-English description and datasets.

Demonstrates the full workflow:

1. **llm_build()** — describe what you want + pass datasets → trained model
2. **Dynamic fidelity** — high detail where you care, coarse elsewhere
3. **Private fine-tuning** — calibrate a general model to private data

Usage:
    # Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in .env / environment
    python examples/07_llm_build.py
"""

import os
import torch

from general_unified_world_model import (
    llm_build,
    llm_project,
    GeneralUnifiedWorldModel,
    DataSource,
    DatasetSpec,
    InputSpec,
    OutputSpec,
)


# ── Helper: build synthetic datasets ─────────────────────────────────────

def make_macro_source(n: int = 500) -> DataSource:
    """Synthetic US macro dataset."""
    return DataSource(
        spec=DatasetSpec(
            name="us_macro",
            description="Quarterly US macroeconomic indicators",
            input_specs=[
                InputSpec(key="gdp", semantic_type="US real GDP growth",
                          field_path="country_us.macro.output.gdp_nowcast"),
                InputSpec(key="cpi", semantic_type="US CPI inflation",
                          field_path="country_us.macro.inflation.cpi_headline"),
                InputSpec(key="ue", semantic_type="US unemployment rate",
                          field_path="country_us.macro.labor.unemployment_rate"),
            ],
            output_specs=[
                OutputSpec(key="gdp", semantic_type="US real GDP growth",
                           field_path="country_us.macro.output.gdp_nowcast"),
            ],
        ),
        data={
            "gdp": torch.randn(n) * 0.5 + 2.0,
            "cpi": torch.randn(n) * 0.3 + 3.0,
            "ue":  torch.randn(n) * 0.2 + 4.0,
        },
    )


def make_equity_source(n: int = 500) -> DataSource:
    """Synthetic equity market dataset."""
    return DataSource(
        spec=DatasetSpec(
            name="equity_prices",
            description="Daily equity index levels and VIX",
            input_specs=[
                InputSpec(key="spx", semantic_type="S&P 500 index level",
                          field_path="financial.equities.large_cap"),
                InputSpec(key="vix", semantic_type="VIX implied volatility",
                          field_path="financial.equities.vix"),
            ],
            output_specs=[
                OutputSpec(key="spx", semantic_type="S&P 500 index level",
                           field_path="financial.equities.large_cap"),
            ],
        ),
        data={
            "spx": torch.cumsum(torch.randn(n) * 0.01, dim=0) + 4500.0,
            "vix": torch.randn(n).abs() * 5 + 15.0,
        },
    )


def make_private_source(n: int = 200) -> DataSource:
    """Synthetic private dataset (e.g. proprietary internal data)."""
    return DataSource(
        spec=DatasetSpec(
            name="private_portfolio",
            description="Internal portfolio P&L and risk metrics",
            input_specs=[
                InputSpec(key="pnl", semantic_type="daily portfolio P&L",
                          field_path="financial.equities.large_cap"),
                InputSpec(key="rate_exp", semantic_type="rate exposure DV01",
                          field_path="financial.yield_curves.ten_year"),
            ],
            output_specs=[
                OutputSpec(key="pnl", semantic_type="daily portfolio P&L",
                           field_path="financial.equities.large_cap"),
            ],
        ),
        data={
            "pnl":      torch.randn(n) * 100_000,
            "rate_exp":  torch.randn(n) * 50 + 200,
        },
    )


# ── Demo 1: llm_build — one call does everything ────────────────────────

def demo_llm_build():
    """Show the single-call workflow: describe + datasets → trained model."""
    print("\n" + "=" * 70)
    print("DEMO 1 — llm_build(description, datasets)")
    print("=" * 70)

    macro   = make_macro_source()
    equities = make_equity_source()

    model = llm_build(
        "I'm a macro strategist at a hedge fund.  I need to model US GDP, "
        "inflation, unemployment, and the equity market together so I can "
        "predict how macro surprises move the S&P 500.",
        datasets=[macro, equities],
        n_steps=200,           # short for demo
        d_model=32,
        batch_size=8,
        log_every=50,
    )

    # Observe some values and predict
    model.observe("country_us.macro.output.gdp_nowcast", 2.5)
    model.observe("country_us.macro.inflation.cpi_headline", 3.2)
    preds = model.predict(n_steps=5)

    print(f"\nPrediction keys ({len(preds)}): "
          f"{list(preds.keys())[:8]} ...")
    return model


# ── Demo 2: dynamic fidelity ────────────────────────────────────────────

def demo_dynamic_fidelity():
    """Show how the same schema gives different fidelity levels.

    High fidelity  = include all sub-fields → more canvas positions
    Low fidelity   = include only the top-level → coarse-grained summary
    """
    print("\n" + "=" * 70)
    print("DEMO 2 — Dynamic fidelity for factors of interest")
    print("=" * 70)

    # High fidelity for financial, low fidelity for macro
    model_high = GeneralUnifiedWorldModel(
        include=[
            "financial",                          # ALL sub-fields (high)
            "country_us.macro.output",            # only GDP sub-tree (medium)
            "regime",                             # compressed (low)
        ],
        d_model=32,
    )
    print(f"\nHigh-detail financial model:")
    print(f"  Fields:    {len(model_high.bound.field_names)}")
    print(f"  Positions: {model_high.bound.layout.num_positions}")

    # Opposite: high fidelity macro, low fidelity financial
    model_macro = GeneralUnifiedWorldModel(
        include=[
            "financial.yield_curves",             # only rates (low financial)
            "country_us",                         # EVERYTHING for US (high)
            "regime",
            "forecasts",
        ],
        d_model=32,
    )
    print(f"\nHigh-detail macro model:")
    print(f"  Fields:    {len(model_macro.bound.field_names)}")
    print(f"  Positions: {model_macro.bound.layout.num_positions}")

    # Demonstrate: the financial model has far more financial fields
    fin_fields_high  = [f for f in model_high.bound.field_names
                        if f.startswith("financial")]
    fin_fields_macro = [f for f in model_macro.bound.field_names
                        if f.startswith("financial")]
    print(f"\nFinancial fields in high-detail model:  {len(fin_fields_high)}")
    print(f"Financial fields in macro-detail model: {len(fin_fields_macro)}")
    print("→ Same schema, different fidelity per domain.\n")


# ── Demo 3: fine-tune on private data ────────────────────────────────────

def demo_private_finetune():
    """Fine-tune a general model on a private dataset.

    The cross-domain associations learned during general training let
    the model extrapolate private patterns to the wider world.
    """
    print("\n" + "=" * 70)
    print("DEMO 3 — Fine-tune general model on private data")
    print("=" * 70)

    # 1. Build a general model
    print("\n1. Building general model ...")
    general = GeneralUnifiedWorldModel(
        include=["financial", "country_us.macro", "regime"],
        d_model=32,
    )
    print(f"   {len(general.bound.field_names)} fields, "
          f"{general.bound.layout.num_positions} positions")

    # 2. Pre-train on public data
    public_macro   = make_macro_source(n=300)
    public_equity  = make_equity_source(n=300)

    print("\n2. Pre-training on public data ...")
    general.finetune(
        datasets=[public_macro, public_equity],
        n_steps=100,
        lr=1e-4,
        batch_size=8,
        log_every=50,
    )

    # 3. Fine-tune on private data (lower LR, optionally freeze backbone)
    private = make_private_source(n=200)

    print("\n3. Fine-tuning on private data (frozen backbone) ...")
    metrics = general.finetune(
        datasets=[private],
        n_steps=100,
        lr=1e-5,
        freeze_backbone=True,    # don't forget public knowledge
        batch_size=8,
        log_every=50,
    )

    print(f"\n   Final loss: {metrics['final_loss']:.4f}")
    print("   Model now calibrated to private data while retaining "
          "general world dynamics.\n")

    # 4. Predict using the calibrated model
    general.observe("financial.equities.large_cap", 4600.0)
    preds = general.predict(n_steps=5)
    print(f"   Predictions: {list(preds.keys())[:5]} ...")
    return general


# ── Demo 4: LLM projection only (no training) ───────────────────────────

def demo_projection_only():
    """Show that llm_project() alone returns the schema design."""
    print("\n" + "=" * 70)
    print("DEMO 4 — LLM projection only (schema design)")
    print("=" * 70)

    result = llm_project(
        "I run a renewable energy fund.  I need commodity prices (oil, gas, "
        "solar panel inputs), US and EU energy policy, carbon credits, and "
        "weather patterns that affect solar/wind output.",
    )

    print(f"\nInclude paths: {result.include}")
    print(f"Entities:      {result.entities or '(none)'}")
    print(f"Reasoning:     {result.reasoning}")

    # Compile and inspect
    bound = result.compile(T=1, d_model=32)
    print(f"\nCompiled schema: {len(bound.field_names)} fields, "
          f"{bound.layout.num_positions} positions")

    # Create a model from the result
    model = result.to_model(d_model=32)
    print(f"Model ready:     {len(model.bound.field_names)} fields")
    return model


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    # Demos that don't need an API key
    demo_dynamic_fidelity()
    demo_private_finetune()

    # Demos that need an API key
    has_key = bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not has_key:
        # Try loading .env
        from general_unified_world_model.llm.projection_builder import _load_dotenv
        _load_dotenv()
        has_key = bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

    if has_key:
        demo_llm_build()
        demo_projection_only()
    else:
        print("\n[SKIP] Set ANTHROPIC_API_KEY or OPENAI_API_KEY for "
              "LLM-powered demos (llm_build, llm_project).")

    print("\nAll demos complete.")


if __name__ == "__main__":
    main()
