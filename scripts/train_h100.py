#!/usr/bin/env python3
"""Production training on H100 GPU using the full DAG curriculum.

Fetches real data from FRED + Yahoo Finance, then trains the standard
4-tier curriculum (foundation → cross-domain → complex → integration).

Usage:
    # Full curriculum (all tiers)
    python scripts/train_h100.py

    # Single tier
    python scripts/train_h100.py --tier 0

    # Resume from checkpoint
    python scripts/train_h100.py --resume checkpoints/dag
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from general_unified_world_model.training.dag_curriculum import (
    DAGCurriculumTrainer,
    STANDARD_DAG,
    TIER_0_FOUNDATION,
    TIER_1_CROSS_DOMAIN,
    TIER_2_COMPLEX,
    TIER_3_INTEGRATION,
    TrainingNode,
)
from general_unified_world_model.training.heterogeneous import (
    DatasetSpec, DataSource, InputSpec, OutputSpec,
)
from general_unified_world_model.data.adapters import (
    fred_adapter, yahoo_finance_adapter, _io_pair, log_return,
)


def fetch_real_data(fred_api_key: str | None = None) -> dict[str, DataSource]:
    """Fetch all real data sources."""
    sources = {}

    # FRED macro data
    if fred_api_key or os.environ.get("FRED_API_KEY"):
        print("Fetching FRED data...")
        try:
            fred_spec, fred_data = fred_adapter(
                api_key=fred_api_key,
                start_date="2000-01-01",
            )
            sources["fred_macro"] = DataSource(spec=fred_spec, data=fred_data)
            print(f"  FRED macro: {len(fred_spec.input_specs)} series, "
                  f"{sum(v.shape[0] for v in fred_data.values())} total rows")

            # Split rates from macro
            rate_keys = ["DFF", "DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y",
                         "T10YIE", "BAMLC0A4CBBB", "BAMLH0A0HYM2",
                         "DFEDTARU", "WALCL", "RRPONTSYD", "WTREGEN", "TOTRESNS"]
            rate_data = {k: v for k, v in fred_data.items() if k in rate_keys}
            rate_specs_in = [s for s in fred_spec.input_specs if s.key in rate_keys]
            rate_specs_out = [s for s in fred_spec.output_specs if s.key in rate_keys]
            if rate_data:
                sources["fred_rates"] = DataSource(
                    spec=DatasetSpec(
                        name="FRED Rates",
                        description="US Treasury yields, credit spreads, monetary policy",
                        input_specs=rate_specs_in,
                        output_specs=rate_specs_out,
                    ),
                    data=rate_data,
                )
                print(f"  FRED rates: {len(rate_specs_in)} series")
        except Exception as e:
            print(f"  FRED fetch failed: {e}")
    else:
        print("No FRED_API_KEY — using synthetic macro data")
        sources.update(_synthetic_macro())

    # Yahoo Finance market data
    print("Fetching Yahoo Finance data...")
    try:
        yahoo_spec, yahoo_data = yahoo_finance_adapter(
            start_date="2000-01-01",
            include_equity=True,
            include_fx=True,
            include_commodities=True,
            include_crypto=True,
        )
        sources["yahoo_finance"] = DataSource(spec=yahoo_spec, data=yahoo_data)
        print(f"  Yahoo Finance: {len(yahoo_spec.input_specs)} tickers")

        # Split commodities
        commodity_keys = ["yahoo_CL=F", "yahoo_NG=F", "yahoo_GC=F", "yahoo_SI=F", "yahoo_HG=F"]
        commodity_data = {k: v for k, v in yahoo_data.items() if k in commodity_keys}
        commodity_specs_in = [s for s in yahoo_spec.input_specs if s.key in commodity_keys]
        commodity_specs_out = [s for s in yahoo_spec.output_specs if s.key in commodity_keys]
        if commodity_data:
            sources["yahoo_commodities"] = DataSource(
                spec=DatasetSpec(
                    name="Yahoo Commodities",
                    description="Commodity futures: energy, metals",
                    input_specs=commodity_specs_in,
                    output_specs=commodity_specs_out,
                ),
                data=commodity_data,
            )
            print(f"  Yahoo Commodities: {len(commodity_specs_in)} tickers")
    except Exception as e:
        print(f"  Yahoo Finance fetch failed: {e}")
        sources.update(_synthetic_markets())

    # News embeddings (synthetic for now)
    print("Generating synthetic news embeddings...")
    n = 5000
    news_emb = torch.randn(n, 32)
    inp, out = _io_pair("news_emb", "events.news_embedding")
    sources["news_embeddings"] = DataSource(
        spec=DatasetSpec(
            name="News Embeddings",
            description="Synthetic news article embeddings",
            input_specs=[inp],
            output_specs=[out],
        ),
        data={"news_emb": news_emb},
    )
    print(f"  News embeddings: {n} samples, dim=32")

    return sources


def _synthetic_macro():
    """Fallback synthetic macro data."""
    n = 5000
    sources = {}
    data = {
        "GDP": torch.cumsum(torch.randn(n // 192) * 0.005, dim=0) + 0.02,
        "CPIAUCSL": torch.cumsum(torch.randn(n // 192) * 0.002, dim=0) + 0.03,
        "UNRATE": (torch.randn(n // 192) * 0.005).abs() + 0.04,
        "PAYEMS": torch.randn(n // 192) * 100 + 150000,
    }
    input_specs = [
        InputSpec(key="GDP", semantic_type="US GDP", field_path="country_us.macro.output.gdp_official.value"),
        InputSpec(key="CPIAUCSL", semantic_type="US CPI", field_path="country_us.macro.inflation.headline_cpi"),
        InputSpec(key="UNRATE", semantic_type="US unemployment", field_path="country_us.macro.labor.unemployment_rate"),
        InputSpec(key="PAYEMS", semantic_type="US nonfarm payrolls", field_path="country_us.macro.labor.nfp_change"),
    ]
    output_specs = [
        OutputSpec(key=s.key, semantic_type=s.semantic_type, field_path=s.field_path)
        for s in input_specs
    ]
    sources["fred_macro"] = DataSource(
        spec=DatasetSpec(name="Synthetic Macro", input_specs=input_specs, output_specs=output_specs),
        data=data,
    )
    return sources


def _synthetic_markets():
    """Fallback synthetic market data."""
    n = 5000
    data = {
        "yahoo_^GSPC": torch.cumsum(torch.randn(n) * 0.01, dim=0) + 8.0,
        "yahoo_^VIX": (torch.randn(n) * 0.1).abs() + 0.15,
        "yahoo_DX-Y.NYB": torch.cumsum(torch.randn(n) * 0.002, dim=0) + 4.6,
    }
    input_specs = [
        InputSpec(key="yahoo_^GSPC", semantic_type="S&P 500", field_path="financial.equities.broad_indices", transform=log_return()),
        InputSpec(key="yahoo_^VIX", semantic_type="VIX", field_path="financial.equities.vix"),
        InputSpec(key="yahoo_DX-Y.NYB", semantic_type="DXY", field_path="financial.fx.dxy", transform=log_return()),
    ]
    output_specs = [
        OutputSpec(key=s.key, semantic_type=s.semantic_type, field_path=s.field_path)
        for s in input_specs
    ]
    return {
        "yahoo_finance": DataSource(
            spec=DatasetSpec(name="Synthetic Markets", input_specs=input_specs, output_specs=output_specs),
            data=data,
        ),
    }


def get_h100_dag() -> list[TrainingNode]:
    """Get the standard DAG with H100-appropriate hyperparameters.

    Scales up from default parameters to use H100 memory budget.
    """
    import copy
    dag = copy.deepcopy(STANDARD_DAG)

    # Tier 0: Foundation — modest, many parallel nodes
    for node in dag:
        if node.name.startswith("basic_"):
            node.d_model = 128
            node.n_steps = 10000
            node.batch_size = 64
            node.lr = 2e-4

    # Tier 1: Cross-domain — bigger canvas
    for node in dag:
        if node.name in ("econ_drives_finance", "geopolitics_commodities", "narratives_drive_markets"):
            node.d_model = 128
            node.n_layers = 8
            node.n_steps = 15000
            node.batch_size = 48
            node.lr = 1e-4

    # Tier 2: Complex — deeper
    for node in dag:
        if node.name in ("corporate_strategy", "policy_impact"):
            node.d_model = 128
            node.n_layers = 10
            node.n_steps = 20000
            node.batch_size = 32
            node.lr = 5e-5

    # Tier 3: Integration — full model
    for node in dag:
        if node.name == "full_integration":
            node.d_model = 128
            node.n_layers = 12
            node.n_loops = 4
            node.n_steps = 30000
            node.batch_size = 24
            node.lr = 3e-5

    return dag


def main():
    parser = argparse.ArgumentParser(description="Train GUWM on H100")
    parser.add_argument("--tier", type=int, default=None,
                        help="Train specific tier (0-3). None=all.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/dag_h100",
                        help="Checkpoint directory")
    parser.add_argument("--fred-key", type=str, default=None,
                        help="FRED API key (or set FRED_API_KEY env)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--nodes", type=str, nargs="+", default=None,
                        help="Train specific nodes by name")
    parser.add_argument("--backbone", type=str, default="cogvideox",
                        choices=["cogvideox", "scratch"],
                        help="Backbone type: cogvideox (graft onto pretrained) or scratch")
    parser.add_argument("--pretrained-model", type=str, default="THUDM/CogVideoX-2b",
                        help="HuggingFace model ID for CogVideoX backbone")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        print(f"Memory: {mem / 1e9:.1f} GB")

    # Fetch data
    print("\n" + "=" * 60)
    print("FETCHING DATA")
    print("=" * 60)
    data_sources = fetch_real_data(fred_api_key=args.fred_key)
    print(f"\nTotal data sources: {len(data_sources)}")
    for name, ds in data_sources.items():
        n_series = len(ds.spec.input_specs)
        n_rows = max((v.shape[0] for v in ds.data.values() if isinstance(v, torch.Tensor) and v.dim() >= 1), default=0)
        print(f"  {name}: {n_series} series, ~{n_rows} rows")

    # Build DAG
    dag = get_h100_dag()
    print(f"\nDAG: {len(dag)} nodes")
    for node in dag:
        has_data = any(ds in data_sources for ds in node.data_sources)
        status = "HAS DATA" if has_data else "NO DATA (untrained init)"
        print(f"  {node.name}: d={node.d_model}, L={node.n_layers}, "
              f"steps={node.n_steps}, [{status}]")

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    trainer = DAGCurriculumTrainer(
        nodes=dag,
        data_sources=data_sources,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        backbone=args.backbone,
        pretrained_model_id=args.pretrained_model,
    )

    start = time.time()

    if args.tier is not None:
        trainer.run_tier(args.tier)
    elif args.nodes:
        trainer.run(args.nodes)
    else:
        trainer.run()

    elapsed = time.time() - start
    print(f"\nTotal training time: {elapsed / 3600:.1f} hours")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for ckpt in trainer.checkpoints:
        print(f"  {ckpt.node_name}: loss={ckpt.loss:.4f}, "
              f"fields={ckpt.n_fields}, params={ckpt.n_params:,}")

    # Save final summary
    summary_path = Path(args.checkpoint_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "device": device,
            "elapsed_hours": elapsed / 3600,
            "nodes": [c.to_dict() for c in trainer.checkpoints],
            "data_sources": list(data_sources.keys()),
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
