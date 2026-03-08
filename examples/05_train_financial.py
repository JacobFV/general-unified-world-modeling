"""Train a world model on real financial data from FRED + Yahoo Finance.

This example shows the full training pipeline:
1. Declare a projection (financial + macro)
2. Fetch real data via adapters
3. Build heterogeneous dataset with masked loss
4. Train with curriculum

Requires: pip install general-unified-world-model[data]
Set FRED_API_KEY environment variable.

Run: python examples/05_train_financial.py
"""

import os
import torch
from canvas_engineering import ConnectivityPolicy

from general_unified_world_model.projection.subset import project
from general_unified_world_model.training.backbone import build_world_model
from general_unified_world_model.training.heterogeneous import (
    FieldEncoder, FieldDecoder, MaskedCanvasTrainer,
    DataSource, build_mixed_dataloader,
)
from general_unified_world_model.data.adapters import fred_adapter, yahoo_finance_adapter


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Projection: financial markets + US macro
    bound = project(
        include=[
            "financial",
            "country_us.macro",
            "resources.energy",
            "regime",
            "events",
            "forecasts.macro",
            "forecasts.financial",
        ],
        T=1, H=48, W=48, d_model=64,
    )
    print(f"Schema: {len(bound.field_names)} fields, "
          f"{bound.layout.num_positions} positions")

    # 2. Fetch data
    sources = []

    # FRED macro data
    if os.environ.get("FRED_API_KEY"):
        print("Fetching FRED data...")
        fred_source = fred_adapter(start_date="2010-01-01")
        sources.append(fred_source)
        print(f"  FRED: {len(fred_source.spec.input_specs)} series")
    else:
        print("FRED_API_KEY not set, generating synthetic data")
        sources.append(_synthetic_macro_source(bound))

    # Yahoo Finance market data
    try:
        print("Fetching Yahoo Finance data...")
        yahoo_source = yahoo_finance_adapter(
            start_date="2010-01-01",
            include_equity=True,
            include_fx=True,
            include_commodities=True,
            include_crypto=True,
        )
        sources.append(yahoo_source)
        print(f"  Yahoo: {len(yahoo_source.spec.input_specs)} tickers")
    except ImportError:
        print("yfinance not installed, using synthetic data")
        sources.append(_synthetic_market_source(bound))

    # 3. Build model
    backbone = build_world_model(bound, n_layers=6, n_loops=3)
    encoder = FieldEncoder(bound)
    decoder = FieldDecoder(bound)

    n_params = sum(p.numel() for p in backbone.parameters())
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"Backbone: {n_params:,} params")
    print(f"Encoder:  {n_enc:,} params")
    print(f"Decoder:  {n_dec:,} params")
    print(f"Total:    {n_params + n_enc + n_dec:,} params")

    # 4. Build training infrastructure
    dataloader = build_mixed_dataloader(bound, sources, batch_size=32)

    all_params = (
        list(backbone.parameters())
        + list(encoder.parameters())
        + list(decoder.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0.01)

    trainer = MaskedCanvasTrainer(
        bound, backbone, encoder, decoder, optimizer, device=device,
    )

    # 5. Train
    print("\nTraining...")
    for step, batch in enumerate(dataloader):
        if step >= 1000:
            break

        metrics = trainer.train_step(batch)

        if step % 50 == 0:
            print(f"Step {step:>4d}: loss={metrics['loss']:.4f}, "
                  f"coverage={metrics['coverage']:.1%}, "
                  f"active={metrics['n_active_positions']}")

    print("\nDone!")


def _synthetic_macro_source(bound):
    """Generate synthetic macro data for testing."""
    from general_unified_world_model.training.heterogeneous import DatasetSpec, DataSource, InputSpec, OutputSpec

    n_rows = 1000
    data = {}
    input_specs = []
    output_specs = []

    # Generate some synthetic series
    for field_name in bound.field_names:
        if "macro" in field_name or "financial" in field_name:
            key = f"synth_{field_name}"
            data[key] = torch.randn(n_rows)
            semantic = field_name.replace("_", " ").replace(".", " > ")
            input_specs.append(InputSpec(
                key=key,
                semantic_type=semantic,
                field_path=field_name,
            ))
            output_specs.append(OutputSpec(
                key=key,
                semantic_type=semantic,
                field_path=field_name,
            ))

    spec = DatasetSpec(name="Synthetic Macro", input_specs=input_specs, output_specs=output_specs)
    return DataSource(spec=spec, data=data)


def _synthetic_market_source(bound):
    """Generate synthetic market data for testing."""
    from general_unified_world_model.training.heterogeneous import DatasetSpec, DataSource, InputSpec, OutputSpec

    n_rows = 1000
    data = {}
    input_specs = []
    output_specs = []

    for field_name in bound.field_names:
        if "equity" in field_name or "fx" in field_name or "yield" in field_name:
            key = f"synth_{field_name}"
            data[key] = torch.cumsum(torch.randn(n_rows) * 0.01, dim=0)
            semantic = field_name.replace("_", " ").replace(".", " > ")
            input_specs.append(InputSpec(
                key=key,
                semantic_type=semantic,
                field_path=field_name,
            ))
            output_specs.append(OutputSpec(
                key=key,
                semantic_type=semantic,
                field_path=field_name,
            ))

    spec = DatasetSpec(name="Synthetic Market", input_specs=input_specs, output_specs=output_specs)
    return DataSource(spec=spec, data=data)


if __name__ == "__main__":
    main()
