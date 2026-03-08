"""Full curriculum training: independent → coupled → integrated.

This example runs the full 3-phase curriculum:
  Phase 1: Train each domain independently (financial, macro, narratives, etc.)
  Phase 2: Couple causally adjacent domains (financial+macro, narratives+financial)
  Phase 3: Full integration on one canvas

Uses synthetic data for demonstration. Replace with real adapters for production.

Run: python examples/06_curriculum_training.py
"""

import torch
from guwm.training.curriculum import (
    CurriculumConfig, CurriculumTrainer, PhaseConfig,
    DomainSpec, STANDARD_DOMAINS,
)
from guwm.training.heterogeneous import DatasetSpec, FieldMapping


def generate_synthetic_sources():
    """Generate synthetic data sources for each standard domain."""
    sources = {}

    # Financial core data
    n = 2000
    financial_data = {
        "sp500": torch.cumsum(torch.randn(n) * 0.01, dim=0) + 4.0,
        "vix": (torch.randn(n) * 0.1).abs() + 0.15,
        "ust_10y": torch.cumsum(torch.randn(n) * 0.005, dim=0) + 0.04,
        "ust_2y": torch.cumsum(torch.randn(n) * 0.005, dim=0) + 0.035,
        "ig_spread": (torch.randn(n) * 0.01).abs() + 0.01,
        "hy_spread": (torch.randn(n) * 0.02).abs() + 0.04,
        "dxy": torch.cumsum(torch.randn(n) * 0.002, dim=0) + 4.6,
    }
    financial_spec = DatasetSpec(
        name="yahoo_finance",
        mappings=[
            FieldMapping("sp500", "financial.equities.broad_indices"),
            FieldMapping("vix", "financial.equities.vix"),
            FieldMapping("ust_10y", "financial.yield_curves.ten_year"),
            FieldMapping("ust_2y", "financial.yield_curves.two_year"),
            FieldMapping("ig_spread", "financial.credit.ig_spread"),
            FieldMapping("hy_spread", "financial.credit.hy_spread"),
            FieldMapping("dxy", "financial.fx.dxy"),
        ],
    )
    sources["yahoo_finance"] = (financial_spec, financial_data)

    # FRED rates
    fred_rates_data = {
        "ffr": torch.cumsum(torch.randn(n) * 0.002, dim=0) + 0.05,
        "t10y2y": torch.randn(n) * 0.005 + 0.01,
    }
    fred_rates_spec = DatasetSpec(
        name="fred_rates",
        mappings=[
            FieldMapping("ffr", "financial.central_banks.policy_rate"),
            FieldMapping("t10y2y", "financial.yield_curves.slope_2s10s"),
        ],
    )
    sources["fred_rates"] = (fred_rates_spec, fred_rates_data)

    # US macro
    macro_data = {
        "gdp": torch.cumsum(torch.randn(n // 192) * 0.005, dim=0) + 0.02,
        "cpi": torch.cumsum(torch.randn(n // 192) * 0.002, dim=0) + 0.03,
        "unrate": (torch.randn(n // 192) * 0.005).abs() + 0.04,
        "pmi": torch.randn(n // 192) * 2.0 + 52.0,
    }
    macro_spec = DatasetSpec(
        name="fred_macro",
        mappings=[
            FieldMapping("gdp", "country_us.macro.output.gdp_nowcast"),
            FieldMapping("cpi", "country_us.macro.inflation.headline_cpi"),
            FieldMapping("unrate", "country_us.macro.labor.unemployment_rate"),
            FieldMapping("pmi", "country_us.macro.output.pmi_manufacturing"),
        ],
        base_period=192,
    )
    sources["fred_macro"] = (macro_spec, macro_data)

    # News embeddings
    news_data = {
        "news_emb": torch.randn(n, 32),  # pre-computed embeddings
    }
    news_spec = DatasetSpec(
        name="news_embeddings",
        mappings=[
            FieldMapping("news_emb", "events.news_embedding"),
        ],
    )
    sources["news_embeddings"] = (news_spec, news_data)

    return sources


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Generate synthetic data
    data_sources = generate_synthetic_sources()
    print(f"Data sources: {list(data_sources.keys())}")

    # Configure curriculum (small for demo)
    config = CurriculumConfig(
        phases=[
            PhaseConfig("phase1_independent", n_steps=200, lr=1e-4, batch_size=32, save_every=100),
            PhaseConfig("phase2_coupling", n_steps=100, lr=5e-5, batch_size=16, save_every=100),
            PhaseConfig("phase3_integration", n_steps=200, lr=3e-5, batch_size=8, save_every=100),
        ],
        checkpoint_dir="checkpoints/curriculum_demo",
        device=device,
    )

    # Build and run curriculum
    trainer = CurriculumTrainer(config, data_sources)
    trainer.run()

    print("\n" + "=" * 60)
    print("Curriculum training complete!")
    print("=" * 60)

    if trainer.integrated_model:
        bound = trainer.integrated_model["bound"]
        print(f"Final model: {len(bound.field_names)} fields, "
              f"{bound.layout.num_positions} positions")


if __name__ == "__main__":
    main()
