"""Data adapters: bridge real-world data sources to world model fields.

Each adapter knows how to:
1. Fetch data from a specific source (FRED, Yahoo Finance, etc.)
2. Map source columns to world model field paths
3. Normalize values to [0, 1] or z-score
4. Return a (DatasetSpec, data_dict) pair ready for HeterogeneousDataset

These adapters are the bridge between messy real-world data and the
clean typed ontology of the world model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from general_unified_world_model.training.heterogeneous import DatasetSpec, FieldMapping


# ── Normalization functions ──────────────────────────────────────────────

def z_score(series_mean: float, series_std: float):
    """Return a transform that z-scores a value."""
    def _transform(x: torch.Tensor) -> torch.Tensor:
        return (x - series_mean) / max(series_std, 1e-8)
    return _transform


def minmax(series_min: float, series_max: float):
    """Return a transform that maps to [0, 1]."""
    def _transform(x: torch.Tensor) -> torch.Tensor:
        return (x - series_min) / max(series_max - series_min, 1e-8)
    return _transform


def log_return():
    """Return a transform that computes log returns."""
    def _transform(x: torch.Tensor) -> torch.Tensor:
        if x.dim() >= 1 and x.shape[0] > 1:
            returns = torch.log(x[1:] / x[:-1].clamp(min=1e-8))
            return torch.cat([torch.zeros(1, device=x.device), returns])
        return x * 0.0
    return _transform


def pct_change():
    """Return a transform that computes percentage change."""
    def _transform(x: torch.Tensor) -> torch.Tensor:
        if x.dim() >= 1 and x.shape[0] > 1:
            changes = (x[1:] - x[:-1]) / x[:-1].clamp(min=1e-8)
            return torch.cat([torch.zeros(1, device=x.device), changes])
        return x * 0.0
    return _transform


def rank_normalize():
    """Return a transform that rank-normalizes to [0, 1]."""
    def _transform(x: torch.Tensor) -> torch.Tensor:
        if x.numel() <= 1:
            return torch.zeros_like(x) + 0.5
        ranks = x.argsort().argsort().float()
        return ranks / max(ranks.max().item(), 1.0)
    return _transform


# ── FRED Adapter ─────────────────────────────────────────────────────────

# Standard FRED series → world model field mappings
FRED_MAPPINGS = {
    # Output & Growth
    "GDP": ("country_us.macro.output.gdp_official.value", 576, None),
    "GDPC1": ("country_us.macro.output.gdp_official.value", 576, None),
    "INDPRO": ("country_us.macro.output.industrial_production", 192, pct_change()),
    "TCU": ("country_us.macro.output.capacity_utilization", 192, None),
    "RSAFS": ("country_us.macro.output.retail_sales", 192, pct_change()),
    "DGORDER": ("country_us.macro.output.new_orders", 192, pct_change()),

    # Inflation
    "CPIAUCSL": ("country_us.macro.inflation.headline_cpi", 192, pct_change()),
    "CPILFESL": ("country_us.macro.inflation.core_cpi", 192, pct_change()),
    "PCEPI": ("country_us.macro.inflation.pce_deflator", 192, pct_change()),
    "PPIFIS": ("country_us.macro.inflation.ppi", 192, pct_change()),
    "CES0500000003": ("country_us.macro.inflation.wage_growth", 192, pct_change()),
    "CUSR0000SEHA": ("country_us.macro.inflation.rent_inflation", 192, pct_change()),
    "MICH": ("country_us.macro.inflation.expectations_1y", 192, None),
    "T5YIE": ("country_us.macro.inflation.expectations_5y", 192, None),

    # Labor
    "UNRATE": ("country_us.macro.labor.unemployment_rate", 192, None),
    "PAYEMS": ("country_us.macro.labor.nfp_change", 192, pct_change()),
    "ICSA": ("country_us.macro.labor.initial_claims", 48, None),
    "CCSA": ("country_us.macro.labor.continuing_claims", 48, None),
    "JTSJOL": ("country_us.macro.labor.job_openings", 192, None),
    "JTSQUR": ("country_us.macro.labor.quits_rate", 192, None),
    "CIVPART": ("country_us.macro.labor.lfpr", 192, None),
    "CES0500000003": ("country_us.macro.labor.avg_hourly_earnings", 192, pct_change()),

    # Fiscal
    "GFDEGDQ188S": ("country_us.macro.fiscal.debt_to_gdp", 576, None),
    "FYFSGDA188S": ("country_us.macro.fiscal.deficit_to_gdp", 576, None),
    "A091RC1Q027SBEA": ("country_us.macro.fiscal.interest_expense_share", 576, None),

    # Housing
    "CSUSHPINSA": ("country_us.macro.housing.home_price_index", 192, pct_change()),
    "HOUST": ("country_us.macro.housing.housing_starts", 192, None),
    "MORTGAGE30US": ("country_us.macro.housing.mortgage_rate", 16, None),
    "EXHOSLUSM495S": ("country_us.macro.housing.existing_home_sales", 192, None),

    # Yield Curve
    "DFF": ("financial.yield_curves.short_rate", 1, None),
    "DGS2": ("financial.yield_curves.two_year", 1, None),
    "DGS5": ("financial.yield_curves.five_year", 1, None),
    "DGS10": ("financial.yield_curves.ten_year", 1, None),
    "DGS30": ("financial.yield_curves.thirty_year", 1, None),
    "T10Y2Y": ("financial.yield_curves.slope_2s10s", 1, None),
    "T10YIE": ("financial.yield_curves.breakeven_inflation", 1, None),

    # Credit
    "BAMLC0A4CBBB": ("financial.credit.ig_spread", 1, None),
    "BAMLH0A0HYM2": ("financial.credit.hy_spread", 1, None),

    # Monetary
    "DFEDTARU": ("financial.central_banks.policy_rate", 192, None),
    "WALCL": ("financial.central_banks.balance_sheet_size", 48, None),

    # Liquidity
    "RRPONTSYD": ("financial.liquidity.fed_reverse_repo", 16, None),
    "WTREGEN": ("financial.liquidity.treasury_general_account", 16, None),
    "TOTRESNS": ("financial.liquidity.bank_reserves", 48, None),

    # Sentiment
    "UMCSENT": ("country_us.domestic_sentiment", 192, None),
    "USSLIND": ("narratives.public.consumer_confidence", 192, None),
}


def fred_adapter(
    series_ids: list[str] | None = None,
    api_key: str | None = None,
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    cache_dir: str | None = None,
) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
    """Build a DatasetSpec + data dict from FRED series.

    Args:
        series_ids: List of FRED series IDs. None = all known mappings.
        api_key: FRED API key. If None, reads from FRED_API_KEY env var.
        start_date: Start date for data fetch.
        end_date: End date (None = today).
        cache_dir: Directory to cache downloaded data.

    Returns:
        (DatasetSpec, data_dict) ready for HeterogeneousDataset.
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("Install fredapi: pip install general-unified-world-model[data]")

    import os
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED API key required. Set FRED_API_KEY env var or pass api_key.")

    fred = Fred(api_key=api_key)

    if series_ids is None:
        series_ids = list(FRED_MAPPINGS.keys())

    mappings = []
    data_dict = {}

    for sid in series_ids:
        if sid not in FRED_MAPPINGS:
            continue

        target_field, period, transform = FRED_MAPPINGS[sid]

        try:
            series = fred.get_series(sid, observation_start=start_date, observation_end=end_date)
            series = series.dropna()
            if len(series) == 0:
                continue

            values = torch.tensor(series.values, dtype=torch.float32)
            data_dict[sid] = values

            mappings.append(FieldMapping(
                source_key=sid,
                target_field=target_field,
                transform=transform,
                frequency=period,
            ))
        except Exception:
            continue

    spec = DatasetSpec(
        name="FRED",
        mappings=mappings,
        base_period=1,  # FRED data comes at natural frequency
        weight=1.0,
    )

    return spec, data_dict


# ── Yahoo Finance Adapter ───────────────────────────────────────────────

YAHOO_EQUITY_FIELDS = {
    "^GSPC": ("financial.equities.broad_indices", 0),
    "^DJI": ("financial.equities.broad_indices", 1),
    "^IXIC": ("financial.equities.broad_indices", 2),
    "^RUT": ("financial.equities.broad_indices", 3),
    "^VIX": ("financial.equities.vix", None),
}

YAHOO_FX_FIELDS = {
    "DX-Y.NYB": ("financial.fx.dxy", None),
    "EURUSD=X": ("financial.fx.eurusd", None),
    "JPY=X": ("financial.fx.usdjpy", None),
    "CNY=X": ("financial.fx.usdcny", None),
}

YAHOO_COMMODITY_FIELDS = {
    "CL=F": ("resources.energy.crude_price", None),
    "NG=F": ("resources.energy.natgas_price", None),
    "GC=F": ("resources.metals.gold", None),
    "SI=F": ("resources.metals.silver", None),
    "HG=F": ("resources.metals.copper", None),
}

YAHOO_CRYPTO_FIELDS = {
    "BTC-USD": ("financial.crypto.btc", None),
    "ETH-USD": ("financial.crypto.eth", None),
}


def yahoo_finance_adapter(
    tickers: list[str] | None = None,
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    include_equity: bool = True,
    include_fx: bool = True,
    include_commodities: bool = True,
    include_crypto: bool = True,
    firm_tickers: dict[str, str] | None = None,
) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
    """Build DatasetSpec from Yahoo Finance.

    Args:
        tickers: Explicit list of tickers. None = use all known mappings.
        start_date: Start date.
        end_date: End date (None = today).
        include_equity: Include equity indices.
        include_fx: Include FX pairs.
        include_commodities: Include commodity futures.
        include_crypto: Include crypto.
        firm_tickers: Dict mapping ticker -> firm field prefix.
            E.g. {"AAPL": "firm_AAPL", "NVDA": "firm_NVDA"}.

    Returns:
        (DatasetSpec, data_dict) ready for HeterogeneousDataset.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install general-unified-world-model[data]")

    # Collect all tickers and their mappings
    all_fields = {}
    if include_equity:
        all_fields.update(YAHOO_EQUITY_FIELDS)
    if include_fx:
        all_fields.update(YAHOO_FX_FIELDS)
    if include_commodities:
        all_fields.update(YAHOO_COMMODITY_FIELDS)
    if include_crypto:
        all_fields.update(YAHOO_CRYPTO_FIELDS)

    if tickers is not None:
        all_fields = {k: v for k, v in all_fields.items() if k in tickers}

    # Add firm-specific tickers
    firm_tickers = firm_tickers or {}
    for ticker, firm_prefix in firm_tickers.items():
        all_fields[ticker] = (f"{firm_prefix}.market.equity_price", None)

    ticker_list = list(all_fields.keys())
    if not ticker_list:
        return DatasetSpec(name="Yahoo Finance", mappings=[]), {}

    # Download
    data = yf.download(
        ticker_list,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    mappings = []
    data_dict = {}

    for ticker in ticker_list:
        target_field, sub_idx = all_fields[ticker]

        try:
            if len(ticker_list) > 1:
                close = data["Close"][ticker].dropna()
            else:
                close = data["Close"].dropna()

            if len(close) == 0:
                continue

            values = torch.tensor(close.values, dtype=torch.float32)
            key = f"yahoo_{ticker}"
            data_dict[key] = values

            mappings.append(FieldMapping(
                source_key=key,
                target_field=target_field,
                transform=log_return(),  # always use log returns for prices
                frequency=1,  # daily
            ))
        except (KeyError, TypeError):
            continue

    spec = DatasetSpec(
        name="Yahoo Finance",
        mappings=mappings,
        base_period=16,  # daily = period 16 in base ticks
        weight=1.0,
    )

    return spec, data_dict


# ── PMI Adapter (ISM) ───────────────────────────────────────────────────

def pmi_adapter(
    data: dict[str, torch.Tensor],
    country: str = "us",
) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
    """Adapter for PMI / ISM survey data.

    Args:
        data: Dict with keys like "manufacturing_pmi", "services_pmi",
              "new_orders", "employment", "prices_paid".
        country: Country code (e.g. "us", "cn", "eu").

    Returns:
        (DatasetSpec, data_dict).
    """
    prefix = f"country_{country}.macro.output"
    mappings = []

    field_map = {
        "manufacturing_pmi": f"{prefix}.pmi_manufacturing",
        "services_pmi": f"{prefix}.pmi_services",
        "new_orders": f"{prefix}.new_orders",
    }

    for key, target in field_map.items():
        if key in data:
            mappings.append(FieldMapping(
                source_key=key,
                target_field=target,
                transform=minmax(30.0, 70.0),  # PMI range
                frequency=192,  # monthly
            ))

    spec = DatasetSpec(
        name=f"PMI ({country.upper()})",
        mappings=mappings,
        base_period=192,
        weight=1.5,  # PMI is highly informative
    )

    return spec, data


# ── Earnings Adapter ─────────────────────────────────────────────────────

def earnings_adapter(
    firm_name: str,
    data: dict[str, torch.Tensor],
) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
    """Adapter for quarterly earnings data for a specific firm.

    Args:
        firm_name: Firm identifier (e.g. "AAPL").
        data: Dict with keys matching FirmFinancials field names.
            E.g. {"revenue": tensor, "gross_margin": tensor, ...}

    Returns:
        (DatasetSpec, data_dict).
    """
    prefix = f"firm_{firm_name}.financials"
    mappings = []

    financial_fields = [
        "revenue", "revenue_growth", "cogs", "gross_margin", "opex",
        "operating_margin", "net_income", "fcf", "cash", "total_debt",
        "net_debt_to_ebitda", "interest_coverage", "capex",
    ]

    for field in financial_fields:
        key = f"earnings_{field}"
        if key in data or field in data:
            source = key if key in data else field
            mappings.append(FieldMapping(
                source_key=source,
                target_field=f"{prefix}.{field}",
                frequency=576,  # quarterly
            ))

    spec = DatasetSpec(
        name=f"Earnings ({firm_name})",
        mappings=mappings,
        base_period=576,
        weight=2.0,
    )

    return spec, data


# ── News Embedding Adapter ──────────────────────────────────────────────

def news_adapter(
    embeddings: torch.Tensor,
    timestamps: Optional[torch.Tensor] = None,
) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
    """Adapter for pre-computed news embeddings.

    Args:
        embeddings: (N, embed_dim) tensor of news article embeddings.
        timestamps: (N,) tensor of tick indices. None = sequential.

    Returns:
        (DatasetSpec, data_dict).
    """
    data = {"news_emb": embeddings}

    mappings = [
        FieldMapping(
            source_key="news_emb",
            target_field="events.news_embedding",
            frequency=1,
        ),
    ]

    spec = DatasetSpec(
        name="News Embeddings",
        mappings=mappings,
        base_period=1,
        weight=0.5,
    )

    return spec, data


# ── Generic CSV/Parquet Adapter ──────────────────────────────────────────

def tabular_adapter(
    name: str,
    data_path: str,
    column_mappings: dict[str, str],
    transforms: dict[str, str] | None = None,
    base_period: int = 1,
    weight: float = 1.0,
) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
    """Generic adapter for CSV/Parquet files with explicit column mappings.

    Args:
        name: Dataset name.
        data_path: Path to CSV or Parquet file.
        column_mappings: Dict mapping source column names to world model field paths.
            E.g. {"gdp_growth": "country_us.macro.output.gdp_nowcast"}
        transforms: Dict mapping source columns to transform names.
            Supported: "z_score", "minmax", "log_return", "pct_change", "rank".
        base_period: Base period for this dataset.
        weight: Dataset weight.

    Returns:
        (DatasetSpec, data_dict).
    """
    import pandas as pd

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    transforms = transforms or {}
    mappings = []
    data_dict = {}

    for source_col, target_field in column_mappings.items():
        if source_col not in df.columns:
            continue

        series = df[source_col].dropna()
        if len(series) == 0:
            continue

        values = torch.tensor(series.values, dtype=torch.float32)

        # Apply transform
        transform_name = transforms.get(source_col)
        transform = None
        if transform_name == "z_score":
            transform = z_score(values.mean().item(), values.std().item())
        elif transform_name == "minmax":
            transform = minmax(values.min().item(), values.max().item())
        elif transform_name == "log_return":
            transform = log_return()
        elif transform_name == "pct_change":
            transform = pct_change()
        elif transform_name == "rank":
            transform = rank_normalize()

        data_dict[source_col] = values
        mappings.append(FieldMapping(
            source_key=source_col,
            target_field=target_field,
            transform=transform,
        ))

    spec = DatasetSpec(
        name=name,
        mappings=mappings,
        base_period=base_period,
        weight=weight,
    )

    return spec, data_dict
