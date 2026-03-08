"""LLM-powered dataset annotation: map any dataset's columns to world model fields.

Given a dataset (columns, dtypes, sample values, description), an LLM selects
the best world model field paths for each column and generates InputSpec/OutputSpec
annotations ready for training.

Usage:
    from general_unified_world_model.llm.dataset_annotator import annotate_dataset

    specs = annotate_dataset(
        name="my_dataset",
        columns={"gdp_growth": "float64", "unemployment": "float64"},
        description="Quarterly US macro data",
        sample_values={"gdp_growth": [2.1, 1.8, -0.3], "unemployment": [3.5, 3.7, 4.1]},
    )
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import urllib.request
import urllib.error
from typing import Any

from general_unified_world_model.training.heterogeneous import InputSpec, OutputSpec, DatasetSpec


def _get_schema_paths() -> list[str]:
    """Get all field paths from the world model schema."""
    from general_unified_world_model.schema.world import World
    from canvas_engineering import Field

    paths = []
    def _walk(obj, prefix):
        if not dataclasses.is_dataclass(obj):
            return
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            fp = f"{prefix}.{f.name}" if prefix else f.name
            if isinstance(val, Field):
                paths.append(fp)
            elif dataclasses.is_dataclass(val):
                _walk(val, fp)
    _walk(World(), "")
    return paths


def _call_llm(system: str, user: str, provider: str, api_key: str, model: str | None) -> str:
    """Call Anthropic or OpenAI and return the raw text response."""
    if provider == "anthropic":
        body = json.dumps({
            "model": model or "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return "".join(
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        )
    elif provider == "openai":
        body = json.dumps({
            "model": model or "gpt-4o-mini",
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Unknown provider: {provider}")


def annotate_dataset(
    name: str,
    columns: dict[str, str],
    description: str = "",
    sample_values: dict[str, list] | None = None,
    provider: str = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
) -> DatasetSpec:
    """Use an LLM to annotate a dataset with InputSpec/OutputSpec mappings.

    Args:
        name: Dataset name.
        columns: Dict mapping column names to dtype strings.
            E.g. {"gdp_growth": "float64", "unemployment": "float64"}.
        description: Natural language description of the dataset.
        sample_values: Optional sample values for each column (helps the LLM).
        provider: "anthropic" or "openai".
        api_key: API key (or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var).
        model: Model to use.

    Returns:
        DatasetSpec with input_specs and output_specs populated.
    """
    all_paths = _get_schema_paths()
    top_level = list({p.split(".")[0] for p in all_paths})

    # Build column info
    col_info = []
    for col, dtype in columns.items():
        entry = f"  - {col} ({dtype})"
        if sample_values and col in sample_values:
            vals = sample_values[col][:5]
            entry += f"  samples: {vals}"
        col_info.append(entry)
    col_block = "\n".join(col_info)

    system_prompt = f"""You are a dataset annotation tool for the General Unified World Model.

The world model has {len(all_paths)} fields across these top-level domains:
{', '.join(sorted(top_level))}

Example field paths:
{chr(10).join(all_paths[:80])}
... and {len(all_paths) - 80} more.

Your job: map each dataset column to the best matching world model field path.

For each column, determine:
1. field_path: The dotted field path in the schema (must be an EXACT match from the list above, or a valid prefix like "financial.equities" or "country_us.macro")
2. semantic_type: A human-readable description of what this column represents
3. dtype: The Python data type (float32, float64, int64, embedding, text)
4. loss_fn: The loss function to use (mse, l1, huber). Default: mse
5. loss_weight: Relative importance weight. Default: 1.0

Respond with ONLY a JSON array:
[
  {{
    "column": "column_name",
    "field_path": "dotted.field.path",
    "semantic_type": "Human readable description",
    "dtype": "float32",
    "loss_fn": "mse",
    "loss_weight": 1.0
  }}
]

Rules:
1. Use EXACT field paths from the schema. Do not invent paths.
2. If a column cannot be mapped to any field, omit it from the output.
3. Use semantic_type to describe the modality in plain English (e.g. "10-year US Treasury yield, daily").
4. Numeric columns that represent prices should use loss_fn "huber" for robustness.
5. For rate/ratio columns (unemployment, yields), use "mse".
"""

    user_msg = f"""Dataset: {name}
Description: {description}

Columns:
{col_block}"""

    if api_key is None:
        env_key = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        api_key = os.environ.get(env_key)
        if not api_key:
            raise ValueError(f"No API key. Set {env_key} or pass api_key=.")

    raw_text = _call_llm(system_prompt, user_msg, provider, api_key, model)

    # Parse JSON
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()
    mappings = json.loads(text)

    input_specs = []
    output_specs = []
    for m in mappings:
        col = m["column"]
        fp = m["field_path"]
        st = m.get("semantic_type", fp.replace("_", " ").replace(".", " > "))
        dtype = m.get("dtype", "float32")
        loss_fn = m.get("loss_fn", "mse")
        loss_weight = m.get("loss_weight", 1.0)

        input_specs.append(InputSpec(
            key=col,
            semantic_type=st,
            field_path=fp,
            dtype=dtype,
        ))
        output_specs.append(OutputSpec(
            key=col,
            semantic_type=st,
            field_path=fp,
            dtype=dtype,
            loss_fn=loss_fn,
            loss_weight=loss_weight,
        ))

    return DatasetSpec(
        name=name,
        description=description,
        input_specs=input_specs,
        output_specs=output_specs,
    )


# ── Hardcoded dataset registry ───────────────────────────────────────────
# Known datasets with pre-computed annotations for production use.
# Each entry is a dict that can be passed to DatasetSpec.

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    # ── FRED Economic Data ──────────────────────────────────────────────
    "fred_macro": {
        "name": "FRED Macro",
        "description": "US macroeconomic fundamentals from Federal Reserve Economic Data",
        "source": "api",
        "update_frequency": "monthly",
        "columns": [
            "GDP", "GDPC1", "INDPRO", "TCU", "RSAFS", "DGORDER",
            "CPIAUCSL", "CPILFESL", "PCEPI", "PPIFIS", "CES0500000003", "MICH", "T5YIE",
            "UNRATE", "PAYEMS", "ICSA", "CCSA", "JTSJOL", "JTSQUR", "CIVPART",
            "GFDEGDQ188S", "FYFSGDA188S",
            "CSUSHPINSA", "HOUST", "MORTGAGE30US",
            "UMCSENT",
        ],
        "input_specs": [
            {"key": "GDP", "semantic_type": "US GDP, quarterly", "field_path": "country_us.macro.output.gdp_official.value", "dtype": "float32"},
            {"key": "INDPRO", "semantic_type": "US industrial production, monthly", "field_path": "country_us.macro.output.industrial_production", "dtype": "float32"},
            {"key": "TCU", "semantic_type": "US capacity utilization, monthly", "field_path": "country_us.macro.output.capacity_utilization", "dtype": "float32"},
            {"key": "CPIAUCSL", "semantic_type": "US headline CPI, monthly", "field_path": "country_us.macro.inflation.headline_cpi", "dtype": "float32"},
            {"key": "CPILFESL", "semantic_type": "US core CPI, monthly", "field_path": "country_us.macro.inflation.core_cpi", "dtype": "float32"},
            {"key": "PCEPI", "semantic_type": "US PCE deflator, monthly", "field_path": "country_us.macro.inflation.pce_deflator", "dtype": "float32"},
            {"key": "PPIFIS", "semantic_type": "US PPI, monthly", "field_path": "country_us.macro.inflation.ppi", "dtype": "float32"},
            {"key": "CES0500000003", "semantic_type": "US average hourly earnings, monthly", "field_path": "country_us.macro.inflation.wage_growth", "dtype": "float32"},
            {"key": "MICH", "semantic_type": "Michigan inflation expectations 1-year", "field_path": "country_us.macro.inflation.expectations_1y", "dtype": "float32"},
            {"key": "T5YIE", "semantic_type": "5-year breakeven inflation", "field_path": "country_us.macro.inflation.expectations_5y", "dtype": "float32"},
            {"key": "UNRATE", "semantic_type": "US unemployment rate, monthly", "field_path": "country_us.macro.labor.unemployment_rate", "dtype": "float32"},
            {"key": "PAYEMS", "semantic_type": "US nonfarm payrolls, monthly", "field_path": "country_us.macro.labor.nfp_change", "dtype": "float32"},
            {"key": "ICSA", "semantic_type": "US initial jobless claims, weekly", "field_path": "country_us.macro.labor.initial_claims", "dtype": "float32"},
            {"key": "CCSA", "semantic_type": "US continuing claims, weekly", "field_path": "country_us.macro.labor.continuing_claims", "dtype": "float32"},
            {"key": "JTSJOL", "semantic_type": "US job openings (JOLTS), monthly", "field_path": "country_us.macro.labor.job_openings", "dtype": "float32"},
            {"key": "JTSQUR", "semantic_type": "US quits rate, monthly", "field_path": "country_us.macro.labor.quits_rate", "dtype": "float32"},
            {"key": "CIVPART", "semantic_type": "US labor force participation rate", "field_path": "country_us.macro.labor.lfpr", "dtype": "float32"},
            {"key": "GFDEGDQ188S", "semantic_type": "US federal debt to GDP ratio", "field_path": "country_us.macro.fiscal.debt_to_gdp", "dtype": "float32"},
            {"key": "CSUSHPINSA", "semantic_type": "Case-Shiller home price index", "field_path": "country_us.macro.housing.home_price_index", "dtype": "float32"},
            {"key": "HOUST", "semantic_type": "US housing starts, monthly", "field_path": "country_us.macro.housing.housing_starts", "dtype": "float32"},
            {"key": "MORTGAGE30US", "semantic_type": "30-year fixed mortgage rate", "field_path": "country_us.macro.housing.mortgage_rate", "dtype": "float32"},
            {"key": "UMCSENT", "semantic_type": "University of Michigan consumer sentiment", "field_path": "country_us.domestic_sentiment", "dtype": "float32"},
        ],
    },

    # ── FRED Rates ──────────────────────────────────────────────────────
    "fred_rates": {
        "name": "FRED Rates",
        "description": "US Treasury yields, credit spreads, and monetary policy rates",
        "source": "api",
        "update_frequency": "daily",
        "input_specs": [
            {"key": "DFF", "semantic_type": "Federal funds effective rate", "field_path": "financial.yield_curves.short_rate", "dtype": "float32"},
            {"key": "DGS2", "semantic_type": "2-year Treasury yield", "field_path": "financial.yield_curves.two_year", "dtype": "float32"},
            {"key": "DGS5", "semantic_type": "5-year Treasury yield", "field_path": "financial.yield_curves.five_year", "dtype": "float32"},
            {"key": "DGS10", "semantic_type": "10-year Treasury yield", "field_path": "financial.yield_curves.ten_year", "dtype": "float32"},
            {"key": "DGS30", "semantic_type": "30-year Treasury yield", "field_path": "financial.yield_curves.thirty_year", "dtype": "float32"},
            {"key": "T10Y2Y", "semantic_type": "2s10s yield curve slope", "field_path": "financial.yield_curves.slope_2s10s", "dtype": "float32"},
            {"key": "T10YIE", "semantic_type": "10-year breakeven inflation rate", "field_path": "financial.yield_curves.breakeven_inflation", "dtype": "float32"},
            {"key": "BAMLC0A4CBBB", "semantic_type": "BBB investment grade credit spread", "field_path": "financial.credit.ig_spread", "dtype": "float32"},
            {"key": "BAMLH0A0HYM2", "semantic_type": "High yield credit spread", "field_path": "financial.credit.hy_spread", "dtype": "float32"},
            {"key": "DFEDTARU", "semantic_type": "Federal funds target rate upper bound", "field_path": "financial.central_banks.policy_rate", "dtype": "float32"},
            {"key": "WALCL", "semantic_type": "Federal Reserve total assets (balance sheet)", "field_path": "financial.central_banks.balance_sheet_size", "dtype": "float32"},
            {"key": "RRPONTSYD", "semantic_type": "Fed overnight reverse repo", "field_path": "financial.liquidity.fed_reverse_repo", "dtype": "float32"},
            {"key": "WTREGEN", "semantic_type": "Treasury General Account balance", "field_path": "financial.liquidity.treasury_general_account", "dtype": "float32"},
            {"key": "TOTRESNS", "semantic_type": "Total bank reserves", "field_path": "financial.liquidity.bank_reserves", "dtype": "float32"},
        ],
    },

    # ── Yahoo Finance Equities ──────────────────────────────────────────
    "yahoo_finance": {
        "name": "Yahoo Finance",
        "description": "Daily equity index prices, VIX, FX, and crypto",
        "source": "api",
        "update_frequency": "daily",
        "input_specs": [
            {"key": "yahoo_^GSPC", "semantic_type": "S&P 500 index price", "field_path": "financial.equities.broad_indices", "dtype": "float32"},
            {"key": "yahoo_^DJI", "semantic_type": "Dow Jones Industrial Average", "field_path": "financial.equities.broad_indices", "dtype": "float32"},
            {"key": "yahoo_^IXIC", "semantic_type": "NASDAQ Composite index", "field_path": "financial.equities.broad_indices", "dtype": "float32"},
            {"key": "yahoo_^RUT", "semantic_type": "Russell 2000 small cap index", "field_path": "financial.equities.broad_indices", "dtype": "float32"},
            {"key": "yahoo_^VIX", "semantic_type": "CBOE VIX volatility index", "field_path": "financial.equities.vix", "dtype": "float32"},
            {"key": "yahoo_DX-Y.NYB", "semantic_type": "US Dollar Index (DXY)", "field_path": "financial.fx.dxy", "dtype": "float32"},
            {"key": "yahoo_EURUSD=X", "semantic_type": "EUR/USD exchange rate", "field_path": "financial.fx.eurusd", "dtype": "float32"},
            {"key": "yahoo_JPY=X", "semantic_type": "USD/JPY exchange rate", "field_path": "financial.fx.usdjpy", "dtype": "float32"},
            {"key": "yahoo_BTC-USD", "semantic_type": "Bitcoin price in USD", "field_path": "financial.crypto.btc", "dtype": "float32"},
            {"key": "yahoo_ETH-USD", "semantic_type": "Ethereum price in USD", "field_path": "financial.crypto.eth", "dtype": "float32"},
        ],
    },

    # ── Yahoo Commodities ───────────────────────────────────────────────
    "yahoo_commodities": {
        "name": "Yahoo Commodities",
        "description": "Daily commodity futures prices: energy, metals",
        "source": "api",
        "update_frequency": "daily",
        "input_specs": [
            {"key": "yahoo_CL=F", "semantic_type": "WTI crude oil futures price", "field_path": "resources.energy.crude_price", "dtype": "float32"},
            {"key": "yahoo_NG=F", "semantic_type": "Natural gas futures price", "field_path": "resources.energy.natgas_price", "dtype": "float32"},
            {"key": "yahoo_GC=F", "semantic_type": "Gold futures price", "field_path": "resources.metals.gold", "dtype": "float32"},
            {"key": "yahoo_SI=F", "semantic_type": "Silver futures price", "field_path": "resources.metals.silver", "dtype": "float32"},
            {"key": "yahoo_HG=F", "semantic_type": "Copper futures price", "field_path": "resources.metals.copper", "dtype": "float32"},
        ],
    },

    # ── HuggingFace: FRED-MD ────────────────────────────────────────────
    "hf/fred_md": {
        "name": "HuggingFace/fred-economic-data/FRED-MD",
        "description": "FRED-MD monthly macroeconomic database (128 series, 1959-present)",
        "source": "huggingface",
        "update_frequency": "monthly",
        "hf_dataset": "fred-economic-data/FRED-MD",
    },

    # ── HuggingFace: World Bank ─────────────────────────────────────────
    "hf/world_bank_wdi": {
        "name": "HuggingFace/worldbank/wdi",
        "description": "World Development Indicators: GDP, population, education, health across 200+ countries",
        "source": "huggingface",
        "hf_dataset": "worldbank/wdi",
    },

    # ── HuggingFace: Climate ────────────────────────────────────────────
    "hf/climate_nasa": {
        "name": "HuggingFace/NASA GISS Temperature",
        "description": "NASA GISS global surface temperature anomalies (monthly, 1880-present)",
        "source": "huggingface",
        "hf_dataset": "climatechange-ai/global-temperature",
    },

    # ── News Embeddings ─────────────────────────────────────────────────
    "news_embeddings": {
        "name": "News Embeddings",
        "description": "Pre-computed embeddings of financial news articles for sentiment and narrative tracking",
        "source": "local",
        "update_frequency": "daily",
        "input_specs": [
            {"key": "news_emb", "semantic_type": "News article embedding vector", "field_path": "events.news_embedding", "dtype": "embedding"},
        ],
    },
}


def get_dataset_registry() -> dict[str, dict[str, Any]]:
    """Return the hardcoded dataset registry.

    Each entry contains metadata and pre-computed InputSpec/OutputSpec definitions
    for known datasets. Entries with 'hf_dataset' can be loaded via hf_adapter().
    Entries with 'input_specs' are ready for DatasetSpec construction.
    """
    return dict(DATASET_REGISTRY)


def registry_to_dataset_spec(key: str) -> DatasetSpec | None:
    """Convert a registry entry to a DatasetSpec (without data).

    Returns None if the entry requires external loading (HuggingFace, API).
    """
    entry = DATASET_REGISTRY.get(key)
    if entry is None:
        return None

    raw_specs = entry.get("input_specs", [])
    if not raw_specs:
        return None

    input_specs = []
    output_specs = []
    for s in raw_specs:
        input_specs.append(InputSpec(
            key=s["key"],
            semantic_type=s["semantic_type"],
            field_path=s["field_path"],
            dtype=s.get("dtype", "float32"),
        ))
        output_specs.append(OutputSpec(
            key=s["key"],
            semantic_type=s["semantic_type"],
            field_path=s["field_path"],
            dtype=s.get("dtype", "float32"),
        ))

    return DatasetSpec(
        name=entry["name"],
        description=entry.get("description", ""),
        input_specs=input_specs,
        output_specs=output_specs,
    )
