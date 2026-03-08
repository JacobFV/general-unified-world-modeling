"""HuggingFace dataset auto-mapper: automatically extract schema mappings.

Given any HuggingFace dataset, this module infers which world model fields
each column maps to using:
  1. Column name keyword matching (fast, no LLM)
  2. Dataset card metadata (tags, description, task categories)
  3. Optional LLM-based resolution for ambiguous columns

The result is a (DatasetSpec, data_dict) ready for heterogeneous training.

Usage:
    from general_unified_world_model.data.huggingface import hf_adapter

    # Auto-map from HuggingFace
    spec, data = hf_adapter("fred-economic-data/FRED-MD")

    # Or with explicit overrides
    spec, data = hf_adapter(
        "nasdaq/nasdaq-100-index",
        column_overrides={"close": "financial.equities.broad_indices"},
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np
import torch

from general_unified_world_model.training.heterogeneous import DatasetSpec, DataSource, InputSpec, OutputSpec, _infer_semantic_type
from general_unified_world_model.data.adapters import (
    z_score, minmax, log_return, pct_change, rank_normalize,
)
from general_unified_world_model.schema.temporal_constants import (
    TICK, HOURLY, DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL,
)

logger = logging.getLogger(__name__)


# ── Column name → field path rules ──────────────────────────────────────
# Ordered by specificity (most specific first). Each rule is:
#   (pattern_regex, target_field, transform_name, frequency)

COLUMN_RULES: list[tuple[str, str, str | None, int]] = [
    # GDP / output
    (r"\bgdp\b.*\bgrowth\b", "country_us.macro.output.gdp_nowcast", "z_score", QUARTERLY),
    (r"\bgdp\b.*\breal\b", "country_us.macro.output.gdp_official.value", "z_score", QUARTERLY),
    (r"\bgdp\b", "country_us.macro.output.gdp_official.value", "z_score", QUARTERLY),
    (r"\bindustrial.?prod", "country_us.macro.output.industrial_production", "pct_change", MONTHLY),
    (r"\bcapacity.?util", "country_us.macro.output.capacity_utilization", "z_score", MONTHLY),
    (r"\bretail.?sales?\b", "country_us.macro.output.retail_sales", "pct_change", MONTHLY),

    # Inflation
    (r"\bcore.?cpi\b", "country_us.macro.inflation.core_cpi", "pct_change", MONTHLY),
    (r"\bcpi\b", "country_us.macro.inflation.headline_cpi", "pct_change", MONTHLY),
    (r"\bpce\b.*\bdeflator\b", "country_us.macro.inflation.pce_deflator", "pct_change", MONTHLY),
    (r"\bppi\b", "country_us.macro.inflation.ppi", "pct_change", MONTHLY),
    (r"\binflation\b.*\bexpect", "country_us.macro.inflation.expectations_1y", "z_score", MONTHLY),
    (r"\binflation\b", "country_us.macro.inflation.headline_cpi", "pct_change", MONTHLY),

    # Labor
    (r"\bunemployment\b.*\brate\b", "country_us.macro.labor.unemployment_rate", "z_score", MONTHLY),
    (r"\bunemployment\b", "country_us.macro.labor.unemployment_rate", "z_score", MONTHLY),
    (r"\bnonfarm\b|\bnfp\b|\bpayroll", "country_us.macro.labor.nfp_change", "pct_change", MONTHLY),
    (r"\binitial.?claims?\b", "country_us.macro.labor.initial_claims", "z_score", WEEKLY),
    (r"\bjob.?open", "country_us.macro.labor.job_openings", "z_score", MONTHLY),
    (r"\bwage\b", "country_us.macro.inflation.wage_growth", "pct_change", MONTHLY),

    # Housing
    (r"\bhome.?price|house.?price|case.?shiller", "country_us.macro.housing.home_price_index", "pct_change", MONTHLY),
    (r"\bhousing.?start", "country_us.macro.housing.housing_starts", "z_score", MONTHLY),
    (r"\bmortgage.?rate", "country_us.macro.housing.mortgage_rate", "z_score", DAILY),

    # Yield curve / rates
    (r"\bfed.?funds?\b|\bpolicy.?rate\b", "financial.central_banks.policy_rate", "z_score", DAILY),
    (r"\b10.?y(?:ear|r)?\b.*\byield\b|\byield\b.*\b10.?y", "financial.yield_curves.ten_year", "z_score", DAILY),
    (r"\b2.?y(?:ear|r)?\b.*\byield\b|\byield\b.*\b2.?y", "financial.yield_curves.two_year", "z_score", DAILY),
    (r"\b30.?y(?:ear|r)?\b.*\byield\b|\byield\b.*\b30.?y", "financial.yield_curves.thirty_year", "z_score", DAILY),
    (r"\bspread\b.*\b2s10s\b|2s10s|term.?spread", "financial.yield_curves.slope_2s10s", "z_score", DAILY),
    (r"\bbreakeven\b", "financial.yield_curves.breakeven_inflation", "z_score", DAILY),

    # Credit
    (r"\big.?spread\b|investment.?grade", "financial.credit.ig_spread", "z_score", DAILY),
    (r"\bhy.?spread\b|high.?yield", "financial.credit.hy_spread", "z_score", DAILY),
    (r"\bcredit.?spread", "financial.credit.ig_spread", "z_score", DAILY),

    # Equity
    (r"\bs&?p.?500\b|\bspx\b|\bgspc\b", "financial.equities.broad_indices", "log_return", DAILY),
    (r"\bnasdaq\b|\bixic\b", "financial.equities.broad_indices", "log_return", DAILY),
    (r"\bdow\b|\bdji\b", "financial.equities.broad_indices", "log_return", DAILY),
    (r"\brussell\b|\brut\b", "financial.equities.broad_indices", "log_return", DAILY),
    (r"\bvix\b|\bvolatility\b", "financial.equities.vix", "z_score", DAILY),
    (r"\bstock\b.*\bprice\b|\bclose\b|\badj.?close\b", "financial.equities.broad_indices", "log_return", DAILY),

    # FX
    (r"\bdxy\b|\bdollar.?index\b", "financial.fx.dxy", "z_score", DAILY),
    (r"\beur.?usd\b", "financial.fx.eurusd", "log_return", DAILY),
    (r"\busd.?jpy\b", "financial.fx.usdjpy", "log_return", DAILY),
    (r"\bexchange.?rate\b|\bfx\b", "financial.fx.dxy", "z_score", DAILY),

    # Commodities
    (r"\bcrude\b|\bwti\b|\bbrent\b|\boil.?price\b", "resources.energy.crude_price", "log_return", DAILY),
    (r"\bnat(?:ural)?.?gas\b", "resources.energy.natgas_price", "log_return", DAILY),
    (r"\bgold\b", "resources.metals.gold", "log_return", DAILY),
    (r"\bsilver\b", "resources.metals.silver", "log_return", DAILY),
    (r"\bcopper\b", "resources.metals.copper", "log_return", DAILY),

    # Crypto
    (r"\bbitcoin\b|\bbtc\b", "financial.crypto.btc", "log_return", DAILY),
    (r"\bethereum\b|\beth\b", "financial.crypto.eth", "log_return", DAILY),

    # Sentiment
    (r"\bconsumer.?sentiment\b", "country_us.domestic_sentiment", "z_score", MONTHLY),
    (r"\bconsumer.?confidence\b", "narratives.public.consumer_confidence", "z_score", MONTHLY),
    (r"\bsentiment\b", "narratives.public.consumer_confidence", "z_score", DAILY),

    # PMI
    (r"\bpmi\b.*\bmanufact", "country_us.macro.output.pmi_manufacturing", "z_score", MONTHLY),
    (r"\bpmi\b.*\bservice", "country_us.macro.output.pmi_services", "z_score", MONTHLY),
    (r"\bpmi\b", "country_us.macro.output.pmi_manufacturing", "z_score", MONTHLY),

    # Demographics
    (r"\bpopulation\b", "country_us.demographics.population", "z_score", ANNUAL),
    (r"\blife.?expect", "country_us.demographics.life_expectancy", "z_score", ANNUAL),
    (r"\burbaniz", "country_us.demographics.urbanization_rate", "z_score", ANNUAL),

    # Climate / environment
    (r"\btemperature\b|\btemp.?anomal", "physical.climate.global_temp_anomaly", "z_score", ANNUAL),
    (r"\bco2\b|\bcarbon.?dioxide", "physical.climate.co2_ppm", "z_score", ANNUAL),
    (r"\bsea.?level\b", "physical.climate.sea_level_trend", "z_score", ANNUAL),
    (r"\benergy.?mix\b|\brenewable\b", "resources.energy.renewable_share", "z_score", ANNUAL),

    # Trade
    (r"\btrade.?balance\b|\bexport|import", "country_us.macro.trade.trade_balance", "z_score", QUARTERLY),
    (r"\bfdi\b|\bforeign.?direct", "country_us.macro.trade.fdi_flows", "z_score", QUARTERLY),
    (r"\btariff\b", "country_us.macro.trade.tariff_rate", "z_score", ANNUAL),
]


# ── Dataset-level tag → domain mapping ──────────────────────────────────
# HuggingFace datasets have tags like "finance", "economics", "climate"

TAG_TO_DOMAIN: dict[str, list[str]] = {
    "finance": ["financial"],
    "financial": ["financial"],
    "economics": ["country_us.macro", "financial"],
    "economic": ["country_us.macro"],
    "macroeconomics": ["country_us.macro"],
    "stock-market": ["financial.equities"],
    "stock": ["financial.equities"],
    "forex": ["financial.fx"],
    "cryptocurrency": ["financial.crypto"],
    "climate": ["physical.climate", "resources.energy"],
    "weather": ["physical.climate"],
    "environment": ["physical.climate", "biology"],
    "health": ["health"],
    "healthcare": ["health"],
    "politics": ["country_us.politics"],
    "geopolitics": ["country_us.politics", "country_cn.politics"],
    "energy": ["resources.energy"],
    "commodities": ["resources"],
    "trade": ["country_us.macro.trade"],
    "labor": ["country_us.macro.labor"],
    "housing": ["country_us.macro.housing"],
    "sentiment": ["narratives"],
    "news": ["events", "narratives"],
    "technology": ["technology"],
    "demographics": ["country_us.demographics"],
    "education": ["education"],
    "legal": ["legal"],
    "infrastructure": ["infrastructure"],
    "cybersecurity": ["cyber"],
    "space": ["space"],
    "agriculture": ["resources.food"],
}


def _make_transform(name: str | None, values: torch.Tensor):
    """Instantiate a transform function from a name."""
    if name is None:
        return None
    if name == "z_score":
        return z_score(values.mean().item(), values.std().item())
    if name == "minmax":
        return minmax(values.min().item(), values.max().item())
    if name == "log_return":
        return log_return()
    if name == "pct_change":
        return pct_change()
    if name == "rank":
        return rank_normalize()
    return None


def _match_column(col_name: str) -> tuple[str, str | None, int] | None:
    """Match a column name to a world model field path.

    Returns (target_field, transform_name, frequency) or None.
    """
    col_lower = col_name.lower().replace("_", " ").replace("-", " ")
    for pattern, target, transform, freq in COLUMN_RULES:
        if re.search(pattern, col_lower):
            return target, transform, freq
    return None


def _infer_frequency_from_metadata(ds_info: dict) -> int:
    """Infer temporal frequency from dataset metadata."""
    desc = (ds_info.get("description", "") or "").lower()
    name = (ds_info.get("id", "") or "").lower()

    for text in [desc, name]:
        if "daily" in text or "day" in text:
            return DAILY
        if "weekly" in text or "week" in text:
            return WEEKLY
        if "monthly" in text or "month" in text:
            return MONTHLY
        if "quarterly" in text or "quarter" in text:
            return QUARTERLY
        if "annual" in text or "yearly" in text or "year" in text:
            return ANNUAL
        if "hourly" in text or "hour" in text:
            return HOURLY
        if "tick" in text or "minute" in text or "intraday" in text:
            return TICK

    return DAILY  # default


def _infer_domains_from_metadata(ds_info: dict) -> list[str]:
    """Infer relevant world model domains from dataset metadata."""
    domains = []
    tags = ds_info.get("tags", []) or []
    desc = (ds_info.get("description", "") or "").lower()
    name = (ds_info.get("id", "") or "").lower()

    for tag in tags:
        tag_lower = tag.lower().strip()
        if tag_lower in TAG_TO_DOMAIN:
            domains.extend(TAG_TO_DOMAIN[tag_lower])

    # Also check description and name for domain hints
    for keyword, domain_paths in TAG_TO_DOMAIN.items():
        if keyword in desc or keyword in name:
            for p in domain_paths:
                if p not in domains:
                    domains.append(p)

    return domains


# ── Main adapter ────────────────────────────────────────────────────────


def hf_adapter(
    dataset_name: str,
    split: str = "train",
    column_overrides: dict[str, str] | None = None,
    exclude_columns: list[str] | None = None,
    max_rows: int | None = None,
    transform_overrides: dict[str, str] | None = None,
    weight: float = 1.0,
    trust_remote_code: bool = False,
) -> DataSource:
    """Auto-map a HuggingFace dataset to world model fields.

    Loads a dataset from HuggingFace Hub, inspects column names and metadata,
    and automatically maps columns to world model field paths.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. "fred-economic-data/FRED-MD").
        split: Dataset split to use.
        column_overrides: Dict mapping column names to explicit field paths.
            Overrides automatic matching.
        exclude_columns: Columns to skip entirely (e.g. ["date", "index"]).
        max_rows: Limit number of rows loaded.
        transform_overrides: Dict mapping column names to transform names
            ("z_score", "log_return", "pct_change", "minmax", "rank").
        weight: Dataset importance weight.
        trust_remote_code: Whether to trust remote code when loading.

    Returns:
        (DatasetSpec, data_dict) ready for HeterogeneousDataset.

    Example:
        >>> spec, data = hf_adapter("fred-economic-data/FRED-MD")
        >>> print(f"{spec.name}: {len(spec.input_specs)} fields mapped")
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install datasets: pip install general-unified-world-model[data]"
        )

    # Load dataset
    ds = load_dataset(
        dataset_name,
        split=split,
        trust_remote_code=trust_remote_code,
    )

    # Get metadata
    ds_info = {}
    if hasattr(ds, "info") and ds.info is not None:
        ds_info = {
            "id": dataset_name,
            "description": getattr(ds.info, "description", ""),
            "tags": getattr(ds.info, "tags", []) or [],
        }
    else:
        ds_info = {"id": dataset_name, "description": "", "tags": []}

    # Extract dataset card info for better context
    inferred_freq = _infer_frequency_from_metadata(ds_info)
    inferred_domains = _infer_domains_from_metadata(ds_info)

    column_overrides = column_overrides or {}
    transform_overrides = transform_overrides or {}
    exclude_columns = set(exclude_columns or [])

    # Common non-data columns to auto-exclude
    auto_exclude = {
        "date", "dates", "timestamp", "time", "datetime", "index", "id",
        "unnamed", "unnamed: 0", "row_id", "observation_date",
    }
    exclude_columns |= auto_exclude

    # Limit rows
    if max_rows is not None and len(ds) > max_rows:
        ds = ds.select(range(max_rows))

    input_specs, output_specs = [], []
    data_dict = {}
    mapped_targets = set()  # avoid duplicate field mappings

    columns = ds.column_names

    for col in columns:
        col_lower = col.lower().strip()
        if col_lower in exclude_columns:
            continue

        # Try to extract numeric data
        try:
            values = ds[col]
            # Filter to numeric values only
            numeric_vals = []
            for v in values:
                if isinstance(v, (int, float)):
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        numeric_vals.append(float(v))
                    else:
                        numeric_vals.append(0.0)
                else:
                    break
            else:
                if len(numeric_vals) < 2:
                    continue
                tensor = torch.tensor(numeric_vals, dtype=torch.float32)
        except (ValueError, TypeError):
            continue

        # Determine field mapping
        target = None
        transform_name = None
        freq = inferred_freq

        # 1. Explicit override
        if col in column_overrides:
            target = column_overrides[col]
            transform_name = transform_overrides.get(col)
        else:
            # 2. Column name matching
            match = _match_column(col)
            if match is not None:
                target, transform_name, freq = match

        # Apply transform override
        if col in transform_overrides:
            transform_name = transform_overrides[col]

        if target is None:
            logger.debug(f"No mapping found for column '{col}'")
            continue

        if target in mapped_targets:
            logger.debug(f"Skipping duplicate mapping for '{target}' from column '{col}'")
            continue

        mapped_targets.add(target)
        transform = _make_transform(transform_name, tensor)

        source_key = f"hf_{col}"
        data_dict[source_key] = tensor

        st = _infer_semantic_type(target)
        input_specs.append(InputSpec(key=source_key, semantic_type=st, field_path=target, transform=transform, frequency=freq))
        output_specs.append(OutputSpec(key=source_key, semantic_type=st, field_path=target, frequency=freq))
        logger.info(f"Mapped '{col}' → {target} (transform={transform_name}, freq={freq})")

    spec = DatasetSpec(
        name=f"HuggingFace/{dataset_name}",
        description=f"HuggingFace dataset: {dataset_name}",
        input_specs=input_specs,
        output_specs=output_specs,
        base_period=inferred_freq,
        weight=weight,
    )

    n_total = len([c for c in columns if c.lower().strip() not in exclude_columns])
    logger.info(
        f"HuggingFace adapter: {len(input_specs)}/{n_total} columns mapped "
        f"from '{dataset_name}' (domains: {inferred_domains})"
    )

    return DataSource(spec=spec, data=data_dict)


def hf_inspect(
    dataset_name: str,
    split: str = "train",
    trust_remote_code: bool = False,
) -> dict:
    """Inspect a HuggingFace dataset and show potential field mappings.

    Useful for previewing what would be auto-mapped before committing.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split.
        trust_remote_code: Whether to trust remote code.

    Returns:
        Dict with dataset info and column → field mapping proposals.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install datasets: pip install general-unified-world-model[data]"
        )

    ds = load_dataset(dataset_name, split=split, trust_remote_code=trust_remote_code)

    ds_info = {}
    if hasattr(ds, "info") and ds.info is not None:
        ds_info = {
            "id": dataset_name,
            "description": getattr(ds.info, "description", ""),
            "tags": getattr(ds.info, "tags", []) or [],
        }
    else:
        ds_info = {"id": dataset_name, "description": "", "tags": []}

    inferred_freq = _infer_frequency_from_metadata(ds_info)
    inferred_domains = _infer_domains_from_metadata(ds_info)

    auto_exclude = {
        "date", "dates", "timestamp", "time", "datetime", "index", "id",
        "unnamed", "unnamed: 0", "row_id", "observation_date",
    }

    proposals = {}
    unmapped = []

    for col in ds.column_names:
        col_lower = col.lower().strip()
        if col_lower in auto_exclude:
            proposals[col] = {"status": "excluded", "reason": "date/index column"}
            continue

        match = _match_column(col)
        if match is not None:
            target, transform, freq = match
            proposals[col] = {
                "status": "mapped",
                "target_field": target,
                "transform": transform,
                "frequency": freq,
            }
        else:
            proposals[col] = {"status": "unmapped"}
            unmapped.append(col)

    return {
        "dataset": dataset_name,
        "n_rows": len(ds),
        "n_columns": len(ds.column_names),
        "inferred_frequency": inferred_freq,
        "inferred_domains": inferred_domains,
        "description": ds_info.get("description", "")[:200],
        "tags": ds_info.get("tags", []),
        "columns": proposals,
        "unmapped_columns": unmapped,
        "mapped_count": sum(1 for v in proposals.values() if v["status"] == "mapped"),
    }
