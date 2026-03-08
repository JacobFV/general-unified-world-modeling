"""Data collectors: download, cache, and return real-world datasets.

Each collector:
1. Downloads data from a free API or public source
2. Caches it locally (in ~/.cache/guwm/ by default)
3. Returns a (DatasetSpec, dict[str, torch.Tensor]) tuple ready for
   the heterogeneous training pipeline

Collectors gracefully degrade if optional dependencies (fredapi, yfinance)
are not installed -- they log a warning and return empty results.
"""

from __future__ import annotations

import hashlib
import warnings
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from general_unified_world_model.training.heterogeneous import DatasetSpec, FieldMapping
from general_unified_world_model.data.adapters import (
    z_score,
    minmax,
    log_return,
    pct_change,
    rank_normalize,
    FRED_MAPPINGS,
    YAHOO_EQUITY_FIELDS,
    YAHOO_FX_FIELDS,
    YAHOO_COMMODITY_FIELDS,
    YAHOO_CRYPTO_FIELDS,
)

logger = logging.getLogger(__name__)

# ── Default cache directory ──────────────────────────────────────────────

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "guwm"


def _ensure_cache_dir(cache_dir: Path) -> Path:
    """Create cache directory if it does not exist."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_key(collector_name: str, params: dict) -> str:
    """Generate a deterministic cache key from collector name and parameters."""
    raw = json.dumps({"collector": collector_name, **params}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _save_cache(cache_path: Path, data_dict: dict[str, torch.Tensor]) -> None:
    """Persist a data dict as a .pt file."""
    torch.save(data_dict, cache_path)
    logger.debug("Cached data to %s", cache_path)


def _load_cache(cache_path: Path) -> dict[str, torch.Tensor] | None:
    """Load cached data if it exists and is not stale."""
    if cache_path.exists():
        logger.debug("Loading cached data from %s", cache_path)
        return torch.load(cache_path, weights_only=False)
    return None


# ── Base Collector ───────────────────────────────────────────────────────

class BaseCollector(ABC):
    """Abstract base class for all data collectors."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        force_refresh: bool = False,
    ):
        self.cache_dir = _ensure_cache_dir(Path(cache_dir or DEFAULT_CACHE_DIR))
        self.force_refresh = force_refresh

    @abstractmethod
    def _fetch(self) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
        """Download and transform data. Implemented by subclasses."""
        ...

    @abstractmethod
    def _cache_params(self) -> dict:
        """Parameters that uniquely identify this collection run."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def collect(self) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
        """Download, cache, and return data.

        Returns:
            (DatasetSpec, dict[str, torch.Tensor]) ready for the training pipeline.
            If the data source is unavailable, returns an empty spec and dict.
        """
        cache_file = self.cache_dir / f"{self.name}_{_cache_key(self.name, self._cache_params())}.pt"

        # Try loading from cache
        if not self.force_refresh:
            cached = _load_cache(cache_file)
            if cached is not None:
                logger.info("[%s] Loaded %d series from cache", self.name, len(cached["data"]))
                spec = cached["spec"]
                data = cached["data"]
                return spec, data

        # Fetch fresh data
        logger.info("[%s] Fetching fresh data...", self.name)
        try:
            spec, data = self._fetch()
        except Exception as exc:
            logger.warning("[%s] Failed to fetch data: %s", self.name, exc)
            spec = DatasetSpec(name=self.name, mappings=[])
            data = {}
            return spec, data

        if data:
            try:
                _save_cache(cache_file, {"spec": spec, "data": data})
            except Exception as exc:
                logger.warning("[%s] Failed to cache data: %s", self.name, exc)

        logger.info("[%s] Collected %d series, %d field mappings",
                     self.name, len(data), len(spec.mappings))
        return spec, data


# ── Z-score helper for full series ───────────────────────────────────────

def _zscore_tensor(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Z-score normalize a 1-D tensor, returning (normalized, mean, std)."""
    mu = t.mean().item()
    sigma = t.std().item()
    normed = (t - mu) / max(sigma, 1e-8)
    return normed, mu, sigma


def _log_returns_tensor(t: torch.Tensor) -> torch.Tensor:
    """Compute log returns for a price series, prepending a zero."""
    if t.shape[0] < 2:
        return torch.zeros_like(t)
    returns = torch.log(t[1:] / t[:-1].clamp(min=1e-8))
    return torch.cat([torch.zeros(1, dtype=t.dtype, device=t.device), returns])


def _pct_change_tensor(t: torch.Tensor) -> torch.Tensor:
    """Compute percentage changes, prepending a zero."""
    if t.shape[0] < 2:
        return torch.zeros_like(t)
    changes = (t[1:] - t[:-1]) / t[:-1].clamp(min=1e-8)
    return torch.cat([torch.zeros(1, dtype=t.dtype, device=t.device), changes])


# ── FRED Collector ───────────────────────────────────────────────────────

# Series that represent levels/rates (z-score normalize, no transformation)
_FRED_LEVEL_SERIES = {
    "GDP", "GDPC1", "TCU", "UNRATE", "ICSA", "CCSA", "JTSJOL", "JTSQUR",
    "CIVPART", "MICH", "T5YIE", "GFDEGDQ188S", "FYFSGDA188S",
    "A091RC1Q027SBEA", "HOUST", "MORTGAGE30US", "EXHOSLUSM495S",
    "DFF", "DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y", "T10YIE",
    "BAMLC0A4CBBB", "BAMLH0A0HYM2", "BAMLC0A0CM",
    "DFEDTARU", "WALCL", "RRPONTSYD", "WTREGEN", "TOTRESNS",
    "UMCSENT", "USSLIND",
}

# Series where percentage change is the natural representation
_FRED_PCT_SERIES = {
    "INDPRO", "RSAFS", "DGORDER",
    "CPIAUCSL", "CPILFESL", "PCEPI", "PPIFIS",
    "CES0500000003", "CUSR0000SEHA",
    "PAYEMS",
    "CSUSHPINSA",
}

# Additional FRED mappings not in adapters.py
_EXTRA_FRED_MAPPINGS = {
    "BAMLC0A0CM": ("financial.credit.ig_spread", 1, None),
}


class FREDCollector(BaseCollector):
    """Downloads macroeconomic data from the FRED API.

    Fetches GDP, CPI, unemployment, industrial production, retail sales,
    housing starts, yield curves, credit spreads, and more.  Data is split
    into monthly and daily buckets.  Each series is normalized:
      - Level/rate series: z-score normalization
      - Growth/change series: percentage change then z-score
      - Yield/spread series: z-score normalization

    Requires the ``fredapi`` package and a FRED API key (free from
    https://fred.stlouisfed.org/docs/api/api_key.html).
    """

    def __init__(
        self,
        api_key: str | None = None,
        series_ids: list[str] | None = None,
        start_date: str = "2000-01-01",
        end_date: str | None = None,
        cache_dir: str | Path | None = None,
        force_refresh: bool = False,
    ):
        super().__init__(cache_dir=cache_dir, force_refresh=force_refresh)
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.series_ids = series_ids
        self.start_date = start_date
        self.end_date = end_date

    @property
    def name(self) -> str:
        return "FRED"

    def _cache_params(self) -> dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date or "latest",
            "series": sorted(self.series_ids) if self.series_ids else "all",
        }

    def _fetch(self) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
        try:
            from fredapi import Fred
        except ImportError:
            logger.warning("fredapi not installed. Install with: pip install fredapi")
            return DatasetSpec(name=self.name, mappings=[]), {}

        if not self.api_key:
            logger.warning("No FRED API key. Set FRED_API_KEY env var or pass api_key.")
            return DatasetSpec(name=self.name, mappings=[]), {}

        fred = Fred(api_key=self.api_key)

        # Merge standard and extra mappings
        all_mappings = dict(FRED_MAPPINGS)
        all_mappings.update(_EXTRA_FRED_MAPPINGS)

        if self.series_ids is not None:
            target_series = [s for s in self.series_ids if s in all_mappings]
        else:
            target_series = list(all_mappings.keys())

        field_mappings: list[FieldMapping] = []
        data_dict: dict[str, torch.Tensor] = {}

        for sid in target_series:
            target_field, period, adapter_transform = all_mappings[sid]
            try:
                logger.debug("[FRED] Fetching %s -> %s", sid, target_field)
                series = fred.get_series(
                    sid,
                    observation_start=self.start_date,
                    observation_end=self.end_date,
                )
                series = series.dropna()
                if len(series) == 0:
                    logger.debug("[FRED] %s returned empty series, skipping", sid)
                    continue

                raw = torch.tensor(series.values, dtype=torch.float32)

                # Apply normalization based on series type
                if sid in _FRED_PCT_SERIES:
                    # Compute pct change first, then z-score
                    transformed = _pct_change_tensor(raw)
                    mu = transformed.mean().item()
                    sigma = transformed.std().item()
                    normalized = (transformed - mu) / max(sigma, 1e-8)
                    transform_fn = None  # already normalized
                else:
                    # z-score the raw level
                    normalized, mu, sigma = _zscore_tensor(raw)
                    transform_fn = None

                data_dict[sid] = normalized

                field_mappings.append(FieldMapping(
                    source_key=sid,
                    target_field=target_field,
                    transform=transform_fn,
                    frequency=period,
                ))

            except Exception as exc:
                logger.debug("[FRED] Failed to fetch %s: %s", sid, exc)
                continue

        spec = DatasetSpec(
            name="FRED",
            mappings=field_mappings,
            base_period=1,
            weight=1.0,
        )

        return spec, data_dict


# ── Yahoo Finance Collector ──────────────────────────────────────────────

class YahooFinanceCollector(BaseCollector):
    """Downloads daily market data from Yahoo Finance.

    Covers major equity indices (SPX, IXIC, DJI), Treasury yields, VIX,
    major FX pairs, commodities (oil, gold, copper), and optionally
    firm-specific tickers.

    Price series are transformed to log returns then z-scored. VIX and
    spreads are z-scored directly.

    Requires the ``yfinance`` package.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        firm_tickers: dict[str, str] | None = None,
        start_date: str = "2000-01-01",
        end_date: str | None = None,
        include_equity: bool = True,
        include_fx: bool = True,
        include_commodities: bool = True,
        include_crypto: bool = True,
        cache_dir: str | Path | None = None,
        force_refresh: bool = False,
    ):
        super().__init__(cache_dir=cache_dir, force_refresh=force_refresh)
        self.tickers = tickers
        self.firm_tickers = firm_tickers or {}
        self.start_date = start_date
        self.end_date = end_date
        self.include_equity = include_equity
        self.include_fx = include_fx
        self.include_commodities = include_commodities
        self.include_crypto = include_crypto

    @property
    def name(self) -> str:
        return "YahooFinance"

    def _cache_params(self) -> dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date or "latest",
            "tickers": sorted(self.tickers) if self.tickers else "auto",
            "firm_tickers": sorted(self.firm_tickers.keys()) if self.firm_tickers else [],
            "equity": self.include_equity,
            "fx": self.include_fx,
            "commodities": self.include_commodities,
            "crypto": self.include_crypto,
        }

    def _fetch(self) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
            return DatasetSpec(name=self.name, mappings=[]), {}

        # Build ticker -> (field_path, sub_idx) map
        all_fields: dict[str, tuple[str, int | None]] = {}
        if self.include_equity:
            all_fields.update(YAHOO_EQUITY_FIELDS)
        if self.include_fx:
            all_fields.update(YAHOO_FX_FIELDS)
        if self.include_commodities:
            all_fields.update(YAHOO_COMMODITY_FIELDS)
        if self.include_crypto:
            all_fields.update(YAHOO_CRYPTO_FIELDS)

        # Filter by explicit tickers if specified
        if self.tickers is not None:
            all_fields = {k: v for k, v in all_fields.items() if k in self.tickers}

        # Add firm-specific tickers
        for ticker, firm_prefix in self.firm_tickers.items():
            all_fields[ticker] = (f"{firm_prefix}.market.equity_price", None)

        ticker_list = list(all_fields.keys())
        if not ticker_list:
            return DatasetSpec(name=self.name, mappings=[]), {}

        # Download all tickers at once
        logger.info("[YahooFinance] Downloading %d tickers...", len(ticker_list))
        try:
            data = yf.download(
                ticker_list,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            logger.warning("[YahooFinance] Download failed: %s", exc)
            return DatasetSpec(name=self.name, mappings=[]), {}

        if data.empty:
            logger.warning("[YahooFinance] Downloaded data is empty")
            return DatasetSpec(name=self.name, mappings=[]), {}

        field_mappings: list[FieldMapping] = []
        data_dict: dict[str, torch.Tensor] = {}

        for ticker in ticker_list:
            target_field, sub_idx = all_fields[ticker]

            try:
                # Handle multi-ticker vs single-ticker DataFrame structure
                if len(ticker_list) > 1:
                    close = data["Close"][ticker].dropna()
                else:
                    close = data["Close"].dropna()

                if len(close) == 0:
                    logger.debug("[YahooFinance] %s has no data, skipping", ticker)
                    continue

                raw = torch.tensor(close.values, dtype=torch.float32)

                # VIX is already a volatility measure -- z-score it directly
                if ticker == "^VIX":
                    normalized, _, _ = _zscore_tensor(raw)
                else:
                    # Compute log returns then z-score
                    log_ret = _log_returns_tensor(raw)
                    normalized, _, _ = _zscore_tensor(log_ret)

                key = f"yahoo_{ticker}"
                data_dict[key] = normalized

                field_mappings.append(FieldMapping(
                    source_key=key,
                    target_field=target_field,
                    transform=None,  # already normalized
                    frequency=1,     # daily
                ))

                logger.debug("[YahooFinance] %s -> %s (%d points)",
                             ticker, target_field, len(normalized))

            except (KeyError, TypeError) as exc:
                logger.debug("[YahooFinance] Failed to process %s: %s", ticker, exc)
                continue

        spec = DatasetSpec(
            name="Yahoo Finance",
            mappings=field_mappings,
            base_period=16,   # daily = period 16 in base ticks
            weight=1.0,
        )

        return spec, data_dict


# ── Synthetic Collector ──────────────────────────────────────────────────

# Groups of field paths for generating correlated synthetic data.
# Within each group, the synthetic signals share a latent factor so
# that the data has realistic cross-field correlations.

_MACRO_FIELDS = [
    ("country_us.macro.output.gdp_nowcast", 16),
    ("country_us.macro.output.industrial_production", 192),
    ("country_us.macro.output.capacity_utilization", 192),
    ("country_us.macro.output.retail_sales", 192),
    ("country_us.macro.output.pmi_manufacturing", 192),
    ("country_us.macro.output.pmi_services", 192),
    ("country_us.macro.inflation.headline_cpi", 192),
    ("country_us.macro.inflation.core_cpi", 192),
    ("country_us.macro.inflation.pce_deflator", 192),
    ("country_us.macro.inflation.wage_growth", 192),
    ("country_us.macro.inflation.expectations_1y", 192),
    ("country_us.macro.inflation.expectations_5y", 192),
    ("country_us.macro.labor.unemployment_rate", 192),
    ("country_us.macro.labor.nfp_change", 192),
    ("country_us.macro.labor.initial_claims", 48),
    ("country_us.macro.labor.job_openings", 192),
    ("country_us.macro.labor.lfpr", 192),
    ("country_us.macro.fiscal.debt_to_gdp", 576),
    ("country_us.macro.fiscal.deficit_to_gdp", 576),
    ("country_us.macro.housing.home_price_index", 192),
    ("country_us.macro.housing.housing_starts", 192),
    ("country_us.macro.housing.mortgage_rate", 16),
    ("country_us.domestic_sentiment", 192),
]

_FINANCIAL_FIELDS = [
    ("financial.yield_curves.short_rate", 1),
    ("financial.yield_curves.two_year", 1),
    ("financial.yield_curves.five_year", 1),
    ("financial.yield_curves.ten_year", 1),
    ("financial.yield_curves.thirty_year", 1),
    ("financial.yield_curves.slope_2s10s", 1),
    ("financial.yield_curves.breakeven_inflation", 1),
    ("financial.credit.ig_spread", 1),
    ("financial.credit.hy_spread", 1),
    ("financial.equities.broad_indices", 1),
    ("financial.equities.vix", 1),
    ("financial.fx.dxy", 1),
    ("financial.fx.eurusd", 1),
    ("financial.fx.usdjpy", 1),
    ("financial.liquidity.fed_reverse_repo", 16),
    ("financial.liquidity.bank_reserves", 48),
    ("financial.central_banks.policy_rate", 192),
    ("financial.central_banks.balance_sheet_size", 48),
]

_COMMODITY_FIELDS = [
    ("resources.energy.crude_price", 1),
    ("resources.energy.natgas_price", 1),
    ("resources.metals.gold", 1),
    ("resources.metals.silver", 1),
    ("resources.metals.copper", 1),
]

_NARRATIVE_FIELDS = [
    ("narratives.media.crisis_framing", 4),
    ("narratives.media.econ_doom_vs_boom", 16),
    ("narratives.media.geopolitical_fear", 16),
    ("narratives.public.consumer_confidence", 192),
    ("narratives.public.economic_anxiety", 48),
    ("narratives.elites.ceo_confidence", 192),
    ("narratives.positioning.equity_fund_flows", 48),
    ("narratives.positioning.bond_fund_flows", 48),
    ("narratives.positioning.retail_sentiment", 16),
]

_REGIME_FIELDS = [
    ("regime.growth_regime", 576),
    ("regime.inflation_regime", 576),
    ("regime.financial_cycle", 576),
    ("regime.liquidity_regime", 192),
    ("regime.fragility", 192),
    ("regime.compressed_world_state", 192),
]

_EVENT_FIELDS = [
    ("events.news_embedding", 1),
    ("events.social_signal", 1),
]


def _generate_correlated_group(
    fields: list[tuple[str, int]],
    n_timesteps: int,
    n_latent_factors: int = 3,
    noise_ratio: float = 0.3,
    rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Generate correlated synthetic data for a group of fields.

    Each field is a linear combination of shared latent factors plus
    idiosyncratic noise, producing realistic cross-correlations.

    Args:
        fields: List of (field_path, period) tuples.
        n_timesteps: Number of time steps to generate.
        n_latent_factors: Number of shared latent factors.
        noise_ratio: Fraction of variance from idiosyncratic noise.
        rng: NumPy random generator for reproducibility.

    Returns:
        Dict mapping field_path -> normalized tensor of shape (n_timesteps,).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_fields = len(fields)
    if n_fields == 0:
        return {}

    # Generate latent factors as auto-correlated random walks
    factors = np.zeros((n_timesteps, n_latent_factors))
    for f in range(n_latent_factors):
        # AR(1) process with different persistence
        persistence = 0.95 + 0.04 * rng.random()
        innovations = rng.standard_normal(n_timesteps) * 0.1
        for t in range(1, n_timesteps):
            factors[t, f] = persistence * factors[t - 1, f] + innovations[t]

    # Generate factor loadings -- each field loads differently on the factors
    loadings = rng.standard_normal((n_fields, n_latent_factors))

    # Compute signals (suppress numpy 2.x matmul warnings with zero rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        signals = factors @ loadings.T  # (n_timesteps, n_fields)

    # Replace any NaN/inf from numerical edge cases with zero
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

    data = {}
    for i, (field_path, period) in enumerate(fields):
        signal = signals[:, i].copy()

        # Add idiosyncratic noise
        noise = rng.standard_normal(n_timesteps) * noise_ratio
        signal += noise

        # For low-frequency fields, apply smoothing to respect the period
        if period > 1:
            kernel_size = min(period // 2, n_timesteps // 4, 20)
            if kernel_size > 1:
                kernel = np.ones(kernel_size) / kernel_size
                signal = np.convolve(signal, kernel, mode="same")

        # Z-score normalize
        mu = signal.mean()
        sigma = signal.std()
        if sigma > 1e-8:
            signal = (signal - mu) / sigma

        data[field_path] = torch.tensor(signal, dtype=torch.float32)

    return data


class SyntheticCollector(BaseCollector):
    """Generates synthetic but structured data for smoke testing.

    Produces correlated random data for all major field groups:
    macro, financial, commodity, narrative, regime, and events.
    Within each group, fields share latent factors so cross-correlations
    are realistic (e.g., GDP and industrial production co-move).

    No API keys or network access needed. Returns short sequences
    (default 100 timesteps) suitable for pipeline validation.
    """

    def __init__(
        self,
        n_timesteps: int = 100,
        seed: int = 42,
        cache_dir: str | Path | None = None,
        force_refresh: bool = False,
    ):
        super().__init__(cache_dir=cache_dir, force_refresh=force_refresh)
        self.n_timesteps = n_timesteps
        self.seed = seed

    @property
    def name(self) -> str:
        return "Synthetic"

    def _cache_params(self) -> dict:
        return {
            "n_timesteps": self.n_timesteps,
            "seed": self.seed,
        }

    def _fetch(self) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed)

        all_data: dict[str, torch.Tensor] = {}
        all_field_info: list[tuple[str, int]] = []

        # Generate each correlated group with its own latent factors,
        # but cross-group correlations are weaker (only via the seed).
        groups = [
            ("macro", _MACRO_FIELDS, 4),
            ("financial", _FINANCIAL_FIELDS, 3),
            ("commodity", _COMMODITY_FIELDS, 2),
            ("narrative", _NARRATIVE_FIELDS, 2),
            ("regime", _REGIME_FIELDS, 2),
            ("events", _EVENT_FIELDS, 1),
        ]

        for group_name, fields, n_factors in groups:
            logger.debug("[Synthetic] Generating %s group (%d fields, %d factors)",
                         group_name, len(fields), n_factors)
            group_data = _generate_correlated_group(
                fields,
                n_timesteps=self.n_timesteps,
                n_latent_factors=n_factors,
                noise_ratio=0.3,
                rng=rng,
            )
            all_data.update(group_data)
            all_field_info.extend(fields)

        # Build field mappings -- source_key == target_field for synthetic
        field_mappings = []
        for field_path, period in all_field_info:
            if field_path in all_data:
                field_mappings.append(FieldMapping(
                    source_key=field_path,
                    target_field=field_path,
                    transform=None,  # already z-scored
                    frequency=period,
                ))

        spec = DatasetSpec(
            name="Synthetic",
            mappings=field_mappings,
            base_period=1,
            weight=0.5,  # lower weight -- it's synthetic
        )

        logger.info("[Synthetic] Generated %d series, %d timesteps each",
                     len(all_data), self.n_timesteps)
        return spec, all_data


# ── Convenience function ─────────────────────────────────────────────────

def collect_all(
    api_keys: dict | None = None,
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    firm_tickers: dict[str, str] | None = None,
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
    include_synthetic: bool = True,
) -> list[tuple[DatasetSpec, dict[str, torch.Tensor]]]:
    """Run all available collectors and return their results.

    This convenience function instantiates each collector with the
    provided parameters, runs ``collect()``, and aggregates the
    results.  Collectors that fail (missing API keys, missing packages)
    are skipped with a warning.

    Args:
        api_keys: Dict of API keys. Recognized keys:
            - ``"fred"``: FRED API key
        start_date: Start date for historical data (YYYY-MM-DD).
        end_date: End date (None = today).
        firm_tickers: Dict mapping ticker symbols to world model firm
            prefixes for Yahoo Finance. E.g. ``{"AAPL": "firm_alpha"}``.
        cache_dir: Directory for caching downloaded data.
        force_refresh: If True, bypass cache and re-download.
        include_synthetic: If True, include SyntheticCollector output.

    Returns:
        List of (DatasetSpec, dict[str, torch.Tensor]) tuples, one per
        collector that returned data.
    """
    api_keys = api_keys or {}
    results: list[tuple[DatasetSpec, dict[str, torch.Tensor]]] = []

    # 1. FRED Collector
    logger.info("Running FREDCollector...")
    fred = FREDCollector(
        api_key=api_keys.get("fred"),
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    spec, data = fred.collect()
    if data:
        results.append((spec, data))
    else:
        logger.info("FREDCollector returned no data (missing key or fredapi?)")

    # 2. Yahoo Finance Collector
    logger.info("Running YahooFinanceCollector...")
    yahoo = YahooFinanceCollector(
        firm_tickers=firm_tickers,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    spec, data = yahoo.collect()
    if data:
        results.append((spec, data))
    else:
        logger.info("YahooFinanceCollector returned no data (missing yfinance?)")

    # 3. Synthetic Collector (always available)
    if include_synthetic:
        logger.info("Running SyntheticCollector...")
        synth = SyntheticCollector(
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
        spec, data = synth.collect()
        if data:
            results.append((spec, data))

    logger.info("collect_all complete: %d sources with data", len(results))
    return results
