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

from general_unified_world_model.training.heterogeneous import DatasetSpec, InputSpec, OutputSpec, _infer_semantic_type
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
from general_unified_world_model.schema.temporal_constants import (
    TICK, HOURLY, DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL, DECADAL,
)

logger = logging.getLogger(__name__)


def _io_pair(key: str, field_path: str, transform=None, frequency=None):
    """Create matched InputSpec + OutputSpec for a field mapping."""
    st = _infer_semantic_type(field_path)
    return (
        InputSpec(key=key, semantic_type=st, field_path=field_path, transform=transform, frequency=frequency),
        OutputSpec(key=key, semantic_type=st, field_path=field_path, frequency=frequency),
    )


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
            spec = DatasetSpec(name=self.name)
            data = {}
            return spec, data

        if data:
            try:
                _save_cache(cache_file, {"spec": spec, "data": data})
            except Exception as exc:
                logger.warning("[%s] Failed to cache data: %s", self.name, exc)

        logger.info("[%s] Collected %d series, %d input specs",
                     self.name, len(data), len(spec.input_specs))
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
    "BAMLC0A0CM": ("financial.credit.ig_spread", TICK, None),
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
            return DatasetSpec(name=self.name), {}

        if not self.api_key:
            logger.warning("No FRED API key. Set FRED_API_KEY env var or pass api_key.")
            return DatasetSpec(name=self.name), {}

        fred = Fred(api_key=self.api_key)

        # Merge standard and extra mappings
        all_mappings = dict(FRED_MAPPINGS)
        all_mappings.update(_EXTRA_FRED_MAPPINGS)

        if self.series_ids is not None:
            target_series = [s for s in self.series_ids if s in all_mappings]
        else:
            target_series = list(all_mappings.keys())

        input_specs: list[InputSpec] = []
        output_specs: list[OutputSpec] = []
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

                inp, out = _io_pair(sid, target_field, transform=transform_fn, frequency=period)
                input_specs.append(inp)
                output_specs.append(out)

            except Exception as exc:
                logger.debug("[FRED] Failed to fetch %s: %s", sid, exc)
                continue

        spec = DatasetSpec(
            name="FRED",
            input_specs=input_specs,
            output_specs=output_specs,
            base_period=TICK,
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
            return DatasetSpec(name=self.name), {}

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
            return DatasetSpec(name=self.name), {}

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
            return DatasetSpec(name=self.name), {}

        if data.empty:
            logger.warning("[YahooFinance] Downloaded data is empty")
            return DatasetSpec(name=self.name), {}

        input_specs: list[InputSpec] = []
        output_specs: list[OutputSpec] = []
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

                inp, out = _io_pair(key, target_field, transform=None, frequency=TICK)
                input_specs.append(inp)
                output_specs.append(out)

                logger.debug("[YahooFinance] %s -> %s (%d points)",
                             ticker, target_field, len(normalized))

            except (KeyError, TypeError) as exc:
                logger.debug("[YahooFinance] Failed to process %s: %s", ticker, exc)
                continue

        spec = DatasetSpec(
            name="Yahoo Finance",
            input_specs=input_specs,
            output_specs=output_specs,
            base_period=DAILY,   # daily = period 16 in base ticks
            weight=1.0,
        )

        return spec, data_dict


# ── Synthetic Collector ──────────────────────────────────────────────────

# Groups of field paths for generating correlated synthetic data.
# Within each group, the synthetic signals share a latent factor so
# that the data has realistic cross-field correlations.

_MACRO_FIELDS = [
    ("country_us.macro.output.gdp_nowcast", DAILY),
    ("country_us.macro.output.industrial_production", MONTHLY),
    ("country_us.macro.output.capacity_utilization", MONTHLY),
    ("country_us.macro.output.retail_sales", MONTHLY),
    ("country_us.macro.output.pmi_manufacturing", MONTHLY),
    ("country_us.macro.output.pmi_services", MONTHLY),
    ("country_us.macro.inflation.headline_cpi", MONTHLY),
    ("country_us.macro.inflation.core_cpi", MONTHLY),
    ("country_us.macro.inflation.pce_deflator", MONTHLY),
    ("country_us.macro.inflation.wage_growth", MONTHLY),
    ("country_us.macro.inflation.expectations_1y", MONTHLY),
    ("country_us.macro.inflation.expectations_5y", MONTHLY),
    ("country_us.macro.labor.unemployment_rate", MONTHLY),
    ("country_us.macro.labor.nfp_change", MONTHLY),
    ("country_us.macro.labor.initial_claims", WEEKLY),
    ("country_us.macro.labor.job_openings", MONTHLY),
    ("country_us.macro.labor.lfpr", MONTHLY),
    ("country_us.macro.fiscal.debt_to_gdp", QUARTERLY),
    ("country_us.macro.fiscal.deficit_to_gdp", QUARTERLY),
    ("country_us.macro.housing.home_price_index", MONTHLY),
    ("country_us.macro.housing.housing_starts", MONTHLY),
    ("country_us.macro.housing.mortgage_rate", DAILY),
    ("country_us.domestic_sentiment", MONTHLY),
]

_FINANCIAL_FIELDS = [
    ("financial.yield_curves.short_rate", TICK),
    ("financial.yield_curves.two_year", TICK),
    ("financial.yield_curves.five_year", TICK),
    ("financial.yield_curves.ten_year", TICK),
    ("financial.yield_curves.thirty_year", TICK),
    ("financial.yield_curves.slope_2s10s", TICK),
    ("financial.yield_curves.breakeven_inflation", TICK),
    ("financial.credit.ig_spread", TICK),
    ("financial.credit.hy_spread", TICK),
    ("financial.equities.broad_indices", TICK),
    ("financial.equities.vix", TICK),
    ("financial.fx.dxy", TICK),
    ("financial.fx.eurusd", TICK),
    ("financial.fx.usdjpy", TICK),
    ("financial.liquidity.fed_reverse_repo", DAILY),
    ("financial.liquidity.bank_reserves", WEEKLY),
    ("financial.central_banks.policy_rate", MONTHLY),
    ("financial.central_banks.balance_sheet_size", WEEKLY),
]

_COMMODITY_FIELDS = [
    ("resources.energy.crude_price", TICK),
    ("resources.energy.natgas_price", TICK),
    ("resources.metals.gold", TICK),
    ("resources.metals.silver", TICK),
    ("resources.metals.copper", TICK),
]

_NARRATIVE_FIELDS = [
    ("narratives.media.crisis_framing", HOURLY),
    ("narratives.media.econ_doom_vs_boom", DAILY),
    ("narratives.media.geopolitical_fear", DAILY),
    ("narratives.public.consumer_confidence", MONTHLY),
    ("narratives.public.economic_anxiety", WEEKLY),
    ("narratives.elites.ceo_confidence", MONTHLY),
    ("narratives.positioning.equity_fund_flows", WEEKLY),
    ("narratives.positioning.bond_fund_flows", WEEKLY),
    ("narratives.positioning.retail_sentiment", DAILY),
]

_REGIME_FIELDS = [
    ("regime.growth_regime", QUARTERLY),
    ("regime.inflation_regime", QUARTERLY),
    ("regime.financial_cycle", QUARTERLY),
    ("regime.liquidity_regime", MONTHLY),
    ("regime.fragility", MONTHLY),
    ("regime.compressed_world_state", MONTHLY),
]

_EVENT_FIELDS = [
    ("events.news_embedding", TICK),
    ("events.social_signal", TICK),
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

        # Build input/output specs -- source_key == target_field for synthetic
        input_specs: list[InputSpec] = []
        output_specs: list[OutputSpec] = []
        for field_path, period in all_field_info:
            if field_path in all_data:
                inp, out = _io_pair(field_path, field_path, transform=None, frequency=period)
                input_specs.append(inp)
                output_specs.append(out)

        spec = DatasetSpec(
            name="Synthetic",
            input_specs=input_specs,
            output_specs=output_specs,
            base_period=TICK,
            weight=0.5,  # lower weight -- it's synthetic
        )

        logger.info("[Synthetic] Generated %d series, %d timesteps each",
                     len(all_data), self.n_timesteps)
        return spec, all_data


# ── World Bank Collector ──────────────────────────────────────────────────

# World Bank indicator code -> (target_field_suffix, frequency)
# The target_field will be prefixed with "country_{code}." at runtime.
_WORLD_BANK_INDICATORS: dict[str, tuple[str, int]] = {
    "NY.GDP.MKTP.KD.ZG": ("macro.output.gdp_nowcast", QUARTERLY),
    "FP.CPI.TOTL.ZG":    ("macro.inflation.headline_cpi", QUARTERLY),
    "SP.POP.TOTL":        ("demographics.population_growth", ANNUAL),
    "SP.DYN.LE00.IN":     ("demographics.life_expectancy", ANNUAL),
    "EN.ATM.CO2E.PC":     ("macro.trade.terms_of_trade", ANNUAL),  # proxy: CO2 per capita
    "NE.TRD.GNFS.ZS":    ("macro.trade.trade_balance", QUARTERLY),
    "BX.KLT.DINV.WD.GD.ZS": ("macro.trade.fdi_flows", ANNUAL),
    "MS.MIL.XPND.GD.ZS": ("macro.fiscal.spending_composition", ANNUAL),
    "IT.NET.USER.ZS":     ("demographics.education_attainment", ANNUAL),  # proxy: internet
    "EG.FEC.RNEW.ZS":     ("macro.output.capacity_utilization", ANNUAL),  # proxy: renewables
}

# World Bank API country codes
_WORLD_BANK_COUNTRIES = {
    "US": "USA",
    "CN": "CHN",
    "EU": "EUU",  # European Union aggregate
    "JP": "JPN",
    "UK": "GBR",
    "IN": "IND",
    "BR": "BRA",
}


class WorldBankCollector(BaseCollector):
    """Downloads macroeconomic data from the World Bank API.

    The World Bank provides free access (no API key required) to a vast
    collection of development indicators including GDP growth, inflation,
    population, life expectancy, CO2 emissions, trade, FDI, military
    spending, internet usage, and renewable energy shares.

    Data is available at annual or quarterly frequency depending on the
    indicator.  All series are z-score normalized.

    Uses the v2 JSON API: ``api.worldbank.org/v2/country/{code}/indicator/{indicator}``
    """

    def __init__(
        self,
        countries: dict[str, str] | None = None,
        indicators: dict[str, tuple[str, int]] | None = None,
        start_year: int = 2000,
        end_year: int | None = None,
        cache_dir: str | Path | None = None,
        force_refresh: bool = False,
    ):
        super().__init__(cache_dir=cache_dir, force_refresh=force_refresh)
        self.countries = countries or _WORLD_BANK_COUNTRIES
        self.indicators = indicators or _WORLD_BANK_INDICATORS
        self.start_year = start_year
        self.end_year = end_year

    @property
    def name(self) -> str:
        return "WorldBank"

    def _cache_params(self) -> dict:
        return {
            "countries": sorted(self.countries.keys()),
            "indicators": sorted(self.indicators.keys()),
            "start_year": self.start_year,
            "end_year": self.end_year or "latest",
        }

    def _fetch(self) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed. Install with: pip install requests")
            return DatasetSpec(name=self.name), {}

        input_specs: list[InputSpec] = []
        output_specs: list[OutputSpec] = []
        data_dict: dict[str, torch.Tensor] = {}

        date_range = f"{self.start_year}:{self.end_year or 2026}"

        for country_key, wb_code in self.countries.items():
            country_prefix = f"country_{country_key.lower()}"

            for indicator_code, (field_suffix, frequency) in self.indicators.items():
                source_key = f"wb_{country_key}_{indicator_code}"
                target_field = f"{country_prefix}.{field_suffix}"

                try:
                    url = (
                        f"https://api.worldbank.org/v2/country/{wb_code}"
                        f"/indicator/{indicator_code}"
                    )
                    params = {
                        "format": "json",
                        "date": date_range,
                        "per_page": 500,
                    }
                    logger.debug("[WorldBank] Fetching %s for %s", indicator_code, wb_code)
                    resp = requests.get(url, params=params, timeout=30)
                    resp.raise_for_status()

                    payload = resp.json()
                    # World Bank returns [metadata, data_array]
                    if (
                        not isinstance(payload, list)
                        or len(payload) < 2
                        or payload[1] is None
                    ):
                        logger.debug(
                            "[WorldBank] No data for %s / %s", wb_code, indicator_code
                        )
                        continue

                    records = payload[1]
                    # Filter out null values and sort by year ascending
                    valid = [
                        (int(r["date"]), float(r["value"]))
                        for r in records
                        if r.get("value") is not None
                    ]
                    if not valid:
                        continue

                    valid.sort(key=lambda x: x[0])
                    values = [v for _, v in valid]

                    raw = torch.tensor(values, dtype=torch.float32)
                    normalized, _, _ = _zscore_tensor(raw)

                    data_dict[source_key] = normalized
                    inp, out = _io_pair(source_key, target_field, transform=None, frequency=frequency)
                    input_specs.append(inp)
                    output_specs.append(out)

                    logger.debug(
                        "[WorldBank] %s -> %s (%d points)",
                        source_key, target_field, len(normalized),
                    )

                except Exception as exc:
                    logger.debug(
                        "[WorldBank] Failed %s / %s: %s",
                        wb_code, indicator_code, exc,
                    )
                    continue

        spec = DatasetSpec(
            name="World Bank",
            input_specs=input_specs,
            output_specs=output_specs,
            base_period=ANNUAL,
            weight=0.8,
        )

        return spec, data_dict


# ── NOAA Climate Collector ───────────────────────────────────────────────

# NOAA CDO dataset IDs and the world model fields they map to.
_NOAA_DATASETS: list[tuple[str, str, str, int]] = [
    # (dataset_id, data_type_id, target_field, frequency)
    ("GSOM", "TAVG", "physical.climate.global_temp_anomaly", ANNUAL),
    ("GSOM", "PRCP", "physical.climate.extreme_weather_freq", QUARTERLY),
]


class NOAAClimateCollector(BaseCollector):
    """Downloads climate data from the NOAA Climate Data Online API.

    Provides global temperature anomaly, precipitation patterns, and
    other climate observations.  Requires a free NOAA API key
    (https://www.ncdc.noaa.gov/cdo-web/token).

    If the API key is not available, the collector gracefully returns
    empty results without raising an error.

    Uses the CDO v2 API: ``https://www.ncdc.noaa.gov/cdo-web/api/v2/``
    """

    def __init__(
        self,
        api_key: str | None = None,
        start_date: str = "2000-01-01",
        end_date: str | None = None,
        cache_dir: str | Path | None = None,
        force_refresh: bool = False,
    ):
        super().__init__(cache_dir=cache_dir, force_refresh=force_refresh)
        self.api_key = api_key or os.environ.get("NOAA_API_KEY")
        self.start_date = start_date
        self.end_date = end_date

    @property
    def name(self) -> str:
        return "NOAAClimate"

    def _cache_params(self) -> dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date or "latest",
        }

    def _fetch(self) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed. Install with: pip install requests")
            return DatasetSpec(name=self.name), {}

        if not self.api_key:
            logger.warning(
                "No NOAA API key. Set NOAA_API_KEY env var or pass api_key. "
                "Get a free key at https://www.ncdc.noaa.gov/cdo-web/token"
            )
            return DatasetSpec(name=self.name), {}

        input_specs: list[InputSpec] = []
        output_specs: list[OutputSpec] = []
        data_dict: dict[str, torch.Tensor] = {}

        headers = {"token": self.api_key}
        base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

        for dataset_id, datatype_id, target_field, frequency in _NOAA_DATASETS:
            source_key = f"noaa_{dataset_id}_{datatype_id}"

            try:
                params = {
                    "datasetid": dataset_id,
                    "datatypeid": datatype_id,
                    "startdate": self.start_date,
                    "enddate": self.end_date or "2026-01-01",
                    "units": "metric",
                    "limit": 1000,
                    "sortfield": "date",
                    "sortorder": "asc",
                    # Use GHCND:USW00094728 (NYC Central Park) as a representative
                    # station for global trends when no station is specified
                    "locationid": "FIPS:US",
                }

                logger.debug("[NOAA] Fetching %s / %s", dataset_id, datatype_id)
                resp = requests.get(
                    base_url, headers=headers, params=params, timeout=30
                )
                resp.raise_for_status()

                payload = resp.json()
                results = payload.get("results", [])
                if not results:
                    logger.debug("[NOAA] No results for %s / %s", dataset_id, datatype_id)
                    continue

                # Aggregate by year-month, taking the mean of all stations
                from collections import defaultdict

                monthly: dict[str, list[float]] = defaultdict(list)
                for record in results:
                    date_key = record["date"][:7]  # YYYY-MM
                    monthly[date_key].append(float(record["value"]))

                sorted_months = sorted(monthly.keys())
                values = [
                    sum(monthly[m]) / len(monthly[m]) for m in sorted_months
                ]

                if not values:
                    continue

                raw = torch.tensor(values, dtype=torch.float32)
                normalized, _, _ = _zscore_tensor(raw)

                data_dict[source_key] = normalized
                inp, out = _io_pair(source_key, target_field, transform=None, frequency=frequency)
                input_specs.append(inp)
                output_specs.append(out)

                logger.debug(
                    "[NOAA] %s -> %s (%d points)",
                    source_key, target_field, len(normalized),
                )

            except Exception as exc:
                logger.debug(
                    "[NOAA] Failed %s / %s: %s", dataset_id, datatype_id, exc
                )
                continue

        # Additionally, try fetching CO2 and sea level data from public
        # NOAA sources (these use different endpoints and need no CDO key,
        # but we attempt them only when the CDO key is valid to keep the
        # collector consistent).
        self._fetch_co2(data_dict, input_specs, output_specs)
        self._fetch_sea_level(data_dict, input_specs, output_specs)

        spec = DatasetSpec(
            name="NOAA Climate",
            input_specs=input_specs,
            output_specs=output_specs,
            base_period=ANNUAL,
            weight=0.6,
        )

        return spec, data_dict

    def _fetch_co2(
        self,
        data_dict: dict[str, torch.Tensor],
        input_specs: list[InputSpec],
        output_specs: list[OutputSpec],
    ) -> None:
        """Fetch Mauna Loa CO2 data from NOAA GML (public, no key)."""
        try:
            import requests

            url = (
                "https://gml.noaa.gov/webdata/ccgg/trends/co2/"
                "co2_annmean_mlo.csv"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            lines = resp.text.strip().split("\n")
            values = []
            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        year = int(parts[0].strip())
                        co2 = float(parts[1].strip())
                        min_year = int(self.start_date.split("-")[0])
                        if year >= min_year:
                            values.append(co2)
                    except (ValueError, IndexError):
                        continue

            if values:
                raw = torch.tensor(values, dtype=torch.float32)
                normalized, _, _ = _zscore_tensor(raw)
                data_dict["noaa_co2_annual"] = normalized
                inp, out = _io_pair("noaa_co2_annual", "physical.climate.carbon_ppm", transform=None, frequency=ANNUAL)
                input_specs.append(inp)
                output_specs.append(out)
                logger.debug("[NOAA] CO2 annual: %d points", len(normalized))

        except Exception as exc:
            logger.debug("[NOAA] Failed to fetch CO2 data: %s", exc)

    def _fetch_sea_level(
        self,
        data_dict: dict[str, torch.Tensor],
        input_specs: list[InputSpec],
        output_specs: list[OutputSpec],
    ) -> None:
        """Fetch global mean sea level data from NOAA (public, no key)."""
        try:
            import requests

            # NOAA Laboratory for Satellite Altimetry — global MSL
            url = (
                "https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/"
                "slr/slr_sla_gbl_free_all_66.csv"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            lines = resp.text.strip().split("\n")
            values = []
            for line in lines:
                if line.startswith("#") or line.startswith("year") or not line.strip():
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        values.append(float(parts[1].strip()))
                    except (ValueError, IndexError):
                        continue

            if values:
                # Subsample to annual means if we have many data points
                if len(values) > 100:
                    chunk_size = max(1, len(values) // (len(values) // 12))
                    annual_means = []
                    for i in range(0, len(values), chunk_size):
                        chunk = values[i : i + chunk_size]
                        annual_means.append(sum(chunk) / len(chunk))
                    values = annual_means

                raw = torch.tensor(values, dtype=torch.float32)
                normalized, _, _ = _zscore_tensor(raw)
                data_dict["noaa_sea_level"] = normalized
                inp, out = _io_pair("noaa_sea_level", "physical.climate.sea_level_trend", transform=None, frequency=ANNUAL)
                input_specs.append(inp)
                output_specs.append(out)
                logger.debug("[NOAA] Sea level: %d points", len(normalized))

        except Exception as exc:
            logger.debug("[NOAA] Failed to fetch sea level data: %s", exc)


# ── IMF Collector ────────────────────────────────────────────────────────

# IMF SDMX dataset codes and their indicators.
# Format: (database_id, indicator_ref, target_field, frequency)
_IMF_INDICATORS: list[tuple[str, str, str, int]] = [
    # World Economic Outlook
    ("WEO", "NGDP_RPCH", "forecasts.macro.gdp_growth_12m", QUARTERLY),
    ("WEO", "PCPIPCH", "forecasts.macro.inflation_path_12m", QUARTERLY),
    ("WEO", "BCA_NGDPD", "country_us.macro.trade.current_account", QUARTERLY),
    # Primary Commodity Price System
    ("PCPS", "POILAPSP", "resources.energy.crude_price", MONTHLY),
    ("PCPS", "PNGAS", "resources.energy.natgas_price", MONTHLY),
    ("PCPS", "PGOLD", "resources.metals.gold", MONTHLY),
    ("PCPS", "PCOPP", "resources.metals.copper", MONTHLY),
    ("PCPS", "PALUM", "resources.metals.aluminum", MONTHLY),
    ("PCPS", "PNICK", "resources.metals.nickel", MONTHLY),
    ("PCPS", "PWHEAMT", "resources.food.wheat", MONTHLY),
    ("PCPS", "PMAIZMT", "resources.food.corn", MONTHLY),
]


class IMFCollector(BaseCollector):
    """Downloads data from the IMF JSON REST API.

    Provides World Economic Outlook (WEO) forecasts and Primary Commodity
    Price System (PCPS) data.  The API is free and requires no key.

    - WEO: GDP growth forecasts, inflation forecasts, current account
    - PCPS: Crude oil, natural gas, gold, copper, aluminum, wheat, corn

    All series are z-score normalized.

    Uses the SDMX JSON API: ``http://dataservices.imf.org/REST/SDMX_JSON.svc/``
    """

    def __init__(
        self,
        indicators: list[tuple[str, str, str, int]] | None = None,
        start_year: int = 2000,
        end_year: int | None = None,
        cache_dir: str | Path | None = None,
        force_refresh: bool = False,
    ):
        super().__init__(cache_dir=cache_dir, force_refresh=force_refresh)
        self.indicators = indicators or _IMF_INDICATORS
        self.start_year = start_year
        self.end_year = end_year

    @property
    def name(self) -> str:
        return "IMF"

    def _cache_params(self) -> dict:
        return {
            "indicators": [
                f"{db}.{ind}" for db, ind, _, _ in self.indicators
            ],
            "start_year": self.start_year,
            "end_year": self.end_year or "latest",
        }

    def _fetch(self) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed. Install with: pip install requests")
            return DatasetSpec(name=self.name), {}

        input_specs: list[InputSpec] = []
        output_specs: list[OutputSpec] = []
        data_dict: dict[str, torch.Tensor] = {}

        base_url = "http://dataservices.imf.org/REST/SDMX_JSON.svc"

        for database_id, indicator_ref, target_field, frequency in self.indicators:
            source_key = f"imf_{database_id}_{indicator_ref}"

            try:
                # Build the compact data request URL.
                # For PCPS: CompactData/PCPS/M..PCOPP
                # For WEO: CompactData/WEO/...
                if database_id == "PCPS":
                    # PCPS uses monthly frequency code "M" and a wildcard country
                    dimension_string = f"M..{indicator_ref}"
                else:
                    # WEO: annual data, US as reference country
                    dimension_string = f"A.US.{indicator_ref}"

                url = f"{base_url}/CompactData/{database_id}/{dimension_string}"
                params = {
                    "startPeriod": str(self.start_year),
                    "endPeriod": str(self.end_year or 2026),
                }

                logger.debug("[IMF] Fetching %s / %s", database_id, indicator_ref)
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()

                payload = resp.json()

                # Navigate the SDMX JSON structure to extract observations
                values = self._extract_observations(payload, database_id)

                if not values:
                    logger.debug(
                        "[IMF] No observations for %s / %s",
                        database_id, indicator_ref,
                    )
                    continue

                raw = torch.tensor(values, dtype=torch.float32)

                # Commodity prices: use log returns then z-score
                if database_id == "PCPS":
                    log_ret = _log_returns_tensor(raw)
                    normalized, _, _ = _zscore_tensor(log_ret)
                else:
                    normalized, _, _ = _zscore_tensor(raw)

                data_dict[source_key] = normalized
                inp, out = _io_pair(source_key, target_field, transform=None, frequency=frequency)
                input_specs.append(inp)
                output_specs.append(out)

                logger.debug(
                    "[IMF] %s -> %s (%d points)",
                    source_key, target_field, len(normalized),
                )

            except Exception as exc:
                logger.debug(
                    "[IMF] Failed %s / %s: %s", database_id, indicator_ref, exc
                )
                continue

        spec = DatasetSpec(
            name="IMF",
            input_specs=input_specs,
            output_specs=output_specs,
            base_period=QUARTERLY,
            weight=0.8,
        )

        return spec, data_dict

    @staticmethod
    def _extract_observations(
        payload: dict, database_id: str
    ) -> list[float]:
        """Extract time-series values from the SDMX JSON response.

        The IMF SDMX JSON structure is:
        ``CompactData -> DataSet -> Series -> Obs``
        where ``Series`` can be a dict (single series) or list (multiple).
        Each ``Obs`` has ``@OBS_VALUE`` and a time period key.

        When multiple series are returned (e.g., multiple countries for PCPS),
        we take the first series as a representative.
        """
        try:
            dataset = payload.get("CompactData", {}).get("DataSet", {})
            series = dataset.get("Series", {})

            # Multiple series -> take the first
            if isinstance(series, list):
                series = series[0] if series else {}

            obs = series.get("Obs", [])
            if isinstance(obs, dict):
                obs = [obs]

            values: list[float] = []
            for ob in obs:
                val = ob.get("@OBS_VALUE")
                if val is not None:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        continue

            return values
        except (AttributeError, KeyError, TypeError):
            return []


# ── BIS Collector ────────────────────────────────────────────────────────

# BIS dataset keys and their indicators.
# Format: (dataset_key, series_key_pattern, target_field, frequency)
# The BIS API uses SDMX-like REST: https://stats.bis.org/api/v1/
_BIS_DATASETS: list[tuple[str, str, str, str, int]] = [
    # Credit-to-GDP ratio (US)
    (
        "TOTAL_CREDIT",
        "Q.US.P.A.M.770.A",
        "credit_gdp_us",
        "financial.credit.credit_impulse",
        QUARTERLY,
    ),
    # Residential property prices (US)
    (
        "SELECTED_PP",
        "Q.US.R.628",
        "property_us",
        "country_us.macro.housing.home_price_index",
        QUARTERLY,
    ),
    # Residential property prices (CN)
    (
        "SELECTED_PP",
        "Q.CN.R.628",
        "property_cn",
        "country_cn.macro.housing.home_price_index",
        QUARTERLY,
    ),
    # Residential property prices (JP)
    (
        "SELECTED_PP",
        "Q.JP.R.628",
        "property_jp",
        "country_jp.macro.housing.home_price_index",
        QUARTERLY,
    ),
    # Effective exchange rates — US broad
    (
        "EER",
        "M.US.R.B",
        "eer_us",
        "financial.fx.dxy",
        MONTHLY,
    ),
    # Debt securities outstanding — total
    (
        "DEBT_SEC2",
        "Q.US.1A.1.TO1.A.A.A.TO1.A.I",
        "debt_sec_us",
        "financial.credit.private_credit_growth",
        QUARTERLY,
    ),
]


class BISCollector(BaseCollector):
    """Downloads data from the Bank for International Settlements statistics API.

    The BIS provides free access (no API key needed) to statistics on:
    - Credit-to-GDP ratios
    - Residential property prices
    - Effective exchange rates
    - Debt securities outstanding

    All series are z-score normalized.

    Uses the BIS API: ``https://stats.bis.org/api/v1/``
    """

    def __init__(
        self,
        datasets: list[tuple[str, str, str, str, int]] | None = None,
        start_period: str = "2000-Q1",
        end_period: str | None = None,
        cache_dir: str | Path | None = None,
        force_refresh: bool = False,
    ):
        super().__init__(cache_dir=cache_dir, force_refresh=force_refresh)
        self.datasets = datasets or _BIS_DATASETS
        self.start_period = start_period
        self.end_period = end_period

    @property
    def name(self) -> str:
        return "BIS"

    def _cache_params(self) -> dict:
        return {
            "datasets": [
                f"{ds[0]}.{ds[1]}" for ds in self.datasets
            ],
            "start_period": self.start_period,
            "end_period": self.end_period or "latest",
        }

    def _fetch(self) -> tuple[DatasetSpec, dict[str, torch.Tensor]]:
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed. Install with: pip install requests")
            return DatasetSpec(name=self.name), {}

        input_specs: list[InputSpec] = []
        output_specs: list[OutputSpec] = []
        data_dict: dict[str, torch.Tensor] = {}

        base_url = "https://stats.bis.org/api/v1"

        for dataset_key, series_key, source_key, target_field, frequency in self.datasets:
            full_source_key = f"bis_{source_key}"

            try:
                # BIS uses SDMX REST: /data/{flow}/{key}
                url = f"{base_url}/data/{dataset_key}/{series_key}"
                params: dict[str, str] = {
                    "format": "json",
                    "startPeriod": self.start_period.replace("-Q", "-Q"),
                }
                if self.end_period:
                    params["endPeriod"] = self.end_period

                logger.debug("[BIS] Fetching %s / %s", dataset_key, series_key)
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()

                payload = resp.json()
                values = self._extract_sdmx_values(payload)

                if not values:
                    logger.debug(
                        "[BIS] No observations for %s / %s",
                        dataset_key, series_key,
                    )
                    continue

                raw = torch.tensor(values, dtype=torch.float32)

                # Property prices and exchange rates: pct change then z-score
                if "property" in source_key or "eer" in source_key:
                    transformed = _pct_change_tensor(raw)
                    normalized, _, _ = _zscore_tensor(transformed)
                else:
                    normalized, _, _ = _zscore_tensor(raw)

                data_dict[full_source_key] = normalized
                inp, out = _io_pair(full_source_key, target_field, transform=None, frequency=frequency)
                input_specs.append(inp)
                output_specs.append(out)

                logger.debug(
                    "[BIS] %s -> %s (%d points)",
                    full_source_key, target_field, len(normalized),
                )

            except Exception as exc:
                logger.debug(
                    "[BIS] Failed %s / %s: %s", dataset_key, series_key, exc
                )
                continue

        spec = DatasetSpec(
            name="BIS",
            input_specs=input_specs,
            output_specs=output_specs,
            base_period=QUARTERLY,
            weight=0.7,
        )

        return spec, data_dict

    @staticmethod
    def _extract_sdmx_values(payload: dict) -> list[float]:
        """Extract observation values from a BIS SDMX JSON response.

        The BIS SDMX-JSON format stores observations indexed by position:
        ``dataSets[0].series["0:0:..."].observations["0"] -> [value]``

        Observations are keyed by their time-dimension position (as strings),
        and we sort by that position to get chronological order.
        """
        try:
            data_sets = payload.get("dataSets", [])
            if not data_sets:
                return []

            series_dict = data_sets[0].get("series", {})
            if not series_dict:
                return []

            # Take the first series available
            first_series = next(iter(series_dict.values()))
            observations = first_series.get("observations", {})

            # Sort by time position key (string ints) and extract values
            sorted_obs = sorted(observations.items(), key=lambda x: int(x[0]))
            values: list[float] = []
            for _, obs_array in sorted_obs:
                if isinstance(obs_array, list) and obs_array:
                    try:
                        values.append(float(obs_array[0]))
                    except (ValueError, TypeError, IndexError):
                        continue

            return values
        except (StopIteration, AttributeError, KeyError, TypeError):
            return []


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
            - ``"noaa"``: NOAA Climate Data Online API key
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

    # Extract start year from start_date for collectors that use year params
    start_year = int(start_date.split("-")[0]) if start_date else 2000

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

    # 3. World Bank Collector
    logger.info("Running WorldBankCollector...")
    world_bank = WorldBankCollector(
        start_year=start_year,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    spec, data = world_bank.collect()
    if data:
        results.append((spec, data))
    else:
        logger.info("WorldBankCollector returned no data")

    # 4. NOAA Climate Collector
    logger.info("Running NOAAClimateCollector...")
    noaa = NOAAClimateCollector(
        api_key=api_keys.get("noaa"),
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    spec, data = noaa.collect()
    if data:
        results.append((spec, data))
    else:
        logger.info("NOAAClimateCollector returned no data (missing NOAA API key?)")

    # 5. IMF Collector
    logger.info("Running IMFCollector...")
    imf = IMFCollector(
        start_year=start_year,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    spec, data = imf.collect()
    if data:
        results.append((spec, data))
    else:
        logger.info("IMFCollector returned no data")

    # 6. BIS Collector
    logger.info("Running BISCollector...")
    bis = BISCollector(
        start_period=f"{start_year}-Q1",
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    spec, data = bis.collect()
    if data:
        results.append((spec, data))
    else:
        logger.info("BISCollector returned no data")

    # 7. Synthetic Collector (always available)
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
