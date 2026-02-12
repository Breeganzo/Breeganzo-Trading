"""
Morning Ranking Engine for Indian Equity Quant Trading Platform.

Computes stock rankings at 8:45 AM IST across multiple categories using a
weighted multi-factor scoring model.  The primary ranking formula is:

    Score = 0.4 * Expected_Return
          + 0.3 * 30d_Momentum
          + 0.2 * Volatility_Inverse
          + 0.1 * Liquidity_Score

Stock universes are broken into Banking, Large Cap, and Small Cap segments.
Rankings are produced for seven categories: top_buy, top_sell, banking,
large_cap, small_cap, high_vol, and overall.

Market data is sourced from Yahoo Finance via *yfinance*.  All blocking
network calls are dispatched through :func:`asyncio.to_thread` to keep the
event loop non-blocking.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR: int = 252
_RISK_FREE_RATE: float = 0.065  # 6.5% annualized (India 10-yr GOI bond)
_RANKING_TIME_HOUR: int = 8
_RANKING_TIME_MINUTE: int = 45

# Score component weights
_W_EXPECTED_RETURN: float = 0.4
_W_MOMENTUM_30D: float = 0.3
_W_VOLATILITY_INV: float = 0.2
_W_LIQUIDITY: float = 0.1

_TOP_N: int = 10

# ---------------------------------------------------------------------------
# Stock Universes
# ---------------------------------------------------------------------------

BANKING_TICKERS: list[str] = [
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK",
    "INDUSINDBK", "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB",
    "BANDHANBNK", "AUBANK",
]

LARGE_CAP_TICKERS: list[str] = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR",
    "ITC", "BHARTIARTL", "SBIN", "BAJFINANCE", "LT", "KOTAKBANK",
    "HCLTECH", "ASIANPAINT", "MARUTI", "TITAN", "WIPRO", "ULTRACEMCO",
    "NESTLEIND", "TATAMOTORS",
]

SMALL_CAP_TICKERS: list[str] = [
    "KTKBANK", "NATIONALUM", "IRFC", "SUZLON", "YESBANK", "IDEA",
    "NHPC", "ZOMATO", "JSWENERGY", "TATAELXSI", "POLYCAB", "DEEPAKNTR",
]

ALL_TICKERS: list[str] = sorted(
    set(BANKING_TICKERS + LARGE_CAP_TICKERS + SMALL_CAP_TICKERS)
)

# Reverse lookup: ticker -> set of categories it belongs to.
_TICKER_CATEGORIES: dict[str, set[str]] = {}
for _t in BANKING_TICKERS:
    _TICKER_CATEGORIES.setdefault(_t, set()).add("banking")
for _t in LARGE_CAP_TICKERS:
    _TICKER_CATEGORIES.setdefault(_t, set()).add("large_cap")
for _t in SMALL_CAP_TICKERS:
    _TICKER_CATEGORIES.setdefault(_t, set()).add("small_cap")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nan_safe(value: Any) -> Any:
    """Convert NaN / Inf to ``None`` for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _ensure_nse_suffix(ticker: str) -> str:
    """Append ``.NS`` if the ticker does not already carry an exchange suffix."""
    if ticker.startswith("^"):
        return ticker
    if not ticker.endswith((".NS", ".BO")):
        return f"{ticker}.NS"
    return ticker


def _fetch_yf_history(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """Blocking helper -- download OHLCV history from Yahoo Finance.

    Imported lazily so the module can be loaded in environments where
    *yfinance* is not yet installed.
    """
    import yfinance as yf  # lazy import

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
    )
    if data is None or data.empty:
        raise ValueError(
            f"No price data returned by yfinance for ticker '{ticker}' "
            f"(period={period})."
        )
    return data


# ---------------------------------------------------------------------------
# Ranking Engine
# ---------------------------------------------------------------------------

class RankingEngine:
    """Multi-factor ranking engine for the Indian equity morning session.

    Computes a composite score for each stock in the universe and produces
    ranked lists across seven categories.  Designed to run at 8:45 AM IST
    before the market opens (9:15 AM IST).

    Typical usage::

        engine = RankingEngine()
        rankings = await engine.compute_all_rankings()
        top_buys = rankings["top_buy"]
    """

    def __init__(self) -> None:
        """Initialise the ranking engine.

        Caches computed scores in-memory for the current ranking run so
        that category slicing does not re-compute scores.
        """
        self._scores_cache: dict[str, dict] = {}
        self._last_computed: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Data fetching (async-safe)
    # ------------------------------------------------------------------

    async def _fetch_history(
        self,
        ticker: str,
        period: str = "3mo",
    ) -> pd.DataFrame:
        """Fetch OHLCV history for *ticker* without blocking the event loop.

        Parameters
        ----------
        ticker : str
            Raw NSE symbol (e.g. ``"RELIANCE"``).  ``.NS`` is appended
            automatically.
        period : str
            Look-back window understood by *yfinance* (default ``"3mo"``).

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame with columns ``Open``, ``High``, ``Low``,
            ``Close``, ``Volume``.

        Raises
        ------
        ValueError
            When no data is returned for the ticker.
        """
        nse_ticker = _ensure_nse_suffix(ticker)
        return await asyncio.to_thread(_fetch_yf_history, nse_ticker, period)

    # ------------------------------------------------------------------
    # Individual factor computations
    # ------------------------------------------------------------------

    async def compute_expected_return(self, ticker: str) -> float:
        """Compute the annualized expected return for *ticker*.

        Uses a simple mean-return extrapolation from the most recent 60
        trading days of daily returns, annualized to 252 trading days,
        and adjusted for the risk-free rate.

        Parameters
        ----------
        ticker : str
            Raw NSE symbol.

        Returns
        -------
        float
            Annualized expected excess return as a decimal, or ``NaN``
            on data failure.
        """
        try:
            df = await self._fetch_history(ticker, period="3mo")
            close = df["Close"].squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            daily_returns = close.pct_change().dropna()

            if len(daily_returns) < 5:
                return float("nan")

            mean_daily = daily_returns.mean()
            annualized = mean_daily * _TRADING_DAYS_PER_YEAR
            expected = annualized - _RISK_FREE_RATE

            return float(expected)
        except Exception:
            logger.warning(
                "Failed to compute expected return for %s.", ticker,
            )
            return float("nan")

    async def compute_momentum_30d(self, ticker: str) -> float:
        """Compute the 30-day price momentum for *ticker*.

        Momentum is defined as the percentage change in closing price
        over the last 30 calendar days (approximately 20-22 trading days).

        Parameters
        ----------
        ticker : str
            Raw NSE symbol.

        Returns
        -------
        float
            30-day momentum as a decimal (e.g. 0.05 for 5%), or ``NaN``
            on failure.
        """
        try:
            df = await self._fetch_history(ticker, period="3mo")
            close = df["Close"].squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            if len(close) < 20:
                return float("nan")

            # Use the last ~22 trading days (approx. 30 calendar days).
            recent_close = close.iloc[-1]
            past_close = close.iloc[-22] if len(close) >= 22 else close.iloc[0]

            momentum = (recent_close - past_close) / past_close

            return float(momentum)
        except Exception:
            logger.warning(
                "Failed to compute 30d momentum for %s.", ticker,
            )
            return float("nan")

    async def compute_volatility(self, ticker: str) -> float:
        """Compute the annualized volatility for *ticker*.

        Uses the standard deviation of daily returns over the available
        look-back window, annualized by multiplying by sqrt(252).

        Parameters
        ----------
        ticker : str
            Raw NSE symbol.

        Returns
        -------
        float
            Annualized volatility as a decimal, or ``NaN`` on failure.
        """
        try:
            df = await self._fetch_history(ticker, period="3mo")
            close = df["Close"].squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            daily_returns = close.pct_change().dropna()

            if len(daily_returns) < 5:
                return float("nan")

            vol = daily_returns.std(ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR)

            return float(vol)
        except Exception:
            logger.warning(
                "Failed to compute volatility for %s.", ticker,
            )
            return float("nan")

    async def compute_liquidity_score(self, ticker: str) -> float:
        """Compute a composite liquidity score for *ticker*.

        The score combines three liquidity proxies, each normalized to
        [0, 1] and then averaged:

        1. **Average Volume (20-day)** -- higher is better.  Normalized
           via a log transform and capped at a reference maximum.
        2. **Bid-Ask Spread Proxy** -- ``(High - Low) / Close``.  Lower
           spread implies *better* liquidity, so we use the inverse.
        3. **Market Depth Proxy** -- ``(Volume * Price) / (Avg_Volume *
           Avg_Price)`` normalized.  A ratio above 1.0 indicates stronger
           depth than average.

        Parameters
        ----------
        ticker : str
            Raw NSE symbol.

        Returns
        -------
        float
            Liquidity score in approximately [0, 1], or ``NaN`` on
            failure.
        """
        try:
            df = await self._fetch_history(ticker, period="3mo")

            close = df["Close"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
            volume = df["Volume"].squeeze()

            # Handle multi-level columns from yfinance.
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if isinstance(high, pd.DataFrame):
                high = high.iloc[:, 0]
            if isinstance(low, pd.DataFrame):
                low = low.iloc[:, 0]
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]

            if len(close) < 20:
                return float("nan")

            # --- Component 1: Average Volume (20-day) ---
            avg_volume_20d = volume.iloc[-20:].mean()
            # Log-normalize: log(vol) / log(reference_max).
            # Reference max is set to 50 million shares -- a very liquid
            # Indian large-cap on an active day.
            _REFERENCE_MAX_VOLUME = 50_000_000
            if avg_volume_20d > 0:
                vol_score = min(
                    np.log1p(avg_volume_20d) / np.log1p(_REFERENCE_MAX_VOLUME),
                    1.0,
                )
            else:
                vol_score = 0.0

            # --- Component 2: Bid-Ask Spread Proxy ---
            # Average (high - low) / close over last 20 days.
            spread_series = (high.iloc[-20:] - low.iloc[-20:]) / close.iloc[-20:]
            avg_spread = spread_series.mean()
            # Invert: lower spread = higher score.  Normalize so that a
            # 0% spread gives 1.0 and >= 5% spread gives ~0.
            if avg_spread >= 0:
                spread_score = max(1.0 - (avg_spread / 0.05), 0.0)
            else:
                spread_score = 0.5

            # --- Component 3: Market Depth Proxy ---
            avg_price = close.mean()
            avg_vol = volume.mean()

            if avg_vol > 0 and avg_price > 0:
                current_depth = (
                    volume.iloc[-1] * close.iloc[-1]
                ) / (avg_vol * avg_price)
                # Normalize around 1.0: depth_score = min(ratio, 2.0) / 2.0
                depth_score = min(float(current_depth), 2.0) / 2.0
            else:
                depth_score = 0.0

            # --- Composite ---
            liquidity = (vol_score + spread_score + depth_score) / 3.0

            return float(liquidity)
        except Exception:
            logger.warning(
                "Failed to compute liquidity score for %s.", ticker,
            )
            return float("nan")

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    async def compute_stock_score(self, ticker: str) -> dict:
        """Compute the full multi-factor score for a single stock.

        Runs all four factor computations concurrently and combines them
        using the fixed weights.

        Parameters
        ----------
        ticker : str
            Raw NSE symbol.

        Returns
        -------
        dict
            ``{"ticker", "expected_return", "momentum_30d", "volatility",
            "volatility_inverse", "liquidity_score", "score",
            "current_price", "categories"}``

            Fields that could not be computed are set to ``None``.
        """
        # Run all factor computations concurrently.
        expected_return, momentum, volatility, liquidity = await asyncio.gather(
            self.compute_expected_return(ticker),
            self.compute_momentum_30d(ticker),
            self.compute_volatility(ticker),
            self.compute_liquidity_score(ticker),
        )

        # Fetch current price.
        current_price: Optional[float] = None
        try:
            df = await self._fetch_history(ticker, period="5d")
            close = df["Close"].squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if len(close) > 0:
                current_price = round(float(close.iloc[-1]), 2)
        except Exception:
            logger.warning("Failed to fetch current price for %s.", ticker)

        # Compute volatility inverse: lower vol -> higher score.
        # Guard against zero / NaN volatility.
        if not math.isnan(volatility) and volatility > 0:
            volatility_inverse = 1.0 / volatility
        else:
            volatility_inverse = float("nan")

        # Composite score.
        components = [
            (_W_EXPECTED_RETURN, expected_return),
            (_W_MOMENTUM_30D, momentum),
            (_W_VOLATILITY_INV, volatility_inverse),
            (_W_LIQUIDITY, liquidity),
        ]

        score = 0.0
        valid_weight = 0.0
        for weight, value in components:
            if not math.isnan(value):
                score += weight * value
                valid_weight += weight

        # Rescale if some components were missing so that scores remain
        # comparable across stocks with partial data.
        if valid_weight > 0 and valid_weight < 1.0:
            score = score / valid_weight
        elif valid_weight == 0:
            score = float("nan")

        categories = _TICKER_CATEGORIES.get(ticker, set())

        result = {
            "ticker": ticker,
            "expected_return": _nan_safe(round(expected_return, 6)),
            "momentum_30d": _nan_safe(round(momentum, 6)),
            "volatility": _nan_safe(round(volatility, 6)),
            "volatility_inverse": _nan_safe(
                round(volatility_inverse, 6)
                if not math.isnan(volatility_inverse)
                else None
            ),
            "liquidity_score": _nan_safe(round(liquidity, 6)),
            "score": _nan_safe(round(score, 6)) if not math.isnan(score) else None,
            "current_price": current_price,
            "categories": sorted(categories),
        }

        return result

    # ------------------------------------------------------------------
    # Bulk ranking
    # ------------------------------------------------------------------

    async def compute_all_rankings(self) -> dict[str, list[dict]]:
        """Compute scores for the entire stock universe and rank by category.

        Runs :meth:`compute_stock_score` concurrently for every ticker in
        :data:`ALL_TICKERS`, then slices and sorts the results into seven
        ranked categories.

        Returns
        -------
        dict[str, list[dict]]
            Mapping from category name to a list of up to 10 ranked
            stock entries.  Categories: ``top_buy``, ``top_sell``,
            ``banking``, ``large_cap``, ``small_cap``, ``high_vol``,
            ``overall``.

            Each entry includes ``rank`` (1-based position within its
            category), plus all fields from :meth:`compute_stock_score`.
        """
        logger.info(
            "Starting ranking computation for %d tickers.", len(ALL_TICKERS),
        )

        # Compute scores concurrently for all tickers.
        tasks = [self.compute_stock_score(ticker) for ticker in ALL_TICKERS]
        raw_results: list[dict] = await asyncio.gather(*tasks)

        # Filter out tickers whose score could not be computed.
        scored: list[dict] = [
            r for r in raw_results if r.get("score") is not None
        ]

        # Cache the scored results.
        self._scores_cache = {r["ticker"]: r for r in scored}
        self._last_computed = datetime.now()

        if not scored:
            logger.warning("No stocks could be scored.  Returning empty rankings.")
            return {
                "top_buy": [],
                "top_sell": [],
                "banking": [],
                "large_cap": [],
                "small_cap": [],
                "high_vol": [],
                "overall": [],
            }

        # ----- Build category rankings -----

        # top_buy: highest score, positive expected returns.
        top_buy_pool = [
            s for s in scored
            if s.get("expected_return") is not None
            and s["expected_return"] > 0
        ]
        top_buy_pool.sort(key=lambda x: x["score"], reverse=True)
        top_buy = self._add_ranks(top_buy_pool[:_TOP_N])

        # top_sell: lowest score, negative expected returns (ascending).
        top_sell_pool = [
            s for s in scored
            if s.get("expected_return") is not None
            and s["expected_return"] < 0
        ]
        top_sell_pool.sort(key=lambda x: x["score"], reverse=False)
        top_sell = self._add_ranks(top_sell_pool[:_TOP_N])

        # banking: banking sector stocks, sorted by score descending.
        banking_pool = [
            s for s in scored if s["ticker"] in set(BANKING_TICKERS)
        ]
        banking_pool.sort(key=lambda x: x["score"], reverse=True)
        banking = self._add_ranks(banking_pool[:_TOP_N])

        # large_cap: large cap stocks, sorted by score descending.
        large_cap_pool = [
            s for s in scored if s["ticker"] in set(LARGE_CAP_TICKERS)
        ]
        large_cap_pool.sort(key=lambda x: x["score"], reverse=True)
        large_cap = self._add_ranks(large_cap_pool[:_TOP_N])

        # small_cap: small cap stocks, sorted by score descending.
        small_cap_pool = [
            s for s in scored if s["ticker"] in set(SMALL_CAP_TICKERS)
        ]
        small_cap_pool.sort(key=lambda x: x["score"], reverse=True)
        small_cap = self._add_ranks(small_cap_pool[:_TOP_N])

        # high_vol: highest volatility stocks.
        high_vol_pool = [
            s for s in scored
            if s.get("volatility") is not None
        ]
        high_vol_pool.sort(
            key=lambda x: x["volatility"], reverse=True,
        )
        high_vol = self._add_ranks(high_vol_pool[:_TOP_N])

        # overall: top by composite score across full universe.
        overall_pool = sorted(scored, key=lambda x: x["score"], reverse=True)
        overall = self._add_ranks(overall_pool[:_TOP_N])

        rankings = {
            "top_buy": top_buy,
            "top_sell": top_sell,
            "banking": banking,
            "large_cap": large_cap,
            "small_cap": small_cap,
            "high_vol": high_vol,
            "overall": overall,
        }

        logger.info(
            "Ranking computation complete.  Scored %d / %d tickers.  "
            "Categories: %s",
            len(scored),
            len(ALL_TICKERS),
            {k: len(v) for k, v in rankings.items()},
        )

        return rankings

    async def get_rankings_by_category(self, category: str) -> list[dict]:
        """Return the ranked list for a single category.

        If rankings have not been computed yet (or the cache is stale),
        a full :meth:`compute_all_rankings` run is triggered first.

        Parameters
        ----------
        category : str
            One of ``"top_buy"``, ``"top_sell"``, ``"banking"``,
            ``"large_cap"``, ``"small_cap"``, ``"high_vol"``,
            ``"overall"``.

        Returns
        -------
        list[dict]
            Ranked entries for the requested category.

        Raises
        ------
        ValueError
            If *category* is not a recognised category name.
        """
        valid_categories = {
            "top_buy", "top_sell", "banking", "large_cap",
            "small_cap", "high_vol", "overall",
        }
        if category not in valid_categories:
            raise ValueError(
                f"Unknown category '{category}'.  "
                f"Valid categories: {sorted(valid_categories)}"
            )

        # Re-compute if the cache is empty or older than 1 hour.
        stale = (
            self._last_computed is None
            or (datetime.now() - self._last_computed) > timedelta(hours=1)
        )

        if stale or not self._scores_cache:
            all_rankings = await self.compute_all_rankings()
            return all_rankings.get(category, [])

        # Rebuild the requested category from the cached scores.
        scored = list(self._scores_cache.values())

        if category == "top_buy":
            pool = [
                s for s in scored
                if s.get("expected_return") is not None
                and s["expected_return"] > 0
            ]
            pool.sort(key=lambda x: x["score"], reverse=True)
            return self._add_ranks(pool[:_TOP_N])

        if category == "top_sell":
            pool = [
                s for s in scored
                if s.get("expected_return") is not None
                and s["expected_return"] < 0
            ]
            pool.sort(key=lambda x: x["score"], reverse=False)
            return self._add_ranks(pool[:_TOP_N])

        if category == "banking":
            pool = [s for s in scored if s["ticker"] in set(BANKING_TICKERS)]
            pool.sort(key=lambda x: x["score"], reverse=True)
            return self._add_ranks(pool[:_TOP_N])

        if category == "large_cap":
            pool = [s for s in scored if s["ticker"] in set(LARGE_CAP_TICKERS)]
            pool.sort(key=lambda x: x["score"], reverse=True)
            return self._add_ranks(pool[:_TOP_N])

        if category == "small_cap":
            pool = [s for s in scored if s["ticker"] in set(SMALL_CAP_TICKERS)]
            pool.sort(key=lambda x: x["score"], reverse=True)
            return self._add_ranks(pool[:_TOP_N])

        if category == "high_vol":
            pool = [
                s for s in scored if s.get("volatility") is not None
            ]
            pool.sort(key=lambda x: x["volatility"], reverse=True)
            return self._add_ranks(pool[:_TOP_N])

        # overall
        pool = sorted(scored, key=lambda x: x["score"], reverse=True)
        return self._add_ranks(pool[:_TOP_N])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_ranks(entries: list[dict]) -> list[dict]:
        """Return a copy of *entries* with a 1-based ``rank`` field added."""
        ranked: list[dict] = []
        for idx, entry in enumerate(entries, start=1):
            ranked_entry = {**entry, "rank": idx}
            ranked.append(ranked_entry)
        return ranked
