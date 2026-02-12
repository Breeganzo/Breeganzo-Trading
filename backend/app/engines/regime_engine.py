"""
Market Regime Detection Engine for Indian Equity Quant Trading Platform.

Determines the current market regime by combining three signals:

1. **Trend (MA crossover)** -- 50-day vs 200-day moving average of the
   NIFTY 50 index (``^NSEI``).  50 MA > 200 MA is bullish; otherwise
   bearish.

2. **Volatility** -- 20-day realised volatility compared against the
   trailing 60-day average volatility.  A ratio above 1.5x flags
   ``high_vol``; below 0.75x flags ``low_vol``.

3. **Market breadth** -- percentage of NIFTY 50 constituent stocks
   trading above their own 50-day MA.  > 60% is bullish breadth;
   < 40% is bearish breadth.

Regime output priority:  ``high_vol`` > ``low_vol`` > ``bull`` / ``bear``.
Volatility regimes always override trend regimes.

Results are cached for 30 minutes to avoid redundant market-data
fetches.  All blocking *yfinance* calls are dispatched through
:func:`asyncio.to_thread` so the event loop stays non-blocking.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR: int = 252
_BENCHMARK_TICKER: str = "^NSEI"  # NIFTY 50 on Yahoo Finance

# Moving-average windows
_MA_SHORT: int = 50
_MA_LONG: int = 200

# Volatility thresholds
_VOL_WINDOW_SHORT: int = 20   # current realised-vol window (trading days)
_VOL_WINDOW_LONG: int = 60    # average-vol reference window (trading days)
_VOL_HIGH_MULT: float = 1.5   # current > 1.5x avg  ->  high_vol
_VOL_LOW_MULT: float = 0.75   # current < 0.75x avg ->  low_vol

# Breadth thresholds
_BREADTH_BULL: float = 0.60   # > 60% above 50-day MA
_BREADTH_BEAR: float = 0.40   # < 40% above 50-day MA

# Cache TTL
_CACHE_TTL_SECONDS: int = 30 * 60  # 30 minutes

# History period -- must cover at least 200 trading days for the long MA.
_INDEX_HISTORY_PERIOD: str = "1y"
_BREADTH_HISTORY_PERIOD: str = "3mo"  # ~63 trading days, enough for 50-day MA

# Representative sample of NIFTY 50 constituents used for breadth proxy.
_NIFTY50_SAMPLE: list[str] = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "HINDUNILVR", "ITC", "BHARTIARTL", "SBIN", "BAJFINANCE",
    "LT", "KOTAKBANK", "HCLTECH", "ASIANPAINT", "MARUTI",
    "TITAN", "WIPRO", "ULTRACEMCO", "NESTLEIND", "TATAMOTORS",
]


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


def _fetch_yf_history(ticker: str, period: str = "1y") -> pd.DataFrame:
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
# Regime Engine
# ---------------------------------------------------------------------------

class RegimeEngine:
    """Market regime detection engine for Indian equity portfolios.

    Combines MA-crossover trend analysis, realised-volatility assessment,
    and market-breadth measurement to classify the current market
    environment into one of four regimes: ``bull``, ``bear``,
    ``high_vol``, or ``low_vol``.

    Typical usage::

        engine = RegimeEngine()
        regime_info = await engine.detect_regime()
        print(regime_info["regime"])  # e.g. "bull"
    """

    def __init__(self) -> None:
        """Initialise the regime engine.

        Sets up an in-memory cache so that repeated calls within the
        30-minute TTL window return the previously computed result
        without re-fetching market data.
        """
        self._cache: Optional[dict] = None
        self._cache_ts: float = 0.0  # epoch seconds of last computation

    # ------------------------------------------------------------------
    # Data fetching (async-safe)
    # ------------------------------------------------------------------

    async def _fetch_history(
        self,
        ticker: str,
        period: str = "1y",
    ) -> pd.DataFrame:
        """Fetch OHLCV history for *ticker* without blocking the event loop.

        Parameters
        ----------
        ticker : str
            Raw symbol (e.g. ``"RELIANCE"`` or ``"^NSEI"``).
            ``.NS`` is appended automatically for non-index tickers.
        period : str
            Look-back window understood by *yfinance* (default ``"1y"``).

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame with at least a ``Close`` column.

        Raises
        ------
        ValueError
            When no data is returned for the ticker.
        """
        nse_ticker = _ensure_nse_suffix(ticker)
        return await asyncio.to_thread(_fetch_yf_history, nse_ticker, period)

    # ------------------------------------------------------------------
    # Signal computations
    # ------------------------------------------------------------------

    def _compute_ma_signal(
        self,
        close: pd.Series,
    ) -> dict[str, Any]:
        """Compute 50/200-day MA crossover signal from *close* prices.

        Parameters
        ----------
        close : pd.Series
            Daily close prices for ``^NSEI``, sorted chronologically.

        Returns
        -------
        dict
            ``{"ma_50": float, "ma_200": float, "ma_signal": str}``
        """
        ma_50 = float(close.rolling(window=_MA_SHORT).mean().iloc[-1])
        ma_200 = float(close.rolling(window=_MA_LONG).mean().iloc[-1])

        if math.isnan(ma_50) or math.isnan(ma_200):
            return {
                "ma_50": _nan_safe(ma_50),
                "ma_200": _nan_safe(ma_200),
                "ma_signal": "neutral",
            }

        ma_signal = "bullish" if ma_50 > ma_200 else "bearish"
        return {
            "ma_50": round(ma_50, 2),
            "ma_200": round(ma_200, 2),
            "ma_signal": ma_signal,
        }

    def _compute_vol_signal(
        self,
        close: pd.Series,
    ) -> dict[str, Any]:
        """Compute volatility regime from *close* prices.

        Uses annualised realised volatility over two windows:

        * **Short** (20-day): current volatility.
        * **Long** (60-day): reference average volatility.

        Parameters
        ----------
        close : pd.Series
            Daily close prices for ``^NSEI``, sorted chronologically.

        Returns
        -------
        dict
            ``{"current_vol": float, "avg_vol": float, "vol_ratio": float,
            "vol_regime": str}``
        """
        daily_returns = close.pct_change().dropna()

        if len(daily_returns) < _VOL_WINDOW_LONG:
            return {
                "current_vol": None,
                "avg_vol": None,
                "vol_ratio": None,
                "vol_regime": "normal",
            }

        # Short-window realised vol (annualised).
        short_ret = daily_returns.iloc[-_VOL_WINDOW_SHORT:]
        current_vol = float(short_ret.std(ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))

        # Long-window realised vol (annualised).
        long_ret = daily_returns.iloc[-_VOL_WINDOW_LONG:]
        avg_vol = float(long_ret.std(ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))

        if avg_vol == 0 or math.isnan(avg_vol):
            return {
                "current_vol": _nan_safe(round(current_vol, 6)),
                "avg_vol": _nan_safe(round(avg_vol, 6)),
                "vol_ratio": None,
                "vol_regime": "normal",
            }

        vol_ratio = current_vol / avg_vol

        if vol_ratio > _VOL_HIGH_MULT:
            vol_regime = "high_vol"
        elif vol_ratio < _VOL_LOW_MULT:
            vol_regime = "low_vol"
        else:
            vol_regime = "normal"

        return {
            "current_vol": round(current_vol, 6),
            "avg_vol": round(avg_vol, 6),
            "vol_ratio": round(vol_ratio, 4),
            "vol_regime": vol_regime,
        }

    async def _compute_breadth_signal(self) -> dict[str, Any]:
        """Compute market breadth proxy from NIFTY 50 sample stocks.

        For each stock in :data:`_NIFTY50_SAMPLE`, checks whether the
        latest close is above the stock's own 50-day moving average.
        The breadth percentage is the ratio of stocks above their 50 MA
        to total stocks successfully fetched.

        Returns
        -------
        dict
            ``{"breadth_pct": float, "breadth_signal": str}``
        """

        async def _is_above_50ma(ticker: str) -> Optional[bool]:
            """Return ``True`` if *ticker*'s last close > its 50-day MA."""
            try:
                df = await self._fetch_history(ticker, period=_BREADTH_HISTORY_PERIOD)
                close = df["Close"].squeeze()
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                if len(close) < _MA_SHORT:
                    return None
                ma_50 = close.rolling(window=_MA_SHORT).mean().iloc[-1]
                last_close = close.iloc[-1]
                if math.isnan(ma_50) or math.isnan(last_close):
                    return None
                return bool(last_close > ma_50)
            except Exception:
                logger.debug(
                    "Breadth check failed for %s; skipping.", ticker,
                )
                return None

        results = await asyncio.gather(
            *[_is_above_50ma(t) for t in _NIFTY50_SAMPLE],
        )

        # Filter out failures.
        valid = [r for r in results if r is not None]
        if not valid:
            return {
                "breadth_pct": None,
                "breadth_signal": "neutral",
            }

        above_count = sum(1 for v in valid if v)
        breadth_pct = above_count / len(valid)

        if breadth_pct > _BREADTH_BULL:
            breadth_signal = "bullish"
        elif breadth_pct < _BREADTH_BEAR:
            breadth_signal = "bearish"
        else:
            breadth_signal = "neutral"

        return {
            "breadth_pct": round(breadth_pct, 4),
            "breadth_signal": breadth_signal,
        }

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        ma_signal: str,
        vol_regime: str,
        breadth_signal: str,
        vol_ratio: Optional[float],
    ) -> float:
        """Heuristic confidence score in [0, 1] for the detected regime.

        The score reflects how strongly the three independent signals
        agree.  Higher agreement and more extreme readings produce
        higher confidence.

        Parameters
        ----------
        ma_signal : str
            ``"bullish"``, ``"bearish"``, or ``"neutral"``.
        vol_regime : str
            ``"high_vol"``, ``"low_vol"``, or ``"normal"``.
        breadth_signal : str
            ``"bullish"``, ``"bearish"``, or ``"neutral"``.
        vol_ratio : float or None
            Current / average volatility ratio.

        Returns
        -------
        float
            Confidence in [0.0, 1.0].
        """
        score = 0.0

        # --- Trend agreement: MA & breadth pointing the same way ---
        if ma_signal == breadth_signal and ma_signal in ("bullish", "bearish"):
            score += 0.40  # strong agreement
        elif ma_signal in ("bullish", "bearish"):
            score += 0.20  # MA alone gives partial confidence
        elif breadth_signal in ("bullish", "bearish"):
            score += 0.15  # breadth alone gives slightly less

        # --- Volatility extremity ---
        if vol_regime in ("high_vol", "low_vol") and vol_ratio is not None:
            # The further vol_ratio from 1.0, the more confident.
            extremity = abs(vol_ratio - 1.0)
            vol_score = min(extremity / 1.0, 0.40)  # cap at 0.40
            score += vol_score
        elif vol_regime == "normal":
            score += 0.10  # some baseline; regime is "not extreme"

        # --- Breadth adds a directional reliability boost ---
        if breadth_signal in ("bullish", "bearish"):
            score += 0.10

        return round(min(score, 1.0), 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect_regime(self) -> dict:
        """Detect the current market regime.

        Fetches NIFTY 50 index data and a representative sample of
        constituent stocks, then combines MA-crossover, volatility, and
        breadth signals to classify the market into one of four regimes.

        Results are cached for 30 minutes.  Subsequent calls within the
        TTL window return the cached result immediately.

        Returns
        -------
        dict
            Regime report::

                {
                    "regime": str,          # bull | bear | high_vol | low_vol | unknown
                    "ma_50": float | None,
                    "ma_200": float | None,
                    "ma_signal": str,
                    "current_vol": float | None,
                    "avg_vol": float | None,
                    "vol_ratio": float | None,
                    "vol_regime": str,
                    "breadth_pct": float | None,
                    "breadth_signal": str,
                    "confidence": float,
                    "detected_at": str,      # ISO-8601
                }
        """
        # ---- Check cache ----
        now = time.monotonic()
        if self._cache is not None and (now - self._cache_ts) < _CACHE_TTL_SECONDS:
            logger.debug("Returning cached regime result (age=%.0fs).", now - self._cache_ts)
            return self._cache

        # ---- Fetch index data & breadth concurrently ----
        try:
            index_df, breadth_info = await asyncio.gather(
                self._fetch_history(_BENCHMARK_TICKER, period=_INDEX_HISTORY_PERIOD),
                self._compute_breadth_signal(),
            )
        except Exception:
            logger.exception("Failed to fetch market data for regime detection.")
            return self._default_result()

        # Extract close series.
        close = index_df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        if close.empty or len(close) < _MA_LONG:
            logger.warning(
                "Insufficient index data for regime detection "
                "(got %d rows, need >= %d).",
                len(close),
                _MA_LONG,
            )
            return self._default_result()

        # ---- Compute signals ----
        ma_info = self._compute_ma_signal(close)
        vol_info = self._compute_vol_signal(close)

        # ---- Determine regime (priority: vol overrides trend) ----
        vol_regime = vol_info["vol_regime"]
        ma_signal = ma_info["ma_signal"]
        breadth_signal = breadth_info["breadth_signal"]

        if vol_regime == "high_vol":
            regime = "high_vol"
        elif vol_regime == "low_vol":
            regime = "low_vol"
        else:
            # Trend regime: MA is primary; breadth can reinforce or
            # break ties when MA is neutral.
            if ma_signal == "bullish":
                regime = "bull"
            elif ma_signal == "bearish":
                regime = "bear"
            elif breadth_signal == "bullish":
                regime = "bull"
            elif breadth_signal == "bearish":
                regime = "bear"
            else:
                regime = "bull"  # default to bull when all neutral

        # ---- Confidence ----
        confidence = self._compute_confidence(
            ma_signal,
            vol_regime,
            breadth_signal,
            vol_info.get("vol_ratio"),
        )

        detected_at = datetime.now(timezone.utc).isoformat()

        result: dict[str, Any] = {
            "regime": regime,
            "ma_50": _nan_safe(ma_info["ma_50"]),
            "ma_200": _nan_safe(ma_info["ma_200"]),
            "ma_signal": ma_info["ma_signal"],
            "current_vol": _nan_safe(vol_info["current_vol"]),
            "avg_vol": _nan_safe(vol_info["avg_vol"]),
            "vol_ratio": _nan_safe(vol_info["vol_ratio"]),
            "vol_regime": vol_info["vol_regime"],
            "breadth_pct": _nan_safe(breadth_info["breadth_pct"]),
            "breadth_signal": breadth_info["breadth_signal"],
            "confidence": confidence,
            "detected_at": detected_at,
        }

        # ---- Update cache ----
        self._cache = result
        self._cache_ts = now

        logger.info(
            "Regime detected: %s (confidence=%.2f, ma_signal=%s, "
            "vol_regime=%s, breadth=%.1f%%).",
            regime,
            confidence,
            ma_signal,
            vol_regime,
            (breadth_info["breadth_pct"] or 0) * 100,
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_result() -> dict:
        """Return a safe default regime when data fetching fails."""
        return {
            "regime": "unknown",
            "ma_50": None,
            "ma_200": None,
            "ma_signal": "neutral",
            "current_vol": None,
            "avg_vol": None,
            "vol_ratio": None,
            "vol_regime": "normal",
            "breadth_pct": None,
            "breadth_signal": "neutral",
            "confidence": 0.0,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }
