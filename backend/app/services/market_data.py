"""
Market Data Service for live price fetching and Redis caching.

Provides real-time and historical price data for NSE-listed equities via
*yfinance*.  Live prices are cached in Redis with adaptive TTLs -- short
(5 s) during market hours for near-real-time updates, longer (5 min)
outside trading hours to reduce unnecessary API calls.

Market hours follow IST (Asia/Kolkata): 09:15 -- 15:30, Monday to Friday.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, time
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo

from app.core.config import get_settings
from app.db.redis import get_redis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IST = ZoneInfo("Asia/Kolkata")
_MARKET_OPEN = time(9, 15)
_MARKET_CLOSE = time(15, 30)
_CACHE_TTL_MARKET_OPEN: int = 5        # seconds
_CACHE_TTL_MARKET_CLOSED: int = 300    # 5 minutes
_CACHE_KEY_PREFIX: str = "price"


def _ensure_nse_suffix(ticker: str) -> str:
    """Append ``.NS`` if the ticker does not already carry an exchange suffix.

    Special tickers (e.g. ``^NSEI``) are returned unchanged.
    """
    if ticker.startswith("^"):
        return ticker
    if not ticker.endswith((".NS", ".BO")):
        return f"{ticker}.NS"
    return ticker


def _nan_safe(value: Any) -> Any:
    """Convert NaN / Inf to ``None`` for JSON serialization."""
    if value is None:
        return None
    try:
        import math
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _make_cache_key(ticker: str) -> str:
    """Build a Redis key for the given *ticker*."""
    return f"{_CACHE_KEY_PREFIX}:{ticker.upper()}"


class MarketDataService:
    """Fetches live and historical market data for NSE equities.

    Prices are sourced from Yahoo Finance via *yfinance*.  All blocking
    network calls are dispatched to a thread via :func:`asyncio.to_thread`
    so the event loop stays responsive.

    A Redis cache sits in front of the Yahoo Finance layer.  During market
    hours the TTL is 5 seconds; outside hours it rises to 5 minutes since
    prices are stale anyway.

    Typical usage::

        svc = MarketDataService()
        price = await svc.get_live_price("RELIANCE")
        batch = await svc.get_batch_prices(["TCS", "INFY", "HDFC"])
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Market hours helpers
    # ------------------------------------------------------------------

    async def is_market_open(self) -> bool:
        """Return ``True`` if the Indian equity market is currently open.

        Checks IST time against 09:15 -- 15:30, Monday -- Friday.  Public
        holidays are **not** accounted for (weekday-only heuristic).
        """
        now = datetime.now(_IST)
        # Monday = 0 ... Friday = 4
        if now.weekday() > 4:
            return False
        current_time = now.time()
        return _MARKET_OPEN <= current_time <= _MARKET_CLOSE

    def _get_cache_ttl(self, market_open: bool) -> int:
        """Return the appropriate Redis TTL in seconds."""
        return _CACHE_TTL_MARKET_OPEN if market_open else _CACHE_TTL_MARKET_CLOSED

    # ------------------------------------------------------------------
    # Redis cache layer
    # ------------------------------------------------------------------

    async def get_cached_price(self, ticker: str) -> dict | None:
        """Retrieve a cached price dict from Redis, or ``None`` on miss."""
        try:
            r = await get_redis()
            raw = await r.get(_make_cache_key(ticker))
            if raw is not None:
                return json.loads(raw)
        except Exception:
            logger.warning("Redis read failed for %s; treating as cache miss.", ticker)
        return None

    async def set_cached_price(self, ticker: str, data: dict) -> None:
        """Store *data* in Redis with an adaptive TTL."""
        try:
            r = await get_redis()
            market_open = await self.is_market_open()
            ttl = self._get_cache_ttl(market_open)
            await r.set(
                _make_cache_key(ticker),
                json.dumps(data, default=str),
                ex=ttl,
            )
        except Exception:
            logger.warning("Redis write failed for %s; continuing without cache.", ticker)

    # ------------------------------------------------------------------
    # yfinance data fetching (blocking -- runs in thread)
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_ticker_info(nse_ticker: str) -> dict | None:
        """Blocking helper -- fetch fast-info / info from yfinance.

        Returns a JSON-serializable dict with price fields, or ``None``
        on failure.
        """
        import yfinance as yf  # lazy import

        try:
            tk = yf.Ticker(nse_ticker)
            info = tk.fast_info

            price = _nan_safe(getattr(info, "last_price", None))
            prev_close = _nan_safe(getattr(info, "previous_close", None) or
                                   getattr(info, "regular_market_previous_close", None))
            open_price = _nan_safe(getattr(info, "open", None) or
                                   getattr(info, "regular_market_open", None))
            day_high = _nan_safe(getattr(info, "day_high", None) or
                                 getattr(info, "regular_market_day_high", None))
            day_low = _nan_safe(getattr(info, "day_low", None) or
                                getattr(info, "regular_market_day_low", None))
            volume = _nan_safe(getattr(info, "last_volume", None) or
                               getattr(info, "regular_market_volume", None))

            # Compute change / change_pct.
            change: float | None = None
            change_pct: float | None = None
            if price is not None and prev_close is not None and prev_close != 0:
                change = round(price - prev_close, 2)
                change_pct = round((change / prev_close) * 100, 2)

            return {
                "price": price,
                "change": change,
                "change_pct": change_pct,
                "volume": int(volume) if volume is not None else None,
                "high": day_high,
                "low": day_low,
                "open": open_price,
                "prev_close": prev_close,
                "timestamp": datetime.now(_IST).isoformat(),
            }
        except Exception:
            logger.exception("yfinance fetch failed for %s.", nse_ticker)
            return None

    @staticmethod
    def _fetch_batch_download(nse_tickers: list[str]) -> dict[str, dict | None]:
        """Blocking helper -- download latest data for multiple tickers.

        Uses :func:`yfinance.download` with a single call for efficiency.
        Returns a mapping of *original NSE ticker* to price dict (or
        ``None`` on per-ticker failure).
        """
        import yfinance as yf  # lazy import

        if not nse_tickers:
            return {}

        results: dict[str, dict | None] = {}
        now_iso = datetime.now(_IST).isoformat()

        try:
            data = yf.download(
                " ".join(nse_tickers),
                period="5d",
                auto_adjust=True,
                progress=False,
                group_by="ticker" if len(nse_tickers) > 1 else "column",
                threads=True,
            )

            if data is None or data.empty:
                return {t: None for t in nse_tickers}

            for ticker in nse_tickers:
                try:
                    if len(nse_tickers) == 1:
                        ticker_data = data
                    else:
                        ticker_data = data[ticker]

                    if ticker_data.empty:
                        results[ticker] = None
                        continue

                    latest = ticker_data.iloc[-1]
                    prev = ticker_data.iloc[-2] if len(ticker_data) >= 2 else latest

                    price = _nan_safe(float(latest["Close"].iloc[0])
                                      if hasattr(latest["Close"], "iloc")
                                      else float(latest["Close"]))
                    prev_close = _nan_safe(float(prev["Close"].iloc[0])
                                           if hasattr(prev["Close"], "iloc")
                                           else float(prev["Close"]))
                    open_price = _nan_safe(float(latest["Open"].iloc[0])
                                           if hasattr(latest["Open"], "iloc")
                                           else float(latest["Open"]))
                    high = _nan_safe(float(latest["High"].iloc[0])
                                     if hasattr(latest["High"], "iloc")
                                     else float(latest["High"]))
                    low = _nan_safe(float(latest["Low"].iloc[0])
                                    if hasattr(latest["Low"], "iloc")
                                    else float(latest["Low"]))
                    volume = _nan_safe(float(latest["Volume"].iloc[0])
                                       if hasattr(latest["Volume"], "iloc")
                                       else float(latest["Volume"]))

                    change: float | None = None
                    change_pct: float | None = None
                    if price is not None and prev_close is not None and prev_close != 0:
                        change = round(price - prev_close, 2)
                        change_pct = round((change / prev_close) * 100, 2)

                    results[ticker] = {
                        "price": price,
                        "change": change,
                        "change_pct": change_pct,
                        "volume": int(volume) if volume is not None else None,
                        "high": high,
                        "low": low,
                        "open": open_price,
                        "prev_close": prev_close,
                        "timestamp": now_iso,
                    }
                except Exception:
                    logger.warning("Failed to parse batch data for %s.", ticker)
                    results[ticker] = None

        except Exception:
            logger.exception("yfinance batch download failed.")
            return {t: None for t in nse_tickers}

        return results

    @staticmethod
    def _fetch_historical(nse_ticker: str, period: str) -> pd.DataFrame:
        """Blocking helper -- download historical OHLCV data.

        Returns a :class:`pandas.DataFrame` with columns
        ``Open, High, Low, Close, Volume`` indexed by date, or an empty
        DataFrame on failure.
        """
        import yfinance as yf  # lazy import

        try:
            data = yf.download(
                nse_ticker,
                period=period,
                auto_adjust=True,
                progress=False,
            )
            if data is None or data.empty:
                logger.warning("No historical data for %s (period=%s).", nse_ticker, period)
                return pd.DataFrame()
            # Flatten multi-level columns when single ticker is downloaded.
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel("Ticker")
            return data
        except Exception:
            logger.exception("Historical data fetch failed for %s.", nse_ticker)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def get_live_price(self, ticker: str) -> dict | None:
        """Fetch the latest price data for a single *ticker*.

        Checks Redis first; on a cache miss the price is fetched from
        Yahoo Finance and written back to Redis.

        Parameters
        ----------
        ticker : str
            NSE symbol (e.g. ``"RELIANCE"``).  ``.NS`` is appended
            automatically if missing.

        Returns
        -------
        dict | None
            Price dict with keys ``price, change, change_pct, volume,
            high, low, open, prev_close, timestamp``, or ``None`` on
            failure.
        """
        # 1. Try cache.
        cached = await self.get_cached_price(ticker)
        if cached is not None:
            return cached

        # 2. Fetch from Yahoo Finance in a thread.
        nse_ticker = _ensure_nse_suffix(ticker)
        data = await asyncio.to_thread(self._fetch_ticker_info, nse_ticker)

        if data is None:
            return None

        # 3. Cache the result.
        await self.set_cached_price(ticker, data)
        return data

    async def get_batch_prices(self, tickers: list[str]) -> dict[str, dict | None]:
        """Fetch latest prices for multiple tickers in one call.

        Cached prices are returned immediately; only missing tickers are
        fetched from Yahoo Finance.  The batch download is performed in a
        background thread to avoid blocking the event loop.

        Parameters
        ----------
        tickers : list[str]
            NSE symbols (e.g. ``["RELIANCE", "TCS", "INFY"]``).

        Returns
        -------
        dict
            Mapping of original ticker name to price dict (or ``None``).
        """
        if not tickers:
            return {}

        results: dict[str, dict | None] = {}
        to_fetch: list[str] = []

        # 1. Check cache for each ticker.
        for ticker in tickers:
            cached = await self.get_cached_price(ticker)
            if cached is not None:
                results[ticker] = cached
            else:
                to_fetch.append(ticker)

        if not to_fetch:
            return results

        # 2. Batch-fetch missing tickers.
        nse_tickers = [_ensure_nse_suffix(t) for t in to_fetch]
        batch_data = await asyncio.to_thread(self._fetch_batch_download, nse_tickers)

        # 3. Map results back to original ticker names and cache.
        for original, nse in zip(to_fetch, nse_tickers):
            data = batch_data.get(nse)
            results[original] = data
            if data is not None:
                await self.set_cached_price(original, data)

        return results

    async def get_historical_data(
        self,
        ticker: str,
        period: str = "1y",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a single *ticker*.

        Parameters
        ----------
        ticker : str
            NSE symbol.  ``.NS`` is appended automatically.
        period : str
            Look-back window understood by *yfinance* (default ``"1y"``).

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame indexed by date, or an empty DataFrame on
            failure.
        """
        nse_ticker = _ensure_nse_suffix(ticker)
        return await asyncio.to_thread(self._fetch_historical, nse_ticker, period)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

market_data_service = MarketDataService()
