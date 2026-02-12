"""
Risk Engine for Indian Equity Portfolio Analytics.

Computes portfolio-level and per-asset risk metrics including Sharpe ratio,
Sortino ratio, portfolio beta (against NIFTY 50), maximum drawdown,
Value-at-Risk, rolling returns, and correlation matrices.

All annualization uses 252 trading days.  The risk-free rate defaults to
6.5% (approximate Indian 10-year government bond yield).  Benchmark data
is sourced from Yahoo Finance via *yfinance* (NIFTY 50 = ``^NSEI``).

NaN values are converted to ``None`` in every public return dict so that
results can be directly serialized to JSON without errors.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR: int = 252
_RISK_FREE_RATE: float = 0.065  # 6.5% annualized (India 10-yr GOI bond)
_DAILY_RISK_FREE: float = _RISK_FREE_RATE / _TRADING_DAYS_PER_YEAR
_BENCHMARK_TICKER: str = "^NSEI"  # NIFTY 50 on Yahoo Finance
_VAR_CONFIDENCE: float = 0.95


def _nan_safe(value: Any) -> Any:
    """Convert NaN / Inf to ``None`` for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _series_from_dicts(
    records: list[dict],
    value_key: str = "return",
    date_key: str = "date",
) -> pd.Series:
    """Build a float Series from a list of ``{date, return}`` dicts.

    Handles both string and datetime date values.  The resulting Series is
    sorted by date in ascending order with a ``DatetimeIndex``.
    """
    df = pd.DataFrame(records)
    if df.empty:
        return pd.Series(dtype=float)
    df[date_key] = pd.to_datetime(df[date_key])
    df = df.sort_values(date_key).set_index(date_key)
    return df[value_key].astype(float)


class RiskEngine:
    """Pure-computation risk engine for Indian equity portfolios.

    All heavy-compute methods accept pre-fetched data so callers retain
    full control over I/O.  The ``async`` methods that *do* fetch (beta,
    correlation matrix) acquire market data via *yfinance* and run the
    blocking download inside :func:`asyncio.to_thread` to stay
    non-blocking on the event loop.

    Typical usage::

        engine = RiskEngine()
        metrics = await engine.compute_portfolio_risk(holdings, daily_returns)
    """

    def __init__(self) -> None:
        """Initialise the risk engine.  No special state is required."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_yf_history(
        ticker: str,
        period: str = "1y",
    ) -> pd.DataFrame:
        """Blocking helper -- download adjusted-close history from Yahoo Finance.

        Always imported lazily to allow the rest of the module to work in
        environments where *yfinance* is not installed (e.g. unit-test
        stubs).
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
        # Flatten MultiIndex columns (yfinance >= 0.2.31 returns MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data

    @staticmethod
    def _ensure_nse_suffix(ticker: str) -> str:
        """Append ``.NS`` if the ticker does not already carry an exchange suffix.

        Special tickers (e.g. ``^NSEI``) are returned unchanged.
        """
        if ticker.startswith("^"):
            return ticker
        if not ticker.endswith((".NS", ".BO")):
            return f"{ticker}.NS"
        return ticker

    # ------------------------------------------------------------------
    # Core metric methods
    # ------------------------------------------------------------------

    def compute_sharpe_ratio(self, returns: pd.Series) -> float:
        """Annualized Sharpe ratio.

        .. math::
            SR = \\frac{\\bar{r} - r_f}{\\sigma} \\times \\sqrt{252}

        Parameters
        ----------
        returns : pd.Series
            Daily simple returns (not log returns).

        Returns
        -------
        float
            Annualized Sharpe ratio, or ``NaN`` when standard deviation is
            zero or the series is empty.
        """
        if returns.empty or len(returns) < 2:
            return float("nan")

        excess = returns - _DAILY_RISK_FREE
        std = excess.std(ddof=1)
        if std == 0 or np.isnan(std):
            return float("nan")

        return float((excess.mean() / std) * np.sqrt(_TRADING_DAYS_PER_YEAR))

    def compute_sortino_ratio(self, returns: pd.Series) -> float:
        """Annualized Sortino ratio (downside deviation denominator).

        Only returns below the daily risk-free rate contribute to the
        downside deviation calculation.

        Parameters
        ----------
        returns : pd.Series
            Daily simple returns.

        Returns
        -------
        float
            Annualized Sortino ratio, or ``NaN`` when downside deviation
            is zero or the series is empty.
        """
        if returns.empty or len(returns) < 2:
            return float("nan")

        excess = returns - _DAILY_RISK_FREE
        downside = excess[excess < 0]

        if downside.empty:
            # No downside observations -- ratio is conceptually infinite;
            # return NaN to avoid misleading callers.
            return float("nan")

        downside_std = np.sqrt((downside ** 2).mean())
        if downside_std == 0 or np.isnan(downside_std):
            return float("nan")

        return float((excess.mean() / downside_std) * np.sqrt(_TRADING_DAYS_PER_YEAR))

    async def compute_beta(
        self,
        ticker: str,
        period: str = "1y",
    ) -> float:
        """Portfolio beta of *ticker* against NIFTY 50.

        Fetches historical closes for both the asset and ``^NSEI`` via
        *yfinance*, computes daily returns, and runs an OLS regression.

        Parameters
        ----------
        ticker : str
            NSE symbol (e.g. ``"RELIANCE"``).  ``.NS`` is appended
            automatically if missing.
        period : str
            Look-back window understood by *yfinance* (default ``"1y"``).

        Returns
        -------
        float
            OLS beta coefficient, or ``NaN`` on failure.
        """
        nse_ticker = self._ensure_nse_suffix(ticker)
        try:
            asset_df, bench_df = await asyncio.gather(
                asyncio.to_thread(self._fetch_yf_history, nse_ticker, period),
                asyncio.to_thread(self._fetch_yf_history, _BENCHMARK_TICKER, period),
            )
        except Exception:
            logger.exception(
                "Failed to fetch price data for beta computation "
                "(ticker=%s, benchmark=%s).",
                nse_ticker,
                _BENCHMARK_TICKER,
            )
            return float("nan")

        asset_close = asset_df["Close"].squeeze()
        bench_close = bench_df["Close"].squeeze()

        # Align on common dates.
        combined = pd.DataFrame({
            "asset": asset_close,
            "bench": bench_close,
        }).dropna()

        if len(combined) < 10:
            logger.warning(
                "Insufficient overlapping data points (%d) for beta "
                "computation of %s.",
                len(combined),
                nse_ticker,
            )
            return float("nan")

        asset_ret = combined["asset"].pct_change().dropna()
        bench_ret = combined["bench"].pct_change().dropna()

        # Align after pct_change.
        asset_ret, bench_ret = asset_ret.align(bench_ret, join="inner")

        if len(asset_ret) < 5:
            return float("nan")

        slope, _intercept, _r, _p, _se = stats.linregress(bench_ret, asset_ret)
        return float(slope)

    def compute_max_drawdown(self, returns: pd.Series) -> float:
        """Maximum peak-to-trough drawdown from a daily return series.

        Parameters
        ----------
        returns : pd.Series
            Daily simple returns.

        Returns
        -------
        float
            Maximum drawdown expressed as a negative decimal (e.g. -0.25
            for a 25% drawdown), or ``NaN`` for empty input.
        """
        if returns.empty:
            return float("nan")

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        max_dd = drawdowns.min()

        return float(max_dd)

    def compute_var_95(self, returns: pd.Series) -> dict:
        """95% Value-at-Risk (single-day) via parametric and historical methods.

        Parameters
        ----------
        returns : pd.Series
            Daily simple returns.

        Returns
        -------
        dict
            ``{"parametric": float, "historical": float}`` -- both
            expressed as negative decimals (loss magnitude).
        """
        if returns.empty or len(returns) < 2:
            return {"parametric": None, "historical": None}

        # Parametric VaR: assume normal distribution.
        mu = returns.mean()
        sigma = returns.std(ddof=1)
        z = stats.norm.ppf(1 - _VAR_CONFIDENCE)  # negative z-score
        parametric_var = float(mu + z * sigma)

        # Historical VaR: empirical quantile.
        historical_var = float(returns.quantile(1 - _VAR_CONFIDENCE))

        return {
            "parametric": _nan_safe(parametric_var),
            "historical": _nan_safe(historical_var),
        }

    def compute_rolling_returns(self, returns: pd.Series) -> dict:
        """Trailing cumulative returns for 30-day, 90-day, and 1-year windows.

        Parameters
        ----------
        returns : pd.Series
            Daily simple returns sorted chronologically.

        Returns
        -------
        dict
            ``{"30d": float | None, "90d": float | None, "1y": float | None}``
        """
        result: dict[str, Any] = {"30d": None, "90d": None, "1y": None}

        windows = {"30d": 30, "90d": 90, "1y": _TRADING_DAYS_PER_YEAR}
        for label, days in windows.items():
            if len(returns) >= days:
                tail = returns.iloc[-days:]
                cumulative = float((1 + tail).prod() - 1)
                result[label] = _nan_safe(cumulative)

        return result

    async def compute_correlation_matrix(
        self,
        tickers: list[str],
        period: str = "6mo",
    ) -> dict:
        """Pearson correlation matrix of daily returns for *tickers*.

        Fetches adjusted-close data from *yfinance* in parallel, computes
        daily returns, and builds the correlation matrix.

        Parameters
        ----------
        tickers : list[str]
            NSE symbols.  ``.NS`` is appended automatically.
        period : str
            Look-back window (default ``"6mo"`` -- approximately 126
            trading days covers weekly rebalance lookback comfortably).

        Returns
        -------
        dict
            JSON-serializable nested dict:
            ``{"tickers": [...], "matrix": [[float, ...], ...]}``.
            ``NaN`` entries are replaced with ``None``.
        """
        if not tickers:
            return {"tickers": [], "matrix": []}

        nse_tickers = [self._ensure_nse_suffix(t) for t in tickers]

        # Fetch all histories concurrently.
        async def _safe_fetch(t: str) -> tuple[str, Optional[pd.DataFrame]]:
            try:
                df = await asyncio.to_thread(self._fetch_yf_history, t, period)
                return t, df
            except Exception:
                logger.warning("Failed to fetch data for %s; skipping.", t)
                return t, None

        results = await asyncio.gather(*[_safe_fetch(t) for t in nse_tickers])

        # Build a combined close-price DataFrame.
        close_frames: dict[str, pd.Series] = {}
        valid_tickers: list[str] = []
        for raw_ticker, orig_ticker in zip(nse_tickers, tickers):
            for fetched_ticker, df in results:
                if fetched_ticker == raw_ticker and df is not None:
                    close_frames[orig_ticker] = df["Close"].squeeze()
                    valid_tickers.append(orig_ticker)
                    break

        if len(valid_tickers) < 2:
            return {
                "tickers": valid_tickers,
                "matrix": [[1.0]] if valid_tickers else [],
            }

        prices = pd.DataFrame(close_frames).dropna()
        if prices.empty or len(prices) < 5:
            return {"tickers": valid_tickers, "matrix": []}

        daily_ret = prices.pct_change().dropna()
        corr = daily_ret.corr()

        # Serialize: replace NaN with None.
        matrix = [
            [_nan_safe(corr.iloc[i, j]) for j in range(len(valid_tickers))]
            for i in range(len(valid_tickers))
        ]

        return {"tickers": valid_tickers, "matrix": matrix}

    # ------------------------------------------------------------------
    # Aggregate portfolio risk
    # ------------------------------------------------------------------

    async def compute_portfolio_risk(
        self,
        holdings: list[dict],
        daily_returns: list[dict],
    ) -> dict:
        """Compute all risk metrics for the current portfolio.

        Parameters
        ----------
        holdings : list[dict]
            Each dict must contain at least ``{"ticker": str, ...}``.
            Optional keys: ``"weight"`` (float, 0-1).  If weights are
            absent, equal weighting is assumed.
        daily_returns : list[dict]
            Portfolio-level daily returns as
            ``[{"date": "YYYY-MM-DD", "return": float}, ...]``.

        Returns
        -------
        dict
            Aggregated risk report::

                {
                    "sharpe_ratio": float | None,
                    "sortino_ratio": float | None,
                    "max_drawdown": float | None,
                    "var_95": {"parametric": ..., "historical": ...},
                    "rolling_returns": {"30d": ..., "90d": ..., "1y": ...},
                    "betas": {"TICKER": float | None, ...},
                    "portfolio_beta": float | None,
                    "correlation_matrix": {...},
                    "holdings_count": int,
                    "observation_days": int,
                }
        """
        returns = _series_from_dicts(daily_returns)
        tickers = [h["ticker"] for h in holdings if "ticker" in h]

        # Determine portfolio weights.
        weights: dict[str, float] = {}
        total_weight = 0.0
        for h in holdings:
            t = h.get("ticker")
            if t is None:
                continue
            w = h.get("weight")
            if w is not None:
                weights[t] = float(w)
                total_weight += float(w)
        # Fall back to equal weights when none supplied or weights are
        # degenerate.
        if not weights or total_weight == 0:
            n = len(tickers) or 1
            weights = {t: 1.0 / n for t in tickers}

        # ---- Synchronous metrics ----
        sharpe = self.compute_sharpe_ratio(returns)
        sortino = self.compute_sortino_ratio(returns)
        max_dd = self.compute_max_drawdown(returns)
        var_95 = self.compute_var_95(returns)
        rolling = self.compute_rolling_returns(returns)

        # ---- Async metrics (beta per ticker + correlation) ----
        beta_tasks = {t: self.compute_beta(t) for t in tickers}
        corr_task = self.compute_correlation_matrix(tickers)

        # Run all async work concurrently.
        beta_results: dict[str, float] = {}
        if beta_tasks:
            completed = await asyncio.gather(
                *beta_tasks.values(), return_exceptions=True,
            )
            for ticker, result in zip(beta_tasks.keys(), completed):
                if isinstance(result, Exception):
                    logger.warning(
                        "Beta computation failed for %s: %s", ticker, result,
                    )
                    beta_results[ticker] = float("nan")
                else:
                    beta_results[ticker] = result

        correlation = await corr_task

        # Weighted portfolio beta.
        portfolio_beta = 0.0
        weight_sum = 0.0
        for t, beta in beta_results.items():
            w = weights.get(t, 0.0)
            if not math.isnan(beta):
                portfolio_beta += w * beta
                weight_sum += w
        if weight_sum > 0:
            portfolio_beta /= weight_sum
        else:
            portfolio_beta = float("nan")

        return {
            "sharpe_ratio": _nan_safe(sharpe),
            "sortino_ratio": _nan_safe(sortino),
            "max_drawdown": _nan_safe(max_dd),
            "var_95": var_95,
            "rolling_returns": rolling,
            "betas": {t: _nan_safe(b) for t, b in beta_results.items()},
            "portfolio_beta": _nan_safe(portfolio_beta),
            "correlation_matrix": correlation,
            "holdings_count": len(tickers),
            "observation_days": len(returns),
        }
