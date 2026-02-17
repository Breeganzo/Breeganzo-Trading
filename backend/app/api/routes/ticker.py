"""
Live Ticker API routes.

Provides endpoints for real-time price data -- single ticker, batch prices
for all portfolio holdings, market open/close status, and a WebSocket
stream that pushes price updates every 5 seconds during market hours.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import math
from datetime import date, datetime, time, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from zoneinfo import ZoneInfo

from app.core.config import get_settings
from app.db.database import get_db
from app.engines.transaction_cost import TransactionCostEngine
from app.middleware.auth import get_current_user
from app.models.models import DailyStockSnapshot, Portfolio, Ranking, User
from app.schemas.schemas import TickerData
from app.services.market_data import market_data_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ticker", tags=["Live Ticker"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IST = ZoneInfo("Asia/Kolkata")
_MARKET_OPEN = time(9, 15)
_MARKET_CLOSE = time(15, 30)

# Default market indices to include alongside portfolio tickers.
_DEFAULT_MARKET_TICKERS: list[str] = [
    "^NSEI",
    "^NSEBANK",
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
]

_BANKING_TICKERS: set[str] = {
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK",
    "INDUSINDBK", "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB",
    "BANDHANBNK", "AUBANK", "KTKBANK",
}
_LARGE_CAP_TICKERS: set[str] = {
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR",
    "ITC", "BHARTIARTL", "SBIN", "BAJFINANCE", "LT", "KOTAKBANK",
    "HCLTECH", "ASIANPAINT", "MARUTI", "TITAN", "WIPRO", "ULTRACEMCO",
    "NESTLEIND", "TATAMOTORS",
}
_COMMODITY_TICKERS: set[str] = {
    "RELIANCE", "ONGC", "COALINDIA", "HINDALCO", "TATASTEEL",
    "JSWSTEEL", "NATIONALUM", "SAIL",
}

# Active WebSocket connection counter.
_active_ws_connections: int = 0
_cost_engine = TransactionCostEngine()

# Strategy-first blend knobs for advisor/top-picks.
_STRATEGY_EDGE_WEIGHT: float = 0.90
_SENTIMENT_EDGE_WEIGHT: float = 0.10

_POSITIVE_SENTIMENT_TERMS = {
    "beat", "growth", "surge", "rally", "upgrade", "profit", "strong",
    "bullish", "record", "outperform", "win", "order", "expand",
}
_NEGATIVE_SENTIMENT_TERMS = {
    "miss", "decline", "drop", "fall", "downgrade", "loss", "weak",
    "bearish", "fraud", "investigation", "warning", "lawsuit", "cut",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_dict_to_ticker_data(ticker: str, data: dict | None) -> TickerData | None:
    """Convert a raw price dict from the market data service into a
    :class:`TickerData` schema instance.

    Returns ``None`` if the data dict is ``None`` or missing the required
    ``price`` field.
    """
    if data is None or data.get("price") is None:
        return None

    return TickerData(
        ticker=ticker,
        price=data["price"],
        change=data.get("change") or 0.0,
        change_pct=data.get("change_pct") or 0.0,
        volume=data.get("volume") or 0,
        high=data.get("high") or data["price"],
        low=data.get("low") or data["price"],
        open=data.get("open") or data["price"],
        prev_close=data.get("prev_close") or data["price"],
        timestamp=data.get("timestamp") or datetime.now(_IST).isoformat(),
    )


def _compute_next_market_open() -> str:
    """Return an ISO-formatted IST datetime string for the next market open.

    If the market is currently open, returns ``"Currently open"``.
    """
    now = datetime.now(_IST)
    current_time = now.time()
    weekday = now.weekday()  # Mon=0 ... Sun=6

    # Market is currently open.
    if weekday <= 4 and _MARKET_OPEN <= current_time <= _MARKET_CLOSE:
        return "Currently open"

    # Determine next open day/time.
    if weekday <= 4 and current_time < _MARKET_OPEN:
        # Before open on a weekday -- opens today.
        next_open = now.replace(
            hour=_MARKET_OPEN.hour, minute=_MARKET_OPEN.minute, second=0, microsecond=0
        )
    elif weekday == 4 and current_time > _MARKET_CLOSE:
        # After close on Friday -- opens next Monday.
        days_ahead = 3
        next_open = (now + timedelta(days=days_ahead)).replace(
            hour=_MARKET_OPEN.hour, minute=_MARKET_OPEN.minute, second=0, microsecond=0
        )
    elif weekday == 5:
        # Saturday -- opens Monday.
        next_open = (now + timedelta(days=2)).replace(
            hour=_MARKET_OPEN.hour, minute=_MARKET_OPEN.minute, second=0, microsecond=0
        )
    elif weekday == 6:
        # Sunday -- opens Monday.
        next_open = (now + timedelta(days=1)).replace(
            hour=_MARKET_OPEN.hour, minute=_MARKET_OPEN.minute, second=0, microsecond=0
        )
    else:
        # After close on a weekday (Mon-Thu) -- opens next day.
        next_open = (now + timedelta(days=1)).replace(
            hour=_MARKET_OPEN.hour, minute=_MARKET_OPEN.minute, second=0, microsecond=0
        )

    return next_open.isoformat()


async def _get_tracked_tickers(db: AsyncSession, user_id) -> list[str]:
    """Return a deduplicated list of portfolio tickers plus default market
    tickers for the given user."""
    result = await db.execute(
        select(Portfolio.ticker).where(Portfolio.user_id == user_id)
    )
    portfolio_tickers = [row[0] for row in result.fetchall()]

    # Merge with defaults, preserving order and avoiding duplicates.
    seen: set[str] = set(portfolio_tickers)
    all_tickers = list(portfolio_tickers)
    for t in _DEFAULT_MARKET_TICKERS:
        if t not in seen:
            all_tickers.append(t)
            seen.add(t)

    return all_tickers


def _base_symbol(ticker: str) -> str:
    t = str(ticker or "").upper().strip()
    if t.endswith(".NS") or t.endswith(".BO"):
        return t.rsplit(".", 1)[0]
    return t


def _sector_bucket_for(base_ticker: str, sector_hint: str | None = None) -> str:
    if sector_hint:
        s = str(sector_hint).strip()
        if s:
            return s
    if base_ticker in _BANKING_TICKERS:
        return "banking"
    if base_ticker in _COMMODITY_TICKERS:
        return "commodity"
    if base_ticker in _LARGE_CAP_TICKERS:
        return "large_cap"
    if base_ticker.endswith("SMALL") or base_ticker.startswith("SMALL"):
        return "small_cap"
    return "other"


def _signal_from_expected_or_change(
    expected_return: float | None,
    change_pct: float | None,
) -> str:
    if expected_return is not None:
        # expected_return is decimal in ranking engine.
        if expected_return >= 0.01:
            return "BUY"
        if expected_return <= -0.01:
            return "SELL"
        return "HOLD"
    if change_pct is not None:
        if change_pct >= 1.0:
            return "BUY"
        if change_pct <= -1.0:
            return "SELL"
    return "HOLD"


async def _latest_ranking_map(db: AsyncSession) -> dict[str, dict]:
    """Map base ticker -> latest overall ranking row for signal hints."""
    rows = (
        await db.execute(
            select(Ranking)
            .where(Ranking.category == "overall")
            .order_by(desc(Ranking.computed_at), Ranking.rank_position.asc())
            .limit(300)
        )
    ).scalars().all()

    if not rows:
        return {}

    latest_ts = rows[0].computed_at
    out: dict[str, dict] = {}
    for row in rows:
        if row.computed_at != latest_ts:
            break
        key = _base_symbol(row.ticker)
        out[key] = {
            "expected_return": row.expected_return,
            "score": row.score,
            "momentum_30d": row.momentum_30d,
            "volatility": row.volatility,
            "liquidity_score": row.liquidity_score,
            "avg_volume": row.avg_volume,
            "current_price": row.current_price,
            "rank_position": row.rank_position,
            "computed_at": row.computed_at.isoformat() if row.computed_at else None,
        }
    return out


async def _portfolio_sector_hints(db: AsyncSession, user_id) -> dict[str, str]:
    rows = (
        await db.execute(
            select(Portfolio.ticker, Portfolio.sector).where(Portfolio.user_id == user_id)
        )
    ).all()
    hints: dict[str, str] = {}
    for ticker, sector in rows:
        if ticker:
            hints[_base_symbol(str(ticker))] = str(sector or "").strip()
    return hints


async def _build_stocks_overview(
    *,
    db: AsyncSession,
    user_id,
    limit: int = 100,
    portfolio_only: bool = False,
) -> list[dict]:
    if portfolio_only:
        raw = (
            await db.execute(select(Portfolio.ticker).where(Portfolio.user_id == user_id))
        ).all()
        tickers = [str(row[0]) for row in raw if row and row[0]]
    else:
        tickers = await _get_tracked_tickers(db, user_id)
    if not tickers:
        return []

    deduped: list[str] = []
    seen_base: set[str] = set()
    for raw_ticker in tickers:
        base = _base_symbol(raw_ticker)
        if not base or base in seen_base:
            continue
        deduped.append(raw_ticker)
        seen_base.add(base)
    tickers = deduped

    ranking_map = await _latest_ranking_map(db)
    sector_hints = await _portfolio_sector_hints(db, user_id)
    live_map = await market_data_service.get_batch_prices(tickers)

    items: list[dict] = []
    for ticker in tickers:
        data = live_map.get(ticker)
        if not data:
            continue
        current_price = float(data.get("price") or 0.0)
        if current_price <= 0:
            continue
        change_pct = data.get("change_pct")
        base = _base_symbol(ticker)
        rank = ranking_map.get(base, {})
        expected_return = rank.get("expected_return")
        signal = _signal_from_expected_or_change(expected_return, change_pct)
        items.append(
            {
                "ticker": ticker,
                "base_ticker": base,
                "sector_bucket": _sector_bucket_for(base, sector_hints.get(base)),
                "current_price": round(current_price, 2),
                "open_price": float(data.get("open") or 0.0) or None,
                "prev_close": float(data.get("prev_close") or 0.0) or None,
                "high": float(data.get("high") or 0.0) or None,
                "low": float(data.get("low") or 0.0) or None,
                "change_pct": float(change_pct) if change_pct is not None else None,
                "volume": int(data.get("volume") or 0),
                "signal": signal,
                "expected_return": expected_return,
                "ranking_score": rank.get("score"),
                "ranking_position": rank.get("rank_position"),
                "ranking_computed_at": rank.get("computed_at"),
                "source": "live_market",
                "captured_at": data.get("timestamp") or datetime.now(_IST).isoformat(),
            }
        )

    items.sort(
        key=lambda x: abs(float(x.get("change_pct") or 0.0)),
        reverse=True,
    )
    return items[: max(1, int(limit))]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sign(value: float, eps: float = 1e-9) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _normalize_return_decimal(expected_return: float | None, change_pct: float | None) -> float:
    """
    Strategy returns are decimals internally.
    Fall back to live change when ranking return is unavailable.
    """
    er = _safe_float(expected_return)
    if er is not None:
        if abs(er) > 2.0:
            er = er / 100.0
        # Ranking engine expected_return is annualized; convert to daily scale
        # when the magnitude is too high for a single-session target.
        if abs(er) > 0.2:
            er = er / 252.0
        return _clamp(er, -0.05, 0.05)

    cp = _safe_float(change_pct)
    if cp is not None:
        return _clamp(cp / 100.0, -0.05, 0.05)
    return 0.0


def _signal_from_return_decimal(return_dec: float) -> str:
    if return_dec >= 0.01:
        return "BUY"
    if return_dec <= -0.01:
        return "SELL"
    return "HOLD"


def _headline_sentiment_score(text: str) -> float:
    low = str(text or "").lower()
    if not low:
        return 0.0
    pos = sum(1 for token in _POSITIVE_SENTIMENT_TERMS if token in low)
    neg = sum(1 for token in _NEGATIVE_SENTIMENT_TERMS if token in low)
    if pos == 0 and neg == 0:
        return 0.0
    return _clamp((pos - neg) / max(pos + neg, 1), -1.0, 1.0)


async def _fetch_same_day_news_sentiment(ticker: str) -> float:
    """
    Same-day-only sentiment in [-1, 1] from Yahoo headlines.
    Returns neutral 0 when news is unavailable.
    """
    import yfinance as yf  # lazy import for startup speed

    raw_ticker = str(ticker or "").strip().upper()
    nse_ticker = raw_ticker if raw_ticker.endswith((".NS", ".BO")) else f"{raw_ticker}.NS"
    today_ist = datetime.now(_IST).date()

    def _load_news() -> list[dict]:
        tk = yf.Ticker(nse_ticker)
        data = tk.news or []
        return data if isinstance(data, list) else []

    try:
        news_items = await asyncio.to_thread(_load_news)
    except Exception:
        return 0.0

    if not news_items:
        return 0.0

    scores: list[float] = []
    for item in news_items[:12]:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        publish_ts = item.get("providerPublishTime")
        if publish_ts is None:
            continue
        try:
            published = datetime.fromtimestamp(float(publish_ts), tz=ZoneInfo("UTC")).astimezone(_IST)
        except Exception:
            continue
        if published.date() != today_ist:
            continue
        scores.append(_headline_sentiment_score(title))

    if not scores:
        return 0.0
    return _clamp(sum(scores) / len(scores), -1.0, 1.0)


def _confidence_from_rank(rank: dict) -> float:
    rank_pos = int(_safe_float(rank.get("rank_position")) or 100)
    rank_score = _safe_float(rank.get("score")) or 0.0
    liq_score = _safe_float(rank.get("liquidity_score")) or 0.0

    pos_component = _clamp(100 - ((rank_pos - 1) * 2.5), 20, 98)
    score_component = _clamp(abs(rank_score) * 12.0, 0, 18)
    liq_component = _clamp(abs(liq_score) * 8.0, 0, 12)
    return round(_clamp(pos_component + score_component + liq_component - 20, 20, 98), 2)


def _agreement_from_signals(rank: dict, final_return_dec: float, change_pct: float | None) -> float:
    target_sign = _sign(final_return_dec)
    comparisons: list[int] = []

    er = _safe_float(rank.get("expected_return"))
    if er is not None:
        comparisons.append(_sign(er))
    mo = _safe_float(rank.get("momentum_30d"))
    if mo is not None:
        comparisons.append(_sign(mo))
    cp = _safe_float(change_pct)
    if cp is not None:
        comparisons.append(_sign(cp / 100.0))

    if not comparisons:
        return 50.0

    matches = 0
    for val in comparisons:
        if target_sign == 0:
            if val == 0:
                matches += 1
        elif val == target_sign:
            matches += 1

    return round(_clamp((matches / len(comparisons)) * 100.0, 5.0, 100.0), 2)


def _liquidity_factor(rank: dict, volume: int) -> float:
    avg_vol = _safe_float(rank.get("avg_volume"))
    if avg_vol is not None and avg_vol > 0 and volume > 0:
        return _clamp(volume / avg_vol, 0.5, 1.5)
    liq = _safe_float(rank.get("liquidity_score"))
    if liq is not None:
        return _clamp(0.7 + (liq / 10.0), 0.6, 1.5)
    return 1.0


def _entry_range(current_price: float, rank: dict) -> tuple[float, float]:
    vol = abs(_safe_float(rank.get("volatility")) or 0.2)
    width_pct = _clamp(vol / 25.0, 0.004, 0.02)
    low = round(current_price * (1 - width_pct), 2)
    high = round(current_price * (1 + width_pct), 2)
    return low, high


def _strategy_target_price(current_price: float, return_dec: float) -> float:
    return round(current_price * (1 + return_dec), 2)


async def _build_top_picks(
    *,
    db: AsyncSession,
    user_id,
    source: str,
    signal: str | None,
    n: int,
) -> list[dict]:
    rows = await _build_stocks_overview(
        db=db,
        user_id=user_id,
        limit=max(120, n * 6),
        portfolio_only=False,
    )
    if not rows:
        return []

    ranking_map = await _latest_ranking_map(db)
    picks: list[dict] = []
    for row in rows:
        ticker = row.get("ticker")
        if not ticker:
            continue
        current_price = float(row.get("current_price") or 0.0)
        if current_price <= 0:
            continue

        base = _base_symbol(ticker)
        rank = ranking_map.get(base, {})
        strategy_return_dec = _normalize_return_decimal(
            rank.get("expected_return"),
            row.get("change_pct"),
        )
        strategy_price = _strategy_target_price(current_price, strategy_return_dec)
        confidence = _confidence_from_rank(rank)
        agreement = _agreement_from_signals(rank, strategy_return_dec, row.get("change_pct"))
        lf = _liquidity_factor(rank, int(row.get("volume") or 0))
        composite = abs(strategy_return_dec) * (confidence / 100.0) * (agreement / 100.0) * lf
        entry_low, entry_high = _entry_range(current_price, rank)

        picks.append(
            {
                "ticker": ticker,
                "base_ticker": base,
                "sector_bucket": row.get("sector_bucket") or "other",
                "current_price": round(current_price, 2),
                "strategy_return_pct": round(strategy_return_dec * 100.0, 2),
                "strategy_price": strategy_price,
                "ai_return_pct": None,
                "ai_price": None,
                "signal_strategy": _signal_from_return_decimal(strategy_return_dec),
                "signal_ai": None,
                "confidence": confidence,
                "agreement": agreement,
                "score": round(composite * 100.0, 4),
                "entry_range_low": entry_low,
                "entry_range_high": entry_high,
                "ranking_position": rank.get("rank_position"),
                "ranking_computed_at": rank.get("computed_at"),
                "captured_at": row.get("captured_at"),
                "source": "strategy",
            }
        )

    picks.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    picks = picks[: max(10, min(len(picks), n * 4))]

    if source == "ai":
        sem = asyncio.Semaphore(6)

        async def _sentiment_task(pick: dict) -> None:
            async with sem:
                sentiment = await _fetch_same_day_news_sentiment(str(pick.get("ticker")))
            strategy_dec = float(pick.get("strategy_return_pct") or 0.0) / 100.0
            ai_dec = _clamp(
                (strategy_dec * _STRATEGY_EDGE_WEIGHT) + (sentiment * _SENTIMENT_EDGE_WEIGHT * 0.03),
                -0.5,
                0.5,
            )
            current_price = float(pick.get("current_price") or 0.0)
            pick["ai_return_pct"] = round(ai_dec * 100.0, 2)
            pick["ai_price"] = _strategy_target_price(current_price, ai_dec)
            pick["signal_ai"] = _signal_from_return_decimal(ai_dec)
            pick["sentiment_score"] = round(sentiment, 4)
            pick["source"] = "ai"
            if abs(ai_dec) > abs(strategy_dec):
                pick["score"] = round(float(pick.get("score") or 0.0) * 1.03, 4)

        await asyncio.gather(*[_sentiment_task(p) for p in picks])
    else:
        for pick in picks:
            pick["ai_return_pct"] = round((float(pick.get("strategy_return_pct") or 0.0) / 100.0) * 100.0, 2)
            pick["ai_price"] = pick.get("strategy_price")
            pick["signal_ai"] = pick.get("signal_strategy")

    if signal:
        sig = signal.upper()
        if source == "ai":
            picks = [p for p in picks if str(p.get("signal_ai") or "").upper() == sig]
        else:
            picks = [p for p in picks if str(p.get("signal_strategy") or "").upper() == sig]

    picks.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return picks[: max(1, int(n))]


async def _build_advisor_open_buy_list(
    *,
    db: AsyncSession,
    user_id,
    n: int,
    budget: float,
) -> dict:
    picks = await _build_top_picks(
        db=db,
        user_id=user_id,
        source="strategy",
        signal="BUY",
        n=max(n * 3, 20),
    )
    if not picks:
        return {
            "budget": round(budget, 2),
            "estimated_total_cost": 0.0,
            "estimated_total_fees": 0.0,
            "picks": [],
        }

    allocated: list[dict] = []
    cash_left = float(max(0.0, budget))
    total_fees = 0.0

    for pick in picks:
        if len(allocated) >= n or cash_left <= 0:
            break

        strategy_price = float(pick.get("strategy_price") or 0.0)
        if strategy_price <= 0:
            continue

        side_budget = cash_left / max(1, (n - len(allocated)))
        try:
            one_cost = _cost_engine.calculate_costs(
                price=strategy_price,
                quantity=1,
                trade_type="BUY",
            )
        except Exception:
            continue
        one_share_total = strategy_price + float(one_cost.get("total_cost") or 0.0)
        if one_share_total <= 0:
            continue

        qty = int(side_budget // one_share_total)
        if qty <= 0:
            continue

        cost_breakdown = _cost_engine.calculate_costs(
            price=strategy_price,
            quantity=qty,
            trade_type="BUY",
        )
        fee = round(float(cost_breakdown.get("total_cost") or 0.0), 2)
        trade_cost = round((strategy_price * qty) + fee, 2)
        if trade_cost > cash_left:
            continue

        stop_loss = round(float(pick.get("entry_range_low") or strategy_price * 0.98), 2)
        risk_distance = max(strategy_price - stop_loss, strategy_price * 0.005)
        target_price = round(strategy_price + (risk_distance * 1.8), 2)
        rr = round(max(target_price - strategy_price, 0.01) / max(risk_distance, 0.01), 2)

        allocated.append(
            {
                "ticker": pick.get("ticker"),
                "sector_bucket": pick.get("sector_bucket"),
                "strategy_price_at_open": round(strategy_price, 2),
                "current_price": round(float(pick.get("current_price") or 0.0), 2),
                "suggested_qty": qty,
                "estimated_fee": fee,
                "est_trade_cost": trade_cost,
                "stop_loss_price": stop_loss,
                "target_price": round(target_price, 2),
                "risk_reward": rr,
                "entry_range_low": float(pick.get("entry_range_low") or 0.0) or None,
                "entry_range_high": float(pick.get("entry_range_high") or 0.0) or None,
                "confidence": pick.get("confidence"),
                "agreement": pick.get("agreement"),
                "score": pick.get("score"),
                "source": "strategy",
            }
        )
        cash_left = round(cash_left - trade_cost, 2)
        total_fees = round(total_fees + fee, 2)

    return {
        "budget": round(budget, 2),
        "estimated_total_cost": round(budget - cash_left, 2),
        "estimated_total_fees": round(total_fees, 2),
        "remaining_cash": round(cash_left, 2),
        "picks": allocated,
    }


def _indicator_snapshot(hist, current_price: float) -> dict:
    if hist is None or getattr(hist, "empty", True):
        return {}
    close = hist["Close"].squeeze()
    volume = hist["Volume"].squeeze()
    high = hist["High"].squeeze()
    low = hist["Low"].squeeze()
    if hasattr(close, "iloc") and len(close) < 20:
        return {}

    sma20 = float(close.tail(20).mean())
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else float(close.mean())
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    typical = (high + low + close) / 3.0
    vwap = float((typical * volume).tail(20).sum() / max(volume.tail(20).sum(), 1))
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain.iloc[-1] / max(loss.iloc[-1], 1e-9)
    rsi14 = float(100 - (100 / (1 + rs)))

    signal = "HOLD"
    if current_price > sma20 and rsi14 < 70:
        signal = "BUY"
    elif current_price < sma20 and rsi14 > 30:
        signal = "SELL"

    return {
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "ema20": round(ema20, 2),
        "vwap": round(vwap, 2),
        "rsi14": round(rsi14, 2),
        "indicator_signal": signal,
    }


async def _build_stock_detail(db: AsyncSession, user_id, ticker: str) -> dict:
    ticker_norm = str(ticker or "").strip().upper()
    if not ticker_norm:
        raise HTTPException(status_code=400, detail="Ticker is required")

    live = await market_data_service.get_live_price(ticker_norm)
    if not live or not live.get("price"):
        raise HTTPException(status_code=404, detail=f"No live market data for {ticker_norm}")

    current_price = float(live.get("price") or 0.0)
    if current_price <= 0:
        raise HTTPException(status_code=404, detail=f"Invalid live price for {ticker_norm}")

    ranking_map = await _latest_ranking_map(db)
    base = _base_symbol(ticker_norm)
    rank = ranking_map.get(base, {})
    strategy_dec = _normalize_return_decimal(rank.get("expected_return"), live.get("change_pct"))
    ai_sentiment = await _fetch_same_day_news_sentiment(ticker_norm)
    ai_dec = _clamp(
        (strategy_dec * _STRATEGY_EDGE_WEIGHT) + (ai_sentiment * _SENTIMENT_EDGE_WEIGHT * 0.03),
        -0.5,
        0.5,
    )

    hist = await market_data_service.get_historical_data(ticker_norm, period="6mo")
    indicators = _indicator_snapshot(hist, current_price)
    entry_low, entry_high = _entry_range(current_price, rank)
    strategy_price = _strategy_target_price(current_price, strategy_dec)
    ai_price = _strategy_target_price(current_price, ai_dec)

    return {
        "ticker": ticker_norm,
        "current_price": round(current_price, 2),
        "open_price": _safe_float(live.get("open")),
        "prev_close": _safe_float(live.get("prev_close")),
        "day_high": _safe_float(live.get("high")),
        "day_low": _safe_float(live.get("low")),
        "volume": int(live.get("volume") or 0),
        "sector_bucket": _sector_bucket_for(base),
        "strategy_price_at_open": strategy_price,
        "strategy_return_pct": round(strategy_dec * 100.0, 2),
        "ai_predicted_price": ai_price,
        "ai_return_pct": round(ai_dec * 100.0, 2),
        "strategy_signal": _signal_from_return_decimal(strategy_dec),
        "ai_signal": _signal_from_return_decimal(ai_dec),
        "entry_range_low": entry_low,
        "entry_range_high": entry_high,
        "confidence": _confidence_from_rank(rank),
        "agreement": _agreement_from_signals(rank, strategy_dec, live.get("change_pct")),
        "indicators": indicators,
        "sentiment_score": round(ai_sentiment, 4),
        "captured_at": live.get("timestamp") or datetime.now(_IST).isoformat(),
    }


async def _build_expected_vs_actual(db: AsyncSession, user_id, day: date) -> list[dict]:
    snapshot_rows = (
        await db.execute(
            select(DailyStockSnapshot)
            .where(
                DailyStockSnapshot.user_id == user_id,
                DailyStockSnapshot.snapshot_date == day,
            )
            .order_by(DailyStockSnapshot.ticker.asc())
        )
    ).scalars().all()
    if not snapshot_rows:
        return []

    tickers = [r.ticker for r in snapshot_rows if r.ticker]
    live_map = await market_data_service.get_batch_prices(tickers)
    ranking_map = await _latest_ranking_map(db)
    results: list[dict] = []
    for row in snapshot_rows:
        ticker = row.ticker
        open_price = float(row.open_price or row.current_price or 0.0)
        if open_price <= 0:
            continue
        current_live = live_map.get(ticker) or {}
        current_price = float(current_live.get("price") or row.current_price or 0.0)
        if current_price <= 0:
            continue

        rank = ranking_map.get(_base_symbol(ticker), {})
        strategy_dec = _normalize_return_decimal(rank.get("expected_return"), row.change_pct)
        sentiment = await _fetch_same_day_news_sentiment(ticker)
        ai_dec = _clamp(
            (strategy_dec * _STRATEGY_EDGE_WEIGHT) + (sentiment * _SENTIMENT_EDGE_WEIGHT * 0.03),
            -0.5,
            0.5,
        )
        actual_return = ((current_price - open_price) / open_price) * 100.0
        strategy_return = strategy_dec * 100.0
        ai_return = ai_dec * 100.0

        strategy_price = round(open_price * (1 + strategy_dec), 2)
        ai_price = round(open_price * (1 + ai_dec), 2)
        direction_match = _sign(actual_return) == _sign(strategy_return)

        results.append(
            {
                "ticker": ticker,
                "open_price": round(open_price, 2),
                "current_price": round(current_price, 2),
                "close_price": round(current_price, 2),
                "strategy_price_at_open": strategy_price,
                "ai_price_at_open": ai_price,
                "strategy_return_pct": round(strategy_return, 2),
                "ai_return_pct": round(ai_return, 2),
                "actual_return_pct": round(actual_return, 2),
                "alpha_pct": round(actual_return - strategy_return, 2),
                "direction_comparison": bool(direction_match),
                "captured_at": row.captured_at.isoformat() if row.captured_at else None,
            }
        )
    return results


# ---------------------------------------------------------------------------
# GET /prices -- Batch prices for all tracked tickers
# ---------------------------------------------------------------------------

@router.get("/prices", response_model=list[TickerData])
async def get_prices(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return current prices for all portfolio tickers plus default market
    indices (NIFTY 50, Bank NIFTY proxies).
    """
    tickers = await _get_tracked_tickers(db, current_user.id)

    if not tickers:
        return []

    try:
        batch_prices = await market_data_service.get_batch_prices(tickers)
    except Exception:
        logger.exception("Failed to fetch batch prices for ticker endpoint")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to fetch market data. Please try again later.",
        )

    ticker_data_list: list[TickerData] = []
    for ticker in tickers:
        td = _price_dict_to_ticker_data(ticker, batch_prices.get(ticker))
        if td is not None:
            ticker_data_list.append(td)

    return ticker_data_list


# ---------------------------------------------------------------------------
# GET /price/{ticker} -- Single ticker price
# ---------------------------------------------------------------------------

@router.get("/price/{ticker}", response_model=TickerData)
async def get_single_price(
    ticker: str,
    current_user: User = Depends(get_current_user),
):
    """Return live price data for a single ticker symbol."""
    try:
        data = await market_data_service.get_live_price(ticker)
    except Exception:
        logger.exception("Failed to fetch price for %s", ticker)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unable to fetch price for {ticker}.",
        )

    td = _price_dict_to_ticker_data(ticker, data)
    if td is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No price data available for ticker '{ticker}'.",
        )

    return td


# ---------------------------------------------------------------------------
# GET /market-status -- Is the market open?
# ---------------------------------------------------------------------------

@router.get("/market-status")
async def get_market_status():
    """Return the current market open/close status, next open time, and
    current IST timestamp.
    """
    is_open = await market_data_service.is_market_open()
    next_open = _compute_next_market_open()
    current_time = datetime.now(_IST).isoformat()

    return {
        "is_open": is_open,
        "next_open": next_open,
        "current_time": current_time,
    }


# ---------------------------------------------------------------------------
# GET /top-picks -- Strategy / AI picks (ranking-driven, no placeholders)
# ---------------------------------------------------------------------------

@router.get("/top-picks")
async def get_top_picks(
    source: str = Query("strategy", pattern="^(strategy|ai)$"),
    signal: str | None = Query(None, pattern="^(BUY|SELL|HOLD)$"),
    n: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    picks = await _build_top_picks(
        db=db,
        user_id=current_user.id,
        source=source,
        signal=signal,
        n=n,
    )
    return {
        "source": source,
        "signal_filter": signal,
        "count": len(picks),
        "captured_at": datetime.now(_IST).isoformat(),
        "items": picks,
    }


# ---------------------------------------------------------------------------
# GET /advisor/open-buy-list -- Strategy-first budget-aware buy list
# ---------------------------------------------------------------------------

@router.get("/advisor/open-buy-list")
async def get_advisor_open_buy_list(
    n: int = Query(10, ge=1, le=25),
    budget: float = Query(40000.0, gt=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    payload = await _build_advisor_open_buy_list(
        db=db,
        user_id=current_user.id,
        n=n,
        budget=budget,
    )
    return {
        "captured_at": datetime.now(_IST).isoformat(),
        "source": "strategy",
        **payload,
    }


# ---------------------------------------------------------------------------
# GET /stock-detail/{ticker} -- Detailed stock page payload
# ---------------------------------------------------------------------------

@router.get("/stock-detail/{ticker}")
async def get_stock_detail(
    ticker: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    payload = await _build_stock_detail(db, current_user.id, ticker)
    return payload


# ---------------------------------------------------------------------------
# GET /expected-vs-actual -- open-vs-current comparison by date
# ---------------------------------------------------------------------------

@router.get("/expected-vs-actual")
async def get_expected_vs_actual(
    snapshot_date: str | None = Query(
        None,
        description="Date in YYYY-MM-DD format. Defaults to today.",
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if snapshot_date:
        try:
            day = date.fromisoformat(snapshot_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    else:
        day = date.today()

    rows = await _build_expected_vs_actual(db, current_user.id, day)
    return {
        "snapshot_date": str(day),
        "count": len(rows),
        "items": rows,
    }


# ---------------------------------------------------------------------------
# GET /stocks/overview -- Live grouped stock overview
# ---------------------------------------------------------------------------

@router.get("/stocks/overview")
async def get_stocks_overview(
    limit: int = Query(100, ge=1, le=500, description="Maximum rows"),
    portfolio_only: bool = Query(
        False,
        description="If true, return only portfolio tickers.",
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        items = await _build_stocks_overview(
            db=db,
            user_id=current_user.id,
            limit=limit,
            portfolio_only=portfolio_only,
        )
    except Exception:
        logger.exception("Failed to build stocks overview")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to fetch live overview data at the moment.",
        )

    grouped: dict[str, list[dict]] = {}
    for row in items:
        grouped.setdefault(row.get("sector_bucket") or "other", []).append(row)

    return {
        "captured_at": datetime.now(_IST).isoformat(),
        "count": len(items),
        "grouped_count": {k: len(v) for k, v in grouped.items()},
        "grouped": grouped,
        "items": items,
    }


# ---------------------------------------------------------------------------
# Snapshot persistence/export
# ---------------------------------------------------------------------------

@router.post("/snapshot/today")
async def capture_today_snapshot(
    limit: int = Query(150, ge=1, le=500, description="Rows to capture"),
    portfolio_only: bool = Query(
        False,
        description="Capture only portfolio tickers if true.",
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    items = await _build_stocks_overview(
        db=db,
        user_id=current_user.id,
        limit=limit,
        portfolio_only=portfolio_only,
    )
    today = date.today()

    await db.execute(
        delete(DailyStockSnapshot).where(
            DailyStockSnapshot.user_id == current_user.id,
            DailyStockSnapshot.snapshot_date == today,
            DailyStockSnapshot.source == "live_market",
        )
    )

    for row in items:
        db.add(
            DailyStockSnapshot(
                user_id=current_user.id,
                snapshot_date=today,
                ticker=row["ticker"],
                sector_bucket=row.get("sector_bucket"),
                current_price=float(row.get("current_price") or 0.0),
                open_price=row.get("open_price"),
                prev_close=row.get("prev_close"),
                high=row.get("high"),
                low=row.get("low"),
                change_pct=row.get("change_pct"),
                volume=row.get("volume"),
                signal=row.get("signal"),
                source="live_market",
            )
        )

    await db.flush()
    return {
        "ok": True,
        "snapshot_date": str(today),
        "saved_count": len(items),
        "source": "live_market",
    }


@router.get("/snapshot/today")
async def get_today_snapshot(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    today = date.today()
    rows = (
        await db.execute(
            select(DailyStockSnapshot)
            .where(
                DailyStockSnapshot.user_id == current_user.id,
                DailyStockSnapshot.snapshot_date == today,
            )
            .order_by(DailyStockSnapshot.ticker.asc())
        )
    ).scalars().all()

    items = [
        {
            "ticker": r.ticker,
            "sector_bucket": r.sector_bucket,
            "current_price": r.current_price,
            "open_price": r.open_price,
            "prev_close": r.prev_close,
            "high": r.high,
            "low": r.low,
            "change_pct": r.change_pct,
            "volume": r.volume,
            "signal": r.signal,
            "source": r.source,
            "captured_at": r.captured_at.isoformat() if r.captured_at else None,
        }
        for r in rows
    ]

    return {"snapshot_date": str(today), "count": len(items), "items": items}


@router.get("/snapshot/today/export.csv")
async def export_today_snapshot_csv(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    today = date.today()
    rows = (
        await db.execute(
            select(DailyStockSnapshot)
            .where(
                DailyStockSnapshot.user_id == current_user.id,
                DailyStockSnapshot.snapshot_date == today,
            )
            .order_by(DailyStockSnapshot.ticker.asc())
        )
    ).scalars().all()

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(
        [
            "snapshot_date",
            "ticker",
            "sector_bucket",
            "current_price",
            "open_price",
            "prev_close",
            "high",
            "low",
            "change_pct",
            "volume",
            "signal",
            "source",
            "captured_at",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                str(today),
                r.ticker,
                r.sector_bucket,
                r.current_price,
                r.open_price,
                r.prev_close,
                r.high,
                r.low,
                r.change_pct,
                r.volume,
                r.signal,
                r.source,
                r.captured_at.isoformat() if r.captured_at else "",
            ]
        )
    payload = out.getvalue().encode("utf-8")

    return StreamingResponse(
        io.BytesIO(payload),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=snapshot-{today}.csv",
        },
    )


@router.get("/snapshot/today/export.xlsx")
async def export_today_snapshot_xlsx(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        from openpyxl import Workbook
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="openpyxl is not installed on server.",
        )

    today = date.today()
    rows = (
        await db.execute(
            select(DailyStockSnapshot)
            .where(
                DailyStockSnapshot.user_id == current_user.id,
                DailyStockSnapshot.snapshot_date == today,
            )
            .order_by(DailyStockSnapshot.ticker.asc())
        )
    ).scalars().all()

    wb = Workbook()
    ws = wb.active
    ws.title = "snapshot"
    ws.append(
        [
            "snapshot_date",
            "ticker",
            "sector_bucket",
            "current_price",
            "open_price",
            "prev_close",
            "high",
            "low",
            "change_pct",
            "volume",
            "signal",
            "source",
            "captured_at",
        ]
    )
    for r in rows:
        ws.append(
            [
                str(today),
                r.ticker,
                r.sector_bucket,
                r.current_price,
                r.open_price,
                r.prev_close,
                r.high,
                r.low,
                r.change_pct,
                r.volume,
                r.signal,
                r.source,
                r.captured_at.isoformat() if r.captured_at else "",
            ]
        )

    buff = io.BytesIO()
    wb.save(buff)
    buff.seek(0)
    return StreamingResponse(
        buff,
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        headers={
            "Content-Disposition": f"attachment; filename=snapshot-{today}.xlsx",
        },
    )


# ---------------------------------------------------------------------------
# WebSocket /ws -- Live price stream
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def websocket_price_stream(websocket: WebSocket):
    """WebSocket endpoint that pushes live price updates every 5 seconds
    during market hours.

    Connection limits are enforced via ``MAX_WS_CONNECTIONS`` from
    application settings.  When the market is closed, updates are paused
    and a status message is sent instead.

    Messages sent to the client:

    * ``{"type": "price_update", "data": [...TickerData...]}``
    * ``{"type": "market_closed", "message": "...", "next_open": "..."}``
    * ``{"type": "error", "message": "..."}``
    """
    global _active_ws_connections

    settings = get_settings()

    # Enforce connection limit.
    if _active_ws_connections >= settings.MAX_WS_CONNECTIONS:
        await websocket.close(
            code=1008,
            reason="Maximum WebSocket connections reached. Please try again later.",
        )
        return

    await websocket.accept()
    _active_ws_connections += 1
    logger.info(
        "WebSocket connected. Active connections: %d / %d",
        _active_ws_connections,
        settings.MAX_WS_CONNECTIONS,
    )

    try:
        # Obtain portfolio tickers for the connected user. Because the WS
        # endpoint bypasses normal Depends injection we open a session
        # manually and fetch default tickers. We include market indices
        # regardless so the feed is always useful.
        from app.db.database import async_session_factory

        tracked_tickers: list[str] = list(_DEFAULT_MARKET_TICKERS)

        try:
            async with async_session_factory() as session:
                result = await session.execute(select(Portfolio.ticker))
                portfolio_tickers = [row[0] for row in result.fetchall()]
                # Prepend portfolio tickers before market indices.
                seen: set[str] = set()
                merged: list[str] = []
                for t in portfolio_tickers + list(_DEFAULT_MARKET_TICKERS):
                    if t not in seen:
                        merged.append(t)
                        seen.add(t)
                tracked_tickers = merged
        except Exception:
            logger.warning(
                "Could not load portfolio tickers for WebSocket; "
                "falling back to market indices only."
            )

        update_interval = settings.TICKER_UPDATE_INTERVAL  # default: 5s

        while True:
            try:
                is_open = await market_data_service.is_market_open()

                if is_open:
                    # Fetch batch prices and stream to the client.
                    batch_prices = await market_data_service.get_batch_prices(
                        tracked_tickers
                    )
                    ticker_data_list: list[dict] = []
                    for ticker in tracked_tickers:
                        td = _price_dict_to_ticker_data(
                            ticker, batch_prices.get(ticker)
                        )
                        if td is not None:
                            ticker_data_list.append(td.model_dump(mode="json"))

                    await websocket.send_json({
                        "type": "price_update",
                        "data": ticker_data_list,
                    })
                else:
                    # Market is closed -- send a status message and wait
                    # longer before the next check.
                    next_open = _compute_next_market_open()
                    await websocket.send_json({
                        "type": "market_closed",
                        "message": "Market is currently closed.",
                        "next_open": next_open,
                    })

                await asyncio.sleep(update_interval)

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected normally.")
                break
            except Exception as exc:
                logger.warning("Error in WebSocket price loop: %s", exc)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Temporary error fetching price data.",
                    })
                except Exception:
                    # Client likely disconnected; break out.
                    break
                await asyncio.sleep(update_interval)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected during setup.")
    except Exception:
        logger.exception("Unexpected WebSocket error.")
    finally:
        _active_ws_connections = max(0, _active_ws_connections - 1)
        logger.info(
            "WebSocket closed. Active connections: %d / %d",
            _active_ws_connections,
            settings.MAX_WS_CONNECTIONS,
        )
