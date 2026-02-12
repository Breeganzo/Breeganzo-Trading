"""
Live Ticker API routes.

Provides endpoints for real-time price data -- single ticker, batch prices
for all portfolio holdings, market open/close status, and a WebSocket
stream that pushes price updates every 5 seconds during market hours.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from zoneinfo import ZoneInfo

from app.core.config import get_settings
from app.db.database import get_db
from app.middleware.auth import get_current_user
from app.models.models import Portfolio, User
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
_DEFAULT_MARKET_TICKERS: list[str] = ["^NSEI", "^NSEBANK"]

# Active WebSocket connection counter.
_active_ws_connections: int = 0


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
