"""
Order Book API routes.

Provides endpoints for creating draft orders, listing orders, confirming
drafts (executing them as trades with full transaction-cost accounting),
cancelling drafts, and viewing an order-book summary.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.engines.transaction_cost import TransactionCostEngine
from app.middleware.auth import get_current_user
from app.models.models import OrderBook, Portfolio, SignalTrigger, Trade, User
from app.schemas.schemas import (
    OrderCreate,
    OrderResponse,
    SignalTriggerCreate,
    SignalTriggerResponse,
    TradeResponse,
)
from app.services.market_data import market_data_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orders", tags=["Order Book"])

# ---------------------------------------------------------------------------
# Shared engine instance
# ---------------------------------------------------------------------------

_cost_engine = TransactionCostEngine()

_POSITIVE_SENTIMENT_TERMS = {
    "beat",
    "growth",
    "surge",
    "rally",
    "upgrade",
    "profit",
    "strong",
    "bullish",
    "record",
    "outperform",
}
_NEGATIVE_SENTIMENT_TERMS = {
    "miss",
    "decline",
    "drop",
    "fall",
    "downgrade",
    "loss",
    "weak",
    "bearish",
    "fraud",
    "investigation",
    "warning",
}


def _ensure_nse_suffix(ticker: str) -> str:
    t = str(ticker or "").strip().upper()
    if not t:
        return t
    if t.startswith("^"):
        return t
    if t.endswith((".NS", ".BO")):
        return t
    return f"{t}.NS"


def _headline_sentiment_score(text: str) -> float:
    lower = str(text or "").lower()
    if not lower:
        return 0.0
    pos = sum(1 for token in _POSITIVE_SENTIMENT_TERMS if token in lower)
    neg = sum(1 for token in _NEGATIVE_SENTIMENT_TERMS if token in lower)
    if pos == 0 and neg == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / max(pos + neg, 1)))


async def _fetch_live_news_sentiment(ticker: str) -> float:
    """
    Lightweight live sentiment score in [-1, 1] using recent Yahoo headlines.

    This is intentionally bounded and conservative. If headlines are missing,
    sentiment defaults to neutral (0.0).
    """
    import yfinance as yf  # lazy import for startup speed

    nse_ticker = _ensure_nse_suffix(ticker)

    def _load_news() -> list[dict]:
        tk = yf.Ticker(nse_ticker)
        raw = tk.news or []
        return raw if isinstance(raw, list) else []

    try:
        news_items = await asyncio.to_thread(_load_news)
    except Exception:
        logger.warning("Failed to fetch ticker news for sentiment: %s", nse_ticker)
        return 0.0

    if not news_items:
        return 0.0

    now_ts = datetime.now(timezone.utc).timestamp()
    weighted_sum = 0.0
    weight_total = 0.0
    for item in news_items[:8]:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        publish_ts = float(item.get("providerPublishTime") or now_ts)
        age_hours = max(0.0, (now_ts - publish_ts) / 3600.0)
        # Fast decay: same-day gets highest influence.
        weight = 1.0 / (1.0 + age_hours / 6.0)
        score = _headline_sentiment_score(title)
        weighted_sum += score * weight
        weight_total += weight

    if weight_total <= 0:
        return 0.0
    return max(-1.0, min(1.0, weighted_sum / weight_total))


def _signal_is_triggered(signal: SignalTrigger, current_price: float) -> bool:
    low = float(signal.trigger_price_low or 0.0)
    high = float(signal.trigger_price_high or 0.0)
    if low > 0 and high > 0:
        if high < low:
            low, high = high, low
        return low <= current_price <= high

    action = str(signal.action or "BUY").upper()
    point = high if high > 0 else low
    if point <= 0:
        return False
    if action == "SELL":
        return current_price >= point
    # BUY / HOLD default: enter near or below threshold.
    return current_price <= point


def _sentiment_allows_signal(signal: SignalTrigger, sentiment_score: float) -> bool:
    floor = signal.sentiment_min
    ceil = signal.sentiment_max
    if floor is not None and sentiment_score < float(floor):
        return False
    if ceil is not None and sentiment_score > float(ceil):
        return False
    return True


async def _execute_portfolio_trade(
    *,
    db: AsyncSession,
    current_user: User,
    ticker: str,
    trade_type: str,
    quantity: int,
    price: float,
    exchange: str = "NSE",
    notes: str | None = None,
) -> Trade:
    """Execute BUY/SELL and update portfolio atomically (same logic as /trades)."""
    try:
        costs = _cost_engine.calculate_costs(
            price=price,
            quantity=quantity,
            trade_type=trade_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    result = await db.execute(
        select(Portfolio).where(
            Portfolio.user_id == current_user.id,
            Portfolio.ticker == ticker,
        )
    )
    portfolio: Portfolio | None = result.scalar_one_or_none()

    if trade_type == "SELL":
        if portfolio is None or portfolio.quantity <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No holdings found for {ticker}. Cannot sell.",
            )
        if quantity > portfolio.quantity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Insufficient holdings for {ticker}. "
                    f"Held: {portfolio.quantity}, requested sell: {quantity}."
                ),
            )

    total_amount = round(price * quantity, 2)
    trade = Trade(
        user_id=current_user.id,
        ticker=ticker,
        exchange=exchange,
        trade_type=trade_type,
        quantity=quantity,
        price=price,
        total_amount=total_amount,
        brokerage=costs["brokerage"],
        stt=costs["stt"],
        exchange_charges=costs["exchange_charges"],
        gst=costs["gst"],
        sebi_charges=costs["sebi_charges"],
        stamp_duty=costs["stamp_duty"],
        slippage_cost=costs["slippage"],
        total_cost=costs["total_cost"],
        net_amount=costs["net_amount"],
        notes=notes,
    )
    db.add(trade)

    if trade_type == "BUY":
        if portfolio is None:
            portfolio = Portfolio(
                user_id=current_user.id,
                ticker=ticker,
                exchange=exchange,
                quantity=quantity,
                avg_buy_price=price,
                total_invested=total_amount,
                realized_pnl=0.0,
                total_buy_costs=costs["total_cost"],
                total_sell_costs=0.0,
            )
            db.add(portfolio)
        else:
            prev_total = portfolio.avg_buy_price * portfolio.quantity
            new_total = price * quantity
            new_quantity = portfolio.quantity + quantity
            portfolio.avg_buy_price = round((prev_total + new_total) / new_quantity, 2)
            portfolio.quantity = new_quantity
            portfolio.total_invested = round(
                (portfolio.total_invested or 0.0) + total_amount, 2
            )
            portfolio.total_buy_costs = round(
                (portfolio.total_buy_costs or 0.0) + costs["total_cost"], 2
            )
    else:
        gross_profit = (price - portfolio.avg_buy_price) * quantity
        realized = round(gross_profit - costs["total_cost"], 2)
        portfolio.realized_pnl = round((portfolio.realized_pnl or 0.0) + realized, 2)
        portfolio.quantity = portfolio.quantity - quantity
        portfolio.total_sell_costs = round(
            (portfolio.total_sell_costs or 0.0) + costs["total_cost"], 2
        )
        if portfolio.quantity == 0:
            portfolio.total_invested = 0.0
        else:
            sold_basis = round(portfolio.avg_buy_price * quantity, 2)
            portfolio.total_invested = round(
                max((portfolio.total_invested or 0.0) - sold_basis, 0.0), 2
            )

    await db.flush()
    await db.refresh(trade)
    return trade


# ---------------------------------------------------------------------------
# GET /summary -- Order book summary  (registered BEFORE /{order_id} routes)
# ---------------------------------------------------------------------------

@router.get("/summary")
async def order_summary(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return aggregate counts by status and total pending buy/sell value.

    Pending value is the sum of ``target_price * quantity`` for all DRAFT
    orders, split by BUY and SELL sides.
    """
    # --- counts by status ---
    count_query = (
        select(OrderBook.status, func.count())
        .where(OrderBook.user_id == current_user.id)
        .group_by(OrderBook.status)
    )
    result = await db.execute(count_query)
    counts: dict[str, int] = {row[0]: row[1] for row in result.all()}

    draft_count = counts.get("DRAFT", 0)
    confirmed_count = counts.get("CONFIRMED", 0)
    cancelled_count = counts.get("CANCELLED", 0)

    # --- pending buy/sell value (DRAFT orders only) ---
    pending_value_query = (
        select(
            OrderBook.order_type,
            func.coalesce(func.sum(OrderBook.target_price * OrderBook.quantity), 0.0),
        )
        .where(
            OrderBook.user_id == current_user.id,
            OrderBook.status == "DRAFT",
        )
        .group_by(OrderBook.order_type)
    )
    pv_result = await db.execute(pending_value_query)
    pending_values: dict[str, float] = {row[0]: round(row[1], 2) for row in pv_result.all()}

    return {
        # Preferred keys for frontend components
        "draft_count": draft_count,
        "confirmed_count": confirmed_count,
        "cancelled_count": cancelled_count,
        # Backward-compatible aliases
        "draft": draft_count,
        "confirmed": confirmed_count,
        "cancelled": cancelled_count,
        "total_pending_buy_value": pending_values.get("BUY", 0.0),
        "total_pending_sell_value": pending_values.get("SELL", 0.0),
    }


# ---------------------------------------------------------------------------
# POST / -- Create a draft order
# ---------------------------------------------------------------------------

@router.post("/", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_draft_order(
    body: OrderCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new order with status ``DRAFT``.

    The order is not executed until it is explicitly confirmed via the
    ``POST /{order_id}/confirm`` endpoint.
    """
    order = OrderBook(
        user_id=current_user.id,
        ticker=body.ticker.upper(),
        exchange=body.exchange,
        order_type=body.order_type.upper(),
        quantity=body.quantity,
        target_price=body.target_price,
        status="DRAFT",
        notes=body.notes,
    )
    db.add(order)

    await db.flush()
    await db.refresh(order)

    logger.info(
        "Draft order created: %s %s x%d @ %.2f for user %s",
        order.order_type,
        order.ticker,
        order.quantity,
        order.target_price,
        current_user.id,
    )

    return order


# ---------------------------------------------------------------------------
# GET / -- List orders
# ---------------------------------------------------------------------------

@router.get("/", response_model=list[OrderResponse])
async def list_orders(
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        pattern="^(DRAFT|CONFIRMED|CANCELLED)$",
        description="Filter by order status",
    ),
    limit: int = Query(50, ge=1, le=500, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return orders for the authenticated user.

    Results are ordered by ``created_at`` descending (most recent first).
    Optionally filter by status.  Supports pagination via *limit* and
    *offset*.
    """
    query = select(OrderBook).where(OrderBook.user_id == current_user.id)

    if status_filter is not None:
        query = query.where(OrderBook.status == status_filter)

    query = query.order_by(OrderBook.created_at.desc()).limit(limit).offset(offset)

    result = await db.execute(query)
    orders = result.scalars().all()

    return orders


# ---------------------------------------------------------------------------
# POST /{order_id}/confirm -- Confirm a draft order (execute as trade)
# ---------------------------------------------------------------------------

@router.post("/{order_id}/confirm", response_model=TradeResponse)
async def confirm_order(
    order_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Confirm a DRAFT order and execute it as a trade.

    The confirmation flow mirrors the trade execution logic:

    1. Transaction costs are calculated via :class:`TransactionCostEngine`.
    2. A :class:`Trade` record is persisted.
    3. The user's :class:`Portfolio` is updated (weighted-average price for
       buys, realized PnL for sells).
    4. The order status is set to ``CONFIRMED`` and ``confirmed_at`` is
       recorded.

    Only orders in ``DRAFT`` status may be confirmed.
    """
    # ------------------------------------------------------------------
    # 1. Load and validate the order
    # ------------------------------------------------------------------
    result = await db.execute(
        select(OrderBook).where(
            OrderBook.id == order_id,
            OrderBook.user_id == current_user.id,
        )
    )
    order: OrderBook | None = result.scalar_one_or_none()

    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found.",
        )

    if order.status != "DRAFT":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only DRAFT orders can be confirmed. Current status: {order.status}.",
        )

    ticker = order.ticker.upper()
    trade_type = order.order_type.upper()

    # ------------------------------------------------------------------
    # 2. Calculate transaction costs
    # ------------------------------------------------------------------
    try:
        costs = _cost_engine.calculate_costs(
            price=order.target_price,
            quantity=order.quantity,
            trade_type=trade_type,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    total_amount = round(order.target_price * order.quantity, 2)

    # ------------------------------------------------------------------
    # 3. Load or create Portfolio record
    # ------------------------------------------------------------------
    port_result = await db.execute(
        select(Portfolio).where(
            Portfolio.user_id == current_user.id,
            Portfolio.ticker == ticker,
        )
    )
    portfolio: Portfolio | None = port_result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # 4. Validate SELL quantity
    # ------------------------------------------------------------------
    if trade_type == "SELL":
        if portfolio is None or portfolio.quantity <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No holdings found for {ticker}. Cannot sell.",
            )
        if order.quantity > portfolio.quantity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Insufficient holdings for {ticker}. "
                    f"Held: {portfolio.quantity}, requested sell: {order.quantity}."
                ),
            )

    # ------------------------------------------------------------------
    # 5. Create the Trade record
    # ------------------------------------------------------------------
    trade = Trade(
        user_id=current_user.id,
        ticker=ticker,
        exchange=order.exchange,
        trade_type=trade_type,
        quantity=order.quantity,
        price=order.target_price,
        total_amount=total_amount,
        brokerage=costs["brokerage"],
        stt=costs["stt"],
        exchange_charges=costs["exchange_charges"],
        gst=costs["gst"],
        sebi_charges=costs["sebi_charges"],
        stamp_duty=costs["stamp_duty"],
        slippage_cost=costs["slippage"],
        total_cost=costs["total_cost"],
        net_amount=costs["net_amount"],
        notes=order.notes,
    )
    db.add(trade)

    # ------------------------------------------------------------------
    # 6. Update Portfolio (same logic as trades route)
    # ------------------------------------------------------------------
    if trade_type == "BUY":
        if portfolio is None:
            portfolio = Portfolio(
                user_id=current_user.id,
                ticker=ticker,
                exchange=order.exchange,
                quantity=order.quantity,
                avg_buy_price=order.target_price,
                total_invested=total_amount,
                realized_pnl=0.0,
                total_buy_costs=costs["total_cost"],
                total_sell_costs=0.0,
            )
            db.add(portfolio)
        else:
            prev_total = portfolio.avg_buy_price * portfolio.quantity
            new_total = order.target_price * order.quantity
            new_quantity = portfolio.quantity + order.quantity
            portfolio.avg_buy_price = round(
                (prev_total + new_total) / new_quantity, 2
            )
            portfolio.quantity = new_quantity
            portfolio.total_invested = round(
                (portfolio.total_invested or 0.0) + total_amount, 2
            )
            portfolio.total_buy_costs = round(
                (portfolio.total_buy_costs or 0.0) + costs["total_cost"], 2
            )
    else:
        # SELL -- portfolio is guaranteed non-None by validation above.
        gross_profit = (order.target_price - portfolio.avg_buy_price) * order.quantity
        realized = round(gross_profit - costs["total_cost"], 2)

        portfolio.realized_pnl = round(
            (portfolio.realized_pnl or 0.0) + realized, 2
        )
        portfolio.quantity = portfolio.quantity - order.quantity
        portfolio.total_sell_costs = round(
            (portfolio.total_sell_costs or 0.0) + costs["total_cost"], 2
        )

        if portfolio.quantity == 0:
            portfolio.total_invested = 0.0
        else:
            sold_basis = round(portfolio.avg_buy_price * order.quantity, 2)
            portfolio.total_invested = round(
                max((portfolio.total_invested or 0.0) - sold_basis, 0.0), 2
            )

    # ------------------------------------------------------------------
    # 7. Update order status
    # ------------------------------------------------------------------
    order.status = "CONFIRMED"
    order.confirmed_at = datetime.now(timezone.utc)

    await db.flush()
    await db.refresh(trade)

    logger.info(
        "Order %s confirmed as trade: %s %s x%d @ %.2f for user %s (cost=%.2f)",
        order_id,
        trade_type,
        ticker,
        order.quantity,
        order.target_price,
        current_user.id,
        costs["total_cost"],
    )

    return trade


# ---------------------------------------------------------------------------
# DELETE /{order_id} -- Cancel a draft order (soft delete)
# ---------------------------------------------------------------------------

@router.delete("/{order_id}", response_model=OrderResponse)
async def cancel_order(
    order_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel a DRAFT order (soft delete).

    The order is not removed from the database; its status is set to
    ``CANCELLED``.  Only orders in ``DRAFT`` status may be cancelled.
    The portfolio is **not** affected.
    """
    result = await db.execute(
        select(OrderBook).where(
            OrderBook.id == order_id,
            OrderBook.user_id == current_user.id,
        )
    )
    order: OrderBook | None = result.scalar_one_or_none()

    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found.",
        )

    if order.status != "DRAFT":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only DRAFT orders can be cancelled. Current status: {order.status}.",
        )

    order.status = "CANCELLED"

    await db.flush()
    await db.refresh(order)

    logger.info(
        "Order %s cancelled for user %s",
        order_id,
        current_user.id,
    )

    return order


# ---------------------------------------------------------------------------
# Signal Trigger Queue (BUY / SELL / HOLD)
# ---------------------------------------------------------------------------


@router.get("/auto-signals", response_model=list[SignalTriggerResponse])
async def list_auto_signals(
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        pattern="^(PENDING|SKIPPED|CANCELLED)$",
        description="Optional signal status filter",
    ),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(SignalTrigger).where(SignalTrigger.user_id == current_user.id)
    if status_filter:
        query = query.where(SignalTrigger.status == status_filter)
    query = query.order_by(SignalTrigger.created_at.desc()).limit(limit)
    rows = (await db.execute(query)).scalars().all()
    return rows


@router.post(
    "/auto-signals",
    response_model=SignalTriggerResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_auto_signal(
    body: SignalTriggerCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    low = body.trigger_price_low
    high = body.trigger_price_high
    if low is None and high is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one trigger price (low/high) is required.",
        )
    if low is not None and high is not None and high < low:
        low, high = high, low

    action = body.action.upper()
    quantity = body.quantity if action in {"BUY", "SELL"} else 1
    signal = SignalTrigger(
        user_id=current_user.id,
        ticker=body.ticker.upper(),
        exchange=body.exchange,
        action=action,
        quantity=quantity,
        trigger_price_low=low,
        trigger_price_high=high,
        sentiment_min=body.sentiment_min,
        sentiment_max=body.sentiment_max,
        status="PENDING",
        source=body.source or "manual",
        notes=body.notes,
    )
    db.add(signal)
    await db.flush()
    await db.refresh(signal)
    return signal


@router.delete("/auto-signals/{signal_id}")
async def delete_auto_signal(
    signal_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    signal = (
        await db.execute(
            select(SignalTrigger).where(
                SignalTrigger.id == signal_id,
                SignalTrigger.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if signal is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Signal not found.")
    await db.delete(signal)
    return {"ok": True, "deleted_signal_id": str(signal_id)}


async def process_pending_signals_for_user(
    *,
    db: AsyncSession,
    current_user: User,
    limit: int = 25,
) -> dict:
    """Process pending auto signals for one user (internal helper)."""
    query = (
        select(SignalTrigger)
        .where(
            SignalTrigger.user_id == current_user.id,
            SignalTrigger.status == "PENDING",
        )
        .order_by(SignalTrigger.created_at.asc())
        .limit(max(1, int(limit or 25)))
    )
    pending = (await db.execute(query)).scalars().all()

    executed: list[dict] = []
    consumed_hold: list[dict] = []
    skipped: list[dict] = []

    for signal in pending:
        ticker = str(signal.ticker or "").upper()
        if not ticker:
            skipped.append({"signal_id": str(signal.id), "reason": "missing_ticker"})
            continue

        try:
            live = await market_data_service.get_live_price(ticker)
        except Exception:
            live = None
        current_price = float((live or {}).get("price") or 0.0)
        if current_price <= 0:
            skipped.append(
                {"signal_id": str(signal.id), "ticker": ticker, "reason": "no_live_price"}
            )
            continue

        if not _signal_is_triggered(signal, current_price):
            skipped.append(
                {
                    "signal_id": str(signal.id),
                    "ticker": ticker,
                    "reason": "price_not_in_trigger_range",
                    "current_price": round(current_price, 2),
                }
            )
            continue

        sentiment_score = await _fetch_live_news_sentiment(ticker)
        signal.sentiment_last = round(sentiment_score, 4)
        if not _sentiment_allows_signal(signal, sentiment_score):
            signal.status = "SKIPPED"
            signal.updated_at = datetime.now(timezone.utc)
            skipped.append(
                {
                    "signal_id": str(signal.id),
                    "ticker": ticker,
                    "reason": "sentiment_blocked",
                    "sentiment": round(sentiment_score, 4),
                }
            )
            continue

        action = str(signal.action or "HOLD").upper()
        if action == "HOLD":
            consumed_hold.append(
                {
                    "signal_id": str(signal.id),
                    "ticker": ticker,
                    "current_price": round(current_price, 2),
                    "sentiment": round(sentiment_score, 4),
                }
            )
            await db.delete(signal)
            continue

        qty = int(signal.quantity or 0)
        if qty <= 0:
            skipped.append(
                {
                    "signal_id": str(signal.id),
                    "ticker": ticker,
                    "reason": "invalid_quantity",
                }
            )
            continue

        note = (
            f"auto_signal:{signal.id} source={signal.source or 'manual'} "
            f"sentiment={sentiment_score:.3f}"
        )
        try:
            trade = await _execute_portfolio_trade(
                db=db,
                current_user=current_user,
                ticker=ticker,
                trade_type=action,
                quantity=qty,
                price=current_price,
                exchange=signal.exchange or "NSE",
                notes=note,
            )
        except HTTPException as exc:
            skipped.append(
                {
                    "signal_id": str(signal.id),
                    "ticker": ticker,
                    "reason": "trade_rejected",
                    "detail": str(exc.detail),
                }
            )
            continue

        executed.append(
            {
                "signal_id": str(signal.id),
                "trade_id": str(trade.id),
                "ticker": ticker,
                "action": action,
                "quantity": qty,
                "price": round(current_price, 2),
                "sentiment": round(sentiment_score, 4),
            }
        )
        await db.delete(signal)

    return {
        "ok": True,
        "processed": len(pending),
        "executed_count": len(executed),
        "hold_consumed_count": len(consumed_hold),
        "skipped_count": len(skipped),
        "executed": executed,
        "hold_consumed": consumed_hold,
        "skipped": skipped,
    }


@router.post("/auto-signals/process")
async def process_auto_signals(
    limit: int = Query(25, ge=1, le=200, description="Max pending signals to process"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Process pending DB triggers once:
      - checks live ticker price
      - applies sentiment gate
      - executes BUY/SELL trades when triggered
      - deletes consumed signals to avoid redundant execution
    """
    return await process_pending_signals_for_user(
        db=db, current_user=current_user, limit=limit
    )
