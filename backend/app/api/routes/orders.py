"""
Order Book API routes.

Provides endpoints for creating draft orders, listing orders, confirming
drafts (executing them as trades with full transaction-cost accounting),
cancelling drafts, and viewing an order-book summary.
"""

from __future__ import annotations

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
from app.models.models import OrderBook, Portfolio, Trade, User
from app.schemas.schemas import OrderCreate, OrderResponse, TradeResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orders", tags=["Order Book"])

# ---------------------------------------------------------------------------
# Shared engine instance
# ---------------------------------------------------------------------------

_cost_engine = TransactionCostEngine()


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
