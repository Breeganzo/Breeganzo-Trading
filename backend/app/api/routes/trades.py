"""
Trades API routes.

Provides endpoints for executing trades, viewing trade history, and
previewing transaction costs before execution.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.engines.transaction_cost import TransactionCostEngine
from app.middleware.auth import get_current_user
from app.models.models import Portfolio, Trade, User
from app.schemas.schemas import TradeCreate, TradeResponse
from app.services.email_service import send_trade_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trades", tags=["Trades"])

# ---------------------------------------------------------------------------
# Shared engine instance
# ---------------------------------------------------------------------------

_cost_engine = TransactionCostEngine()


# ---------------------------------------------------------------------------
# POST / -- Execute a new trade
# ---------------------------------------------------------------------------

@router.post("/", response_model=TradeResponse, status_code=status.HTTP_201_CREATED)
async def execute_trade(
    body: TradeCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Execute a new trade and update the portfolio accordingly.

    For **BUY** trades the portfolio quantity is increased, the weighted
    average buy price is recalculated, ``total_invested`` grows by the trade
    turnover, and cumulative buy-side transaction costs are updated.

    For **SELL** trades the portfolio quantity is decreased, realized PnL is
    computed as ``(sell_price - avg_buy_price) * quantity - sell_costs``, and
    cumulative sell-side transaction costs are updated.  The portfolio record
    is kept even when all shares are sold so that realized PnL history is
    preserved.

    Validation ensures the user cannot sell more shares than currently held.
    """
    ticker = body.ticker.upper()
    trade_type = body.trade_type.upper()

    # ------------------------------------------------------------------
    # 1. Calculate transaction costs
    # ------------------------------------------------------------------
    try:
        costs = _cost_engine.calculate_costs(
            price=body.price,
            quantity=body.quantity,
            trade_type=trade_type,
            slippage_pct=body.slippage_pct,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    total_amount = round(body.price * body.quantity, 2)

    # ------------------------------------------------------------------
    # 2. Load or create Portfolio record
    # ------------------------------------------------------------------
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.user_id == current_user.id,
            Portfolio.ticker == ticker,
        )
    )
    portfolio: Portfolio | None = result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # 3. Validate SELL quantity
    # ------------------------------------------------------------------
    if trade_type == "SELL":
        if portfolio is None or portfolio.quantity <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No holdings found for {ticker}. Cannot sell.",
            )
        if body.quantity > portfolio.quantity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Insufficient holdings for {ticker}. "
                    f"Held: {portfolio.quantity}, requested sell: {body.quantity}."
                ),
            )

    # ------------------------------------------------------------------
    # 4. Create the Trade record
    # ------------------------------------------------------------------
    trade = Trade(
        user_id=current_user.id,
        ticker=ticker,
        exchange=body.exchange,
        trade_type=trade_type,
        quantity=body.quantity,
        price=body.price,
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
        notes=body.notes,
    )
    if body.executed_at is not None:
        trade.executed_at = body.executed_at

    db.add(trade)

    # ------------------------------------------------------------------
    # 5. Update Portfolio
    # ------------------------------------------------------------------
    if trade_type == "BUY":
        if portfolio is None:
            # First purchase of this ticker -- create a new record.
            portfolio = Portfolio(
                user_id=current_user.id,
                ticker=ticker,
                exchange=body.exchange,
                quantity=body.quantity,
                avg_buy_price=body.price,
                total_invested=total_amount,
                realized_pnl=0.0,
                total_buy_costs=costs["total_cost"],
                total_sell_costs=0.0,
            )
            db.add(portfolio)
        else:
            # Existing holding -- recalculate weighted average price.
            prev_total = portfolio.avg_buy_price * portfolio.quantity
            new_total = body.price * body.quantity
            new_quantity = portfolio.quantity + body.quantity
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
        # Realized PnL: gross profit minus sell-side costs.
        gross_profit = (body.price - portfolio.avg_buy_price) * body.quantity
        realized = round(gross_profit - costs["total_cost"], 2)

        portfolio.realized_pnl = round(
            (portfolio.realized_pnl or 0.0) + realized, 2
        )
        portfolio.quantity = portfolio.quantity - body.quantity
        portfolio.total_sell_costs = round(
            (portfolio.total_sell_costs or 0.0) + costs["total_cost"], 2
        )

        # Reduce total_invested proportionally when selling.
        if portfolio.quantity == 0:
            portfolio.total_invested = 0.0
        else:
            # Reduce invested amount by the cost-basis of the shares sold.
            sold_basis = round(portfolio.avg_buy_price * body.quantity, 2)
            portfolio.total_invested = round(
                max((portfolio.total_invested or 0.0) - sold_basis, 0.0), 2
            )
        # avg_buy_price stays the same on sell (weighted avg doesn't change).

    # Flush so that server-generated defaults (id, executed_at) are populated.
    await db.flush()
    await db.refresh(trade)

    logger.info(
        "Trade executed: %s %s x%d @ %.2f for user %s (cost=%.2f)",
        trade_type,
        ticker,
        body.quantity,
        body.price,
        current_user.id,
        costs["total_cost"],
    )

    if trade_type == "BUY":
        await send_trade_email(
            action=trade_type,
            ticker=ticker,
            quantity=body.quantity,
            price=body.price,
            total_amount=total_amount,
            total_cost=costs["total_cost"],
            net_amount=costs["net_amount"],
            user_email=current_user.email,
            source="trades_api",
            executed_at=trade.executed_at,
        )

    return trade


# ---------------------------------------------------------------------------
# GET / -- Trade history
# ---------------------------------------------------------------------------

@router.get("/", response_model=list[TradeResponse])
async def get_trade_history(
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
    limit: int = Query(50, ge=1, le=500, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return trade history for the authenticated user.

    Results are ordered by ``executed_at`` descending (most recent first).
    Optionally filter by ticker symbol.  Supports pagination via *limit*
    and *offset*.
    """
    query = select(Trade).where(Trade.user_id == current_user.id)

    if ticker is not None:
        query = query.where(Trade.ticker == ticker.upper())

    query = query.order_by(Trade.executed_at.desc()).limit(limit).offset(offset)

    result = await db.execute(query)
    trades = result.scalars().all()

    return trades


# ---------------------------------------------------------------------------
# GET /cost-preview -- Preview transaction costs
# ---------------------------------------------------------------------------

@router.get("/cost-preview")
async def preview_transaction_costs(
    ticker: str = Query(..., description="Ticker symbol"),
    trade_type: str = Query(..., pattern="^(BUY|SELL)$", description="BUY or SELL"),
    quantity: int = Query(..., gt=0, description="Number of shares"),
    price: float = Query(..., gt=0, description="Price per share"),
    slippage_pct: Optional[float] = Query(
        None, ge=0.001, le=0.003, description="Slippage percentage (0.001-0.003)"
    ),
    current_user: User = Depends(get_current_user),
):
    """Preview the full transaction cost breakdown without executing a trade.

    Useful for the front-end to display estimated costs before the user
    confirms a trade.
    """
    try:
        costs = _cost_engine.calculate_costs(
            price=price,
            quantity=quantity,
            trade_type=trade_type.upper(),
            slippage_pct=slippage_pct,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return {
        "ticker": ticker.upper(),
        "trade_type": trade_type.upper(),
        "quantity": quantity,
        "price": price,
        "total_amount": round(price * quantity, 2),
        "brokerage": costs["brokerage"],
        "stt": costs["stt"],
        "exchange_charges": costs["exchange_charges"],
        "gst": costs["gst"],
        "sebi_charges": costs["sebi_charges"],
        "stamp_duty": costs["stamp_duty"],
        "slippage_cost": costs["slippage"],
        "total_cost": costs["total_cost"],
        "net_amount": costs["net_amount"],
    }
