"""
Portfolio API routes.

Provides endpoints for viewing portfolio holdings with live market data,
risk metrics, daily return history, and initial portfolio seeding.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.engines.risk_engine import RiskEngine
from app.engines.transaction_cost import TransactionCostEngine
from app.middleware.auth import get_current_user
from app.models.models import DailyReturn, Portfolio, User
from app.schemas.schemas import PortfolioHolding, PortfolioSummary
from app.services.market_data import market_data_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

# ---------------------------------------------------------------------------
# Shared engine instances
# ---------------------------------------------------------------------------

_risk_engine = RiskEngine()
_cost_engine = TransactionCostEngine()


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

class SeedHolding(BaseModel):
    """Schema for a single holding in the seed request body."""

    ticker: str
    quantity: int = Field(..., gt=0)
    avg_buy_price: float = Field(..., gt=0)
    sector: Optional[str] = None
    exchange: str = "NSE"


class SeedRequest(BaseModel):
    """Body for POST /portfolio/seed."""

    holdings: list[SeedHolding]


class DailyReturnRecord(BaseModel):
    """A single daily return data point."""

    date: datetime
    portfolio_value: float
    daily_return_pct: float
    total_invested: float
    total_pnl: float

    class Config:
        from_attributes = True


class SeedResponse(BaseModel):
    """Response from the seed endpoint."""

    created: int
    message: str


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _nan_safe(value) -> Optional[float]:
    """Convert NaN / Inf to ``None`` for safe JSON serialization."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
    except (TypeError, ValueError):
        return None
    return value


# ---------------------------------------------------------------------------
# GET / -- Full portfolio summary with live prices and risk metrics
# ---------------------------------------------------------------------------

@router.get("/", response_model=PortfolioSummary)
async def get_portfolio_summary(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the complete portfolio summary with live prices and risk metrics.

    For each holding the endpoint fetches the current market price, computes
    unrealized PnL, day change percentage, individual beta and volatility.
    It also computes sector exposure (value-weighted) and portfolio-level
    beta (value-weighted average of individual betas).
    """
    # 1. Fetch all portfolio holdings for the user.
    result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == current_user.id)
    )
    holdings_db = result.scalars().all()

    if not holdings_db:
        return PortfolioSummary(
            total_value=0.0,
            total_invested=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            total_transaction_costs=0.0,
            day_pnl=0.0,
            day_pnl_pct=0.0,
            holdings=[],
            sector_exposure={},
            beta_exposure=0.0,
        )

    # 2. Batch-fetch live prices for all tickers.
    tickers = [h.ticker for h in holdings_db]
    try:
        live_prices = await market_data_service.get_batch_prices(tickers)
    except Exception:
        logger.exception("Failed to fetch batch prices")
        live_prices = {}

    # 3. Fetch beta for each ticker (concurrently via the risk engine).
    betas: dict[str, Optional[float]] = {}
    for ticker in tickers:
        try:
            beta = await _risk_engine.compute_beta(ticker)
            betas[ticker] = _nan_safe(beta)
        except Exception:
            logger.warning("Beta computation failed for %s", ticker)
            betas[ticker] = None

    # 4. Fetch historical data to compute per-ticker volatility.
    volatilities: dict[str, Optional[float]] = {}
    for ticker in tickers:
        try:
            hist = await market_data_service.get_historical_data(ticker, period="1y")
            if not hist.empty:
                close_col = hist["Close"].squeeze()
                daily_ret = close_col.pct_change().dropna()
                if len(daily_ret) > 1:
                    vol = float(daily_ret.std() * (252 ** 0.5))
                    volatilities[ticker] = _nan_safe(vol)
                else:
                    volatilities[ticker] = None
            else:
                volatilities[ticker] = None
        except Exception:
            logger.warning("Volatility computation failed for %s", ticker)
            volatilities[ticker] = None

    # 5. Build enriched holdings list and accumulate totals.
    enriched_holdings: list[PortfolioHolding] = []
    total_value = 0.0
    total_invested = 0.0
    total_unrealized_pnl = 0.0
    total_realized_pnl = 0.0
    total_transaction_costs = 0.0
    day_pnl = 0.0

    # Sector exposure accumulators: sector -> sum of current value.
    sector_values: dict[str, float] = {}
    # For weighted beta: list of (value, beta) tuples.
    weighted_beta_parts: list[tuple[float, float]] = []

    for holding in holdings_db:
        price_data = live_prices.get(holding.ticker)
        current_price = price_data.get("price") if price_data else None
        day_change_pct = price_data.get("change_pct") if price_data else None

        holding_value = 0.0
        unrealized_pnl = None
        pnl_pct = None

        if current_price is not None:
            holding_value = current_price * holding.quantity
            unrealized_pnl = (current_price - holding.avg_buy_price) * holding.quantity
            if holding.total_invested and holding.total_invested > 0:
                pnl_pct = (unrealized_pnl / holding.total_invested) * 100
            elif holding.avg_buy_price > 0:
                pnl_pct = ((current_price - holding.avg_buy_price) / holding.avg_buy_price) * 100

            total_value += holding_value
            total_unrealized_pnl += unrealized_pnl

            # Day PnL: change_pct applied to holding value at previous close.
            if day_change_pct is not None:
                prev_close = price_data.get("prev_close")
                if prev_close is not None:
                    day_pnl += (current_price - prev_close) * holding.quantity

        total_invested += holding.total_invested or 0.0
        total_realized_pnl += holding.realized_pnl or 0.0
        total_transaction_costs += (holding.total_buy_costs or 0.0) + (holding.total_sell_costs or 0.0)

        # Sector exposure.
        sector = holding.sector or "Unknown"
        sector_values[sector] = sector_values.get(sector, 0.0) + holding_value

        # Beta weighting.
        ticker_beta = betas.get(holding.ticker)
        if ticker_beta is not None and holding_value > 0:
            weighted_beta_parts.append((holding_value, ticker_beta))

        enriched_holdings.append(
            PortfolioHolding(
                id=holding.id,
                ticker=holding.ticker,
                exchange=holding.exchange or "NSE",
                quantity=holding.quantity,
                avg_buy_price=holding.avg_buy_price,
                total_invested=holding.total_invested or 0.0,
                realized_pnl=holding.realized_pnl or 0.0,
                current_price=current_price,
                unrealized_pnl=round(unrealized_pnl, 2) if unrealized_pnl is not None else None,
                pnl_pct=round(pnl_pct, 2) if pnl_pct is not None else None,
                day_change_pct=day_change_pct,
                total_buy_costs=holding.total_buy_costs or 0.0,
                total_sell_costs=holding.total_sell_costs or 0.0,
                sector=holding.sector,
                beta=betas.get(holding.ticker),
                volatility=volatilities.get(holding.ticker),
            )
        )

    # 6. Compute sector exposure as percentage of total portfolio value.
    sector_exposure: dict[str, float] = {}
    if total_value > 0:
        for sector, value in sector_values.items():
            sector_exposure[sector] = round((value / total_value) * 100, 2)

    # 7. Compute portfolio beta (value-weighted average).
    beta_exposure = 0.0
    if weighted_beta_parts:
        total_beta_weight = sum(w for w, _ in weighted_beta_parts)
        if total_beta_weight > 0:
            beta_exposure = sum(w * b for w, b in weighted_beta_parts) / total_beta_weight

    # 8. Overall PnL.
    total_pnl = total_unrealized_pnl + total_realized_pnl
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0
    day_pnl_pct = (day_pnl / (total_value - day_pnl) * 100) if (total_value - day_pnl) > 0 else 0.0

    return PortfolioSummary(
        total_value=round(total_value, 2),
        total_invested=round(total_invested, 2),
        total_unrealized_pnl=round(total_unrealized_pnl, 2),
        total_realized_pnl=round(total_realized_pnl, 2),
        total_pnl=round(total_pnl, 2),
        total_pnl_pct=round(total_pnl_pct, 2),
        total_transaction_costs=round(total_transaction_costs, 2),
        day_pnl=round(day_pnl, 2),
        day_pnl_pct=round(day_pnl_pct, 2),
        holdings=enriched_holdings,
        sector_exposure=sector_exposure,
        beta_exposure=round(beta_exposure, 4),
    )


# ---------------------------------------------------------------------------
# GET /holdings -- Simplified holdings list (no risk metrics)
# ---------------------------------------------------------------------------

@router.get("/holdings", response_model=list[PortfolioHolding])
async def get_holdings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return a simplified list of portfolio holdings with live prices.

    This is a lighter-weight alternative to the full summary endpoint --
    it skips beta, volatility, and sector-exposure calculations.
    """
    result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == current_user.id)
    )
    holdings_db = result.scalars().all()

    if not holdings_db:
        return []

    # Batch-fetch live prices.
    tickers = [h.ticker for h in holdings_db]
    try:
        live_prices = await market_data_service.get_batch_prices(tickers)
    except Exception:
        logger.exception("Failed to fetch batch prices for holdings")
        live_prices = {}

    holdings_out: list[PortfolioHolding] = []
    for holding in holdings_db:
        price_data = live_prices.get(holding.ticker)
        current_price = price_data.get("price") if price_data else None
        day_change_pct = price_data.get("change_pct") if price_data else None

        unrealized_pnl = None
        pnl_pct = None
        if current_price is not None:
            unrealized_pnl = (current_price - holding.avg_buy_price) * holding.quantity
            if holding.total_invested and holding.total_invested > 0:
                pnl_pct = (unrealized_pnl / holding.total_invested) * 100
            elif holding.avg_buy_price > 0:
                pnl_pct = ((current_price - holding.avg_buy_price) / holding.avg_buy_price) * 100

        holdings_out.append(
            PortfolioHolding(
                id=holding.id,
                ticker=holding.ticker,
                exchange=holding.exchange or "NSE",
                quantity=holding.quantity,
                avg_buy_price=holding.avg_buy_price,
                total_invested=holding.total_invested or 0.0,
                realized_pnl=holding.realized_pnl or 0.0,
                current_price=current_price,
                unrealized_pnl=round(unrealized_pnl, 2) if unrealized_pnl is not None else None,
                pnl_pct=round(pnl_pct, 2) if pnl_pct is not None else None,
                day_change_pct=day_change_pct,
                total_buy_costs=holding.total_buy_costs or 0.0,
                total_sell_costs=holding.total_sell_costs or 0.0,
                sector=holding.sector,
                beta=None,
                volatility=None,
            )
        )

    return holdings_out


# ---------------------------------------------------------------------------
# GET /daily-returns -- Historical daily return records
# ---------------------------------------------------------------------------

@router.get("/daily-returns", response_model=list[DailyReturnRecord])
async def get_daily_returns(
    days: int = Query(default=90, ge=1, le=365, description="Number of days of history"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return daily portfolio return records for the requested look-back window.

    Defaults to 90 days.  Maximum is 365 days.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    result = await db.execute(
        select(DailyReturn)
        .where(
            DailyReturn.user_id == current_user.id,
            DailyReturn.date >= cutoff,
        )
        .order_by(DailyReturn.date.asc())
    )
    records = result.scalars().all()

    return [
        DailyReturnRecord(
            date=r.date,
            portfolio_value=r.portfolio_value,
            daily_return_pct=r.daily_return_pct or 0.0,
            total_invested=r.total_invested or 0.0,
            total_pnl=r.total_pnl or 0.0,
        )
        for r in records
    ]


# ---------------------------------------------------------------------------
# POST /seed -- Seed initial portfolio holdings
# ---------------------------------------------------------------------------

@router.post("/seed", response_model=SeedResponse, status_code=status.HTTP_201_CREATED)
async def seed_portfolio(
    body: SeedRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Seed the portfolio with an initial set of holdings.

    Creates ``Portfolio`` records for each holding in the request body.
    If a holding with the same ticker already exists for the user, it is
    skipped to avoid duplicates (the unique constraint on ``user_id`` +
    ``ticker`` would reject it anyway).

    Example seed payload for the initial KTK Bank holding::

        {
            "holdings": [
                {
                    "ticker": "KTKBANK",
                    "quantity": 196,
                    "avg_buy_price": 211.48,
                    "sector": "Banking"
                }
            ]
        }
    """
    if not body.holdings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one holding is required",
        )

    # Check which tickers already exist for this user.
    existing_result = await db.execute(
        select(Portfolio.ticker).where(Portfolio.user_id == current_user.id)
    )
    existing_tickers = {row[0] for row in existing_result.fetchall()}

    created_count = 0
    skipped: list[str] = []

    for item in body.holdings:
        if item.ticker.upper() in existing_tickers:
            skipped.append(item.ticker)
            logger.info(
                "Skipping duplicate ticker %s for user %s",
                item.ticker,
                current_user.id,
            )
            continue

        total_invested = item.avg_buy_price * item.quantity

        # Estimate initial buy-side transaction costs.
        try:
            buy_costs = _cost_engine.calculate_costs(
                price=item.avg_buy_price,
                quantity=item.quantity,
                trade_type="BUY",
            )
            total_buy_costs = buy_costs["total_cost"]
        except Exception:
            logger.warning(
                "Failed to compute transaction costs for %s; defaulting to 0",
                item.ticker,
            )
            total_buy_costs = 0.0

        new_holding = Portfolio(
            user_id=current_user.id,
            ticker=item.ticker.upper(),
            exchange=item.exchange,
            quantity=item.quantity,
            avg_buy_price=item.avg_buy_price,
            total_invested=round(total_invested, 2),
            realized_pnl=0.0,
            total_buy_costs=round(total_buy_costs, 2),
            total_sell_costs=0.0,
            sector=item.sector,
        )
        db.add(new_holding)
        existing_tickers.add(item.ticker.upper())
        created_count += 1

    # The session is committed by the get_db dependency on success.
    await db.flush()

    message = f"Successfully seeded {created_count} holding(s)"
    if skipped:
        message += f"; skipped {len(skipped)} duplicate(s): {', '.join(skipped)}"

    return SeedResponse(created=created_count, message=message)
