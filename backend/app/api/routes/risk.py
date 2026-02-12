"""
Risk Analytics API routes.

Provides endpoints for portfolio risk metrics, correlation matrices,
market regime detection, and forced recalculation.  Results are cached
in Redis to avoid redundant heavy computations against yfinance.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.redis import get_redis
from app.engines.regime_engine import RegimeEngine
from app.engines.risk_engine import RiskEngine
from app.middleware.auth import get_current_user
from app.models.models import DailyReturn, Portfolio, User
from app.schemas.schemas import RiskMetrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Risk Analytics"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_TTL_METRICS = 30 * 60       # 30 minutes
CACHE_TTL_CORRELATION = 60 * 60   # 1 hour (weekly recalc cadence)
CACHE_TTL_REGIME = 30 * 60        # 30 minutes

# Shared engine instances
_risk_engine = RiskEngine()
_regime_engine = RegimeEngine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_key_metrics(user_id: str) -> str:
    """Return the Redis cache key for a user's full risk metrics."""
    return f"risk:metrics:{user_id}"


def _cache_key_correlation(user_id: str) -> str:
    """Return the Redis cache key for a user's correlation matrix."""
    return f"risk:correlation:{user_id}"


CACHE_KEY_REGIME = "risk:regime"


async def _fetch_holdings(
    db: AsyncSession,
    user_id: Any,
) -> list[dict]:
    """Load portfolio holdings for a user and return them as dicts.

    Each dict contains ``ticker``, ``quantity``, ``avg_buy_price``,
    ``total_invested``, and ``sector``.  A weight field is derived from
    ``total_invested`` relative to the portfolio total.
    """
    result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == user_id)
    )
    rows = result.scalars().all()

    if not rows:
        return []

    total_invested = sum(r.total_invested or 0.0 for r in rows) or 1.0

    holdings: list[dict] = []
    for row in rows:
        invested = row.total_invested or 0.0
        holdings.append({
            "ticker": row.ticker,
            "quantity": row.quantity,
            "avg_buy_price": row.avg_buy_price,
            "total_invested": invested,
            "sector": row.sector,
            "weight": invested / total_invested,
        })

    return holdings


async def _fetch_daily_returns(
    db: AsyncSession,
    user_id: Any,
) -> list[dict]:
    """Load daily return records for a user, sorted chronologically.

    Returns a list of ``{"date": str, "return": float}`` dicts that the
    :class:`RiskEngine` expects.
    """
    result = await db.execute(
        select(DailyReturn)
        .where(DailyReturn.user_id == user_id)
        .order_by(DailyReturn.date.asc())
    )
    rows = result.scalars().all()

    return [
        {
            "date": row.date.isoformat() if isinstance(row.date, datetime) else str(row.date),
            "return": row.daily_return_pct or 0.0,
        }
        for row in rows
    ]


def _build_risk_metrics_response(
    risk_data: dict,
    regime_data: dict,
) -> RiskMetrics:
    """Combine risk engine and regime engine outputs into a RiskMetrics schema."""
    var_95_data = risk_data.get("var_95", {})
    # Prefer the parametric VaR as the single-number summary.
    var_95_value = (
        var_95_data.get("parametric")
        if isinstance(var_95_data, dict)
        else var_95_data
    )

    rolling = risk_data.get("rolling_returns", {})

    return RiskMetrics(
        sharpe_ratio=risk_data.get("sharpe_ratio"),
        sortino_ratio=risk_data.get("sortino_ratio"),
        portfolio_beta=risk_data.get("portfolio_beta"),
        max_drawdown=risk_data.get("max_drawdown"),
        var_95=var_95_value,
        rolling_return_30d=rolling.get("30d") if isinstance(rolling, dict) else None,
        rolling_return_90d=rolling.get("90d") if isinstance(rolling, dict) else None,
        rolling_return_1y=rolling.get("1y") if isinstance(rolling, dict) else None,
        regime=regime_data.get("regime", "unknown"),
        regime_details=regime_data,
        correlation_matrix=risk_data.get("correlation_matrix"),
        last_updated=datetime.now(timezone.utc),
    )


def _serialize(data: Any) -> str:
    """Serialize arbitrary data to a JSON string for Redis storage."""
    return json.dumps(data, default=str)


def _deserialize(raw: str) -> Any:
    """Deserialize a JSON string from Redis."""
    return json.loads(raw)


# ---------------------------------------------------------------------------
# GET / -- Full risk metrics for the authenticated user's portfolio
# ---------------------------------------------------------------------------

@router.get("/", response_model=RiskMetrics)
async def get_risk_metrics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RiskMetrics:
    """Return full risk metrics for the authenticated user's portfolio.

    Checks Redis cache first.  On a miss, fetches portfolio holdings and
    daily returns from the database, runs the risk engine and regime
    engine, caches the result for 30 minutes, and returns it.
    """
    user_id = str(current_user.id)
    cache_key = _cache_key_metrics(user_id)

    # 1. Check Redis cache.
    try:
        redis = await get_redis()
        cached = await redis.get(cache_key)
        if cached:
            logger.debug("Risk metrics cache hit for user %s", user_id)
            data = _deserialize(cached)
            return RiskMetrics(**data)
    except Exception:
        logger.warning("Redis unavailable for risk metrics cache lookup", exc_info=True)

    # 2. Fetch data from DB.
    holdings = await _fetch_holdings(db, current_user.id)
    daily_returns = await _fetch_daily_returns(db, current_user.id)

    if not holdings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No portfolio holdings found. Add holdings before requesting risk metrics.",
        )

    # 3. Compute risk metrics and detect regime.
    try:
        risk_data = await _risk_engine.compute_portfolio_risk(holdings, daily_returns)
    except Exception:
        logger.exception("Risk engine computation failed for user %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk computation failed. Check server logs for details.",
        )

    try:
        regime_data = await _regime_engine.detect_regime()
    except Exception:
        logger.exception("Regime detection failed for user %s", user_id)
        regime_data = RegimeEngine._default_result()

    # 4. Build response.
    metrics = _build_risk_metrics_response(risk_data, regime_data)

    # 5. Cache in Redis.
    try:
        redis = await get_redis()
        await redis.setex(cache_key, CACHE_TTL_METRICS, _serialize(metrics.model_dump()))
    except Exception:
        logger.warning("Failed to cache risk metrics in Redis", exc_info=True)

    return metrics


# ---------------------------------------------------------------------------
# GET /correlation -- Correlation matrix
# ---------------------------------------------------------------------------

@router.get("/correlation")
async def get_correlation_matrix(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the correlation matrix for the authenticated user's portfolio tickers.

    Cached in Redis for 1 hour (intended for weekly recalculation cadence).
    """
    user_id = str(current_user.id)
    cache_key = _cache_key_correlation(user_id)

    # 1. Check Redis cache.
    try:
        redis = await get_redis()
        cached = await redis.get(cache_key)
        if cached:
            logger.debug("Correlation matrix cache hit for user %s", user_id)
            return _deserialize(cached)
    except Exception:
        logger.warning("Redis unavailable for correlation cache lookup", exc_info=True)

    # 2. Get portfolio tickers.
    holdings = await _fetch_holdings(db, current_user.id)
    if not holdings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No portfolio holdings found. Add holdings before requesting correlation data.",
        )

    tickers = [h["ticker"] for h in holdings]

    # 3. Compute correlation matrix.
    try:
        correlation = await _risk_engine.compute_correlation_matrix(tickers)
    except Exception:
        logger.exception("Correlation matrix computation failed for user %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Correlation computation failed. Check server logs for details.",
        )

    result = {
        "tickers": correlation.get("tickers", []),
        "matrix": correlation.get("matrix", []),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    # 4. Cache in Redis.
    try:
        redis = await get_redis()
        await redis.setex(cache_key, CACHE_TTL_CORRELATION, _serialize(result))
    except Exception:
        logger.warning("Failed to cache correlation matrix in Redis", exc_info=True)

    return result


# ---------------------------------------------------------------------------
# GET /regime -- Current market regime (no auth required)
# ---------------------------------------------------------------------------

@router.get("/regime")
async def get_market_regime() -> dict:
    """Return the current market regime detected by the regime engine.

    This endpoint does not require authentication as regime data is
    market-wide, not user-specific.  Cached in Redis for 30 minutes.
    """
    # 1. Check Redis cache.
    try:
        redis = await get_redis()
        cached = await redis.get(CACHE_KEY_REGIME)
        if cached:
            logger.debug("Market regime cache hit")
            return _deserialize(cached)
    except Exception:
        logger.warning("Redis unavailable for regime cache lookup", exc_info=True)

    # 2. Detect regime.
    try:
        regime_data = await _regime_engine.detect_regime()
    except Exception:
        logger.exception("Regime detection failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Regime detection failed. Check server logs for details.",
        )

    # 3. Cache in Redis.
    try:
        redis = await get_redis()
        await redis.setex(CACHE_KEY_REGIME, CACHE_TTL_REGIME, _serialize(regime_data))
    except Exception:
        logger.warning("Failed to cache regime data in Redis", exc_info=True)

    return regime_data


# ---------------------------------------------------------------------------
# POST /recalculate -- Force recalculate all risk metrics
# ---------------------------------------------------------------------------

@router.post("/recalculate", response_model=RiskMetrics)
async def recalculate_risk_metrics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RiskMetrics:
    """Force recalculate all risk metrics, clearing any cached data first.

    Requires authentication.  Invalidates all risk-related cache keys for
    the user, recomputes everything from scratch, and returns fresh metrics.
    """
    user_id = str(current_user.id)

    logger.info(
        "Risk recalculation triggered by user %s (%s)",
        user_id,
        current_user.email,
    )

    # 1. Clear cached data for this user.
    keys_to_clear = [
        _cache_key_metrics(user_id),
        _cache_key_correlation(user_id),
        CACHE_KEY_REGIME,
    ]

    try:
        redis = await get_redis()
        for key in keys_to_clear:
            await redis.delete(key)
        logger.debug("Cleared %d risk cache keys for user %s", len(keys_to_clear), user_id)
    except Exception:
        logger.warning("Failed to clear risk cache keys in Redis", exc_info=True)

    # 2. Also clear the regime engine's in-memory cache so it refetches.
    _regime_engine._cache = None
    _regime_engine._cache_ts = 0.0

    # 3. Fetch fresh data from DB.
    holdings = await _fetch_holdings(db, current_user.id)
    daily_returns = await _fetch_daily_returns(db, current_user.id)

    if not holdings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No portfolio holdings found. Add holdings before recalculating risk.",
        )

    # 4. Recompute risk metrics.
    try:
        risk_data = await _risk_engine.compute_portfolio_risk(holdings, daily_returns)
    except Exception:
        logger.exception("Risk recalculation failed for user %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk recalculation failed. Check server logs for details.",
        )

    # 5. Redetect regime.
    try:
        regime_data = await _regime_engine.detect_regime()
    except Exception:
        logger.exception("Regime detection failed during recalculation for user %s", user_id)
        regime_data = RegimeEngine._default_result()

    # 6. Build response.
    metrics = _build_risk_metrics_response(risk_data, regime_data)

    # 7. Cache fresh results.
    try:
        redis = await get_redis()
        await redis.setex(
            _cache_key_metrics(user_id),
            CACHE_TTL_METRICS,
            _serialize(metrics.model_dump()),
        )
        await redis.setex(
            CACHE_KEY_REGIME,
            CACHE_TTL_REGIME,
            _serialize(regime_data),
        )

        # Also cache the correlation matrix from the risk data.
        correlation = risk_data.get("correlation_matrix")
        if correlation:
            corr_result = {
                "tickers": correlation.get("tickers", []),
                "matrix": correlation.get("matrix", []),
                "computed_at": datetime.now(timezone.utc).isoformat(),
            }
            await redis.setex(
                _cache_key_correlation(user_id),
                CACHE_TTL_CORRELATION,
                _serialize(corr_result),
            )
    except Exception:
        logger.warning("Failed to cache recalculated risk data in Redis", exc_info=True)

    logger.info("Risk recalculation completed for user %s", user_id)

    return metrics
