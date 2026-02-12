"""
AI Analytics API routes.

Provides endpoints for AI-powered explanations of risk metrics and
holistic portfolio risk assessments.  All AI interactions are constrained
to risk education and management suggestions -- never trade signals.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.middleware.auth import get_current_user
from app.models.models import Portfolio, User
from app.schemas.schemas import AIExplainRequest, AIExplainResponse
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["AI Analytics"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _fetch_holdings(db: AsyncSession, user_id: Any) -> list[dict]:
    """Load portfolio holdings for a user and return them as dicts.

    Each dict contains ``ticker``, ``quantity``, ``avg_buy_price``,
    ``total_invested``, ``sector``, and a computed ``weight``.
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
            "weight": round(invested / total_invested, 4),
        })

    return holdings


async def _fetch_risk_metrics(db: AsyncSession, user_id: Any) -> dict[str, Any]:
    """Attempt to compute risk metrics for the user's portfolio.

    Imports the risk engine lazily and returns default values on failure
    so the AI endpoint can still provide a useful response.
    """
    from app.engines.risk_engine import RiskEngine

    holdings = await _fetch_holdings(db, user_id)
    if not holdings:
        return {}

    risk_engine = RiskEngine()

    try:
        risk_data = await risk_engine.compute_portfolio_risk(holdings, [])
    except Exception:
        logger.warning(
            "Risk computation failed for AI portfolio analysis; "
            "proceeding with empty metrics."
        )
        return {}

    # Flatten to simple metric -> value mapping for the AI service.
    metrics: dict[str, Any] = {}
    for key in (
        "sharpe_ratio",
        "sortino_ratio",
        "portfolio_beta",
        "max_drawdown",
        "var_95",
        "volatility",
    ):
        value = risk_data.get(key)
        if value is not None:
            metrics[key] = value

    return metrics


# ---------------------------------------------------------------------------
# POST /explain -- Explain a single risk metric
# ---------------------------------------------------------------------------

@router.post("/explain", response_model=AIExplainResponse)
async def explain_metric(
    body: AIExplainRequest,
    current_user: User = Depends(get_current_user),
):
    """Ask the AI to explain a risk metric in plain language and suggest
    improvements.

    The ``metric`` field should be one of the supported metric keys (e.g.
    ``sharpe_ratio``, ``beta``, ``max_drawdown``).  An optional
    ``context`` dict can provide the metric's current ``value`` and
    additional portfolio context to tailor the explanation.
    """
    metric = body.metric
    context = body.context or {}

    # Extract the metric value from context if provided.
    value = context.pop("value", 0.0)

    try:
        result = await ai_service.explain_metric(
            metric=metric,
            value=value,
            portfolio_context=context if context else None,
        )
    except Exception:
        logger.exception("AI explain_metric failed for '%s'", metric)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI explanation service is currently unavailable.",
        )

    return AIExplainResponse(
        metric=result.get("metric", metric),
        explanation=result.get("explanation", "Explanation unavailable."),
        suggestions=result.get("suggestions", []),
    )


# ---------------------------------------------------------------------------
# POST /portfolio-analysis -- Holistic portfolio risk explanation
# ---------------------------------------------------------------------------

@router.post("/portfolio-analysis")
async def portfolio_analysis(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Ask the AI for a holistic portfolio risk assessment.

    Fetches the user's current holdings and risk metrics, then sends
    them to the AI service for a plain-language analysis with actionable
    risk-management suggestions.
    """
    # 1. Fetch holdings.
    holdings = await _fetch_holdings(db, current_user.id)

    if not holdings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No portfolio holdings found. Add holdings before requesting AI analysis.",
        )

    # 2. Fetch risk metrics.
    risk_metrics = await _fetch_risk_metrics(db, current_user.id)

    # 3. Call the AI service.
    try:
        result = await ai_service.explain_portfolio_risk(
            risk_metrics=risk_metrics,
            holdings=holdings,
        )
    except Exception:
        logger.exception("AI portfolio analysis failed for user %s", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI portfolio analysis is currently unavailable.",
        )

    return {
        "explanation": result.get("explanation", "Analysis unavailable."),
        "risk_assessment": result.get("risk_assessment", {}),
        "suggestions": result.get("suggestions", []),
        "source": result.get("source", "unknown"),
    }
