"""
System API routes.

Provides health checks, liveness probes, and full system validation for
monitoring and operational visibility.  Checks database, Redis, data feed
connectivity, and reports model/correlation freshness alongside uptime.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.redis import get_redis
from app.models.models import SystemStatus
from app.schemas.schemas import SystemHealthResponse
from app.services.market_data import market_data_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["System"])

# ---------------------------------------------------------------------------
# Module-level start time -- set once when the module is first imported.
# The main application can overwrite this via ``set_start_time`` if needed.
# ---------------------------------------------------------------------------

_start_time: float = time.time()


def set_start_time(t: float) -> None:
    """Allow the application entry-point to record the true startup time."""
    global _start_time
    _start_time = t


def _uptime_seconds() -> float:
    """Return seconds elapsed since the recorded start time."""
    return round(time.time() - _start_time, 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _check_database(db: AsyncSession) -> str:
    """Run a lightweight query to verify database connectivity.

    Returns ``"healthy"`` on success, ``"unhealthy"`` otherwise.
    """
    try:
        await db.execute(text("SELECT 1"))
        return "healthy"
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return "unhealthy"


async def _check_redis() -> str:
    """Send a PING to Redis and return ``"healthy"`` or ``"unhealthy"``."""
    try:
        r = await get_redis()
        pong = await r.ping()
        return "healthy" if pong else "unhealthy"
    except Exception as exc:
        logger.error("Redis health check failed: %s", exc)
        return "unhealthy"


async def _check_data_feed() -> str:
    """Attempt to fetch a single well-known ticker to verify the data feed.

    Uses ``^NSEI`` (NIFTY 50 index) as a lightweight probe.
    Returns ``"healthy"`` or ``"unhealthy"``.
    """
    try:
        data = await market_data_service.get_live_price("^NSEI")
        return "healthy" if data is not None else "unhealthy"
    except Exception as exc:
        logger.error("Data feed health check failed: %s", exc)
        return "unhealthy"


async def _get_system_timestamp(
    db: AsyncSession, component: str
) -> datetime | None:
    """Fetch ``last_updated`` from the ``system_status`` table for a given
    *component* name.  Returns ``None`` if no matching row exists.
    """
    try:
        result = await db.execute(
            select(SystemStatus.last_updated).where(
                SystemStatus.component == component
            )
        )
        row = result.scalar_one_or_none()
        return row
    except Exception as exc:
        logger.warning(
            "Could not fetch system timestamp for '%s': %s", component, exc
        )
        return None


# ---------------------------------------------------------------------------
# GET /health -- Full system health check
# ---------------------------------------------------------------------------

@router.get("/health", response_model=SystemHealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Comprehensive health check covering database, Redis, data feed,
    model freshness, and correlation freshness.
    """
    db_status = await _check_database(db)
    redis_status = await _check_redis()
    feed_status = await _check_data_feed()

    model_last_updated = await _get_system_timestamp(db, "ranking_model")
    correlation_last_calculated = await _get_system_timestamp(
        db, "correlation_matrix"
    )

    # Overall status: healthy only when all critical components are up.
    if all(s == "healthy" for s in (db_status, redis_status, feed_status)):
        overall = "healthy"
    elif db_status == "unhealthy":
        overall = "unhealthy"
    else:
        overall = "degraded"

    return SystemHealthResponse(
        status=overall,
        database=db_status,
        redis=redis_status,
        data_feed=feed_status,
        model_last_updated=model_last_updated,
        correlation_last_calculated=correlation_last_calculated,
        uptime_seconds=_uptime_seconds(),
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# GET /ping -- Simple liveness probe
# ---------------------------------------------------------------------------

@router.get("/ping")
async def ping():
    """Lightweight liveness probe.  No dependency checks -- if the server
    can respond, it returns 200.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# POST /validate -- Full system validation
# ---------------------------------------------------------------------------

@router.post("/validate")
async def validate_system(db: AsyncSession = Depends(get_db)):
    """Run a thorough validation of every subsystem and return a detailed
    breakdown suitable for ops dashboards or CI smoke tests.
    """
    results: dict[str, Any] = {}
    errors: list[str] = []

    # 1. Database ----------------------------------------------------------
    db_status = await _check_database(db)
    results["database"] = {
        "status": db_status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    if db_status != "healthy":
        errors.append("Database connectivity check failed")

    # 2. Redis -------------------------------------------------------------
    redis_status = await _check_redis()
    redis_detail: dict[str, Any] = {
        "status": redis_status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    if redis_status == "healthy":
        try:
            r = await get_redis()
            info = await r.info("server")
            redis_detail["redis_version"] = info.get("redis_version", "unknown")
        except Exception:
            redis_detail["redis_version"] = "unknown"
    results["redis"] = redis_detail
    if redis_status != "healthy":
        errors.append("Redis connectivity check failed")

    # 3. Data feed ---------------------------------------------------------
    feed_status = await _check_data_feed()
    results["data_feed"] = {
        "status": feed_status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    if feed_status != "healthy":
        errors.append("Data feed check failed (could not fetch ^NSEI)")

    # 4. Model freshness ---------------------------------------------------
    model_ts = await _get_system_timestamp(db, "ranking_model")
    results["ranking_model"] = {
        "last_updated": model_ts.isoformat() if model_ts else None,
        "status": "available" if model_ts else "not_available",
    }

    # 5. Correlation freshness ---------------------------------------------
    corr_ts = await _get_system_timestamp(db, "correlation_matrix")
    results["correlation_matrix"] = {
        "last_calculated": corr_ts.isoformat() if corr_ts else None,
        "status": "available" if corr_ts else "not_available",
    }

    # 6. Uptime ------------------------------------------------------------
    results["uptime_seconds"] = _uptime_seconds()

    # Overall verdict ------------------------------------------------------
    if not errors:
        overall = "healthy"
    elif db_status == "unhealthy":
        overall = "unhealthy"
    else:
        overall = "degraded"

    return {
        "status": overall,
        "components": results,
        "errors": errors,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }
