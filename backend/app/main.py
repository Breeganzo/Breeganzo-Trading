"""
QuantDesk Pro API -- main application entry-point.

Creates the FastAPI application, configures CORS, registers all routers,
and manages the async lifespan (database initialisation, Redis verification,
graceful shutdown).
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import get_settings
from app.db.database import close_db, init_db
from app.db.redis import close_redis, get_redis

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

settings = get_settings()
API_PREFIX: str = settings.API_PREFIX  # e.g. "/api/v1"

# ---------------------------------------------------------------------------
# Application start time (used for uptime calculation)
# ---------------------------------------------------------------------------

_app_start_time: float = 0.0
_auto_signal_worker_task: asyncio.Task | None = None


async def _auto_signal_worker_loop() -> None:
    """
    Background worker: processes pending DB signal triggers server-side.

    Runs independently of UI sessions so queued BUY/SELL/HOLD triggers can be
    consumed even when dashboard is closed.
    """
    # Lazy imports to avoid startup cycles.
    from sqlalchemy import select

    from app.api.routes.orders import process_pending_signals_for_user
    from app.db.database import async_session_factory
    from app.models.models import SignalTrigger, User
    from app.services.market_data import market_data_service

    interval = max(2, int(settings.AUTO_SIGNAL_INTERVAL_SEC or 10))
    batch_size = max(1, int(settings.AUTO_SIGNAL_BATCH_SIZE or 50))

    logger.info(
        "Auto-signal worker started (interval=%ss, batch=%s, market_hours_only=%s)",
        interval,
        batch_size,
        bool(settings.AUTO_SIGNAL_WORKER_MARKET_HOURS_ONLY),
    )
    while True:
        try:
            if settings.AUTO_SIGNAL_WORKER_MARKET_HOURS_ONLY:
                try:
                    if not await market_data_service.is_market_open():
                        await asyncio.sleep(interval)
                        continue
                except Exception:
                    logger.warning("Market status check failed in auto-signal worker")
                    await asyncio.sleep(interval)
                    continue

            async with async_session_factory() as session:
                user_ids = (
                    await session.execute(
                        select(SignalTrigger.user_id)
                        .where(SignalTrigger.status == "PENDING")
                        .distinct()
                    )
                ).scalars().all()

                total_exec = 0
                total_processed = 0
                for user_id in user_ids:
                    user = (
                        await session.execute(
                            select(User).where(User.id == user_id, User.is_active.is_(True))
                        )
                    ).scalar_one_or_none()
                    if user is None:
                        continue
                    result = await process_pending_signals_for_user(
                        db=session,
                        current_user=user,
                        limit=batch_size,
                    )
                    total_exec += int(result.get("executed_count", 0) or 0)
                    total_processed += int(result.get("processed", 0) or 0)
                await session.commit()
                if total_processed > 0:
                    logger.info(
                        "Auto-signal worker cycle processed=%s executed=%s users=%s",
                        total_processed,
                        total_exec,
                        len(user_ids),
                    )
        except asyncio.CancelledError:
            logger.info("Auto-signal worker stopped.")
            break
        except Exception:
            logger.exception("Auto-signal worker cycle failed")
        await asyncio.sleep(interval)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager executed on startup and shutdown.

    Startup:
        1. Initialise the database (create tables if needed).
        2. Verify Redis connectivity.
        3. Record application start time.

    Shutdown:
        1. Close the database engine.
        2. Close the Redis connection pool.
    """
    global _app_start_time, _auto_signal_worker_task

    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("Starting QuantDesk Pro API v1.0.0 ...")
    if settings.APP_ENV.lower() == "production" and settings.AUTH_BYPASS_LOCAL:
        logger.warning(
            "AUTH_BYPASS_LOCAL=true while APP_ENV=production. "
            "Google OAuth is effectively bypassed."
        )

    # Database
    try:
        await init_db()
        logger.info("Database initialised successfully.")
    except Exception as exc:
        logger.error("Database initialisation failed: %s", exc)
        raise

    # Redis
    try:
        r = await get_redis()
        pong = await r.ping()
        if pong:
            logger.info("Redis connection verified (PONG received).")
        else:
            logger.warning("Redis PING did not return expected response.")
    except Exception as exc:
        logger.warning("Redis verification failed -- continuing without cache: %s", exc)

    # Record start time and propagate to system router
    _app_start_time = time.time()
    try:
        from app.api.routes.system import set_start_time
        set_start_time(_app_start_time)
    except ImportError:
        pass

    logger.info("QuantDesk Pro API startup complete.")

    if settings.AUTO_SIGNAL_WORKER_ENABLED:
        _auto_signal_worker_task = asyncio.create_task(_auto_signal_worker_loop())

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Shutting down QuantDesk Pro API ...")

    if _auto_signal_worker_task is not None:
        _auto_signal_worker_task.cancel()
        try:
            await _auto_signal_worker_task
        except asyncio.CancelledError:
            pass
        _auto_signal_worker_task = None

    try:
        await close_db()
        logger.info("Database connections closed.")
    except Exception as exc:
        logger.error("Error closing database: %s", exc)

    try:
        await close_redis()
        logger.info("Redis connections closed.")
    except Exception as exc:
        logger.error("Error closing Redis: %s", exc)

    logger.info("QuantDesk Pro API shutdown complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="QuantDesk Pro API",
    description=(
        "Professional-grade portfolio analytics, live market data, "
        "risk management, and AI-powered insights for Indian equities."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

_allowed_origins = [
    origin.strip()
    for origin in settings.ALLOWED_ORIGINS.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

from app.api.routes import (  # noqa: E402
    auth,
    portfolio,
    trades,
    orders,
    rankings,
    risk,
    ticker,
    ai,
    system,
)

app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(portfolio.router, prefix=API_PREFIX)
app.include_router(trades.router, prefix=API_PREFIX)
app.include_router(orders.router, prefix=API_PREFIX)
app.include_router(rankings.router, prefix=API_PREFIX)
app.include_router(risk.router, prefix=API_PREFIX)
app.include_router(ticker.router, prefix=API_PREFIX)
app.include_router(ai.router, prefix=API_PREFIX)
app.include_router(system.router, prefix=API_PREFIX)

# ---------------------------------------------------------------------------
# Root endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    """Redirect the bare root URL to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def root_health():
    """Simple top-level health check (liveness probe).

    Does not inspect downstream dependencies -- if the server can serve
    this response, it is alive.  For a comprehensive check use
    ``/api/v1/system/health``.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - _app_start_time, 2) if _app_start_time else 0,
    }
