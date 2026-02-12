"""
Rankings API routes.

Provides endpoints for retrieving and computing multi-factor stock rankings
across seven categories: top_buy, top_sell, banking, large_cap, small_cap,
high_vol, and overall.  Results are cached in Redis and persisted to the
database for historical reference.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.redis import get_redis
from app.engines.ranking_engine import RankingEngine
from app.middleware.auth import get_current_user
from app.models.models import Ranking, User
from app.schemas.schemas import RankingEntry, RankingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rankings", tags=["Rankings"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CATEGORIES: set[str] = {
    "top_buy", "top_sell", "banking", "large_cap",
    "small_cap", "high_vol", "overall",
}

CACHE_KEY_ALL = "rankings:all"
CACHE_TTL_SECONDS = 3600  # 1 hour

# Shared engine instance
_ranking_engine = RankingEngine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_key_for_category(category: str) -> str:
    """Return the Redis cache key for a single ranking category."""
    return f"rankings:{category}"


def _serialize_rankings(rankings: dict[str, list[dict]]) -> str:
    """Serialize rankings dict to a JSON string for Redis storage."""
    return json.dumps(rankings, default=str)


def _deserialize_rankings(raw: str) -> dict[str, list[dict]]:
    """Deserialize a JSON string from Redis back to a rankings dict."""
    return json.loads(raw)


def _today_start() -> datetime:
    """Return the start of today (midnight UTC) for database queries."""
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _db_rows_to_ranking_entries(rows: list[Ranking]) -> list[dict]:
    """Convert a list of Ranking ORM instances to dicts matching engine output."""
    entries = []
    for row in rows:
        entries.append({
            "ticker": row.ticker,
            "rank": row.rank_position,
            "rank_position": row.rank_position,
            "score": row.score,
            "expected_return": row.expected_return,
            "momentum_30d": row.momentum_30d,
            "volatility": row.volatility,
            "liquidity_score": row.liquidity_score,
            "current_price": row.current_price,
            "computed_at": row.computed_at.isoformat() if row.computed_at else None,
        })
    return entries


def _build_ranking_response(
    category: str,
    entries: list[dict],
) -> RankingResponse:
    """Build a RankingResponse schema from raw entry dicts."""
    computed_at = None
    ranking_entries = []
    for entry in entries:
        entry_computed_at = entry.get("computed_at")
        if entry_computed_at and computed_at is None:
            if isinstance(entry_computed_at, str):
                try:
                    computed_at = datetime.fromisoformat(entry_computed_at)
                except (ValueError, TypeError):
                    computed_at = None
            elif isinstance(entry_computed_at, datetime):
                computed_at = entry_computed_at

        ranking_entries.append(
            RankingEntry(
                ticker=entry.get("ticker", ""),
                rank_position=entry.get("rank") or entry.get("rank_position", 0),
                score=entry.get("score", 0.0),
                expected_return=entry.get("expected_return"),
                momentum_30d=entry.get("momentum_30d"),
                volatility=entry.get("volatility"),
                liquidity_score=entry.get("liquidity_score"),
                current_price=entry.get("current_price"),
                computed_at=computed_at or datetime.now(timezone.utc),
            )
        )

    return RankingResponse(
        category=category,
        entries=ranking_entries,
        computed_at=computed_at or (datetime.now(timezone.utc) if ranking_entries else None),
    )


async def _load_rankings_from_db(
    db: AsyncSession,
    category: str | None = None,
) -> dict[str, list[dict]]:
    """Load today's rankings from the database.

    Parameters
    ----------
    db : AsyncSession
        Active database session.
    category : str or None
        If provided, load only this category.  Otherwise load all.

    Returns
    -------
    dict[str, list[dict]]
        Mapping of category -> ranked entries.  Empty dict if nothing found.
    """
    today_start = _today_start()

    query = (
        select(Ranking)
        .where(Ranking.computed_at >= today_start)
        .order_by(Ranking.category, Ranking.rank_position)
    )
    if category:
        query = query.where(Ranking.category == category)

    result = await db.execute(query)
    rows = result.scalars().all()

    if not rows:
        return {}

    # Group by category.
    grouped: dict[str, list[Ranking]] = {}
    for row in rows:
        grouped.setdefault(row.category, []).append(row)

    rankings: dict[str, list[dict]] = {}
    for cat, cat_rows in grouped.items():
        rankings[cat] = _db_rows_to_ranking_entries(cat_rows)

    return rankings


async def _store_rankings_to_db(
    db: AsyncSession,
    rankings: dict[str, list[dict]],
) -> None:
    """Persist computed rankings to the database.

    Deletes any existing rankings for each category before inserting fresh
    rows to avoid duplicates.
    """
    computed_at = datetime.now(timezone.utc)

    for category, entries in rankings.items():
        # Delete old rankings for this category.
        await db.execute(
            delete(Ranking).where(Ranking.category == category)
        )

        # Insert new rankings.
        for entry in entries:
            ranking_row = Ranking(
                ticker=entry.get("ticker", ""),
                exchange="NSE",
                category=category,
                rank_position=entry.get("rank") or entry.get("rank_position", 0),
                score=entry.get("score", 0.0),
                expected_return=entry.get("expected_return"),
                momentum_30d=entry.get("momentum_30d"),
                volatility=entry.get("volatility"),
                liquidity_score=entry.get("liquidity_score"),
                current_price=entry.get("current_price"),
                computed_at=computed_at,
            )
            db.add(ranking_row)

    await db.flush()


async def _cache_rankings(rankings: dict[str, list[dict]]) -> None:
    """Cache rankings in Redis with the configured TTL."""
    try:
        redis = await get_redis()

        # Cache the combined result under the "all" key.
        await redis.setex(
            CACHE_KEY_ALL,
            CACHE_TTL_SECONDS,
            _serialize_rankings(rankings),
        )

        # Cache each category individually.
        for category, entries in rankings.items():
            await redis.setex(
                _cache_key_for_category(category),
                CACHE_TTL_SECONDS,
                json.dumps(entries, default=str),
            )
    except Exception:
        logger.warning("Failed to cache rankings in Redis", exc_info=True)


# ---------------------------------------------------------------------------
# GET / -- All rankings (every category)
# ---------------------------------------------------------------------------

@router.get("/", response_model=dict[str, list[RankingEntry]])
async def get_all_rankings(
    db: AsyncSession = Depends(get_db),
) -> dict[str, list[RankingEntry]]:
    """Return rankings for all categories.

    Checks Redis cache first, then the database for today's rankings, and
    falls back to a live computation if neither source has data.
    """
    # 1. Check Redis cache.
    try:
        redis = await get_redis()
        cached = await redis.get(CACHE_KEY_ALL)
        if cached:
            logger.debug("Rankings cache hit (all categories)")
            all_rankings = _deserialize_rankings(cached)
            result: dict[str, list[RankingEntry]] = {}
            for cat, entries in all_rankings.items():
                result[cat] = [
                    RankingEntry(
                        ticker=e.get("ticker", ""),
                        rank_position=e.get("rank") or e.get("rank_position", 0),
                        score=e.get("score", 0.0),
                        expected_return=e.get("expected_return"),
                        momentum_30d=e.get("momentum_30d"),
                        volatility=e.get("volatility"),
                        liquidity_score=e.get("liquidity_score"),
                        current_price=e.get("current_price"),
                        computed_at=e.get("computed_at", datetime.now(timezone.utc)),
                    )
                    for e in entries
                ]
            return result
    except Exception:
        logger.warning("Redis unavailable for rankings cache lookup", exc_info=True)

    # 2. Check database for today's rankings.
    db_rankings = await _load_rankings_from_db(db)
    if db_rankings:
        logger.debug("Rankings loaded from database")
        # Re-populate cache from DB data.
        await _cache_rankings(db_rankings)

        result = {}
        for cat, entries in db_rankings.items():
            result[cat] = [
                RankingEntry(
                    ticker=e.get("ticker", ""),
                    rank_position=e.get("rank") or e.get("rank_position", 0),
                    score=e.get("score", 0.0),
                    expected_return=e.get("expected_return"),
                    momentum_30d=e.get("momentum_30d"),
                    volatility=e.get("volatility"),
                    liquidity_score=e.get("liquidity_score"),
                    current_price=e.get("current_price"),
                    computed_at=e.get("computed_at", datetime.now(timezone.utc)),
                )
                for e in entries
            ]
        return result

    # 3. No cached or DB data -- trigger live computation.
    logger.info("No cached rankings found; triggering live computation")
    computed = await _ranking_engine.compute_all_rankings()
    await _store_rankings_to_db(db, computed)
    await _cache_rankings(computed)

    result = {}
    for cat, entries in computed.items():
        computed_at = datetime.now(timezone.utc)
        result[cat] = [
            RankingEntry(
                ticker=e.get("ticker", ""),
                rank_position=e.get("rank") or e.get("rank_position", 0),
                score=e.get("score", 0.0),
                expected_return=e.get("expected_return"),
                momentum_30d=e.get("momentum_30d"),
                volatility=e.get("volatility"),
                liquidity_score=e.get("liquidity_score"),
                current_price=e.get("current_price"),
                computed_at=computed_at,
            )
            for e in entries
        ]
    return result


# ---------------------------------------------------------------------------
# GET /last-computed -- When rankings were last computed
# NOTE: This route MUST be defined before /{category} to avoid being
# intercepted by the generic path parameter.
# ---------------------------------------------------------------------------

@router.get("/last-computed")
async def get_last_computed(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return the timestamp of the most recent ranking computation.

    Checks the database for the latest ``computed_at`` value across all
    ranking records.
    """
    result = await db.execute(
        select(Ranking.computed_at)
        .order_by(desc(Ranking.computed_at))
        .limit(1)
    )
    row = result.scalar_one_or_none()

    if row is None:
        return {
            "last_computed": None,
            "message": "No rankings have been computed yet.",
        }

    return {
        "last_computed": row.isoformat() if isinstance(row, datetime) else str(row),
    }


# ---------------------------------------------------------------------------
# GET /{category} -- Rankings for a specific category
# ---------------------------------------------------------------------------

@router.get("/{category}", response_model=RankingResponse)
async def get_rankings_by_category(
    category: str,
    db: AsyncSession = Depends(get_db),
) -> RankingResponse:
    """Return rankings for a specific category.

    Valid categories: top_buy, top_sell, banking, large_cap, small_cap,
    high_vol, overall.
    """
    # Validate category.
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid category '{category}'. "
                f"Valid categories: {sorted(VALID_CATEGORIES)}"
            ),
        )

    # 1. Check Redis cache for this category.
    try:
        redis = await get_redis()
        cached = await redis.get(_cache_key_for_category(category))
        if cached:
            logger.debug("Rankings cache hit for category '%s'", category)
            entries = json.loads(cached)
            return _build_ranking_response(category, entries)
    except Exception:
        logger.warning(
            "Redis unavailable for category '%s' cache lookup",
            category,
            exc_info=True,
        )

    # 2. Check database for today's rankings in this category.
    db_rankings = await _load_rankings_from_db(db, category=category)
    if db_rankings and category in db_rankings:
        logger.debug("Rankings for '%s' loaded from database", category)
        entries = db_rankings[category]
        # Re-populate cache.
        await _cache_rankings({category: entries})
        return _build_ranking_response(category, entries)

    # 3. No cached or DB data -- trigger live computation for all categories.
    logger.info(
        "No cached rankings for '%s'; triggering live computation", category,
    )
    computed = await _ranking_engine.compute_all_rankings()
    await _store_rankings_to_db(db, computed)
    await _cache_rankings(computed)

    entries = computed.get(category, [])
    return _build_ranking_response(category, entries)


# ---------------------------------------------------------------------------
# POST /compute -- Trigger ranking computation
# ---------------------------------------------------------------------------

@router.post("/compute", status_code=status.HTTP_200_OK)
async def compute_rankings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Trigger a full ranking computation.

    Requires authentication.  Intended for cron jobs or manual triggers.
    Computes rankings for all categories, persists results to the database,
    and caches them in Redis.
    """
    logger.info(
        "Ranking computation triggered by user %s (%s)",
        current_user.id,
        current_user.email,
    )

    start_time = datetime.now(timezone.utc)

    try:
        rankings = await _ranking_engine.compute_all_rankings()
    except Exception:
        logger.exception("Ranking computation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ranking computation failed. Check server logs for details.",
        )

    # Persist to database.
    await _store_rankings_to_db(db, rankings)

    # Cache in Redis.
    await _cache_rankings(rankings)

    end_time = datetime.now(timezone.utc)
    duration_seconds = (end_time - start_time).total_seconds()

    # Build summary of what was computed.
    category_counts = {cat: len(entries) for cat, entries in rankings.items()}
    total_entries = sum(category_counts.values())

    logger.info(
        "Ranking computation completed in %.2fs. %d total entries across %d categories.",
        duration_seconds,
        total_entries,
        len(category_counts),
    )

    return {
        "status": "completed",
        "computed_at": end_time.isoformat(),
        "duration_seconds": round(duration_seconds, 2),
        "categories": category_counts,
        "total_entries": total_entries,
        "triggered_by": str(current_user.id),
    }
