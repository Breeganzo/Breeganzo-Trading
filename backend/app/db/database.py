from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

settings = get_settings()

engine = None
async_session_factory = None


def _get_db_url() -> str:
    db_url = settings.DATABASE_URL
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif not db_url.startswith("postgresql+asyncpg://"):
        db_url = f"postgresql+asyncpg://{db_url}" if db_url else ""
    return db_url


def _ensure_engine():
    global engine, async_session_factory
    if engine is None:
        db_url = _get_db_url()
        if not db_url:
            raise RuntimeError(
                "DATABASE_URL is not configured. "
                "Set it in backend/.env or as an environment variable."
            )
        engine = create_async_engine(
            db_url,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=settings.DEBUG,
        )
        async_session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    _ensure_engine()
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    _ensure_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    global engine
    if engine is not None:
        await engine.dispose()
        engine = None
