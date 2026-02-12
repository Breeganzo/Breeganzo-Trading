from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255))
    picture = Column(String(512))
    totp_secret = Column(String(64))
    totp_enabled = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    portfolio = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="user", cascade="all, delete-orphan")
    orders = relationship("OrderBook", back_populates="user", cascade="all, delete-orphan")
    daily_returns = relationship("DailyReturn", back_populates="user", cascade="all, delete-orphan")


class Portfolio(Base):
    __tablename__ = "portfolio"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(String(20), nullable=False, index=True)
    exchange = Column(String(10), default="NSE")
    quantity = Column(Integer, nullable=False, default=0)
    avg_buy_price = Column(Float, nullable=False, default=0.0)
    total_invested = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    total_buy_costs = Column(Float, default=0.0)  # cumulative transaction costs on buys
    total_sell_costs = Column(Float, default=0.0)  # cumulative transaction costs on sells
    sector = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="portfolio")

    __table_args__ = (
        UniqueConstraint("user_id", "ticker", name="uq_user_ticker"),
        Index("ix_portfolio_user_ticker", "user_id", "ticker"),
    )


class Trade(Base):
    __tablename__ = "trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(String(20), nullable=False, index=True)
    exchange = Column(String(10), default="NSE")
    trade_type = Column(String(4), nullable=False)  # BUY or SELL
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False)
    # Transaction cost breakdown
    brokerage = Column(Float, default=0.0)
    stt = Column(Float, default=0.0)
    exchange_charges = Column(Float, default=0.0)
    gst = Column(Float, default=0.0)
    sebi_charges = Column(Float, default=0.0)
    stamp_duty = Column(Float, default=0.0)
    slippage_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    net_amount = Column(Float, default=0.0)  # total_amount + costs for buy, total_amount - costs for sell
    notes = Column(Text)
    executed_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="trades")

    __table_args__ = (
        Index("ix_trades_user_ticker", "user_id", "ticker"),
        Index("ix_trades_executed", "executed_at"),
    )


class OrderBook(Base):
    __tablename__ = "order_book"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(String(20), nullable=False, index=True)
    exchange = Column(String(10), default="NSE")
    order_type = Column(String(4), nullable=False)  # BUY or SELL
    quantity = Column(Integer, nullable=False)
    target_price = Column(Float, nullable=False)
    status = Column(String(20), default="DRAFT")  # DRAFT, CONFIRMED, CANCELLED
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    confirmed_at = Column(DateTime(timezone=True))

    user = relationship("User", back_populates="orders")

    __table_args__ = (
        Index("ix_orders_user_status", "user_id", "status"),
    )


class DailyReturn(Base):
    __tablename__ = "daily_returns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)
    portfolio_value = Column(Float, nullable=False)
    daily_return_pct = Column(Float, default=0.0)
    total_invested = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="daily_returns")

    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_user_date_return"),
        Index("ix_daily_returns_user_date", "user_id", "date"),
    )


class Ranking(Base):
    __tablename__ = "rankings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(20), nullable=False, index=True)
    exchange = Column(String(10), default="NSE")
    category = Column(String(50), nullable=False, index=True)  # top_buy, top_sell, banking, large_cap, small_cap, high_vol, overall
    rank_position = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)
    expected_return = Column(Float)
    momentum_30d = Column(Float)
    volatility = Column(Float)
    liquidity_score = Column(Float)
    avg_volume = Column(Float)
    bid_ask_spread = Column(Float)
    market_depth_proxy = Column(Float)
    current_price = Column(Float)
    computed_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_rankings_category_rank", "category", "rank_position"),
        Index("ix_rankings_computed", "computed_at"),
    )


class SystemStatus(Base):
    __tablename__ = "system_status"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    component = Column(String(100), unique=True, nullable=False)
    status = Column(String(20), default="healthy")  # healthy, degraded, down
    last_checked = Column(DateTime(timezone=True), server_default=func.now())
    last_updated = Column(DateTime(timezone=True), server_default=func.now())
    details = Column(Text)
    metadata_json = Column(Text)  # JSON string for flexible data
