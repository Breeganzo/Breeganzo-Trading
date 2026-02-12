from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Auth ──
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    requires_totp: bool = False
    totp_setup_uri: Optional[str] = None


class TOTPVerify(BaseModel):
    code: str = Field(..., min_length=6, max_length=6)


class UserResponse(BaseModel):
    id: UUID
    email: str
    name: Optional[str]
    picture: Optional[str]
    totp_enabled: bool
    is_active: bool

    class Config:
        from_attributes = True


# ── Portfolio ──
class PortfolioHolding(BaseModel):
    id: UUID
    ticker: str
    exchange: str
    quantity: int
    avg_buy_price: float
    total_invested: float
    realized_pnl: float
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    day_change_pct: Optional[float] = None
    total_buy_costs: float = 0.0
    total_sell_costs: float = 0.0
    sector: Optional[str] = None
    beta: Optional[float] = None
    volatility: Optional[float] = None

    class Config:
        from_attributes = True


class PortfolioSummary(BaseModel):
    total_value: float
    total_invested: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    total_pnl_pct: float
    total_transaction_costs: float
    day_pnl: float
    day_pnl_pct: float
    holdings: list[PortfolioHolding]
    sector_exposure: dict[str, float]
    beta_exposure: float


# ── Trades ──
class TradeCreate(BaseModel):
    ticker: str
    exchange: str = "NSE"
    trade_type: str = Field(..., pattern="^(BUY|SELL)$")
    quantity: int = Field(..., gt=0)
    price: float = Field(..., gt=0)
    slippage_pct: Optional[float] = Field(None, ge=0.001, le=0.003)
    notes: Optional[str] = None
    executed_at: Optional[datetime] = None


class TradeResponse(BaseModel):
    id: UUID
    ticker: str
    exchange: str
    trade_type: str
    quantity: int
    price: float
    total_amount: float
    brokerage: float
    stt: float
    exchange_charges: float
    gst: float
    sebi_charges: float
    stamp_duty: float
    slippage_cost: float
    total_cost: float
    net_amount: float
    notes: Optional[str]
    executed_at: datetime

    class Config:
        from_attributes = True


# ── Order Book ──
class OrderCreate(BaseModel):
    ticker: str
    exchange: str = "NSE"
    order_type: str = Field(..., pattern="^(BUY|SELL)$")
    quantity: int = Field(..., gt=0)
    target_price: float = Field(..., gt=0)
    notes: Optional[str] = None


class OrderResponse(BaseModel):
    id: UUID
    ticker: str
    exchange: str
    order_type: str
    quantity: int
    target_price: float
    status: str
    notes: Optional[str]
    created_at: datetime
    confirmed_at: Optional[datetime]

    class Config:
        from_attributes = True


# ── Rankings ──
class RankingEntry(BaseModel):
    ticker: str
    rank_position: int
    score: float
    expected_return: Optional[float]
    momentum_30d: Optional[float]
    volatility: Optional[float]
    liquidity_score: Optional[float]
    current_price: Optional[float]
    computed_at: datetime

    class Config:
        from_attributes = True


class RankingResponse(BaseModel):
    category: str
    entries: list[RankingEntry]
    computed_at: Optional[datetime]


# ── Risk ──
class RiskMetrics(BaseModel):
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    portfolio_beta: Optional[float]
    max_drawdown: Optional[float]
    var_95: Optional[float]
    rolling_return_30d: Optional[float]
    rolling_return_90d: Optional[float]
    rolling_return_1y: Optional[float]
    regime: str  # bull, bear, high_vol, low_vol
    regime_details: dict
    correlation_matrix: Optional[dict]
    last_updated: datetime


# ── Ticker ──
class TickerData(BaseModel):
    ticker: str
    price: float
    change: float
    change_pct: float
    volume: int
    high: float
    low: float
    open: float
    prev_close: float
    timestamp: datetime


# ── System ──
class SystemHealthResponse(BaseModel):
    status: str
    database: str
    redis: str
    data_feed: str
    model_last_updated: Optional[datetime]
    correlation_last_calculated: Optional[datetime]
    uptime_seconds: float
    version: str


# ── AI ──
class AIExplainRequest(BaseModel):
    metric: str
    context: Optional[dict] = None


class AIExplainResponse(BaseModel):
    explanation: str
    suggestions: list[str]
    metric: str
