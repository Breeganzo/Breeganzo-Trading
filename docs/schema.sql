-- ============================================================================
-- QuantDesk Pro — Database Schema (Supabase PostgreSQL)
-- This SQL is auto-generated as reference. Tables are created automatically
-- by SQLAlchemy on first startup via Base.metadata.create_all()
-- ============================================================================

-- Enable UUID extension (Supabase has this by default)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    picture VARCHAR(512),
    totp_secret VARCHAR(64),
    totp_enabled BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);

-- Portfolio
CREATE TABLE IF NOT EXISTS portfolio (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ticker VARCHAR(20) NOT NULL,
    exchange VARCHAR(10) DEFAULT 'NSE',
    quantity INTEGER NOT NULL DEFAULT 0,
    avg_buy_price FLOAT NOT NULL DEFAULT 0.0,
    total_invested FLOAT DEFAULT 0.0,
    realized_pnl FLOAT DEFAULT 0.0,
    total_buy_costs FLOAT DEFAULT 0.0,
    total_sell_costs FLOAT DEFAULT 0.0,
    sector VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_user_ticker UNIQUE(user_id, ticker)
);
CREATE INDEX IF NOT EXISTS ix_portfolio_ticker ON portfolio(ticker);
CREATE INDEX IF NOT EXISTS ix_portfolio_user_ticker ON portfolio(user_id, ticker);

-- Trades
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ticker VARCHAR(20) NOT NULL,
    exchange VARCHAR(10) DEFAULT 'NSE',
    trade_type VARCHAR(4) NOT NULL, -- BUY or SELL
    quantity INTEGER NOT NULL,
    price FLOAT NOT NULL,
    total_amount FLOAT NOT NULL,
    brokerage FLOAT DEFAULT 0.0,
    stt FLOAT DEFAULT 0.0,
    exchange_charges FLOAT DEFAULT 0.0,
    gst FLOAT DEFAULT 0.0,
    sebi_charges FLOAT DEFAULT 0.0,
    stamp_duty FLOAT DEFAULT 0.0,
    slippage_cost FLOAT DEFAULT 0.0,
    total_cost FLOAT DEFAULT 0.0,
    net_amount FLOAT DEFAULT 0.0,
    notes TEXT,
    executed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_trades_ticker ON trades(ticker);
CREATE INDEX IF NOT EXISTS ix_trades_user_ticker ON trades(user_id, ticker);
CREATE INDEX IF NOT EXISTS ix_trades_executed ON trades(executed_at);

-- Order Book
CREATE TABLE IF NOT EXISTS order_book (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ticker VARCHAR(20) NOT NULL,
    exchange VARCHAR(10) DEFAULT 'NSE',
    order_type VARCHAR(4) NOT NULL, -- BUY or SELL
    quantity INTEGER NOT NULL,
    target_price FLOAT NOT NULL,
    status VARCHAR(20) DEFAULT 'DRAFT', -- DRAFT, CONFIRMED, CANCELLED
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    confirmed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_orders_ticker ON order_book(ticker);
CREATE INDEX IF NOT EXISTS ix_orders_user_status ON order_book(user_id, status);

-- Auto signal trigger queue (consume-once BUY/SELL/HOLD triggers)
CREATE TABLE IF NOT EXISTS signal_triggers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ticker VARCHAR(20) NOT NULL,
    exchange VARCHAR(10) DEFAULT 'NSE',
    action VARCHAR(6) NOT NULL, -- BUY / SELL / HOLD
    quantity INTEGER NOT NULL DEFAULT 1,
    trigger_price_low FLOAT,
    trigger_price_high FLOAT,
    sentiment_min FLOAT,
    sentiment_max FLOAT,
    sentiment_last FLOAT,
    status VARCHAR(20) DEFAULT 'PENDING', -- PENDING / SKIPPED / CANCELLED
    source VARCHAR(40) DEFAULT 'manual',
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    triggered_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_signal_triggers_ticker ON signal_triggers(ticker);
CREATE INDEX IF NOT EXISTS ix_signal_triggers_user_status ON signal_triggers(user_id, status);
CREATE INDEX IF NOT EXISTS ix_signal_triggers_user_created ON signal_triggers(user_id, created_at);

-- Daily Returns
CREATE TABLE IF NOT EXISTS daily_returns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date TIMESTAMPTZ NOT NULL,
    portfolio_value FLOAT NOT NULL,
    daily_return_pct FLOAT DEFAULT 0.0,
    total_invested FLOAT DEFAULT 0.0,
    total_pnl FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_user_date_return UNIQUE(user_id, date)
);
CREATE INDEX IF NOT EXISTS ix_daily_returns_user_date ON daily_returns(user_id, date);

-- Rankings
CREATE TABLE IF NOT EXISTS rankings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL,
    exchange VARCHAR(10) DEFAULT 'NSE',
    category VARCHAR(50) NOT NULL, -- top_buy, top_sell, banking, large_cap, small_cap, high_vol, overall
    rank_position INTEGER NOT NULL,
    score FLOAT NOT NULL,
    expected_return FLOAT,
    momentum_30d FLOAT,
    volatility FLOAT,
    liquidity_score FLOAT,
    avg_volume FLOAT,
    bid_ask_spread FLOAT,
    market_depth_proxy FLOAT,
    current_price FLOAT,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_rankings_ticker ON rankings(ticker);
CREATE INDEX IF NOT EXISTS ix_rankings_category ON rankings(category);
CREATE INDEX IF NOT EXISTS ix_rankings_category_rank ON rankings(category, rank_position);
CREATE INDEX IF NOT EXISTS ix_rankings_computed ON rankings(computed_at);

-- System Status
CREATE TABLE IF NOT EXISTS system_status (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component VARCHAR(100) UNIQUE NOT NULL,
    status VARCHAR(20) DEFAULT 'healthy', -- healthy, degraded, down
    last_checked TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    details TEXT,
    metadata_json TEXT
);

-- Daily stock snapshot (persisted live market view for replay/export)
CREATE TABLE IF NOT EXISTS daily_stock_snapshot (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    sector_bucket VARCHAR(60),
    current_price FLOAT NOT NULL,
    open_price FLOAT,
    prev_close FLOAT,
    high FLOAT,
    low FLOAT,
    change_pct FLOAT,
    volume INTEGER,
    signal VARCHAR(12),
    source VARCHAR(40) DEFAULT 'live_market',
    captured_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    CONSTRAINT uq_daily_snapshot_user_date_ticker_source
      UNIQUE(user_id, snapshot_date, ticker, source)
);
CREATE INDEX IF NOT EXISTS ix_daily_snapshot_user_date
  ON daily_stock_snapshot(user_id, snapshot_date);
CREATE INDEX IF NOT EXISTS ix_daily_snapshot_ticker
  ON daily_stock_snapshot(ticker);
