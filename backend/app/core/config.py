from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "QuantDesk Pro"
    APP_ENV: str = "production"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION"
    ALLOWED_ORIGINS: str = "http://localhost:3000,https://quantdesk.vercel.app"

    # Database - Supabase
    DATABASE_URL: str = ""
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 2

    # Redis - Upstash
    REDIS_URL: str = ""

    # Auth
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/callback"
    ALLOWED_EMAIL: str = "anthonybreeganzo02@gmail.com"
    AUTH_BYPASS_LOCAL: bool = False
    LOCAL_BYPASS_EMAIL: str = "anthonybreeganzo02@gmail.com"
    LOCAL_BYPASS_NAME: str = "Local User"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 720  # 12 hours
    TOTP_SECRET: str = ""  # Generated on first setup

    # Groq AI
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Market Data
    TICKER_UPDATE_INTERVAL: int = 5  # seconds
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 15
    MARKET_CLOSE_HOUR: int = 15
    MARKET_CLOSE_MINUTE: int = 30
    TIMEZONE: str = "Asia/Kolkata"

    # Transaction Costs (Groww structure)
    BROKERAGE_RATE: float = 0.0  # Groww zero brokerage for delivery
    STT_BUY_RATE: float = 0.001  # 0.1%
    STT_SELL_RATE: float = 0.001
    EXCHANGE_CHARGE_RATE: float = 0.0000345
    GST_RATE: float = 0.18
    SEBI_CHARGE_RATE: float = 0.000001
    STAMP_DUTY_RATE: float = 0.00015
    DEFAULT_SLIPPAGE: float = 0.001  # 0.1%

    # Free tier optimization
    MAX_TRACKED_TICKERS: int = 50
    CACHE_TTL_SECONDS: int = 300  # 5 min default
    PRICE_CACHE_TTL: int = 5
    RANKING_CACHE_TTL: int = 3600  # 1 hour
    RISK_CACHE_TTL: int = 1800  # 30 min
    MAX_WS_CONNECTIONS: int = 3

    # Auto signal background worker
    AUTO_SIGNAL_WORKER_ENABLED: bool = True
    AUTO_SIGNAL_INTERVAL_SEC: int = 10
    AUTO_SIGNAL_BATCH_SIZE: int = 50
    AUTO_SIGNAL_WORKER_MARKET_HOURS_ONLY: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
