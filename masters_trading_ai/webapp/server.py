"""
Masters AI Trading Bot — Flask Web Application
================================================
Groww-inspired stock cards with live prices, ML predictions,
expected vs actual tracking, and alpha calculation.

Run:  python webapp/server.py
Open: http://localhost:5001
"""

import sys
import os
import json
import csv
import time
import logging
import threading
from uuid import uuid4
from io import StringIO
from pathlib import Path
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Load .env from project root BEFORE any other imports that need env vars
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Validate critical environment variables at startup ──
_REQUIRED_ENV = []  # Add keys here if they should be mandatory
_OPTIONAL_ENV = {
    "GROQ_API_KEY": "AI explanations disabled",
    "FLASK_SECRET_KEY": "Using dev fallback",
}
for _key in _REQUIRED_ENV:
    if not os.environ.get(_key):
        raise EnvironmentError(
            f"Missing required env var: {_key}. Copy .env.example → .env and fill in values."
        )
for _key, _msg in _OPTIONAL_ENV.items():
    if not os.environ.get(_key):
        print(f"⚠️  {_key} not set — {_msg}. See .env.example")

# ── CRITICAL: import torch FIRST to avoid segfault with statsmodels C extensions ──
import torch  # noqa: F401

from flask import Flask, jsonify, render_template, request, make_response
from flask_cors import CORS
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import LivePredictor, get_market_status, get_intraday_data
from src.tracking.prediction_logger import PredictionLogger
from src.tracking.performance_reporter import PerformanceReporter

from webapp.groq_explainer import (
    explain_fundamental,
    explain_greek,
    explain_indicator,
    get_groq_strategy,
    get_combined_strategy,
    get_stock_overview,
    get_news_sentiment,
    explain_risk_term,
    get_groq_price_forecast,
    explain_model,
    stock_chat_response,
    portfolio_profit_suggestion,
    suggest_ticker_shortlist,
    review_trade_plan,
    ai_risk_assessment,
)
from webapp.prediction_tracker import PredictionTracker

import yaml

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trading_bot")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-fallback-not-for-production")
CORS(app)


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Flask 3.0+ removed json_encoder; use json_provider_class instead
from flask.json.provider import DefaultJSONProvider


class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

IST = ZoneInfo("Asia/Kolkata")
CONFIG_DIR = PROJECT_ROOT / "config"
CACHE_DIR = PROJECT_ROOT / "cache"
PREDICTION_LOG_DIR = CACHE_DIR / "prediction_log"
PORTFOLIO_FILE = CACHE_DIR / "portfolio.json"
PORTFOLIO_TRADES_FILE = CACHE_DIR / "portfolio_trades.json"
DELISTED_TICKERS_FILE = CACHE_DIR / "delisted_tickers.csv"
PREMARKET_OUTLOOK_FILE = CACHE_DIR / "premarket_outlook.json"
PREDICTION_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _load_settings() -> dict:
    """Load config/settings.yaml if available."""
    settings_path = CONFIG_DIR / "settings.yaml"
    if not settings_path.exists():
        return {}
    try:
        with open(settings_path) as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log.warning("Failed to load settings.yaml: %s", e)
        return {}


_settings = _load_settings()
_webapp_settings = (
    _settings.get("webapp", {}) if isinstance(_settings.get("webapp", {}), dict) else {}
)
PREMARKET_MAX_BUFFER_MINUTES = int(
    os.environ.get(
        "PREMARKET_MAX_BUFFER_MINUTES",
        _webapp_settings.get("premarket_max_buffer_minutes", 30),
    )
)
PREMARKET_MAX_BUFFER_MINUTES = int(np.clip(PREMARKET_MAX_BUFFER_MINUTES, 0, 180))
PREMARKET_DEFAULT_TICKERS = int(_webapp_settings.get("premarket_default_tickers", 10))
PREMARKET_DEFAULT_TICKERS = max(1, PREMARKET_DEFAULT_TICKERS)
NEXT_DAY_PREDICTION_HOUR = int(
    os.environ.get(
        "NEXT_DAY_PREDICTION_HOUR",
        _webapp_settings.get("next_day_prediction_hour", 15),
    )
)
NEXT_DAY_PREDICTION_MINUTE = int(
    os.environ.get(
        "NEXT_DAY_PREDICTION_MINUTE",
        _webapp_settings.get("next_day_prediction_minute", 45),
    )
)
NEXT_DAY_PREDICTION_HOUR = int(np.clip(NEXT_DAY_PREDICTION_HOUR, 0, 23))
NEXT_DAY_PREDICTION_MINUTE = int(np.clip(NEXT_DAY_PREDICTION_MINUTE, 0, 59))
AFTER_HOURS_UI_HOUR = int(
    os.environ.get(
        "AFTER_HOURS_UI_HOUR",
        _webapp_settings.get("after_hours_ui_hour", 16),
    )
)
AFTER_HOURS_UI_MINUTE = int(
    os.environ.get(
        "AFTER_HOURS_UI_MINUTE",
        _webapp_settings.get("after_hours_ui_minute", 0),
    )
)
AFTER_HOURS_UI_HOUR = int(np.clip(AFTER_HOURS_UI_HOUR, 0, 23))
AFTER_HOURS_UI_MINUTE = int(np.clip(AFTER_HOURS_UI_MINUTE, 0, 59))

# Thread-safe lock for file I/O
_log_lock = threading.Lock()
_portfolio_lock = threading.Lock()
_delisted_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
predictor: LivePredictor | None = None
logger = PredictionLogger()
tickers_by_sector: dict[str, list[str]] = {}
all_tickers: list[str] = []
ticker_names: dict[str, str] = {}  # RELIANCE.NS → Reliance Industries
models_loaded = False
load_error = ""
models_loading = False
models_load_started_at: float | None = None
models_load_completed_at: float | None = None
_groq_forecast_cache: dict[str, dict] = {}
_groq_forecast_cache_time: dict[str, float] = {}
GROQ_FORECAST_TTL = 900  # 15m


def _clean_name(ticker: str) -> str:
    """Convert RELIANCE.NS → Reliance"""
    name = ticker.replace(".NS", "").replace(".BO", "")
    # Make it title-case with common replacements
    name = name.replace("_", " ")
    return name


def load_tickers():
    """Load ticker universe from config."""
    global tickers_by_sector, all_tickers, ticker_names
    cfg_path = CONFIG_DIR / "tickers.yaml"
    with open(cfg_path) as f:
        data = yaml.safe_load(f)

    tradeable_sectors = [
        "large_cap",
        "banking",
        "mid_cap",
        "high_volatility",
        "commodities",
    ]
    for sec in tradeable_sectors:
        syms = data.get(sec, [])
        if isinstance(syms, list):
            tickers_by_sector[sec] = syms
            all_tickers.extend(syms)

    all_tickers_set = sorted(set(all_tickers))
    all_tickers.clear()
    all_tickers.extend(all_tickers_set)

    for t in all_tickers:
        ticker_names[t] = _clean_name(t)


def load_models_background():
    """Load ML models in background thread so server starts fast."""
    global predictor, models_loaded, load_error, models_loading, models_load_started_at, models_load_completed_at
    try:
        models_loading = True
        models_load_started_at = time.time()
        models_load_completed_at = None
        log.info("Loading ML models...")
        t0 = time.time()
        predictor = LivePredictor()
        predictor.load_models()
        elapsed = time.time() - t0
        models_loaded = True
        models_loading = False
        models_load_completed_at = time.time()
        log.info(f"All models loaded in {elapsed:.1f}s")
    except Exception as e:
        models_loading = False
        load_error = str(e)
        log.error(f"Model loading failed: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SECTOR_DISPLAY = {
    "large_cap": "Large Cap",
    "banking": "Banking & Finance",
    "mid_cap": "Mid Cap Growth",
    "high_volatility": "High Volatility",
    "commodities": "Commodities",
}


def _log_prediction(ticker: str, pred: dict):
    """Log prediction to daily JSON file for end-of-day comparison."""
    now_ist = datetime.now(IST)
    now_iso = now_ist.isoformat()
    today_str = now_ist.strftime("%Y-%m-%d")
    log_file = PREDICTION_LOG_DIR / f"{today_str}.json"
    predicted_return_pct = float(pred.get("predicted_return", 0) or 0)
    current_price = float(pred.get("current_price", 0) or 0)
    open_price = float(
        _opening_prices.get(ticker, {}).get("open", current_price) or current_price
    )
    strategy_price_at_open = (
        round(open_price * (1 + predicted_return_pct / 100.0), 2)
        if open_price > 0
        else float(pred.get("predicted_price", 0) or 0)
    )
    premarket_row = _get_premarket_row_for_ticker(ticker, today_str)
    ai_meta = _resolve_ai_forecast_price(
        ticker,
        open_price=float(open_price or 0),
        strategy_price=float(strategy_price_at_open or 0),
        current_price=float(current_price or 0),
        allow_generate=False,
    )
    ai_last_prediction = float(
        premarket_row.get("ai_predicted_price") or ai_meta.get("price") or 0
    )
    ai_source = str(
        premarket_row.get("ai_source") or ai_meta.get("source", "none") or "none"
    )
    strategy_direction = (
        "UP"
        if strategy_price_at_open > open_price
        else "DOWN" if strategy_price_at_open < open_price else "FLAT"
    )
    ai_direction = (
        (
            "UP"
            if ai_last_prediction > current_price
            else "DOWN" if ai_last_prediction < current_price else "FLAT"
        )
        if ai_last_prediction > 0
        else "N/A"
    )
    strategy_predicted_at_open = _normalize_open_window_timestamp(
        premarket_row.get("strategy_predicted_at_open")
        or premarket_row.get("captured_at"),
        date_hint=today_str,
        default_offset_minutes=5,
    )
    ai_predicted_at_open = _normalize_open_window_timestamp(
        premarket_row.get("ai_predicted_at_open") or premarket_row.get("captured_at"),
        date_hint=today_str,
        default_offset_minutes=7,
    )

    with _log_lock:
        existing = {}
        if log_file.exists():
            try:
                with open(log_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}

        existing[ticker] = {
            "predicted_return": pred.get("predicted_return", 0),
            "predicted_price": pred.get("predicted_price", 0),
            "current_price": pred.get("current_price", 0),
            "open_price": round(open_price, 2) if open_price > 0 else 0.0,
            "strategy_price_at_open": strategy_price_at_open,
            "ai_last_prediction": (
                round(ai_last_prediction, 2) if ai_last_prediction > 0 else None
            ),
            "ai_source": ai_source if ai_last_prediction > 0 else "none",
            "strategy_direction_at_open": strategy_direction,
            "ai_direction_last": ai_direction,
            "strategy_vs_ai_direction": (
                strategy_direction == ai_direction if ai_last_prediction > 0 else None
            ),
            "direction_comparison": None,
            "strategy_predicted_at_open": strategy_predicted_at_open,
            "ai_predicted_at_open": ai_predicted_at_open,
            "ai_last_prediction_at": now_iso,
            "signal": pred.get("signal", "HOLD"),
            "confidence": pred.get("confidence", 50),
            "model_predictions": pred.get("model_predictions", {}),
            "timestamp": now_iso,
        }

        with open(log_file, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    # Also record to prediction tracker for hit/miss tracking
    try:
        tracked = dict(pred)
        tracked["open_price"] = round(open_price, 2) if open_price > 0 else 0.0
        tracked["strategy_price_at_open"] = strategy_price_at_open
        tracked["ai_last_prediction"] = (
            round(ai_last_prediction, 2) if ai_last_prediction > 0 else 0.0
        )
        tracked["ai_source"] = ai_source if ai_last_prediction > 0 else "none"
        tracked["strategy_predicted_at_open"] = strategy_predicted_at_open
        tracked["ai_predicted_at_open"] = ai_predicted_at_open
        tracked["ai_last_prediction_at"] = now_iso
        tracked["strategy_direction_at_open"] = strategy_direction
        tracked["ai_direction_last"] = ai_direction
        tracked["strategy_vs_ai_direction"] = (
            strategy_direction == ai_direction if ai_last_prediction > 0 else None
        )
        tracked["direction_comparison"] = None
        PredictionTracker.record_prediction(ticker, tracked)
    except Exception as e:
        log.warning(f"Prediction tracking failed for {ticker}: {e}")


def _read_portfolio() -> list[dict]:
    """Read user portfolio entries from cache file."""
    with _portfolio_lock:
        if not PORTFOLIO_FILE.exists():
            return []
        try:
            data = json.loads(PORTFOLIO_FILE.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []


def _write_portfolio(entries: list[dict]) -> None:
    """Persist user portfolio entries to cache file."""
    with _portfolio_lock:
        PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
        PORTFOLIO_FILE.write_text(json.dumps(entries, indent=2))


def _read_portfolio_trades() -> list[dict]:
    with _portfolio_lock:
        if not PORTFOLIO_TRADES_FILE.exists():
            return []
        try:
            data = json.loads(PORTFOLIO_TRADES_FILE.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []


def _write_portfolio_trades(entries: list[dict]) -> None:
    with _portfolio_lock:
        PORTFOLIO_TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
        PORTFOLIO_TRADES_FILE.write_text(json.dumps(entries, indent=2))


def _load_delisted_registry_unlocked() -> dict[str, dict]:
    registry: dict[str, dict] = {}
    if not DELISTED_TICKERS_FILE.exists():
        return registry
    try:
        with open(DELISTED_TICKERS_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = str(row.get("ticker", "")).strip().upper()
                if not ticker:
                    continue
                try:
                    hit_count = int(float(row.get("hit_count", 0) or 0))
                except Exception:
                    hit_count = 0
                registry[ticker] = {
                    "ticker": ticker,
                    "first_seen": row.get("first_seen", ""),
                    "last_seen": row.get("last_seen", ""),
                    "hit_count": hit_count,
                    "last_reason": row.get("last_reason", ""),
                    "last_source": row.get("last_source", ""),
                    "status": row.get("status", "watchlist"),
                }
    except Exception as e:
        log.warning(f"Failed to read delisted ticker registry: {e}")
    return registry


def _save_delisted_registry_unlocked(registry: dict[str, dict]) -> None:
    DELISTED_TICKERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ticker",
        "first_seen",
        "last_seen",
        "hit_count",
        "last_reason",
        "last_source",
        "status",
    ]
    with open(DELISTED_TICKERS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ticker in sorted(registry):
            row = registry[ticker]
            writer.writerow(
                {
                    "ticker": row.get("ticker", ticker),
                    "first_seen": row.get("first_seen", ""),
                    "last_seen": row.get("last_seen", ""),
                    "hit_count": int(row.get("hit_count", 0) or 0),
                    "last_reason": row.get("last_reason", ""),
                    "last_source": row.get("last_source", ""),
                    "status": row.get("status", "watchlist"),
                }
            )


def _record_unavailable_ticker(
    ticker: str, reason: str, source: str = "yfinance"
) -> None:
    """Track tickers that repeatedly fail to return valid price data."""
    t = str(ticker or "").strip().upper()
    if not t:
        return
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    with _delisted_lock:
        registry = _load_delisted_registry_unlocked()
        row = registry.get(t)
        if row is None:
            row = {
                "ticker": t,
                "first_seen": now,
                "last_seen": now,
                "hit_count": 1,
                "last_reason": reason,
                "last_source": source,
                "status": "watchlist",
            }
        else:
            row["last_seen"] = now
            row["hit_count"] = int(row.get("hit_count", 0) or 0) + 1
            row["last_reason"] = reason
            row["last_source"] = source
        if int(row.get("hit_count", 0) or 0) >= 3:
            row["status"] = "delisted_candidate"
        elif row.get("status") != "recovered":
            row["status"] = "watchlist"
        registry[t] = row
        _save_delisted_registry_unlocked(registry)


def _mark_ticker_recovered(ticker: str, source: str = "yfinance") -> None:
    """Mark previously unavailable ticker as recovered once data is back."""
    t = str(ticker or "").strip().upper()
    if not t:
        return
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    with _delisted_lock:
        registry = _load_delisted_registry_unlocked()
        row = registry.get(t)
        if not row:
            return
        row["last_seen"] = now
        row["last_source"] = source
        row["last_reason"] = "price_available"
        row["status"] = "recovered"
        registry[t] = row
        _save_delisted_registry_unlocked(registry)


def _prune_delisted_registry() -> dict:
    """
    Remove noisy batch artifacts from delisted registry.
    Keep:
      - symbols not in configured universe (e.g., accidental dummy entries),
      - entries created from explicit single-ticker failures.
    """
    with _delisted_lock:
        registry = _load_delisted_registry_unlocked()
        if not registry:
            return {"changed": False, "removed": 0, "remaining": 0}

        allowed_universe = set(all_tickers)
        kept: dict[str, dict] = {}
        removed = 0
        for ticker, row in registry.items():
            src = str(row.get("last_source", ""))
            if ticker not in allowed_universe and not ticker.startswith("^"):
                kept[ticker] = row
                continue
            if src == "yfinance_single":
                kept[ticker] = row
                continue
            removed += 1

        changed = len(kept) != len(registry)
        if changed:
            _save_delisted_registry_unlocked(kept)
        return {"changed": changed, "removed": removed, "remaining": len(kept)}


def _is_tradeable_ticker(ticker: str) -> bool:
    t = str(ticker or "").strip().upper()
    if not t:
        return False
    if t in set(all_tickers) or t in ticker_names:
        return True
    # Test/dev fallback before ticker universe is loaded.
    if not all_tickers and (t.endswith(".NS") or t.endswith(".BO")) and len(t) >= 4:
        return True
    return False


def _new_trade_entry(ticker: str, side: str, qty: float, price: float) -> dict:
    return {
        "id": uuid4().hex[:12],
        "ticker": ticker,
        "name": ticker_names.get(ticker, _clean_name(ticker)),
        "side": side,
        "quantity": round(float(qty), 4),
        "price": round(float(price), 2),
        "timestamp": datetime.now(IST).isoformat(),
    }


def _ensure_trade_ids(trades: list[dict]) -> bool:
    """Backfill IDs for older trade rows. Returns True if modified."""
    changed = False
    seen: set[str] = set()
    for tr in trades:
        tid = str(tr.get("id", "")).strip()
        if not tid or tid in seen:
            tr["id"] = uuid4().hex[:12]
            tid = tr["id"]
            changed = True
        seen.add(tid)
    return changed


def _validate_trade_sequence(trades: list[dict]) -> tuple[bool, str]:
    """Simple balance validation: cumulative SELL qty cannot exceed BUY qty per ticker."""
    qty_open: dict[str, float] = {}
    for tr in trades:
        ticker = str(tr.get("ticker", "")).strip().upper()
        side = str(tr.get("side", "")).strip().upper()
        if not ticker or side not in {"BUY", "SELL"}:
            return False, "Invalid trade row"
        try:
            qty = float(tr.get("quantity", 0) or 0)
            price = float(tr.get("price", 0) or 0)
        except Exception:
            return False, f"Invalid quantity/price for {ticker}"
        if qty <= 0 or price <= 0:
            return False, f"Non-positive quantity/price for {ticker}"
        if side == "BUY":
            qty_open[ticker] = qty_open.get(ticker, 0.0) + qty
        else:
            have = qty_open.get(ticker, 0.0)
            if have + 1e-9 < qty:
                return False, f"SELL exceeds open quantity for {ticker}"
            qty_open[ticker] = have - qty
    return True, ""


def _sanitize_portfolio_storage() -> dict:
    """
    Remove invalid/dummy/unknown portfolio rows and backfill missing trade IDs.
    This prevents placeholders like ABC.NS/B.NS from polluting portfolio logic.
    """
    trades = _read_portfolio_trades()
    holdings = _read_portfolio()
    removed_trade_rows = 0
    removed_holding_rows = 0
    changed = False

    valid_trades: list[dict] = []
    for tr in trades:
        ticker = str(tr.get("ticker", "")).strip().upper()
        side = str(tr.get("side", "")).strip().upper()
        try:
            qty = float(tr.get("quantity", 0) or 0)
            price = float(tr.get("price", 0) or 0)
        except Exception:
            qty, price = 0, 0
        if (
            not _is_tradeable_ticker(ticker)
            or side not in {"BUY", "SELL"}
            or qty <= 0
            or price <= 0
        ):
            removed_trade_rows += 1
            changed = True
            continue
        tr["ticker"] = ticker
        tr["side"] = side
        tr["name"] = ticker_names.get(ticker, _clean_name(ticker))
        valid_trades.append(tr)

    if _ensure_trade_ids(valid_trades):
        changed = True

    ok, _err = _validate_trade_sequence(valid_trades)
    if not ok:
        # Keep only BUY rows if sequence got corrupted beyond recovery.
        rebuilt = [t for t in valid_trades if t.get("side") == "BUY"]
        if len(rebuilt) != len(valid_trades):
            changed = True
            removed_trade_rows += len(valid_trades) - len(rebuilt)
            valid_trades = rebuilt

    valid_holdings: list[dict] = []
    for row in holdings:
        ticker = str(row.get("ticker", "")).strip().upper()
        if not _is_tradeable_ticker(ticker):
            removed_holding_rows += 1
            changed = True
            continue
        row["ticker"] = ticker
        row["name"] = ticker_names.get(ticker, _clean_name(ticker))
        valid_holdings.append(row)

    if changed:
        _write_portfolio_trades(valid_trades)
        summary = _portfolio_summary_from_trades(valid_trades)
        _write_portfolio(
            [
                {
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "quantity": row["quantity"],
                    "entry_price": row["avg_buy_price"],
                    "updated_at": datetime.now(IST).isoformat(),
                }
                for row in summary["positions"]
            ]
        )

    return {
        "changed": changed,
        "removed_trade_rows": removed_trade_rows,
        "removed_holding_rows": removed_holding_rows,
        "remaining_trades": len(valid_trades),
    }


def _portfolio_summary_from_trades(
    trades: list[dict], include_live_prices: bool = True
) -> dict:
    positions: dict[str, dict] = {}
    realized_total = 0.0

    # FIFO lot accounting for accurate realized pnl
    lots: dict[str, list[dict]] = {}
    for tr in trades:
        ticker = tr.get("ticker")
        side = str(tr.get("side", "BUY")).upper()
        qty = float(tr.get("quantity", 0) or 0)
        price = float(tr.get("price", 0) or 0)
        if not ticker or qty <= 0 or price <= 0:
            continue
        lots.setdefault(ticker, [])
        if side == "BUY":
            lots[ticker].append({"qty": qty, "price": price})
        elif side == "SELL":
            remaining = qty
            while remaining > 1e-9 and lots[ticker]:
                lot = lots[ticker][0]
                consume = min(remaining, lot["qty"])
                realized_total += (price - lot["price"]) * consume
                lot["qty"] -= consume
                remaining -= consume
                if lot["qty"] <= 1e-9:
                    lots[ticker].pop(0)

    for ticker, ticker_lots in lots.items():
        qty = sum(l["qty"] for l in ticker_lots)
        if qty <= 1e-9:
            continue
        cost = sum(l["qty"] * l["price"] for l in ticker_lots)
        avg_price = cost / qty
        positions[ticker] = {
            "ticker": ticker,
            "name": ticker_names.get(ticker, _clean_name(ticker)),
            "quantity": round(qty, 4),
            "avg_buy_price": round(avg_price, 2),
            "cost_value": round(cost, 2),
        }

    if include_live_prices and positions:
        live = _get_live_prices_batch(list(positions.keys()))
    else:
        live = {}

    unrealized_total = 0.0
    for ticker, row in positions.items():
        l = live.get(ticker, {})
        curr = float(l.get("price", 0) or 0)
        row["current_price"] = round(curr, 2)
        m2m = curr * row["quantity"] if curr > 0 else 0.0
        row["market_value"] = round(m2m, 2)
        row["unrealized_pnl"] = round(m2m - row["cost_value"], 2)
        row["unrealized_pnl_pct"] = (
            round(((m2m - row["cost_value"]) / row["cost_value"] * 100), 3)
            if row["cost_value"] > 0 and m2m > 0
            else 0.0
        )
        unrealized_total += m2m - row["cost_value"]

    return {
        "positions": sorted(
            positions.values(), key=lambda x: x["cost_value"], reverse=True
        ),
        "position_count": len(positions),
        "realized_pnl": round(realized_total, 2),
        "unrealized_pnl": round(unrealized_total, 2),
        "total_pnl": round(realized_total + unrealized_total, 2),
    }


def _get_live_prices_batch(tickers: list[str]) -> dict:
    """Get live prices for multiple tickers using yfinance."""
    import yfinance as yf

    result: dict[str, dict] = {}

    clean_tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not clean_tickers:
        return result

    def _extract_field_series(df: pd.DataFrame, field: str, ticker: str):
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            if field not in lvl0:
                return None
            obj = df[field]
            if isinstance(obj, pd.DataFrame):
                if ticker in obj.columns:
                    return obj[ticker]
                if obj.shape[1] == 1:
                    return obj.iloc[:, 0]
                return None
            return obj
        return df[field] if field in df.columns else None

    try:
        # Use unadjusted prices for live parity with broker/app quotes.
        data = yf.download(
            clean_tickers if len(clean_tickers) > 1 else clean_tickers[0],
            period="2d",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=True,
        )
        if data is None or data.empty:
            if len(clean_tickers) == 1:
                _record_unavailable_ticker(
                    clean_tickers[0], "empty_batch_data", "yfinance_single"
                )
            return result

        successful: set[str] = set()
        for ticker in clean_tickers:
            try:
                close_col = _extract_field_series(data, "Close", ticker)

                if close_col is None or close_col.dropna().empty:
                    if len(clean_tickers) == 1:
                        _record_unavailable_ticker(
                            ticker, "missing_close_series", "yfinance_single"
                        )
                    continue

                close_vals = close_col.dropna()
                current = float(close_vals.iloc[-1])
                prev = float(close_vals.iloc[-2]) if len(close_vals) >= 2 else current
                change = current - prev
                change_pct = (change / prev * 100) if prev != 0 else 0

                vol_col = _extract_field_series(data, "Volume", ticker)
                high_col = _extract_field_series(data, "High", ticker)
                low_col = _extract_field_series(data, "Low", ticker)
                open_col = _extract_field_series(data, "Open", ticker)

                vol = (
                    float(vol_col.dropna().iloc[-1])
                    if vol_col is not None and not vol_col.dropna().empty
                    else 0
                )
                high = (
                    float(high_col.dropna().iloc[-1])
                    if high_col is not None and not high_col.dropna().empty
                    else current
                )
                low = (
                    float(low_col.dropna().iloc[-1])
                    if low_col is not None and not low_col.dropna().empty
                    else current
                )
                open_p = (
                    float(open_col.dropna().iloc[-1])
                    if open_col is not None and not open_col.dropna().empty
                    else current
                )

                result[ticker] = {
                    "price": round(current, 2),
                    "prev_close": round(prev, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "volume": int(vol),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "open": round(open_p, 2),
                }
                successful.add(ticker)
                if len(clean_tickers) == 1:
                    _mark_ticker_recovered(ticker, "yfinance_single")
            except Exception as exc:
                log.debug(f"Price fetch failed for {ticker}: {exc}")
                if len(clean_tickers) == 1:
                    _record_unavailable_ticker(
                        ticker,
                        f"price_fetch_exception:{type(exc).__name__}",
                        "yfinance_single",
                    )
                continue

        if not successful and len(clean_tickers) == 1:
            _record_unavailable_ticker(
                clean_tickers[0], "all_missing_single_ticker", "yfinance_single"
            )
    except Exception as e:
        log.warning(f"Batch price fetch error: {e}")

    return result


def _get_close_prices_for_date(tickers: list[str], date_str: str) -> dict[str, float]:
    """Fetch close price for the specific trading date for each ticker."""
    import yfinance as yf

    try:
        day = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return {}

    # yfinance end is exclusive; add 2 days to safely capture next trading day.
    start = day.strftime("%Y-%m-%d")
    end = (day + timedelta(days=2)).strftime("%Y-%m-%d")

    prices: dict[str, float] = {}
    if not tickers:
        return prices

    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=True,
        )
        if data is None or data.empty:
            return prices

        if len(tickers) == 1:
            ticker = tickers[0]
            close_series = data["Close"] if "Close" in data.columns else None
            if close_series is not None and not close_series.dropna().empty:
                prices[ticker] = round(float(close_series.dropna().iloc[0]), 2)
            return prices

        if isinstance(data.columns, pd.MultiIndex):
            lvl0 = data.columns.get_level_values(0)
            if "Close" in lvl0:
                close_df = data["Close"]
                for ticker in tickers:
                    if ticker in close_df.columns:
                        ser = close_df[ticker].dropna()
                        if not ser.empty:
                            prices[ticker] = round(float(ser.iloc[0]), 2)
    except Exception as e:
        log.warning(f"Date-close fetch failed for {date_str}: {e}")

    return prices


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Main dashboard — Groww-style stock cards."""
    return render_template("index.html")


@app.route("/stock/<ticker>")
def stock_detail(ticker: str):
    """Individual stock detail page with live chart and predictions."""
    name = ticker_names.get(ticker, _clean_name(ticker))
    return render_template("stock.html", ticker=ticker, name=name)


@app.route("/portfolio")
def portfolio_page():
    """Portfolio page with positions and full trade ledger."""
    return render_template("portfolio.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------
@app.route("/api/status")
def api_status():
    """Return server + market status."""
    mkt = get_market_status()
    load_elapsed = 0.0
    if models_load_started_at:
        ref = models_load_completed_at or time.time()
        load_elapsed = max(0.0, ref - models_load_started_at)

    load_progress = predictor.get_load_status() if predictor else {}
    with _premarket_snapshot_lock:
        premarket_meta = (
            {
                "date": _premarket_snapshot.get("date"),
                "captured_at": _premarket_snapshot.get("captured_at"),
                "snapshot_type": _premarket_snapshot.get("snapshot_type"),
                "captured_within_buffer": _premarket_snapshot.get(
                    "captured_within_buffer"
                ),
            }
            if _premarket_snapshot
            else {}
        )
    snapshot_now_iso = datetime.now(IST).isoformat()
    return jsonify(
        {
            "models_loaded": models_loaded,
            "models_loading": models_loading,
            "load_error": load_error,
            "model_count": len(predictor.models) if predictor else 0,
            "model_load_elapsed_sec": round(load_elapsed, 1),
            "model_load_progress": load_progress,
            "market": {
                "status": mkt["status"],
                "description": mkt["description"],
                "next_open": mkt.get("next_open", ""),
                "ist_now": mkt["ist_now"].strftime("%d %b %Y, %I:%M %p IST"),
            },
            "ticker_count": len(all_tickers),
            "premarket_config": {
                "max_buffer_minutes": PREMARKET_MAX_BUFFER_MINUTES,
                "default_tickers": PREMARKET_DEFAULT_TICKERS,
                "after_hours_ui_hour": AFTER_HOURS_UI_HOUR,
                "after_hours_ui_minute": AFTER_HOURS_UI_MINUTE,
            },
            "premarket_snapshot": premarket_meta,
        }
    )


@app.route("/api/sectors")
def api_sectors():
    """Return tickers organized by sector."""
    result = {}
    for sec, tickers in tickers_by_sector.items():
        result[sec] = {
            "display_name": SECTOR_DISPLAY.get(sec, sec),
            "tickers": [
                {"symbol": t, "name": ticker_names.get(t, _clean_name(t))}
                for t in tickers
            ],
        }
    return jsonify(result)


@app.route("/api/prices")
def api_prices():
    """Get live prices for tickers. Query: ?sector=large_cap or ?tickers=RELIANCE.NS,TCS.NS"""
    sector = request.args.get("sector")
    tickers_param = request.args.get("tickers")

    if tickers_param:
        tickers = [t.strip() for t in tickers_param.split(",")]
    elif sector and sector in tickers_by_sector:
        tickers = tickers_by_sector[sector]
    else:
        # Default: large_cap
        tickers = tickers_by_sector.get("large_cap", all_tickers[:30])

    prices = _get_live_prices_batch(tickers)

    # Attach names
    for t in prices:
        prices[t]["name"] = ticker_names.get(t, _clean_name(t))
        prices[t]["symbol"] = t

    return jsonify(prices)


@app.route("/api/predict/<ticker>")
def api_predict(ticker: str):
    """Get ML prediction for a single ticker."""
    if not models_loaded or predictor is None:
        return jsonify({"error": "Models still loading, please wait..."}), 503

    force = request.args.get("force", "").lower() in ("true", "1", "yes")
    try:
        pred = predictor.predict_single(ticker, use_cache=not force)
        if pred is None:
            return jsonify({"error": f"Prediction failed for {ticker}"}), 404

        # Log for end-of-day comparison
        _log_prediction(ticker, pred)

        pred["name"] = ticker_names.get(ticker, _clean_name(ticker))
        return jsonify(pred)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/top-picks")
def api_top_picks():
    """Get top ML picks across sectors."""
    if not models_loaded or predictor is None:
        return jsonify({"error": "Models still loading..."}), 503

    sectors = request.args.getlist("sectors") or ["large_cap", "banking"]
    top_n = int(request.args.get("n", 5))
    grouped = request.args.get("grouped", "").lower() in ("1", "true", "yes")

    try:
        if grouped:
            groups = predictor.predict_top_picks_grouped(sectors=sectors, top_n=top_n)
            for group in groups.values():
                for pick in group:
                    if pick.get("current_price", 0) > 0:
                        pick["name"] = ticker_names.get(pick.get("ticker", ""), "")
            groups = {
                key: [p for p in value if p.get("current_price", 0) > 0]
                for key, value in groups.items()
            }
            return jsonify(groups)

        picks = predictor.predict_top_picks(sectors=sectors, top_n=top_n)
        picks = [p for p in picks if p.get("current_price", 0) > 0]
        for p in picks:
            p["name"] = ticker_names.get(p.get("ticker", ""), "")
        return jsonify(picks)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/premarket-outlook")
def api_premarket_outlook():
    """
    Premarket snapshot endpoint.
    Returns:
      ticker, current_price, strategy_price_at_open, ai_predicted_price,
      strategy_direction, ai_direction, captured_at,
      strategy_predicted_at_open, ai_predicted_at_open.
    """
    if predictor is None:
        return jsonify({"error": "Predictor not initialized"}), 503

    force = request.args.get("force", "").lower() in ("1", "true", "yes")
    try:
        snapshot = _capture_premarket_snapshot_if_due(force=force)
        items = snapshot.get("items", [])
        return jsonify(
            {
                "date": snapshot.get("date"),
                "captured_at": snapshot.get("captured_at"),
                "captured_at_actual": snapshot.get("captured_at_actual"),
                "capture_cutoff": snapshot.get("capture_cutoff"),
                "captured_within_buffer": snapshot.get("captured_within_buffer", True),
                "buffer_minutes": snapshot.get(
                    "buffer_minutes", PREMARKET_MAX_BUFFER_MINUTES
                ),
                "snapshot_type": snapshot.get("snapshot_type", "market_open"),
                "capture_note": snapshot.get("capture_note"),
                "items": items,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/intraday/<ticker>")
def api_intraday(ticker: str):
    """Get intraday price data for charting."""
    period = request.args.get("period", "5d")
    interval = request.args.get("interval", "15m")

    try:
        df = get_intraday_data(ticker, period=period, interval=interval)
        if (df is None or df.empty) and interval == "1m":
            # Fallback when 1m bars are not available for the requested period window.
            df = get_intraday_data(ticker, period=period, interval="5m")
        if df is None or df.empty:
            return jsonify({"error": "No intraday data"}), 404

        records = []
        for idx, row in df.iterrows():
            records.append(
                {
                    "time": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                    "open": round(float(row.get("Open", 0)), 2),
                    "high": round(float(row.get("High", 0)), 2),
                    "low": round(float(row.get("Low", 0)), 2),
                    "close": round(float(row.get("Close", 0)), 2),
                    "volume": int(row.get("Volume", 0)),
                }
            )
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/<ticker>")
def api_history(ticker: str):
    """Get daily price history for charting."""
    period = request.args.get("period", "1y")
    import yfinance as yf

    try:
        df = yf.download(
            ticker, period=period, interval="1d", progress=False, auto_adjust=False
        )
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 404
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        records = []
        for idx, row in df.iterrows():
            records.append(
                {
                    "time": idx.strftime("%Y-%m-%d"),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                }
            )
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expected-vs-actual")
def api_expected_vs_actual():
    """
    Compare today's (or a given date's) predictions vs actual closing prices.
    Returns hit rate, alpha, and per-stock results.
    """
    date_str = request.args.get("date", datetime.now(IST).strftime("%Y-%m-%d"))
    log_file = PREDICTION_LOG_DIR / f"{date_str}.json"

    if not log_file.exists():
        return jsonify(
            {"error": f"No predictions logged for {date_str}", "date": date_str}
        )

    with open(log_file) as f:
        predictions = json.load(f)

    if not predictions:
        return jsonify({"error": "Empty prediction log", "date": date_str})

    # Persist and hydrate tracker outcomes so expected-vs-actual stays retraining-ready.
    tracker_results = PredictionTracker.check_outcomes(date_str)
    tracker_by_ticker = tracker_results if isinstance(tracker_results, dict) else {}

    # Fetch actual prices
    tickers_to_check = list(predictions.keys())
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    market_status = get_market_status().get("status", "")
    after_hours_mode = _is_after_hours_ui_mode(datetime.now(IST), market_status)
    if date_str == today_str and not after_hours_mode:
        actuals = _get_live_prices_batch(tickers_to_check)
        actual_prices = {k: v.get("price", 0) for k, v in actuals.items()}
    else:
        actual_prices = _get_close_prices_for_date(tickers_to_check, date_str)

    # Fetch benchmark (Nifty 50) return for alpha
    benchmark_data = _get_live_prices_batch(["^NSEI"])
    benchmark_return = 0.0
    if "^NSEI" in benchmark_data:
        bd = benchmark_data["^NSEI"]
        if bd["prev_close"] > 0:
            benchmark_return = (bd["price"] - bd["prev_close"]) / bd["prev_close"]
    market_status = get_market_status().get("status", "")

    def _rescale_logged_price(raw_price: float, actual_price: float) -> float:
        """
        Repair stale logged prices after corporate actions or unit drift.
        Keeps normal prices unchanged and only rescales obvious outliers.
        """
        if raw_price <= 0 or actual_price <= 0:
            return raw_price
        ratio = raw_price / actual_price
        scale_factors = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100)
        for factor in scale_factors:
            if abs(ratio - factor) / factor <= 0.35:
                return raw_price / factor
            inv = 1.0 / factor
            if abs(ratio - inv) / inv <= 0.35:
                return raw_price * factor
        if ratio > 5.0 or ratio < 0.2:
            return actual_price
        return raw_price

    results = []
    for ticker, pred in predictions.items():
        if ticker not in actual_prices:
            continue

        tracker_row = (
            tracker_by_ticker.get(ticker, {})
            if isinstance(tracker_by_ticker, dict)
            else {}
        )
        market_open_row = _get_premarket_row_for_ticker(
            ticker,
            date_str,
            allowed_snapshot_types={"market_open", "near_open_fallback"},
        )

        pred_price_at_prediction = float(pred.get("current_price", 0) or 0)
        pred_return_pct = _normalize_predicted_return_pct(
            pred.get("predicted_return", 0)
        )
        actual_price = float(actual_prices[ticker] or 0)

        if pred_price_at_prediction <= 0 or actual_price <= 0:
            continue

        pred_price_at_prediction = _rescale_logged_price(
            pred_price_at_prediction, actual_price
        )

        actual_return = (
            actual_price - pred_price_at_prediction
        ) / pred_price_at_prediction
        actual_return_pct = actual_return * 100

        # Direction check uses market-open baseline vs strategy-open prediction.
        pred_dir = (
            "UP" if pred_return_pct > 0 else "DOWN" if pred_return_pct < 0 else "FLAT"
        )
        strategy_price_at_open = float(
            tracker_row.get(
                "strategy_price_at_open",
                market_open_row.get(
                    "strategy_price_at_open",
                    pred.get("strategy_price_at_open", pred.get("predicted_price", 0)),
                ),
            )
            or 0
        )
        ai_last_prediction = float(
            tracker_row.get(
                "ai_last_prediction",
                pred.get("ai_last_prediction", 0),
            )
            or 0
        )
        ai_source_hint = str(
            tracker_row.get("ai_source") or pred.get("ai_source") or "none"
        ).lower()
        open_price = float(
            tracker_row.get(
                "open_price",
                market_open_row.get(
                    "open_price",
                    pred.get("open_price", pred_price_at_prediction),
                ),
            )
            or pred_price_at_prediction
        )
        open_price = _rescale_logged_price(open_price, actual_price)
        strategy_price_at_open = _rescale_logged_price(
            strategy_price_at_open, actual_price
        )
        if ai_last_prediction > 0:
            ai_last_prediction = _rescale_logged_price(ai_last_prediction, actual_price)
        if (
            ai_last_prediction > 0
            and abs(ai_last_prediction - strategy_price_at_open) <= 0.01
            and ai_source_hint in {"", "none", "fallback", "strategy_fallback"}
        ):
            ai_last_prediction = 0.0
        if actual_price > 0 and open_price <= 0:
            open_price = pred_price_at_prediction
        if actual_price > 0 and strategy_price_at_open <= 0:
            strategy_price_at_open = 0
        if actual_price > 0 and ai_last_prediction <= 0:
            ai_last_prediction = 0
        if open_price > 0 and (
            strategy_price_at_open <= 0 or strategy_price_at_open > open_price * 5
        ):
            strategy_price_at_open = round(
                open_price * (1 + pred_return_pct / 100.0), 2
            )
        if strategy_price_at_open <= 0:
            strategy_price_at_open = (
                round(open_price, 2)
                if open_price > 0
                else round(float(pred.get("predicted_price", 0) or 0), 2)
            )
        if ai_last_prediction <= 0 or ai_last_prediction > pred_price_at_prediction * 5:
            ai_last_prediction = 0.0
        strategy_predicted_at_open = _normalize_open_window_timestamp(
            tracker_row.get("strategy_predicted_at_open")
            or market_open_row.get("strategy_predicted_at_open")
            or market_open_row.get("captured_at")
            or pred.get("strategy_predicted_at_open")
            or pred.get("timestamp"),
            date_hint=date_str,
            default_offset_minutes=5,
        )
        ai_predicted_at_open = _normalize_open_window_timestamp(
            tracker_row.get("ai_predicted_at_open")
            or market_open_row.get("ai_predicted_at_open")
            or market_open_row.get("captured_at")
            or pred.get("ai_predicted_at_open")
            or pred.get("timestamp"),
            date_hint=date_str,
            default_offset_minutes=7,
        )
        ai_last_prediction_at = (
            tracker_row.get("ai_last_prediction_at")
            or pred.get("ai_last_prediction_at")
            or pred.get("timestamp")
        )
        strategy_direction = tracker_row.get(
            "strategy_direction_at_open"
        ) or _direction_from_prices(open_price, strategy_price_at_open)
        ai_direction = tracker_row.get("ai_direction_last") or (
            _direction_from_prices(pred_price_at_prediction, ai_last_prediction)
            if ai_last_prediction > 0
            else "N/A"
        )
        actual_dir = _direction_from_prices(open_price, actual_price)
        strategy_vs_actual = strategy_direction == actual_dir
        direction_comparison = bool(strategy_vs_actual)
        strategy_vs_ai_direction = tracker_row.get("strategy_vs_ai_direction")
        if strategy_vs_ai_direction not in (True, False):
            strategy_vs_ai_direction = (
                strategy_direction == ai_direction if ai_last_prediction > 0 else None
            )
        predicted_price = float(pred.get("predicted_price", 0) or 0)
        if predicted_price <= 0 or predicted_price > pred_price_at_prediction * 5:
            predicted_price = round(
                pred_price_at_prediction * (1 + pred_return_pct / 100.0), 2
            )
        predicted_price = _rescale_logged_price(predicted_price, actual_price)
        strategy_vs_actual_price_diff = (
            actual_price - strategy_price_at_open if strategy_price_at_open > 0 else 0.0
        )
        strategy_vs_actual_pct = (
            (strategy_vs_actual_price_diff / open_price * 100)
            if open_price > 0
            else 0.0
        )
        market_open_price = (
            round(open_price, 2)
            if open_price > 0
            else round(pred_price_at_prediction, 2)
        )
        strategy_price_display = (
            round(strategy_price_at_open, 2)
            if strategy_price_at_open > 0
            else market_open_price
        )

        # Alpha = actual return - benchmark return
        alpha = actual_return - benchmark_return
        alpha_pct = alpha * 100

        results.append(
            {
                "ticker": ticker,
                "name": ticker_names.get(ticker, _clean_name(ticker)),
                "signal": pred.get("signal", "N/A"),
                "predicted_return_pct": round(pred_return_pct, 3),
                "predicted_price": round(predicted_price, 2),
                "actual_price": round(actual_price, 2),
                "actual_close": round(actual_price, 2),
                "actual_return_pct": round(actual_return_pct, 3),
                "market_open_price": market_open_price,
                "open_price": market_open_price,
                "strategy_vs_actual_price_diff": round(
                    strategy_vs_actual_price_diff, 2
                ),
                "strategy_vs_actual_pct": round(strategy_vs_actual_pct, 3),
                "direction_predicted": pred_dir,
                "direction_actual": actual_dir,
                "strategy_direction_at_open": strategy_direction,
                "ai_direction_last": ai_direction,
                "strategy_vs_ai_direction": strategy_vs_ai_direction,
                "direction_comparison": direction_comparison,
                "direction_correct": direction_comparison,
                "last_prediction_basis": "strategy_vs_actual",
                "strategy_price_at_open": strategy_price_display,
                "ai_last_prediction": (
                    round(ai_last_prediction, 2) if ai_last_prediction > 0 else None
                ),
                "strategy_predicted_at_open": strategy_predicted_at_open,
                "strategy_predicted_at_open_display": _format_ist_timestamp(
                    strategy_predicted_at_open
                ),
                "ai_predicted_at_open": ai_predicted_at_open,
                "ai_predicted_at_open_display": _format_ist_timestamp(
                    ai_predicted_at_open
                ),
                "ai_last_prediction_at": ai_last_prediction_at,
                "ai_last_prediction_at_display": _format_ist_timestamp(
                    ai_last_prediction_at
                ),
                "alpha_pct": round(alpha_pct, 3),
                "confidence": pred.get("confidence", 50),
                "checked_at": tracker_row.get("checked_at"),
                "market_status": market_status,
                "after_hours_mode": after_hours_mode,
                "snapshot_type": market_open_row.get("snapshot_type", "market_open"),
            }
        )

    if not results:
        return jsonify({"error": "Could not fetch actuals", "date": date_str})

    # Summary
    total = len(results)
    hits = sum(1 for r in results if r["direction_comparison"])
    hit_rate = (hits / total * 100) if total > 0 else 0
    avg_alpha = float(np.mean([r["alpha_pct"] for r in results]))
    total_alpha = float(np.sum([r["alpha_pct"] for r in results]))
    avg_confidence = float(np.mean([r["confidence"] for r in results]))

    return jsonify(
        {
            "date": date_str,
            "total_predictions": total,
            "direction_hits": hits,
            "hit_rate_pct": round(hit_rate, 1),
            "avg_alpha_pct": round(avg_alpha, 3),
            "total_alpha_pct": round(total_alpha, 3),
            "benchmark_return_pct": round(benchmark_return * 100, 3),
            "avg_confidence": round(avg_confidence, 1),
            "schema_version": PredictionTracker.SCHEMA_VERSION,
            "market_status": market_status,
            "after_hours_mode": after_hours_mode,
            "results": sorted(results, key=lambda x: abs(x["alpha_pct"]), reverse=True),
        }
    )


@app.route("/api/prediction-dates")
def api_prediction_dates():
    """List all dates that have prediction logs."""
    if not PREDICTION_LOG_DIR.exists():
        return jsonify([])
    dates = sorted(
        [f.stem for f in PREDICTION_LOG_DIR.glob("*.json")],
        reverse=True,
    )
    return jsonify(dates)


@app.route("/api/ensemble-weights")
def api_ensemble_weights():
    """Return current ensemble model weights."""
    if predictor and predictor.ensemble:
        weights = predictor.ensemble.learned_weights or {}
        return jsonify(weights)
    return jsonify({"error": "Ensemble not loaded"}), 503


# ---------------------------------------------------------------------------
# Routes — Groq AI Explainer
# ---------------------------------------------------------------------------
@app.route("/api/explain", methods=["POST"])
def api_explain():
    """Get AI explanation for a metric/indicator/greek."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    explain_type = data.get("type", "")
    metric = data.get("metric", "")
    value = data.get("value", "")
    ticker = data.get("ticker", "")
    stock_name = data.get("stock_name", "")

    try:
        if explain_type == "fundamental":
            explanation = explain_fundamental(metric, value, ticker, stock_name)
        elif explain_type == "greek":
            explanation = explain_greek(
                metric, float(value) if value else 0, "call", ticker, stock_name
            )
        elif explain_type == "indicator":
            explanation = explain_indicator(metric, value, None, ticker, stock_name)
        else:
            explanation = f"Unknown explanation type: {explain_type}"

        return jsonify({"explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategy/<ticker>")
def api_strategy(ticker: str):
    """Get Groq AI strategy analysis for a ticker."""
    if not models_loaded or predictor is None:
        return jsonify({"error": "Models still loading..."}), 503

    try:
        # Get prediction data (from cache if available)
        pred = predictor.predict_single(ticker, use_cache=True)
        if pred is None:
            return jsonify({"error": f"No prediction for {ticker}"}), 404

        stock_name = ticker_names.get(ticker, _clean_name(ticker))

        # Get Groq strategy
        groq_strat = get_groq_strategy(ticker, stock_name, pred)

        # Get combined strategy
        combined = get_combined_strategy(ticker, stock_name, pred, groq_strat)

        return jsonify(
            {
                "groq_strategy": groq_strat,
                "combined_strategy": combined,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/overview/<ticker>")
def api_stock_overview(ticker: str):
    """Get comprehensive stock overview with company info and sentiment."""
    if not models_loaded or predictor is None:
        return jsonify({"error": "Models still loading..."}), 503

    try:
        # Get prediction data (from cache if available)
        pred = predictor.predict_single(ticker, use_cache=True)
        stock_name = ticker_names.get(ticker, _clean_name(ticker))

        fundamentals = pred.get("fundamentals", {}) if pred else {}
        current_price = pred.get("current_price", 0) if pred else 0
        signal = pred.get("signal", "") if pred else ""

        # Get stock overview
        overview = get_stock_overview(
            ticker, stock_name, fundamentals, current_price, signal
        )

        # Get news sentiment
        sentiment = get_news_sentiment(ticker, stock_name)

        return jsonify(
            {
                "overview": overview,
                "sentiment": sentiment,
                "stock_name": stock_name,
                "ticker": ticker,
            }
        )
    except Exception as e:
        log.warning(f"Overview error for {ticker}: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Routes — Prediction Tracking
# ---------------------------------------------------------------------------
@app.route("/api/tracking/daily")
def api_tracking_daily():
    """Get daily prediction tracking summary."""
    date_str = request.args.get("date")
    summary = PredictionTracker.get_daily_summary(date_str)
    return jsonify(summary)


@app.route("/api/tracking/check")
def api_tracking_check():
    """Check outcomes for a date's predictions."""
    date_str = request.args.get("date")
    results = PredictionTracker.check_outcomes(date_str)
    return jsonify(results)


@app.route("/api/tracking/monthly")
def api_tracking_monthly():
    """Get monthly accuracy report."""
    year = request.args.get("year", type=int)
    month = request.args.get("month", type=int)
    report = PredictionTracker.get_monthly_report(year, month)
    return jsonify(report)


@app.route("/api/tracking/history")
def api_tracking_history():
    """Get running monthly accuracy history."""
    history = PredictionTracker.get_accuracy_history()
    return jsonify(history)


@app.route("/api/tracking/feedback")
def api_tracking_feedback():
    """Get training feedback data for model improvement."""
    feedback = PredictionTracker.get_training_feedback_data()
    return jsonify(feedback)


@app.route("/api/training-feedback")
def api_training_feedback():
    """Alias endpoint for retraining export in a stable schema."""
    feedback = PredictionTracker.get_training_feedback_data()
    return jsonify(feedback)


# ---------------------------------------------------------------------------
# Routes — Daily Analysis (Enhanced Dashboard)
# ---------------------------------------------------------------------------
# In-memory cache of opening prices captured at market open
_opening_prices: dict[str, dict] = {}
_opening_prices_date: str = ""
_daily_prediction_baseline: dict[str, dict] = {}
_daily_prediction_baseline_date: str = ""
_premarket_snapshot: dict = {}
_premarket_snapshot_lock = threading.Lock()
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15

# Background-computed daily analysis cache (avoids blocking requests)
_daily_analysis_cache: dict = {}
_daily_analysis_cache_time: float = 0
_daily_analysis_lock = threading.Lock()
DAILY_ANALYSIS_TTL = 30  # seconds — serve cached result within this window

# Price cache with TTL (shared across endpoints)
_price_cache: dict[str, dict] = {}
_price_cache_time: float = 0
PRICE_CACHE_TTL = 15  # seconds


def _get_prices_chunked(tickers: list[str], chunk_size: int = 25) -> dict:
    """Fetch prices in parallel-friendly chunks to avoid yfinance timeouts."""
    all_prices: dict = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        try:
            prices = _get_live_prices_batch(chunk)
            all_prices.update(prices)
        except Exception as e:
            log.warning(f"Price chunk {i//chunk_size} failed: {e}")
    return all_prices


def _get_cached_prices() -> dict:
    """Return cached prices if fresh, otherwise refetch in chunks."""
    global _price_cache, _price_cache_time
    now = time.time()
    if _price_cache and (now - _price_cache_time) < PRICE_CACHE_TTL:
        return _price_cache
    prices = _get_prices_chunked(all_tickers)
    _price_cache = prices
    _price_cache_time = time.time()
    return prices


def _capture_opening_prices():
    """Capture opening prices once at start of day (uses chunked fetch)."""
    global _opening_prices, _opening_prices_date
    today = datetime.now(IST).strftime("%Y-%m-%d")
    if _opening_prices_date == today and _opening_prices:
        return  # Already captured today

    log.info("Capturing opening prices for today...")
    prices = _get_cached_prices()
    captured_at = datetime.now(IST).isoformat()
    _opening_prices = {}
    for t, p in prices.items():
        _opening_prices[t] = {
            "open": p.get("open", p.get("price", 0)),
            "prev_close": p.get("prev_close", 0),
            "captured_at": captured_at,
        }
    _opening_prices_date = today
    log.info(f"Captured opening prices for {len(_opening_prices)} tickers")


def _market_open_dt(now: datetime) -> datetime:
    return now.replace(
        hour=MARKET_OPEN_HOUR,
        minute=MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0,
    )


def _premarket_cutoff_dt(now: datetime) -> datetime:
    return _market_open_dt(now) - timedelta(minutes=PREMARKET_MAX_BUFFER_MINUTES)


def _open_window_bounds(trading_day: date) -> tuple[datetime, datetime]:
    start = datetime(
        trading_day.year,
        trading_day.month,
        trading_day.day,
        MARKET_OPEN_HOUR,
        MARKET_OPEN_MINUTE,
        tzinfo=IST,
    )
    end = start + timedelta(minutes=15)  # 09:15 -> 09:30 IST window
    return start, end


def _next_day_prediction_switch_dt(now: datetime) -> datetime:
    return now.replace(
        hour=NEXT_DAY_PREDICTION_HOUR,
        minute=NEXT_DAY_PREDICTION_MINUTE,
        second=0,
        microsecond=0,
    )


def _is_next_day_prediction_window(now: datetime) -> bool:
    return now >= _next_day_prediction_switch_dt(now)


def _after_hours_ui_dt(now: datetime) -> datetime:
    return now.replace(
        hour=AFTER_HOURS_UI_HOUR,
        minute=AFTER_HOURS_UI_MINUTE,
        second=0,
        microsecond=0,
    )


def _is_after_hours_ui_mode(now: datetime, market_status: str = "") -> bool:
    return market_status in ("after_hours", "weekend") or now >= _after_hours_ui_dt(now)


def _next_trading_day(d: date) -> date:
    out = d + timedelta(days=1)
    while out.weekday() >= 5:
        out += timedelta(days=1)
    return out


def _direction_from_prices(base_price: float, predicted_price: float) -> str:
    if base_price <= 0 or predicted_price <= 0:
        return "FLAT"
    if predicted_price > base_price:
        return "UP"
    if predicted_price < base_price:
        return "DOWN"
    return "FLAT"


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=IST)
    return parsed.astimezone(IST)


def _format_ist_timestamp(value: str | None) -> str | None:
    parsed = _parse_iso_datetime(value)
    if not parsed:
        return value if value else None
    return parsed.strftime("%d %b %Y, %I:%M:%S %p IST")


def _normalize_open_window_timestamp(
    value: str | None,
    *,
    date_hint: str | None = None,
    default_offset_minutes: int = 5,
) -> str:
    """
    Clamp/derive an "at-open" timestamp into 09:15–09:30 IST for a trading day.
    """
    parsed = _parse_iso_datetime(value)
    if parsed is not None:
        trading_day = parsed.date()
    elif date_hint:
        try:
            trading_day = datetime.strptime(date_hint, "%Y-%m-%d").date()
        except Exception:
            trading_day = datetime.now(IST).date()
    else:
        trading_day = datetime.now(IST).date()

    start, end = _open_window_bounds(trading_day)
    offset = int(np.clip(default_offset_minutes, 0, 15))

    if parsed is None:
        return (start + timedelta(minutes=offset)).isoformat()
    if parsed < start:
        return start.isoformat()
    if parsed > end:
        return end.isoformat()
    return parsed.isoformat()


def _get_premarket_row_for_ticker(
    ticker: str,
    date_str: str | None = None,
    allowed_snapshot_types: set[str] | None = None,
) -> dict:
    """
    Best-effort read of cached premarket row for a ticker (no live fetch).
    """
    target_date = date_str or datetime.now(IST).strftime("%Y-%m-%d")
    snapshot: dict = {}
    with _premarket_snapshot_lock:
        if (
            _premarket_snapshot
            and _premarket_snapshot.get("date") == target_date
            and _premarket_snapshot.get("items")
        ):
            snapshot = dict(_premarket_snapshot)

    if not snapshot:
        disk_snapshot = _load_premarket_snapshot_from_disk()
        if (
            disk_snapshot
            and disk_snapshot.get("date") == target_date
            and disk_snapshot.get("items")
        ):
            snapshot = disk_snapshot

    snapshot = _normalize_premarket_snapshot(snapshot)

    allowed = (
        {str(v).strip().lower() for v in allowed_snapshot_types}
        if allowed_snapshot_types
        else None
    )
    for row in snapshot.get("items", []):
        if row.get("ticker") != ticker:
            continue
        row_type = str(
            row.get("snapshot_type") or snapshot.get("snapshot_type") or ""
        ).lower()
        if allowed is not None and row_type not in allowed:
            continue
        return row
    return {}


def _load_premarket_snapshot_from_disk() -> dict:
    if not PREMARKET_OUTLOOK_FILE.exists():
        return {}
    try:
        data = json.loads(PREMARKET_OUTLOOK_FILE.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_premarket_snapshot_to_disk(snapshot: dict) -> None:
    PREMARKET_OUTLOOK_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREMARKET_OUTLOOK_FILE.write_text(json.dumps(snapshot, indent=2, default=str))


def _normalize_premarket_snapshot(snapshot: dict) -> dict:
    if not isinstance(snapshot, dict):
        return {}
    out = dict(snapshot)
    snap_date = out.get("date") or datetime.now(IST).strftime("%Y-%m-%d")
    snapshot_type = str(out.get("snapshot_type", "")).strip().lower()
    allowed_types = {
        "pending_market_open",
        "premarket_preview",
        "market_open",
        "near_open_fallback",
        "market_open_live",
        "market_open_backfilled",
    }
    if snapshot_type not in allowed_types:
        snapshot_type = ""
    out["snapshot_type"] = snapshot_type or "market_open"

    if snapshot_type == "pending_market_open":
        out["date"] = snap_date
        out.setdefault("captured_at", None)
        out["captured_at_actual"] = out.get("captured_at_actual")
        out["items"] = []
        out.setdefault("buffer_minutes", PREMARKET_MAX_BUFFER_MINUTES)
        out.setdefault("captured_within_buffer", None)
        out.setdefault("capture_note", "Waiting for market open snapshot window")
        return out

    snap_captured_actual = out.get("captured_at_actual") or out.get("captured_at")
    snap_captured = _normalize_open_window_timestamp(
        out.get("captured_at") or snap_captured_actual,
        date_hint=snap_date,
        default_offset_minutes=20,
    )
    out["date"] = snap_date
    out["captured_at"] = snap_captured
    out["captured_at_actual"] = snap_captured_actual or snap_captured
    out["buffer_minutes"] = int(out.get("buffer_minutes", PREMARKET_MAX_BUFFER_MINUTES))

    parsed_actual = _parse_iso_datetime(out["captured_at_actual"])
    if snapshot_type in {"", "market_open"}:
        if parsed_actual is not None:
            open_start, open_end = _open_window_bounds(parsed_actual.date())
            snapshot_type = (
                "market_open_backfilled"
                if parsed_actual > open_end
                else "market_open_live"
            )
            out["capture_cutoff"] = (
                out.get("capture_cutoff")
                or (open_start - timedelta(minutes=out["buffer_minutes"])).isoformat()
            )
        else:
            snapshot_type = "market_open"
    elif snapshot_type not in {
        "near_open_fallback",
        "premarket_preview",
        "market_open_live",
        "market_open_backfilled",
    }:
        snapshot_type = "market_open"
    out["snapshot_type"] = snapshot_type

    if out.get("captured_within_buffer") not in (True, False):
        if parsed_actual is not None:
            out["captured_within_buffer"] = bool(
                parsed_actual <= _premarket_cutoff_dt(parsed_actual)
            )
        else:
            out["captured_within_buffer"] = True
    if not out.get("capture_note"):
        out["capture_note"] = (
            "Captured in pre-open buffer"
            if out["captured_within_buffer"]
            else "Backfilled from later session using market-open window timestamps"
        )

    rows = []
    for raw in out.get("items", []):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row_captured_actual = (
            row.get("captured_at_actual")
            or row.get("captured_at")
            or out["captured_at_actual"]
        )
        row_captured = _normalize_open_window_timestamp(
            row.get("captured_at") or row_captured_actual or snap_captured,
            date_hint=snap_date,
            default_offset_minutes=20,
        )
        row["captured_at"] = row_captured
        row["captured_at_actual"] = row_captured_actual or out["captured_at_actual"]
        row["snapshot_type"] = row.get("snapshot_type") or out["snapshot_type"]
        row["strategy_predicted_at_open"] = _normalize_open_window_timestamp(
            row.get("strategy_predicted_at_open") or row_captured,
            date_hint=snap_date,
            default_offset_minutes=5,
        )
        row["ai_predicted_at_open"] = _normalize_open_window_timestamp(
            row.get("ai_predicted_at_open") or row_captured,
            date_hint=snap_date,
            default_offset_minutes=7,
        )
        try:
            ai_px = float(row.get("ai_predicted_price") or 0)
        except Exception:
            ai_px = 0.0
        row["strategy_source"] = row.get("strategy_source") or "ensemble_models"
        row["ai_source"] = row.get("ai_source") or (
            "groq_cache" if ai_px > 0 else "none"
        )
        if ai_px <= 0:
            row["ai_direction"] = "N/A"
        if row.get("strategy_vs_ai_direction") not in (True, False):
            if ai_px > 0:
                row["strategy_vs_ai_direction"] = (
                    str(row.get("strategy_direction", "FLAT")).upper()
                    == str(row.get("ai_direction", "N/A")).upper()
                )
            else:
                row["strategy_vs_ai_direction"] = None
        rows.append(row)
    out["items"] = rows
    return out


def _build_premarket_snapshot(
    tickers: list[str] | None = None,
    *,
    snapshot_type: str = "market_open",
    captured_at: str | None = None,
) -> dict:
    """
    Build a premarket snapshot payload with strategy-open and AI-now predictions.
    Internal prices are numeric INR, returns are %.
    """
    snapshot_type = str(snapshot_type or "market_open").strip().lower()
    captured_iso = captured_at or datetime.now(IST).isoformat()
    if predictor is None:
        return {
            "date": datetime.now(IST).strftime("%Y-%m-%d"),
            "captured_at": captured_iso,
            "snapshot_type": snapshot_type,
            "schema_version": 2,
            "items": [],
        }

    today = datetime.now(IST).strftime("%Y-%m-%d")
    scan_tickers = (
        tickers or tickers_by_sector.get("large_cap", [])[:PREMARKET_DEFAULT_TICKERS]
    )
    prices = _get_live_prices_batch(scan_tickers)
    captured_at_actual = datetime.now(IST).isoformat()
    captured_at_open_window = _normalize_open_window_timestamp(
        captured_at_actual,
        date_hint=today,
        default_offset_minutes=20,
    )
    rows: list[dict] = []

    for ticker in scan_tickers:
        quote = prices.get(ticker, {})
        current_price = float(quote.get("price", 0) or 0)
        open_price = float(quote.get("open", 0) or 0)
        if open_price <= 0:
            open_price = float(
                _opening_prices.get(ticker, {}).get("open", current_price)
                or current_price
            )
        if current_price <= 0:
            continue

        try:
            pred = predictor.predict_single(ticker, use_cache=True) or {}
        except Exception:
            pred = {}

        predicted_return_pct = _normalize_predicted_return_pct(
            pred.get("predicted_return", 0)
        )
        strategy_price_at_open = (
            round(open_price * (1 + predicted_return_pct / 100.0), 2)
            if open_price > 0
            else _safe_float(pred.get("predicted_price", 0))
        )
        ai_meta = _resolve_ai_forecast_price(
            ticker,
            open_price=open_price,
            strategy_price=strategy_price_at_open,
            current_price=current_price,
            allow_generate=True,
        )
        ai_predicted_price = _safe_float(ai_meta.get("price"))
        ai_direction = (
            _direction_from_prices(current_price, ai_predicted_price)
            if ai_predicted_price > 0
            else "N/A"
        )
        strategy_direction = _direction_from_prices(open_price, strategy_price_at_open)

        row = {
            "ticker": ticker,
            "name": ticker_names.get(ticker, _clean_name(ticker)),
            "current_price": round(current_price, 2),
            "open_price": round(open_price, 2) if open_price > 0 else None,
            "strategy_price_at_open": (
                round(strategy_price_at_open, 2) if strategy_price_at_open > 0 else None
            ),
            "ai_predicted_price": (
                round(ai_predicted_price, 2) if ai_predicted_price > 0 else None
            ),
            "strategy_direction": strategy_direction,
            "ai_direction": ai_direction,
            "predicted_return_pct": round(predicted_return_pct, 4),
            "captured_at": captured_at_open_window,
            "captured_at_actual": captured_at_actual,
            "strategy_predicted_at_open": _normalize_open_window_timestamp(
                captured_at_open_window,
                date_hint=today,
                default_offset_minutes=5,
            ),
            "ai_predicted_at_open": _normalize_open_window_timestamp(
                ai_meta.get("generated_at_iso") or captured_at_open_window,
                date_hint=today,
                default_offset_minutes=7,
            ),
            "strategy_source": "ensemble_models",
            "ai_source": ai_meta.get("source", "none"),
            "snapshot_type": snapshot_type,
        }
        rows.append(row)

        # Track for expected-vs-actual feedback.
        try:
            tracked = dict(pred)
            tracked.update(
                {
                    "open_price": row["open_price"] or row["current_price"],
                    "current_price": row["current_price"],
                    "strategy_price_at_open": row["strategy_price_at_open"]
                    or row["current_price"],
                    "ai_last_prediction": row["ai_predicted_price"] or 0,
                    "strategy_predicted_at_open": row["strategy_predicted_at_open"],
                    "ai_predicted_at_open": row["ai_predicted_at_open"],
                    "ai_last_prediction_at": ai_meta.get("generated_at_iso")
                    or captured_at_actual,
                    "strategy_direction_at_open": strategy_direction,
                    "ai_direction_last": ai_direction,
                    "strategy_vs_ai_direction": (
                        strategy_direction == ai_direction
                        if row["ai_predicted_price"] is not None
                        else None
                    ),
                }
            )
            PredictionTracker.record_prediction(
                ticker,
                tracked,
                snapshot_type=snapshot_type,
            )
        except Exception as e:
            log.warning("Premarket tracking record failed for %s: %s", ticker, e)

    return _normalize_premarket_snapshot(
        {
            "date": today,
            "captured_at": captured_at_open_window,
            "captured_at_actual": captured_at_actual,
            "snapshot_type": snapshot_type,
            "items": rows,
        }
    )


def _capture_premarket_snapshot_if_due(force: bool = False) -> dict:
    """
    Capture today's strategy snapshot around market open.

    Priority:
      1) market_open snapshot inside 09:15–09:30 IST
      2) near_open_fallback shortly after open if the app starts late
      3) premarket_preview only when explicitly forced before open
    """
    global _premarket_snapshot
    now = datetime.now(IST)
    today = now.strftime("%Y-%m-%d")
    open_start, open_end = _open_window_bounds(now.date())
    fallback_deadline = open_end + timedelta(minutes=PREMARKET_MAX_BUFFER_MINUTES)

    open_types = {
        "market_open",
        "near_open_fallback",
        "market_open_live",
        "market_open_backfilled",
    }

    with _premarket_snapshot_lock:
        if _premarket_snapshot and _premarket_snapshot.get("date") == today:
            cached = _normalize_premarket_snapshot(dict(_premarket_snapshot))
            cached_type = str(cached.get("snapshot_type") or "").lower()
            if not force and (
                (cached.get("items") and cached_type in open_types)
                or (now < open_start and cached_type == "pending_market_open")
            ):
                return cached

        disk_snapshot = _load_premarket_snapshot_from_disk()
        if disk_snapshot and disk_snapshot.get("date") == today:
            normalized = _normalize_premarket_snapshot(dict(disk_snapshot))
            disk_type = str(normalized.get("snapshot_type") or "").lower()
            if not force and (
                (normalized.get("items") and disk_type in open_types)
                or (now < open_start and disk_type == "pending_market_open")
            ):
                _premarket_snapshot = normalized
                return dict(_premarket_snapshot)

    if now < open_start and not force:
        pending = _normalize_premarket_snapshot(
            {
                "date": today,
                "captured_at": None,
                "captured_at_actual": None,
                "snapshot_type": "pending_market_open",
                "schema_version": 2,
                "capture_cutoff": open_start.isoformat(),
                "buffer_minutes": PREMARKET_MAX_BUFFER_MINUTES,
                "captured_within_buffer": None,
                "capture_note": "Waiting for market open snapshot window",
                "items": [],
            }
        )
        with _premarket_snapshot_lock:
            _premarket_snapshot = pending
            _save_premarket_snapshot_to_disk(pending)
            return dict(_premarket_snapshot)

    if now < open_start and force:
        snapshot_type = "premarket_preview"
    elif now <= open_end:
        snapshot_type = "market_open"
    elif now <= fallback_deadline:
        snapshot_type = "near_open_fallback"
    else:
        snapshot_type = "near_open_fallback"
        log.warning(
            "Market-open snapshot captured late. now=%s open_end=%s fallback_deadline=%s",
            now.isoformat(),
            open_end.isoformat(),
            fallback_deadline.isoformat(),
        )

    snapshot = _normalize_premarket_snapshot(
        _build_premarket_snapshot(
            snapshot_type=snapshot_type,
            captured_at=now.isoformat(),
        )
    )
    snapshot["snapshot_type"] = snapshot_type
    snapshot["captured_at_actual"] = (
        snapshot.get("captured_at_actual") or now.isoformat()
    )
    snapshot["captured_at"] = _normalize_open_window_timestamp(
        snapshot.get("captured_at") or snapshot["captured_at_actual"],
        date_hint=today,
        default_offset_minutes=20,
    )
    snapshot["capture_cutoff"] = open_start.isoformat()
    snapshot["buffer_minutes"] = PREMARKET_MAX_BUFFER_MINUTES
    snapshot["captured_within_buffer"] = now <= fallback_deadline
    snapshot["open_window_end"] = open_end.isoformat()
    snapshot["market_open_time"] = open_start.isoformat()
    if not snapshot["captured_within_buffer"]:
        log.warning(
            "Premarket snapshot captured after fallback buffer. now=%s deadline=%s buffer=%sm",
            now.isoformat(),
            fallback_deadline.isoformat(),
            PREMARKET_MAX_BUFFER_MINUTES,
        )

    for row in snapshot.get("items", []):
        if isinstance(row, dict):
            row["snapshot_type"] = snapshot_type

    with _premarket_snapshot_lock:
        _premarket_snapshot = snapshot
        _save_premarket_snapshot_to_disk(snapshot)
        return dict(_premarket_snapshot)


def _normalize_predicted_return_pct(value) -> float:
    """Normalize return to a safe percent range [-50, 50]."""
    try:
        ret_pct = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(ret_pct):
        return 0.0

    if abs(ret_pct) > 500:
        ret_pct = ret_pct / 100.0

    return float(np.clip(ret_pct, -50.0, 50.0))


def _safe_float(value) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return 0.0


def _get_cached_ai_forecast_meta(ticker: str) -> dict:
    now = time.time()
    cached = _groq_forecast_cache.get(ticker, {})
    cached_at = _groq_forecast_cache_time.get(ticker, 0)
    if not cached:
        return {
            "available": False,
            "price": 0.0,
            "source": "none",
            "generated_at_iso": None,
        }
    if (now - cached_at) >= GROQ_FORECAST_TTL:
        return {
            "available": False,
            "price": 0.0,
            "source": "stale",
            "generated_at_iso": cached.get("generated_at_iso"),
        }
    ai_price = _safe_float(cached.get("ai_predicted_price"))
    return {
        "available": ai_price > 0,
        "price": round(ai_price, 2) if ai_price > 0 else 0.0,
        "source": str(cached.get("ai_source", "groq")),
        "generated_at_iso": cached.get("generated_at_iso"),
    }


def _resolve_ai_forecast_price(
    ticker: str,
    *,
    open_price: float,
    strategy_price: float,
    current_price: float,
    allow_generate: bool = False,
) -> dict:
    """
    Resolve AI forecast price.
    - Uses cached Groq forecast first.
    - Optionally generates a fresh Groq forecast (small ticker sets only).
    - Never synthesizes AI values from strategy values.
    """
    cached_meta = _get_cached_ai_forecast_meta(ticker)
    if cached_meta.get("available"):
        return cached_meta

    if not allow_generate or not os.environ.get("GROQ_API_KEY"):
        return cached_meta

    strategy_px = _safe_float(strategy_price)
    open_px = _safe_float(open_price)
    current_px = _safe_float(current_price)
    if strategy_px <= 0:
        strategy_px = open_px if open_px > 0 else current_px
    if current_px <= 0:
        current_px = open_px if open_px > 0 else strategy_px
    if strategy_px <= 0 or current_px <= 0:
        return cached_meta

    stock_name = ticker_names.get(ticker, _clean_name(ticker))
    try:
        sentiment = get_news_sentiment(ticker, stock_name)
        forecast = get_groq_price_forecast(
            ticker=ticker,
            stock_name=stock_name,
            open_price=open_px,
            strategy_predicted_price=strategy_px,
            current_price=current_px,
            sentiment_text=sentiment or "",
        )
        ai_price = _safe_float(forecast.get("ai_predicted_price"))
        if ai_price <= 0:
            return cached_meta

        generated_at_iso = datetime.now(IST).isoformat()
        payload = {
            "ticker": ticker,
            "name": stock_name,
            "strategy_predicted_price": round(strategy_px, 2),
            "current_price": round(current_px, 2),
            "ai_predicted_price": round(ai_price, 2),
            "open_price": round(open_px, 2) if open_px > 0 else None,
            "open_to_ai_predicted_pct": (
                round(((ai_price - open_px) / open_px * 100), 3)
                if open_px > 0
                else None
            ),
            "outlook": forecast.get("outlook", "Neutral"),
            "rationale": forecast.get("rationale", ""),
            "news_sentiment": sentiment,
            "ai_available": True,
            "ai_source": forecast.get("source", "groq"),
            "generated_at_iso": generated_at_iso,
            "generated_at": _format_ist_timestamp(generated_at_iso),
        }
        _groq_forecast_cache[ticker] = payload
        _groq_forecast_cache_time[ticker] = time.time()
        return {
            "available": True,
            "price": round(ai_price, 2),
            "source": payload["ai_source"],
            "generated_at_iso": generated_at_iso,
        }
    except Exception as exc:
        log.debug("Groq AI forecast generation failed for %s: %s", ticker, exc)
        return cached_meta


def _get_prediction_for_ticker(
    ticker: str,
    fallback: dict | None = None,
    allow_live_refresh: bool = False,
    baseline_price: float = 0.0,
) -> dict:
    """Get a robust prediction object from log or predictor cache/live output."""
    global _daily_prediction_baseline, _daily_prediction_baseline_date
    today = datetime.now(IST).strftime("%Y-%m-%d")
    if _daily_prediction_baseline_date != today:
        _daily_prediction_baseline = {}
        _daily_prediction_baseline_date = today

    if ticker in _daily_prediction_baseline:
        return dict(_daily_prediction_baseline[ticker])

    pred = dict(fallback or {})

    needs_refresh = (
        not pred
        or pred.get("predicted_price", 0) <= 0
        or abs(pred.get("predicted_return", 0) or 0) > 50
    )

    if needs_refresh and allow_live_refresh and predictor is not None:
        try:
            refreshed = predictor.predict_single(ticker, use_cache=True)
            if refreshed:
                pred = refreshed
        except Exception:
            pass

    if not pred:
        pred = {
            "predicted_return": 0.0,
            "predicted_price": baseline_price,
            "signal": "HOLD",
            "confidence": 50.0,
            "model_agreement": 50.0,
            "risk_reward": 1.0,
            "model_predictions": {},
        }

    pred_return = _normalize_predicted_return_pct(pred.get("predicted_return", 0))
    pred["predicted_return"] = pred_return
    _daily_prediction_baseline[ticker] = dict(pred)
    return pred


def _build_daily_analysis() -> dict:
    """Build the full daily analysis payload (called by background thread)."""
    _capture_opening_prices()
    current_prices = _get_cached_prices()
    now_ist = datetime.now(IST)
    market = get_market_status()
    market_status = market.get("status", "")
    after_hours_mode = _is_after_hours_ui_mode(now_ist, market_status)
    next_day_mode = _is_next_day_prediction_window(now_ist) or after_hours_mode
    prediction_mode = "next_day_after_close" if next_day_mode else "market_open_window"
    predicted_for_date = (
        _next_trading_day(now_ist.date()).strftime("%Y-%m-%d")
        if next_day_mode
        else now_ist.strftime("%Y-%m-%d")
    )
    prediction_generated_at = now_ist.isoformat()
    price_label = "Close Price" if after_hours_mode else "Current Price"

    # Get all predictions (from cache)
    today_str = now_ist.strftime("%Y-%m-%d")
    log_file = PREDICTION_LOG_DIR / f"{today_str}.json"
    predictions = {}
    if log_file.exists():
        try:
            with open(log_file) as f:
                predictions = json.load(f)
        except (json.JSONDecodeError, OSError):
            predictions = {}

    stocks = []
    prediction_count = 0
    refresh_budget = 0
    for ticker in all_tickers:
        if ticker.startswith("^") or ticker in ("USDINR=X", "GC=F", "CL=F"):
            continue

        curr = current_prices.get(ticker, {})
        opening = _opening_prices.get(ticker, {})

        current_price = curr.get("price", 0)
        open_price = opening.get("open", curr.get("open", 0))
        prev_close = opening.get("prev_close", curr.get("prev_close", 0))
        if open_price <= 0:
            open_price = current_price
        pred = _get_prediction_for_ticker(
            ticker,
            predictions.get(ticker, {}),
            allow_live_refresh=refresh_budget > 0,
            baseline_price=open_price if open_price > 0 else current_price,
        )
        if refresh_budget > 0 and ticker not in predictions:
            refresh_budget -= 1
        predicted_return = _normalize_predicted_return_pct(
            pred.get("predicted_return", 0)
        )
        signal = pred.get("signal", "HOLD")
        confidence = float(pred.get("confidence", 50) or 50)
        premarket_row = _get_premarket_row_for_ticker(
            ticker,
            today_str,
            allowed_snapshot_types={"market_open", "near_open_fallback"},
        )
        strategy_price_at_open = float(
            premarket_row.get(
                "strategy_price_at_open",
                pred.get("strategy_price_at_open", 0),
            )
            or 0
        )
        if strategy_price_at_open <= 0 and open_price > 0:
            strategy_price_at_open = round(
                open_price * (1 + predicted_return / 100.0), 2
            )
        if strategy_price_at_open <= 0:
            strategy_price_at_open = float(current_price or 0)

        ai_meta_open = _resolve_ai_forecast_price(
            ticker,
            open_price=float(open_price or 0),
            strategy_price=float(strategy_price_at_open or 0),
            current_price=float(current_price or 0),
            allow_generate=False,
        )
        ai_price_at_open = float(
            premarket_row.get("ai_predicted_price") or ai_meta_open.get("price") or 0
        )
        ai_source_open = str(
            premarket_row.get("ai_source") or ai_meta_open.get("source", "none")
        )

        strategy_predicted_at_open = _normalize_open_window_timestamp(
            premarket_row.get("strategy_predicted_at_open")
            or premarket_row.get("captured_at")
            or pred.get("strategy_predicted_at_open")
            or pred.get("timestamp"),
            date_hint=today_str,
            default_offset_minutes=5,
        )
        ai_predicted_at_open = _normalize_open_window_timestamp(
            premarket_row.get("ai_predicted_at_open")
            or premarket_row.get("captured_at")
            or pred.get("ai_predicted_at_open")
            or pred.get("timestamp"),
            date_hint=today_str,
            default_offset_minutes=7,
        )

        if next_day_mode:
            reference_price = float(current_price or open_price or 0)
            strategy_predicted_price = (
                round(reference_price * (1 + predicted_return / 100.0), 2)
                if reference_price > 0
                else strategy_price_at_open
            )
            ai_meta_now = _resolve_ai_forecast_price(
                ticker,
                open_price=float(open_price or 0),
                strategy_price=float(strategy_predicted_price or 0),
                current_price=float(current_price or 0),
                allow_generate=False,
            )
            ai_predicted_price = float(ai_meta_now.get("price") or 0)
            strategy_predicted_at = prediction_generated_at
            ai_predicted_at = ai_meta_now.get("generated_at_iso")
            ai_source = str(ai_meta_now.get("source", ai_source_open))
            prediction_context = "next_day"
        else:
            reference_price = float(open_price or current_price or 0)
            strategy_predicted_price = float(strategy_price_at_open or 0)
            ai_predicted_price = float(ai_price_at_open or 0)
            strategy_predicted_at = strategy_predicted_at_open
            ai_predicted_at = ai_meta_open.get("generated_at_iso") or (
                ai_predicted_at_open if ai_predicted_price > 0 else None
            )
            ai_source = ai_source_open
            prediction_context = "market_open"

        if current_price <= 0 or open_price <= 0:
            continue

        prediction_count += 1

        # Calculate percentage changes
        open_to_current_pct = (
            ((current_price - open_price) / open_price * 100) if open_price > 0 else 0
        )
        open_to_predicted_pct = (
            ((strategy_predicted_price - open_price) / open_price * 100)
            if open_price > 0 and strategy_predicted_price > 0
            else 0
        )
        open_to_ai_predicted_pct = (
            ((ai_predicted_price - open_price) / open_price * 100)
            if open_price > 0 and ai_predicted_price > 0
            else None
        )
        reference_to_strategy_pct = (
            ((strategy_predicted_price - reference_price) / reference_price * 100)
            if reference_price > 0 and strategy_predicted_price > 0
            else 0
        )
        reference_to_ai_pct = (
            ((ai_predicted_price - reference_price) / reference_price * 100)
            if reference_price > 0 and ai_predicted_price > 0
            else None
        )
        close_to_strategy_pct = (
            ((strategy_predicted_price - current_price) / current_price * 100)
            if current_price > 0 and strategy_predicted_price > 0
            else 0
        )
        close_to_ai_pct = (
            ((ai_predicted_price - current_price) / current_price * 100)
            if current_price > 0 and ai_predicted_price > 0
            else None
        )
        prev_to_current_pct = (
            ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
        )

        # Composite score for ranking (higher = better opportunity)
        # Factors: predicted return (40%), confidence (25%), model agreement (20%), R:R (15%)
        model_agreement = float(pred.get("model_agreement", 50) or 50)
        risk_reward = pred.get("risk_reward", 1.0) if pred.get("risk_reward") else 1.0

        score = (
            abs(predicted_return) * 4.0
            + confidence * 0.25
            + model_agreement * 0.20
            + min(risk_reward, 5) * 3.0
        )

        model_preds = pred.get("model_predictions", {})

        stocks.append(
            {
                "ticker": ticker,
                "name": ticker_names.get(ticker, _clean_name(ticker)),
                "open_price": round(open_price, 2),
                "predicted_price": round(strategy_predicted_price, 2),
                "strategy_predicted_price": round(strategy_predicted_price, 2),
                "ai_predicted_price": (
                    round(ai_predicted_price, 2) if ai_predicted_price > 0 else None
                ),
                "strategy_price_at_open": round(strategy_price_at_open, 2),
                "ai_price_at_open": (
                    round(ai_price_at_open, 2) if ai_price_at_open > 0 else None
                ),
                "current_price": round(current_price, 2),
                "prev_close": round(prev_close, 2),
                "display_price_label": price_label,
                "open_to_current_pct": round(open_to_current_pct, 3),
                "open_to_predicted_pct": round(open_to_predicted_pct, 3),
                "open_to_ai_predicted_pct": (
                    round(open_to_ai_predicted_pct, 3)
                    if open_to_ai_predicted_pct is not None
                    else None
                ),
                "reference_price": round(reference_price, 2),
                "reference_to_strategy_pct": round(reference_to_strategy_pct, 3),
                "reference_to_ai_pct": (
                    round(reference_to_ai_pct, 3)
                    if reference_to_ai_pct is not None
                    else None
                ),
                "close_to_strategy_pct": round(close_to_strategy_pct, 3),
                "close_to_ai_pct": (
                    round(close_to_ai_pct, 3) if close_to_ai_pct is not None else None
                ),
                "prediction_mode": prediction_mode,
                "prediction_context": prediction_context,
                "predicted_for_date": predicted_for_date,
                "after_hours_mode": after_hours_mode,
                "snapshot_type": premarket_row.get("snapshot_type", "market_open"),
                "strategy_predicted_at": strategy_predicted_at,
                "ai_predicted_at": ai_predicted_at,
                "strategy_predicted_at_open": strategy_predicted_at_open,
                "ai_predicted_at_open": ai_predicted_at_open,
                "strategy_source": "ensemble_models",
                "ai_source": ai_source,
                "prev_to_current_pct": round(prev_to_current_pct, 3),
                "predicted_return": round(predicted_return, 3),
                "signal": signal,
                "confidence": round(confidence, 1),
                "model_agreement": round(model_agreement, 1),
                "risk_reward": round(risk_reward, 1) if risk_reward else None,
                "volume": curr.get("volume", 0),
                "high": curr.get("high", 0),
                "low": curr.get("low", 0),
                "change": curr.get("change", 0),
                "change_pct": curr.get("change_pct", 0),
                "composite_score": round(score, 2),
                "model_predictions": model_preds,
            }
        )

    stocks.sort(key=lambda x: x["composite_score"], reverse=True)
    top_10 = stocks[:10]

    gainers = [s for s in stocks if s["change_pct"] > 0]
    losers = [s for s in stocks if s["change_pct"] < 0]

    return {
        "date": today_str,
        "market_status": market_status,
        "after_hours_mode": after_hours_mode,
        "prediction_mode": prediction_mode,
        "predicted_for_date": predicted_for_date,
        "prediction_generated_at": prediction_generated_at,
        "total_stocks": len(stocks),
        "prediction_coverage": {
            "with_predictions": prediction_count,
            "coverage_pct": round((prediction_count / max(len(stocks), 1)) * 100, 1),
        },
        "market_summary": {
            "gainers": len(gainers),
            "losers": len(losers),
            "unchanged": len(stocks) - len(gainers) - len(losers),
            "avg_change_pct": round(
                float(np.mean([s["change_pct"] for s in stocks])) if stocks else 0, 3
            ),
        },
        "top_10": top_10,
        "all_stocks": stocks,
        "cached_at": time.time(),
    }


def _daily_analysis_background_loop():
    """Background thread: recompute daily analysis every DAILY_ANALYSIS_TTL seconds."""
    global _daily_analysis_cache, _daily_analysis_cache_time
    while True:
        if not models_loaded or not all_tickers:
            time.sleep(5)
            continue
        try:
            _capture_premarket_snapshot_if_due(force=False)
            result = _build_daily_analysis()
            with _daily_analysis_lock:
                _daily_analysis_cache = result
                _daily_analysis_cache_time = time.time()
            log.info(
                f"Daily analysis cache refreshed — {result['total_stocks']} stocks"
            )
        except Exception as e:
            log.error(f"Daily analysis background error: {e}")
        time.sleep(DAILY_ANALYSIS_TTL)


@app.route("/api/daily-analysis")
def api_daily_analysis():
    """
    Comprehensive daily analysis — served from background-computed cache.
    Never blocks on live price fetching; returns cached data instantly.
    """
    if not models_loaded or predictor is None:
        return jsonify({"error": "Models still loading..."}), 503

    with _daily_analysis_lock:
        if _daily_analysis_cache:
            return jsonify(_daily_analysis_cache)

    # Cache not ready yet — return a loading indicator
    return (
        jsonify(
            {"error": "Analysis is being computed, please retry in ~15 seconds..."}
        ),
        202,
    )


@app.route("/api/price-tracker/<ticker>")
def api_price_tracker(ticker: str):
    """
    Lightweight endpoint for individual stock price tracking.
    Uses cached prices first, falls back to single-ticker fetch.
    """
    if not models_loaded or predictor is None:
        return jsonify({"error": "Models still loading..."}), 503

    # Try cached prices first (from background analysis), then live fetch
    curr = _price_cache.get(ticker, {})
    if not curr:
        prices = _get_live_prices_batch([ticker])
        curr = prices.get(ticker, {})
    if not curr:
        # Last resort: pull from daily analysis cache
        with _daily_analysis_lock:
            for s in _daily_analysis_cache.get("all_stocks", []):
                if s["ticker"] == ticker:
                    return jsonify(s)
        return jsonify({"error": f"Price not available for {ticker}"}), 404

    # Get opening price from cache
    opening = _opening_prices.get(ticker, {})
    open_price = opening.get("open", curr.get("open", 0))
    prev_close = opening.get("prev_close", curr.get("prev_close", 0))
    open_price_captured_at = opening.get("captured_at")

    # Get prediction from today's log
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    log_file = PREDICTION_LOG_DIR / f"{today_str}.json"
    pred = {}
    if log_file.exists():
        try:
            with open(log_file) as f:
                all_preds = json.load(f)
                pred = all_preds.get(ticker, {})
        except (json.JSONDecodeError, OSError):
            pass

    current_price = float(curr.get("price", 0) or 0)
    predicted_return = _normalize_predicted_return_pct(pred.get("predicted_return", 0))
    strategy_predicted_price = (
        round(open_price * (1 + predicted_return / 100.0), 2)
        if open_price > 0
        else pred.get("predicted_price", 0)
    )
    premarket_row = _get_premarket_row_for_ticker(
        ticker,
        today_str,
        allowed_snapshot_types={"market_open", "near_open_fallback"},
    )
    strategy_price_at_open = float(
        premarket_row.get(
            "strategy_price_at_open",
            pred.get("strategy_price_at_open", strategy_predicted_price),
        )
        or 0
    )
    if strategy_price_at_open <= 0:
        strategy_price_at_open = float(strategy_predicted_price or 0)
    ai_meta_open = _resolve_ai_forecast_price(
        ticker,
        open_price=float(open_price or 0),
        strategy_price=float(strategy_price_at_open or 0),
        current_price=float(current_price or 0),
        allow_generate=False,
    )
    ai_predicted_price = float(ai_meta_open.get("price") or 0)
    ai_available = ai_predicted_price > 0
    ai_open_price = float(
        premarket_row.get("ai_predicted_price") or ai_predicted_price or 0
    )
    signal = pred.get("signal", "")
    confidence = pred.get("confidence", 0)
    market = get_market_status()
    market_status = market.get("status", "")
    now_ist = datetime.now(IST)
    after_hours_mode = _is_after_hours_ui_mode(now_ist, market_status)
    next_day_mode = _is_next_day_prediction_window(now_ist) or after_hours_mode
    prediction_mode = "next_day_after_close" if next_day_mode else "market_open_window"
    predicted_for_date = (
        _next_trading_day(now_ist.date()).strftime("%Y-%m-%d")
        if next_day_mode
        else now_ist.strftime("%Y-%m-%d")
    )
    is_market_closed = after_hours_mode
    display_price_label = "Close Price" if is_market_closed else "Current Price"
    strategy_predicted_at_open = _normalize_open_window_timestamp(
        premarket_row.get("strategy_predicted_at_open")
        or premarket_row.get("captured_at")
        or pred.get("strategy_predicted_at_open")
        or pred.get("timestamp"),
        date_hint=today_str,
        default_offset_minutes=5,
    )
    ai_predicted_at_open = _normalize_open_window_timestamp(
        premarket_row.get("ai_predicted_at_open")
        or premarket_row.get("captured_at")
        or pred.get("ai_predicted_at_open")
        or pred.get("timestamp"),
        date_hint=today_str,
        default_offset_minutes=7,
    )
    current_strategy_predicted_at = (
        pred.get("timestamp")
        or pred.get("ai_last_prediction_at")
        or datetime.now(IST).isoformat()
    )
    current_ai_predicted_at = (
        ai_meta_open.get("generated_at_iso")
        or pred.get("ai_last_prediction_at")
        or (pred.get("timestamp") if ai_open_price > 0 else None)
        or None
    )

    if open_price <= 0:
        open_price = current_price
    if strategy_price_at_open <= 0:
        strategy_price_at_open = strategy_predicted_price or open_price
    if ai_open_price <= 0:
        ai_open_price = ai_predicted_price if ai_predicted_price > 0 else 0.0

    if next_day_mode:
        reference_price = current_price if current_price > 0 else open_price
        next_day_strategy_price = (
            round(reference_price * (1 + predicted_return / 100.0), 2)
            if reference_price > 0
            else strategy_price_at_open
        )
        ai_meta_now = _resolve_ai_forecast_price(
            ticker,
            open_price=float(open_price or 0),
            strategy_price=float(next_day_strategy_price or 0),
            current_price=float(current_price or 0),
            allow_generate=False,
        )
        next_day_ai_price = float(ai_meta_now.get("price") or 0)
        next_day_predicted_at = now_ist.isoformat()
        current_strategy_predicted_at = next_day_predicted_at
        current_ai_predicted_at = ai_meta_now.get("generated_at_iso") or None
        ai_source = str(ai_meta_now.get("source", ai_meta_open.get("source", "none")))
    else:
        reference_price = open_price
        next_day_strategy_price = strategy_price_at_open
        next_day_ai_price = ai_open_price
        next_day_predicted_at = strategy_predicted_at_open
        ai_source = str(
            premarket_row.get("ai_source") or ai_meta_open.get("source", "none")
        )

    open_to_current_pct = (
        ((current_price - open_price) / open_price * 100) if open_price > 0 else 0
    )
    open_to_predicted_pct = (
        ((strategy_price_at_open - open_price) / open_price * 100)
        if open_price > 0 and strategy_price_at_open > 0
        else 0
    )

    open_to_ai_predicted_pct = (
        ((ai_open_price - open_price) / open_price * 100)
        if open_price > 0 and ai_open_price > 0
        else None
    )
    close_to_strategy_pct = (
        ((next_day_strategy_price - current_price) / current_price * 100)
        if current_price > 0 and next_day_strategy_price > 0
        else None
    )
    close_to_ai_pct = (
        ((next_day_ai_price - current_price) / current_price * 100)
        if current_price > 0 and next_day_ai_price > 0
        else None
    )

    snapshot_now_iso = datetime.now(IST).isoformat()
    return jsonify(
        {
            "ticker": ticker,
            "name": ticker_names.get(ticker, _clean_name(ticker)),
            "open_price": round(open_price, 2),
            "open_price_captured_at": open_price_captured_at,
            "open_price_captured_at_display": _format_ist_timestamp(
                open_price_captured_at
            ),
            "predicted_price": round(strategy_price_at_open, 2),
            "strategy_predicted_price": round(strategy_price_at_open, 2),
            "ai_predicted_price": (
                round(ai_open_price, 2) if ai_open_price > 0 else None
            ),
            "current_strategy_predicted_price": round(next_day_strategy_price, 2),
            "current_ai_predicted_price": (
                round(next_day_ai_price, 2) if next_day_ai_price > 0 else None
            ),
            "current_price": round(current_price, 2),
            "close_price": round(current_price, 2) if is_market_closed else None,
            "display_price_label": display_price_label,
            "market_status": market_status,
            "prediction_mode": prediction_mode,
            "after_hours_mode": after_hours_mode,
            "snapshot_type": premarket_row.get("snapshot_type", "market_open"),
            "predicted_for_date": predicted_for_date,
            "prediction_reference_price": (
                round(reference_price, 2) if reference_price > 0 else None
            ),
            "next_day_strategy_predicted_price": (
                round(next_day_strategy_price, 2)
                if next_day_strategy_price > 0
                else None
            ),
            "next_day_ai_predicted_price": (
                round(next_day_ai_price, 2) if next_day_ai_price > 0 else None
            ),
            "next_day_predicted_at": next_day_predicted_at,
            "next_day_predicted_at_display": _format_ist_timestamp(
                next_day_predicted_at
            ),
            "strategy_predicted_at_open": strategy_predicted_at_open,
            "strategy_predicted_at_open_display": _format_ist_timestamp(
                strategy_predicted_at_open
            ),
            "ai_predicted_at_open": ai_predicted_at_open,
            "ai_predicted_at_open_display": _format_ist_timestamp(ai_predicted_at_open),
            "current_strategy_predicted_at": current_strategy_predicted_at,
            "current_strategy_predicted_at_display": _format_ist_timestamp(
                current_strategy_predicted_at
            ),
            "current_ai_predicted_at": current_ai_predicted_at,
            "current_ai_predicted_at_display": _format_ist_timestamp(
                current_ai_predicted_at
            ),
            "current_snapshot_at": snapshot_now_iso,
            "current_snapshot_at_display": _format_ist_timestamp(snapshot_now_iso),
            "prev_close": round(prev_close, 2),
            "open_to_current_pct": round(open_to_current_pct, 3),
            "open_to_predicted_pct": round(open_to_predicted_pct, 3),
            "open_to_ai_predicted_pct": (
                round(open_to_ai_predicted_pct, 3)
                if open_to_ai_predicted_pct is not None
                else None
            ),
            "close_to_strategy_pct": (
                round(close_to_strategy_pct, 3)
                if close_to_strategy_pct is not None
                else None
            ),
            "close_to_ai_pct": (
                round(close_to_ai_pct, 3) if close_to_ai_pct is not None else None
            ),
            "predicted_return": round(predicted_return, 3),
            "signal": signal,
            "confidence": round(confidence, 1),
            "ai_forecast_available": bool(ai_open_price > 0 or next_day_ai_price > 0),
            "ai_forecast_source": (
                ai_source if (ai_open_price > 0 or next_day_ai_price > 0) else "none"
            ),
            "strategy_source": "ensemble_models",
            "volume": curr.get("volume", 0),
            "high": curr.get("high", 0),
            "low": curr.get("low", 0),
            "open": curr.get("open", 0),
            "change": curr.get("change", 0),
            "change_pct": curr.get("change_pct", 0),
        }
    )


@app.route("/api/news/<ticker>")
def api_news(ticker: str):
    """Get real-world news and sentiment for a specific stock."""
    try:
        stock_name = ticker_names.get(ticker, _clean_name(ticker))
        sentiment = get_news_sentiment(ticker, stock_name)
        return jsonify(
            {
                "ticker": ticker,
                "name": stock_name,
                "sentiment": sentiment,
                "generated_at": datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST"),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/groq-price-forecast/<ticker>")
def api_groq_price_forecast(ticker: str):
    """Return Groq-based AI price forecast using strategy + current context."""
    if not models_loaded or predictor is None:
        return jsonify({"error": "Models still loading..."}), 503

    now = time.time()
    if (
        ticker in _groq_forecast_cache
        and (now - _groq_forecast_cache_time.get(ticker, 0)) < GROQ_FORECAST_TTL
    ):
        return jsonify(_groq_forecast_cache[ticker])

    try:
        tracker = api_price_tracker(ticker)
        if isinstance(tracker, tuple):
            tracker_payload = tracker[0].get_json()
            status = tracker[1]
            if status != 200:
                return jsonify(tracker_payload), status
        else:
            tracker_payload = tracker.get_json()

        if tracker_payload.get("error"):
            return jsonify(tracker_payload), 404

        stock_name = ticker_names.get(ticker, _clean_name(ticker))
        sentiment = get_news_sentiment(ticker, stock_name)
        forecast = get_groq_price_forecast(
            ticker=ticker,
            stock_name=stock_name,
            open_price=float(tracker_payload.get("open_price") or 0),
            strategy_predicted_price=float(
                tracker_payload.get("strategy_predicted_price")
                or tracker_payload.get("predicted_price")
                or 0
            ),
            current_price=float(tracker_payload.get("current_price") or 0),
            sentiment_text=sentiment or "",
        )
        ai_raw = forecast.get("ai_predicted_price")
        try:
            ai_price = float(ai_raw) if ai_raw not in (None, "") else 0.0
        except Exception:
            ai_price = 0.0
        ai_available = ai_price > 0
        open_price = float(tracker_payload.get("open_price") or 0)
        open_to_ai_pct = (
            ((ai_price - open_price) / open_price * 100)
            if open_price > 0 and ai_available
            else None
        )

        generated_at_iso = datetime.now(IST).isoformat()
        payload = {
            "ticker": ticker,
            "name": stock_name,
            "strategy_predicted_price": tracker_payload.get("strategy_predicted_price"),
            "current_price": tracker_payload.get("current_price"),
            "ai_predicted_price": round(ai_price, 2) if ai_available else None,
            "open_price": tracker_payload.get("open_price"),
            "open_to_ai_predicted_pct": (
                round(open_to_ai_pct, 3) if open_to_ai_pct is not None else None
            ),
            "outlook": forecast.get("outlook", "Neutral"),
            "rationale": forecast.get("rationale", ""),
            "news_sentiment": sentiment,
            "ai_available": ai_available,
            "ai_source": forecast.get("source", "groq"),
            "generated_at_iso": generated_at_iso,
            "generated_at": _format_ist_timestamp(generated_at_iso),
        }
        _groq_forecast_cache[ticker] = payload
        _groq_forecast_cache_time[ticker] = now
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/explain-risk-term")
def api_explain_risk_term():
    """Get Groq explanation for a risk metric/term."""
    term = request.args.get("term", "").strip()
    context = request.args.get("context", "").strip()
    if not term:
        return jsonify({"error": "term is required"}), 400
    try:
        explanation = explain_risk_term(term, context)
        return jsonify({"term": term, "explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/delisted-tickers")
def api_delisted_tickers():
    """
    Return tracked unavailable tickers.
    Query params:
      - min_hits (default 1)
      - status (optional: watchlist|delisted_candidate|recovered)
    """
    try:
        min_hits = int(request.args.get("min_hits", 1))
    except Exception:
        min_hits = 1
    status_filter = request.args.get("status", "").strip().lower()

    _prune_delisted_registry()
    with _delisted_lock:
        registry = _load_delisted_registry_unlocked()

    rows = []
    for row in registry.values():
        hits = int(row.get("hit_count", 0) or 0)
        status = str(row.get("status", "")).lower()
        if hits < min_hits:
            continue
        if status_filter and status != status_filter:
            continue
        rows.append(row)

    rows.sort(
        key=lambda r: (int(r.get("hit_count", 0) or 0), r.get("last_seen", "")),
        reverse=True,
    )
    return jsonify(
        {"count": len(rows), "tickers": rows, "file": str(DELISTED_TICKERS_FILE)}
    )


@app.route("/api/delisted-tickers/export.csv")
def api_delisted_tickers_export_csv():
    """Export unavailable/delisted tracking sheet as CSV."""
    _prune_delisted_registry()
    with _delisted_lock:
        registry = _load_delisted_registry_unlocked()

    fieldnames = [
        "ticker",
        "first_seen",
        "last_seen",
        "hit_count",
        "last_reason",
        "last_source",
        "status",
    ]
    out = StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for ticker in sorted(registry):
        writer.writerow(registry[ticker])

    resp = make_response(out.getvalue())
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=delisted_tickers.csv"
    return resp


@app.route("/api/portfolio", methods=["GET", "POST", "DELETE"])
def api_portfolio():
    """Portfolio positions derived from trade ledger; backward compatible POST adds BUY."""
    _sanitize_portfolio_storage()

    if request.method == "GET":
        summary = _portfolio_summary_from_trades(_read_portfolio_trades())
        entries = summary["positions"]
        ticker = request.args.get("ticker", "").strip()
        if ticker:
            entries = [e for e in entries if e.get("ticker") == ticker]
        return jsonify({"holdings": entries, "count": len(entries)})

    if request.method == "DELETE":
        ticker = request.args.get("ticker", "").strip()
        if not ticker:
            _write_portfolio([])
            _write_portfolio_trades([])
            return jsonify({"ok": True, "holdings": [], "count": 0})
        trades = [e for e in _read_portfolio_trades() if e.get("ticker") != ticker]
        _write_portfolio_trades(trades)
        summary = _portfolio_summary_from_trades(trades)
        return jsonify(
            {
                "ok": True,
                "holdings": summary["positions"],
                "count": summary["position_count"],
            }
        )

    payload = request.get_json(silent=True) or {}
    ticker = str(payload.get("ticker", "")).strip().upper()
    qty = payload.get("quantity")
    entry_price = payload.get("entry_price")
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    if not _is_tradeable_ticker(ticker):
        return jsonify({"error": f"{ticker} is not in configured ticker universe"}), 400
    try:
        qty = float(qty)
        entry_price = float(entry_price)
    except (TypeError, ValueError):
        return jsonify({"error": "quantity and entry_price must be numeric"}), 400
    if qty <= 0 or entry_price <= 0:
        return jsonify({"error": "quantity and entry_price must be > 0"}), 400

    # Backward-compatible path: this endpoint creates a BUY trade.
    trades = _read_portfolio_trades()
    trades.append(
        _new_trade_entry(ticker=ticker, side="BUY", qty=qty, price=entry_price)
    )
    _write_portfolio_trades(trades)
    summary = _portfolio_summary_from_trades(trades)
    _write_portfolio(
        [
            {
                "ticker": row["ticker"],
                "name": row["name"],
                "quantity": row["quantity"],
                "entry_price": row["avg_buy_price"],
                "updated_at": datetime.now(IST).isoformat(),
            }
            for row in summary["positions"]
        ]
    )
    return jsonify(
        {
            "ok": True,
            "holdings": summary["positions"],
            "count": summary["position_count"],
        }
    )


@app.route("/api/portfolio/trade", methods=["POST"])
def api_portfolio_trade():
    """Record BUY/SELL trade and return updated P&L summary."""
    _sanitize_portfolio_storage()
    payload = request.get_json(silent=True) or {}
    ticker = str(payload.get("ticker", "")).strip().upper()
    side = str(payload.get("side", "BUY")).strip().upper()
    qty = payload.get("quantity")
    price = payload.get("price")
    if side not in {"BUY", "SELL"}:
        return jsonify({"error": "side must be BUY or SELL"}), 400
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    if not _is_tradeable_ticker(ticker):
        return jsonify({"error": f"{ticker} is not in configured ticker universe"}), 400
    try:
        qty = float(qty)
        price = float(price)
    except (TypeError, ValueError):
        return jsonify({"error": "quantity and price must be numeric"}), 400
    if qty <= 0 or price <= 0:
        return jsonify({"error": "quantity and price must be > 0"}), 400

    trades = _read_portfolio_trades()
    summary_before = _portfolio_summary_from_trades(trades, include_live_prices=False)
    pos_before = {p["ticker"]: p for p in summary_before["positions"]}
    if side == "SELL":
        open_qty = float(pos_before.get(ticker, {}).get("quantity", 0) or 0)
        if open_qty + 1e-9 < qty:
            return (
                jsonify({"error": f"Cannot SELL {qty}; open quantity is {open_qty}"}),
                400,
            )

    trades.append(_new_trade_entry(ticker=ticker, side=side, qty=qty, price=price))
    _write_portfolio_trades(trades)
    summary = _portfolio_summary_from_trades(trades)
    _write_portfolio(
        [
            {
                "ticker": row["ticker"],
                "name": row["name"],
                "quantity": row["quantity"],
                "entry_price": row["avg_buy_price"],
                "updated_at": datetime.now(IST).isoformat(),
            }
            for row in summary["positions"]
        ]
    )
    return jsonify({"ok": True, "summary": summary})


@app.route("/api/portfolio/trade/<trade_id>", methods=["DELETE", "PATCH"])
def api_portfolio_trade_edit(trade_id: str):
    """Edit or delete a trade row by ID."""
    _sanitize_portfolio_storage()
    trades = _read_portfolio_trades()
    idx = next(
        (i for i, tr in enumerate(trades) if str(tr.get("id", "")) == trade_id), None
    )
    if idx is None:
        return jsonify({"error": f"trade_id {trade_id} not found"}), 404

    if request.method == "DELETE":
        trades.pop(idx)
        _write_portfolio_trades(trades)
        summary = _portfolio_summary_from_trades(trades)
        _write_portfolio(
            [
                {
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "quantity": row["quantity"],
                    "entry_price": row["avg_buy_price"],
                    "updated_at": datetime.now(IST).isoformat(),
                }
                for row in summary["positions"]
            ]
        )
        return jsonify({"ok": True, "summary": summary, "trade_count": len(trades)})

    payload = request.get_json(silent=True) or {}
    old = dict(trades[idx])

    ticker = str(payload.get("ticker", old.get("ticker", ""))).strip().upper()
    side = str(payload.get("side", old.get("side", "BUY"))).strip().upper()
    try:
        qty = float(payload.get("quantity", old.get("quantity", 0)) or 0)
        price = float(payload.get("price", old.get("price", 0)) or 0)
    except Exception:
        return jsonify({"error": "quantity and price must be numeric"}), 400

    if side not in {"BUY", "SELL"}:
        return jsonify({"error": "side must be BUY or SELL"}), 400
    if qty <= 0 or price <= 0:
        return jsonify({"error": "quantity and price must be > 0"}), 400
    if not _is_tradeable_ticker(ticker):
        return jsonify({"error": f"{ticker} is not in configured ticker universe"}), 400

    trades[idx].update(
        {
            "ticker": ticker,
            "name": ticker_names.get(ticker, _clean_name(ticker)),
            "side": side,
            "quantity": round(qty, 4),
            "price": round(price, 2),
            "edited_at": datetime.now(IST).isoformat(),
        }
    )
    valid, err = _validate_trade_sequence(trades)
    if not valid:
        trades[idx] = old
        return jsonify({"error": err}), 400

    _write_portfolio_trades(trades)
    summary = _portfolio_summary_from_trades(trades)
    _write_portfolio(
        [
            {
                "ticker": row["ticker"],
                "name": row["name"],
                "quantity": row["quantity"],
                "entry_price": row["avg_buy_price"],
                "updated_at": datetime.now(IST).isoformat(),
            }
            for row in summary["positions"]
        ]
    )
    return jsonify({"ok": True, "summary": summary, "trade_count": len(trades)})


@app.route("/api/portfolio/clean", methods=["POST"])
def api_portfolio_clean():
    """Force cleanup of invalid/dummy portfolio rows."""
    cleaned = _sanitize_portfolio_storage()
    return jsonify({"ok": True, **cleaned})


@app.route("/api/portfolio/summary")
def api_portfolio_summary():
    """Return portfolio summary with optional Groq suggestion."""
    _sanitize_portfolio_storage()
    suggest = request.args.get("suggest", "").lower() in ("1", "true", "yes")
    trades = _read_portfolio_trades()
    summary = _portfolio_summary_from_trades(trades)
    if suggest:
        try:
            summary["strategy_suggestion"] = portfolio_profit_suggestion(summary)
        except Exception as e:
            summary["strategy_suggestion"] = f"Suggestion unavailable: {e}"
    summary["trade_count"] = len(trades)
    return jsonify(summary)


@app.route("/api/portfolio/trades")
def api_portfolio_trades():
    """Return full trade history."""
    _sanitize_portfolio_storage()
    ticker = request.args.get("ticker", "").strip().upper()
    limit = request.args.get("limit")
    trades = _read_portfolio_trades()
    if _ensure_trade_ids(trades):
        _write_portfolio_trades(trades)
    if ticker:
        trades = [t for t in trades if str(t.get("ticker", "")).upper() == ticker]
    trades = sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True)
    if limit:
        try:
            n = max(1, int(limit))
            trades = trades[:n]
        except ValueError:
            pass
    return jsonify({"trades": trades, "count": len(trades)})


@app.route("/api/portfolio/export.csv")
def api_portfolio_export_csv():
    """Export trade ledger as CSV."""
    _sanitize_portfolio_storage()
    trades = sorted(_read_portfolio_trades(), key=lambda x: x.get("timestamp", ""))
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(
        ["id", "timestamp", "ticker", "name", "side", "quantity", "price", "notional"]
    )
    for tr in trades:
        qty = float(tr.get("quantity", 0) or 0)
        price = float(tr.get("price", 0) or 0)
        writer.writerow(
            [
                tr.get("id", ""),
                tr.get("timestamp", ""),
                tr.get("ticker", ""),
                tr.get("name", ""),
                tr.get("side", ""),
                round(qty, 4),
                round(price, 2),
                round(qty * price, 2),
            ]
        )
    resp = make_response(out.getvalue())
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=portfolio_trades.csv"
    return resp


@app.route("/api/groq-ticker-suggestions")
def api_groq_ticker_suggestions():
    """Suggest shortlist of tickers and reasoning from current model outputs."""
    if not models_loaded or predictor is None:
        return jsonify({"error": "Models still loading..."}), 503
    try:
        n = int(request.args.get("n", 8))
        n = max(3, min(n, 20))
    except Exception:
        n = 8

    try:
        groups = predictor.predict_top_picks_grouped(
            sectors=[
                "large_cap",
                "banking",
                "mid_cap",
                "high_volatility",
                "commodities",
            ],
            top_n=n,
        )
        candidates: list[dict] = []
        for bucket in ("top_buy", "top_hold", "top_sell"):
            for p in groups.get(bucket, []):
                if float(p.get("current_price", 0) or 0) <= 0:
                    continue
                candidates.append(
                    {
                        "ticker": p.get("ticker"),
                        "name": ticker_names.get(
                            p.get("ticker", ""), _clean_name(p.get("ticker", ""))
                        ),
                        "signal": p.get("signal", "HOLD"),
                        "predicted_return": round(
                            float(p.get("predicted_return", 0) or 0), 3
                        ),
                        "confidence": round(float(p.get("confidence", 0) or 0), 1),
                        "model_agreement": round(
                            float(p.get("model_agreement", 0) or 0), 1
                        ),
                        "current_price": round(
                            float(p.get("current_price", 0) or 0), 2
                        ),
                    }
                )
        # Keep unique symbols and cap length.
        seen: set[str] = set()
        unique_candidates = []
        for c in candidates:
            t = c.get("ticker")
            if not t or t in seen:
                continue
            seen.add(t)
            unique_candidates.append(c)
            if len(unique_candidates) >= n:
                break

        recommendation = suggest_ticker_shortlist(unique_candidates)
        return jsonify(
            {
                "count": len(unique_candidates),
                "candidates": unique_candidates,
                "recommendation": recommendation,
                "generated_at": datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST"),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/groq-trade-review", methods=["POST"])
def api_groq_trade_review():
    """Review user-selected ticker + entry plan with model/news context."""
    if not models_loaded or predictor is None:
        return jsonify({"error": "Models still loading..."}), 503

    payload = request.get_json(silent=True) or {}
    ticker = str(payload.get("ticker", "")).strip().upper()
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    if not _is_tradeable_ticker(ticker):
        return jsonify({"error": f"{ticker} is not in configured ticker universe"}), 400
    try:
        entry_price = float(payload.get("entry_price", 0) or 0)
        quantity = float(payload.get("quantity", 0) or 0)
    except Exception:
        return jsonify({"error": "entry_price and quantity must be numeric"}), 400
    if entry_price <= 0 or quantity <= 0:
        return jsonify({"error": "entry_price and quantity must be > 0"}), 400

    try:
        pred = predictor.predict_single(ticker, use_cache=True) or {}
        px = _get_live_prices_batch([ticker]).get(ticker, {})
        current_price = float(px.get("price", 0) or pred.get("current_price", 0) or 0)
        signal = pred.get("signal", "HOLD")
        predicted_return = float(pred.get("predicted_return", 0) or 0)
        confidence = float(pred.get("confidence", 0) or 0)
        agreement = float(pred.get("model_agreement", 0) or 0)
        stock_name = ticker_names.get(ticker, _clean_name(ticker))
        sentiment = get_news_sentiment(ticker, stock_name)
        review = review_trade_plan(
            ticker=ticker,
            stock_name=stock_name,
            entry_price=entry_price,
            quantity=quantity,
            current_price=current_price,
            signal=signal,
            predicted_return_pct=predicted_return,
            confidence=confidence,
            agreement=agreement,
            sentiment_text=sentiment,
        )
        return jsonify(
            {
                "ticker": ticker,
                "name": stock_name,
                "entry_price": round(entry_price, 2),
                "quantity": quantity,
                "current_price": round(current_price, 2) if current_price > 0 else None,
                "signal": signal,
                "predicted_return": round(predicted_return, 3),
                "confidence": round(confidence, 1),
                "model_agreement": round(agreement, 1),
                "review": review,
                "generated_at": datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST"),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai-risk-analysis")
def api_ai_risk_analysis():
    """AI-generated risk analysis page payload based on current portfolio summary."""
    _sanitize_portfolio_storage()
    summary = _portfolio_summary_from_trades(_read_portfolio_trades())
    try:
        analysis = ai_risk_assessment(summary)
    except Exception as e:
        analysis = f"AI risk analysis unavailable: {e}"
    return jsonify(
        {
            "summary": summary,
            "analysis": analysis,
            "generated_at": datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST"),
        }
    )


@app.route("/api/explain-model")
def api_explain_model():
    """Explain model purpose and behavior."""
    model = request.args.get("model", "").strip()
    if not model:
        return jsonify({"error": "model is required"}), 400
    try:
        return jsonify({"model": model, "explanation": explain_model(model)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stock-chat/<ticker>", methods=["POST"])
def api_stock_chat(ticker: str):
    """Contextual stock Q&A assistant."""
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    if not models_loaded or predictor is None:
        return jsonify({"error": "Models still loading..."}), 503
    try:
        pred = predictor.predict_single(ticker, use_cache=True) or {}
        indicators = pred.get("indicators", {})
        stock_name = ticker_names.get(ticker, _clean_name(ticker))
        answer = stock_chat_response(
            ticker=ticker,
            stock_name=stock_name,
            question=question,
            prediction_data=pred,
            indicator_snapshot=indicators,
        )
        return jsonify({"ticker": ticker, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/feature-importance/<ticker>")
def api_feature_importance(ticker: str):
    """Return feature importance for a stock's prediction."""
    if not models_loaded or predictor is None:
        return jsonify({"error": "Models still loading..."}), 503

    try:
        pred = predictor.predict_single(ticker, use_cache=True)
        if pred is None:
            return jsonify({"error": f"No prediction for {ticker}"}), 404

        # Extract feature importances from tree-based models
        importances = {}
        if hasattr(predictor, "models") and predictor.models:
            for name, model in predictor.models.items():
                inner = getattr(model, "model", model)
                if hasattr(inner, "feature_importances_"):
                    fi = inner.feature_importances_
                    feat_names = getattr(inner, "feature_names_in_", None) or getattr(
                        model, "feature_names", None
                    )
                    if feat_names is not None:
                        # Pair names with importances, sort by importance desc
                        paired = sorted(
                            zip(feat_names, fi), key=lambda x: x[1], reverse=True
                        )[:20]
                        importances[name] = {
                            str(n): round(float(v), 4) for n, v in paired
                        }

        # Provide top indicators affecting price
        indicators = pred.get("indicators", {})
        fundamentals = pred.get("fundamentals", {})

        key_factors = []
        # RSI
        rsi = indicators.get("rsi")
        if rsi is not None:
            if rsi > 70:
                key_factors.append(
                    {
                        "factor": "RSI (Overbought)",
                        "value": f"{rsi:.1f}",
                        "impact": "bearish",
                        "weight": 0.15,
                    }
                )
            elif rsi < 30:
                key_factors.append(
                    {
                        "factor": "RSI (Oversold)",
                        "value": f"{rsi:.1f}",
                        "impact": "bullish",
                        "weight": 0.15,
                    }
                )
            else:
                key_factors.append(
                    {
                        "factor": "RSI",
                        "value": f"{rsi:.1f}",
                        "impact": "neutral",
                        "weight": 0.10,
                    }
                )

        # MACD
        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                key_factors.append(
                    {
                        "factor": "MACD Crossover",
                        "value": f"{macd:.3f}",
                        "impact": "bullish",
                        "weight": 0.12,
                    }
                )
            else:
                key_factors.append(
                    {
                        "factor": "MACD Crossover",
                        "value": f"{macd:.3f}",
                        "impact": "bearish",
                        "weight": 0.12,
                    }
                )

        # Volume
        vol_ratio = indicators.get("volume_ratio")
        if vol_ratio is not None:
            if vol_ratio > 1.5:
                key_factors.append(
                    {
                        "factor": "High Volume",
                        "value": f"{vol_ratio:.2f}x",
                        "impact": "bullish",
                        "weight": 0.10,
                    }
                )
            elif vol_ratio < 0.5:
                key_factors.append(
                    {
                        "factor": "Low Volume",
                        "value": f"{vol_ratio:.2f}x",
                        "impact": "bearish",
                        "weight": 0.08,
                    }
                )

        # ADX
        adx = indicators.get("adx")
        if adx is not None:
            key_factors.append(
                {
                    "factor": "ADX (Trend Strength)",
                    "value": f"{adx:.1f}",
                    "impact": "bullish" if adx > 25 else "neutral",
                    "weight": 0.08,
                }
            )

        # ATR
        atr = indicators.get("atr_pct")
        if atr is not None:
            key_factors.append(
                {
                    "factor": "Volatility (ATR%)",
                    "value": f"{atr*100:.2f}%",
                    "impact": "neutral",
                    "weight": 0.07,
                }
            )

        # Fundamentals
        pe = fundamentals.get("pe_ratio")
        if pe is not None and pe > 0:
            key_factors.append(
                {
                    "factor": "P/E Ratio",
                    "value": f"{pe:.1f}",
                    "impact": (
                        "bearish" if pe > 40 else "neutral" if pe > 20 else "bullish"
                    ),
                    "weight": 0.10,
                }
            )

        # Sort by weight
        key_factors.sort(key=lambda x: x["weight"], reverse=True)

        return jsonify(
            {
                "ticker": ticker,
                "key_factors": key_factors,
                "model_importances": importances,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Helpers — JSON Sanitization for numpy types
# ---------------------------------------------------------------------------


def _sanitize(obj):
    """Recursively convert numpy types to Python-native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Routes — Portfolio Risk Analytics (Institutional Grade)
# ---------------------------------------------------------------------------


@app.route("/api/risk-analytics")
def api_risk_analytics():
    """
    Comprehensive portfolio-level risk analytics dashboard.
    Returns correlation matrix, sector exposure, risk metrics,
    Monte Carlo simulation, and statistical validation.
    """
    import yfinance as yf
    from src.backtest.metrics import (
        sharpe_ratio as calc_sharpe,
        sortino_ratio as calc_sortino,
        max_drawdown as calc_mdd,
        max_drawdown_duration as calc_mdd_dur,
        value_at_risk as calc_var,
        conditional_var as calc_cvar,
        parametric_var as calc_pvar,
        return_skewness,
        return_kurtosis,
        tail_ratio as calc_tail,
        jarque_bera_test,
        monte_carlo_backtest,
        information_ratio as calc_ir,
    )

    try:
        _sanitize_portfolio_storage()
        portfolio_rows = _read_portfolio()
        if not portfolio_rows:
            return (
                jsonify(
                    {
                        "error": "No portfolio holdings found. Add portfolio positions to run risk analytics.",
                        "portfolio_tickers": [],
                    }
                ),
                400,
            )

        custom_weights = {}
        portfolio_universe = []
        if portfolio_rows:
            for row in portfolio_rows:
                t = row.get("ticker")
                q = float(row.get("quantity", 0) or 0)
                p = float(row.get("entry_price", 0) or 0)
                if t and q > 0 and p > 0:
                    portfolio_universe.append(t)
                    custom_weights[t] = custom_weights.get(t, 0.0) + (q * p)
        portfolio_universe = sorted(set(portfolio_universe))
        if not portfolio_universe:
            return (
                jsonify(
                    {
                        "error": "No valid portfolio holdings found. Remove invalid tickers and add holdings again.",
                        "portfolio_tickers": [],
                    }
                ),
                400,
            )

        raw_requested = request.args.getlist("tickers")
        requested: list[str] = []
        for raw in raw_requested:
            requested.extend(
                [t.strip().upper() for t in str(raw).split(",") if t.strip()]
            )
        if not requested:
            tickers_param = request.args.get("tickers", "")
            if tickers_param:
                requested = [
                    t.strip().upper()
                    for t in str(tickers_param).split(",")
                    if t.strip()
                ]
        if requested:
            portfolio_tickers = [t for t in requested if t in portfolio_universe]
            ignored = [t for t in requested if t not in portfolio_universe]
        else:
            portfolio_tickers = list(portfolio_universe)
            ignored = []

        if not portfolio_tickers:
            return (
                jsonify(
                    {
                        "error": "Requested tickers are not in portfolio holdings.",
                        "portfolio_tickers": [],
                        "ignored_tickers": ignored,
                    }
                ),
                400,
            )

        # Download 6 months of history for robust risk calcs
        ticker_str = " ".join(portfolio_tickers + ["^NSEI"])
        data = yf.download(
            ticker_str,
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=True,
        )

        if data is None or data.empty:
            return jsonify(
                _sanitize(
                    {
                        "portfolio_tickers": portfolio_tickers,
                        "portfolio_holdings": [
                            r
                            for r in portfolio_rows
                            if r.get("ticker") in portfolio_tickers
                        ],
                        "initial_capital": round(
                            float(sum(custom_weights.values()) or 100000.0), 2
                        ),
                        "ignored_tickers": ignored,
                        "n_stocks": len(portfolio_tickers),
                        "period": "0 days",
                        "warning": "Unable to fetch historical data",
                        "risk_metrics": {},
                        "correlation_matrix": {},
                        "sector_exposure": {},
                        "statistical_tests": {},
                        "monte_carlo": {
                            "error": "Insufficient data for analytics",
                            "initial_capital": round(
                                float(sum(custom_weights.values()) or 100000.0), 2
                            ),
                        },
                        "equity_curve": [],
                    }
                )
            )

        # Extract closing prices
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].dropna(how="all")
        else:
            close = data[["Close"]].dropna()

        # Separate benchmark before filtering
        bench_close = close["^NSEI"] if "^NSEI" in close.columns else None
        port_close = close.drop(columns=["^NSEI"], errors="ignore")

        # Drop columns with too many NaNs (50% threshold for wider coverage)
        port_close = port_close.dropna(axis=1, thresh=int(len(port_close) * 0.5))
        actual_period_days = len(port_close)

        # Recombine with benchmark for returns calculation
        if bench_close is not None:
            combined = pd.concat([port_close, bench_close.rename("^NSEI")], axis=1)
        else:
            combined = port_close
        returns = combined.pct_change().dropna()

        # Update portfolio_tickers to only surviving tickers
        surviving_tickers = [c for c in port_close.columns if c in portfolio_tickers]

        # --- Correlation Matrix (portfolio tickers only) ---
        port_returns_only = returns[[c for c in returns.columns if c != "^NSEI"]]
        corr_matrix = port_returns_only.corr()
        corr_data = {}
        for col in corr_matrix.columns:
            corr_data[col] = {
                c: round(float(v), 3) for c, v in corr_matrix[col].items()
            }

        # --- Sector Exposure ---
        sector_exposure = {}
        for sec, tickers in tickers_by_sector.items():
            overlap = [t for t in portfolio_tickers if t in tickers]
            if overlap:
                sector_exposure[SECTOR_DISPLAY.get(sec, sec)] = {
                    "count": len(overlap),
                    "tickers": overlap,
                    "weight_pct": round(
                        len(overlap) / max(len(surviving_tickers), 1) * 100, 1
                    ),
                }

        # --- Portfolio-level returns (weighted by user holdings if available) ---
        portfolio_cols = [
            c for c in returns.columns if c != "^NSEI" and c in surviving_tickers
        ]
        if not portfolio_cols:
            return jsonify(
                _sanitize(
                    {
                        "portfolio_tickers": surviving_tickers,
                        "portfolio_holdings": [
                            r
                            for r in portfolio_rows
                            if r.get("ticker") in surviving_tickers
                        ],
                        "initial_capital": round(
                            float(
                                sum(
                                    custom_weights.get(t, 0.0)
                                    for t in surviving_tickers
                                )
                                or 100000.0
                            ),
                            2,
                        ),
                        "ignored_tickers": ignored,
                        "n_stocks": len(surviving_tickers),
                        "period": f"{actual_period_days} days",
                        "warning": "Insufficient data for analytics",
                        "risk_metrics": {},
                        "correlation_matrix": corr_data,
                        "sector_exposure": sector_exposure,
                        "statistical_tests": {},
                        "monte_carlo": {
                            "error": "Insufficient data for analytics",
                            "initial_capital": round(
                                float(
                                    sum(
                                        custom_weights.get(t, 0.0)
                                        for t in surviving_tickers
                                    )
                                    or 100000.0
                                ),
                                2,
                            ),
                        },
                        "equity_curve": [],
                    }
                )
            )

        if custom_weights:
            total_cost = sum(custom_weights.get(t, 0.0) for t in portfolio_cols)
            if total_cost > 0:
                normalized = {
                    t: custom_weights.get(t, 0.0) / total_cost for t in portfolio_cols
                }
            else:
                normalized = {t: 1.0 / len(portfolio_cols) for t in portfolio_cols}
            weighted = pd.Series(0.0, index=returns.index)
            for t in portfolio_cols:
                weighted = weighted + returns[t] * normalized.get(t, 0.0)
            port_returns = weighted
        else:
            port_returns = returns[portfolio_cols].mean(axis=1)
        initial_capital = float(sum(custom_weights.get(t, 0.0) for t in portfolio_cols))
        if initial_capital <= 0:
            initial_capital = 100000.0
        port_equity = (1 + port_returns).cumprod() * initial_capital

        benchmark_returns = returns["^NSEI"] if "^NSEI" in returns.columns else None

        def _safe(fn, *args, default=0.0):
            """Safely compute a metric, returning default on NaN/error."""
            try:
                val = float(fn(*args))
                return default if (np.isnan(val) or np.isinf(val)) else round(val, 6)
            except Exception:
                return default

        # --- Risk Metrics ---
        risk_metrics = {
            "sharpe_ratio": _safe(calc_sharpe, port_returns),
            "sortino_ratio": _safe(calc_sortino, port_returns),
            "max_drawdown": _safe(calc_mdd, port_equity),
            "max_drawdown_duration_days": int(calc_mdd_dur(port_equity)),
            "daily_var_95": _safe(calc_var, port_returns, 0.95),
            "daily_cvar_95": _safe(calc_cvar, port_returns, 0.95),
            "parametric_var_cf_95": _safe(
                calc_pvar, port_returns, 0.95, "cornish_fisher"
            ),
            "skewness": _safe(return_skewness, port_returns),
            "excess_kurtosis": _safe(return_kurtosis, port_returns),
            "tail_ratio": _safe(calc_tail, port_returns, default=1.0),
            "annualized_volatility": round(float(port_returns.std() * np.sqrt(252)), 4),
        }

        # Add benchmark-relative metrics
        if benchmark_returns is not None and not benchmark_returns.empty:
            risk_metrics["information_ratio"] = _safe(
                calc_ir, port_returns, benchmark_returns
            )

        # --- Statistical Tests ---
        stat_tests = {
            "jarque_bera": jarque_bera_test(port_returns),
        }

        # --- Monte Carlo ---
        mc_result = monte_carlo_backtest(
            port_returns,
            n_simulations=500,
            initial_capital=initial_capital,
            equity_curve=port_equity,
        )

        # --- Equity Curve (for chart) ---
        equity_data = [
            {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
            for d, v in port_equity.items()
        ]

        return jsonify(
            _sanitize(
                {
                    "portfolio_tickers": surviving_tickers,
                    "portfolio_holdings": [
                        r
                        for r in portfolio_rows
                        if r.get("ticker") in surviving_tickers
                    ],
                    "initial_capital": round(initial_capital, 2),
                    "ignored_tickers": ignored,
                    "n_stocks": len(surviving_tickers),
                    "period": f"{actual_period_days} days",
                    "risk_metrics": risk_metrics,
                    "correlation_matrix": corr_data,
                    "sector_exposure": sector_exposure,
                    "statistical_tests": stat_tests,
                    "monte_carlo": mc_result,
                    "equity_curve": equity_data,
                }
            )
        )

    except Exception as e:
        log.error(f"Risk analytics error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/risk")
def risk_dashboard():
    """Portfolio risk analytics dashboard page."""
    return render_template("risk.html")


@app.route("/ai-risk")
def ai_risk_dashboard():
    """AI-only risk interpretation page."""
    return render_template("ai_risk.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_tickers()
    log.info(
        f"Loaded {len(all_tickers)} tickers across {len(tickers_by_sector)} sectors"
    )
    cleaned = _sanitize_portfolio_storage()
    if cleaned.get("changed"):
        log.info(
            "Portfolio sanitized at startup: removed_trades=%s removed_holdings=%s remaining_trades=%s",
            cleaned.get("removed_trade_rows"),
            cleaned.get("removed_holding_rows"),
            cleaned.get("remaining_trades"),
        )
    pruned = _prune_delisted_registry()
    if pruned.get("changed"):
        log.info(
            "Delisted registry pruned at startup: removed=%s remaining=%s",
            pruned.get("removed"),
            pruned.get("remaining"),
        )

    # Load models in background thread
    model_thread = threading.Thread(target=load_models_background, daemon=True)
    model_thread.start()

    # Start daily analysis background computation thread
    analysis_thread = threading.Thread(
        target=_daily_analysis_background_loop, daemon=True
    )
    analysis_thread.start()
    log.info("Daily analysis background thread started")

    port = int(os.environ.get("FLASK_PORT", 5001))
    log.info(f"Starting Flask server at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
