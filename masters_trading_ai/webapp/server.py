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
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Load .env from project root BEFORE any other imports that need env vars
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Validate critical environment variables at startup ──
_REQUIRED_ENV = []  # Add keys here if they should be mandatory
_OPTIONAL_ENV = {"GROQ_API_KEY": "AI explanations disabled", "FLASK_SECRET_KEY": "Using dev fallback"}
for _key in _REQUIRED_ENV:
    if not os.environ.get(_key):
        raise EnvironmentError(f"Missing required env var: {_key}. Copy .env.example → .env and fill in values.")
for _key, _msg in _OPTIONAL_ENV.items():
    if not os.environ.get(_key):
        print(f"⚠️  {_key} not set — {_msg}. See .env.example")

# ── CRITICAL: import torch FIRST to avoid segfault with statsmodels C extensions ──
import torch  # noqa: F401

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import LivePredictor, get_market_status, get_intraday_data
from src.tracking.prediction_logger import PredictionLogger
from src.tracking.performance_reporter import PerformanceReporter

from webapp.groq_explainer import (
    explain_fundamental, explain_greek, explain_indicator,
    get_groq_strategy, get_combined_strategy,
    get_stock_overview, get_news_sentiment,
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
PREDICTION_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Thread-safe lock for file I/O
_log_lock = threading.Lock()

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

    tradeable_sectors = ["large_cap", "banking", "mid_cap", "high_volatility", "commodities"]
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
    global predictor, models_loaded, load_error
    try:
        log.info("Loading ML models...")
        t0 = time.time()
        predictor = LivePredictor()
        predictor.load_models()
        elapsed = time.time() - t0
        models_loaded = True
        log.info(f"All models loaded in {elapsed:.1f}s")
    except Exception as e:
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
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    log_file = PREDICTION_LOG_DIR / f"{today_str}.json"

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
            "signal": pred.get("signal", "HOLD"),
            "confidence": pred.get("confidence", 50),
            "model_predictions": pred.get("model_predictions", {}),
            "timestamp": datetime.now(IST).isoformat(),
        }

        with open(log_file, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    # Also record to prediction tracker for hit/miss tracking
    try:
        PredictionTracker.record_prediction(ticker, pred)
    except Exception as e:
        log.warning(f"Prediction tracking failed for {ticker}: {e}")


def _get_live_prices_batch(tickers: list[str]) -> dict:
    """Get live prices for multiple tickers using yfinance."""
    import yfinance as yf
    result = {}
    try:
        data = yf.download(
            tickers, period="2d", interval="1d",
            progress=False, auto_adjust=True, threads=True,
        )
        if data is None or data.empty:
            return result

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    if isinstance(data.columns, pd.MultiIndex):
                        cols = data.columns.get_level_values(0)
                    else:
                        cols = data.columns
                    close_col = data["Close"]
                else:
                    close_col = data["Close"][ticker] if "Close" in data.columns.get_level_values(0) else None

                if close_col is None or close_col.dropna().empty:
                    continue

                close_vals = close_col.dropna()
                current = float(close_vals.values[-1])
                prev = float(close_vals.values[-2]) if len(close_vals) >= 2 else current
                change = current - prev
                change_pct = (change / prev * 100) if prev != 0 else 0

                # Get volume, high, low, open
                if len(tickers) == 1:
                    vol = float(data["Volume"].dropna().values[-1]) if "Volume" in data.columns else 0
                    high = float(data["High"].dropna().values[-1]) if "High" in data.columns else current
                    low = float(data["Low"].dropna().values[-1]) if "Low" in data.columns else current
                    open_p = float(data["Open"].dropna().values[-1]) if "Open" in data.columns else current
                else:
                    vol = float(data["Volume"][ticker].dropna().values[-1]) if "Volume" in data.columns.get_level_values(0) else 0
                    high = float(data["High"][ticker].dropna().values[-1]) if "High" in data.columns.get_level_values(0) else current
                    low = float(data["Low"][ticker].dropna().values[-1]) if "Low" in data.columns.get_level_values(0) else current
                    open_p = float(data["Open"][ticker].dropna().values[-1]) if "Open" in data.columns.get_level_values(0) else current

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
            except Exception as exc:
                log.debug(f"Price fetch failed for {ticker}: {exc}")
                continue
    except Exception as e:
        log.warning(f"Batch price fetch error: {e}")

    return result


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


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------
@app.route("/api/status")
def api_status():
    """Return server + market status."""
    mkt = get_market_status()
    return jsonify({
        "models_loaded": models_loaded,
        "load_error": load_error,
        "model_count": len(predictor.models) if predictor else 0,
        "market": {
            "status": mkt["status"],
            "description": mkt["description"],
            "next_open": mkt.get("next_open", ""),
            "ist_now": mkt["ist_now"].strftime("%d %b %Y, %I:%M %p IST"),
        },
        "ticker_count": len(all_tickers),
    })


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

    try:
        picks = predictor.predict_top_picks(sectors=sectors, top_n=top_n)
        picks = [p for p in picks if p.get("current_price", 0) > 0]
        for p in picks:
            p["name"] = ticker_names.get(p.get("ticker", ""), "")
        return jsonify(picks)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/intraday/<ticker>")
def api_intraday(ticker: str):
    """Get intraday price data for charting."""
    period = request.args.get("period", "5d")
    interval = request.args.get("interval", "15m")

    try:
        df = get_intraday_data(ticker, period=period, interval=interval)
        if df is None or df.empty:
            return jsonify({"error": "No intraday data"}), 404

        records = []
        for idx, row in df.iterrows():
            records.append({
                "time": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                "open": round(float(row.get("Open", 0)), 2),
                "high": round(float(row.get("High", 0)), 2),
                "low": round(float(row.get("Low", 0)), 2),
                "close": round(float(row.get("Close", 0)), 2),
                "volume": int(row.get("Volume", 0)),
            })
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/<ticker>")
def api_history(ticker: str):
    """Get daily price history for charting."""
    period = request.args.get("period", "1y")
    import yfinance as yf
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 404
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        records = []
        for idx, row in df.iterrows():
            records.append({
                "time": idx.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })
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
        return jsonify({"error": f"No predictions logged for {date_str}", "date": date_str})

    with open(log_file) as f:
        predictions = json.load(f)

    if not predictions:
        return jsonify({"error": "Empty prediction log", "date": date_str})

    # Fetch actual prices
    tickers_to_check = list(predictions.keys())
    actuals = _get_live_prices_batch(tickers_to_check)

    # Fetch benchmark (Nifty 50) return for alpha
    benchmark_data = _get_live_prices_batch(["^NSEI"])
    benchmark_return = 0.0
    if "^NSEI" in benchmark_data:
        bd = benchmark_data["^NSEI"]
        if bd["prev_close"] > 0:
            benchmark_return = (bd["price"] - bd["prev_close"]) / bd["prev_close"]

    results = []
    for ticker, pred in predictions.items():
        if ticker not in actuals:
            continue

        actual = actuals[ticker]
        pred_price_at_prediction = pred.get("current_price", 0)
        pred_return_pct = pred.get("predicted_return", 0)  # already in %
        actual_price = actual["price"]

        if pred_price_at_prediction <= 0:
            continue

        actual_return = (actual_price - pred_price_at_prediction) / pred_price_at_prediction
        actual_return_pct = actual_return * 100

        # Direction check
        pred_dir = "UP" if pred_return_pct > 0 else "DOWN"
        actual_dir = "UP" if actual_return > 0 else "DOWN"
        direction_correct = pred_dir == actual_dir

        # Alpha = actual return - benchmark return
        alpha = actual_return - benchmark_return
        alpha_pct = alpha * 100

        results.append({
            "ticker": ticker,
            "name": ticker_names.get(ticker, _clean_name(ticker)),
            "signal": pred.get("signal", "N/A"),
            "predicted_return_pct": round(pred_return_pct, 3),
            "predicted_price": round(pred.get("predicted_price", 0), 2),
            "actual_price": round(actual_price, 2),
            "actual_return_pct": round(actual_return_pct, 3),
            "direction_predicted": pred_dir,
            "direction_actual": actual_dir,
            "direction_correct": direction_correct,
            "alpha_pct": round(alpha_pct, 3),
            "confidence": pred.get("confidence", 50),
        })

    if not results:
        return jsonify({"error": "Could not fetch actuals", "date": date_str})

    # Summary
    total = len(results)
    hits = sum(1 for r in results if r["direction_correct"])
    hit_rate = (hits / total * 100) if total > 0 else 0
    avg_alpha = float(np.mean([r["alpha_pct"] for r in results]))
    total_alpha = float(np.sum([r["alpha_pct"] for r in results]))
    avg_confidence = float(np.mean([r["confidence"] for r in results]))

    return jsonify({
        "date": date_str,
        "total_predictions": total,
        "direction_hits": hits,
        "hit_rate_pct": round(hit_rate, 1),
        "avg_alpha_pct": round(avg_alpha, 3),
        "total_alpha_pct": round(total_alpha, 3),
        "benchmark_return_pct": round(benchmark_return * 100, 3),
        "avg_confidence": round(avg_confidence, 1),
        "results": sorted(results, key=lambda x: abs(x["alpha_pct"]), reverse=True),
    })


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
            explanation = explain_greek(metric, float(value) if value else 0, "call", ticker, stock_name)
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

        return jsonify({
            "groq_strategy": groq_strat,
            "combined_strategy": combined,
        })
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

        return jsonify({
            "overview": overview,
            "sentiment": sentiment,
            "stock_name": stock_name,
            "ticker": ticker,
        })
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


# ---------------------------------------------------------------------------
# Routes — Daily Analysis (Enhanced Dashboard)
# ---------------------------------------------------------------------------
# In-memory cache of opening prices captured at market open
_opening_prices: dict[str, dict] = {}
_opening_prices_date: str = ""

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
    _opening_prices = {}
    for t, p in prices.items():
        _opening_prices[t] = {
            "open": p.get("open", p.get("price", 0)),
            "prev_close": p.get("prev_close", 0),
        }
    _opening_prices_date = today
    log.info(f"Captured opening prices for {len(_opening_prices)} tickers")


def _build_daily_analysis() -> dict:
    """Build the full daily analysis payload (called by background thread)."""
    _capture_opening_prices()
    current_prices = _get_cached_prices()

    # Get all predictions (from cache)
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    log_file = PREDICTION_LOG_DIR / f"{today_str}.json"
    predictions = {}
    if log_file.exists():
        try:
            with open(log_file) as f:
                predictions = json.load(f)
        except (json.JSONDecodeError, OSError):
            predictions = {}

    stocks = []
    for ticker in all_tickers:
        if ticker.startswith("^") or ticker in ("USDINR=X", "GC=F", "CL=F"):
            continue

        curr = current_prices.get(ticker, {})
        opening = _opening_prices.get(ticker, {})
        pred = predictions.get(ticker, {})

        current_price = curr.get("price", 0)
        open_price = opening.get("open", curr.get("open", 0))
        prev_close = opening.get("prev_close", curr.get("prev_close", 0))
        predicted_price = pred.get("predicted_price", 0)
        predicted_return = pred.get("predicted_return", 0)
        signal = pred.get("signal", "")
        confidence = pred.get("confidence", 0)

        if current_price <= 0 or open_price <= 0:
            continue

        # Calculate percentage changes
        open_to_current_pct = ((current_price - open_price) / open_price * 100) if open_price > 0 else 0
        open_to_predicted_pct = ((predicted_price - open_price) / open_price * 100) if open_price > 0 and predicted_price > 0 else 0
        prev_to_current_pct = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0

        # Composite score for ranking (higher = better opportunity)
        # Factors: predicted return (40%), confidence (25%), model agreement (20%), R:R (15%)
        model_agreement = pred.get("model_agreement", 50)
        risk_reward = pred.get("risk_reward", 1.0) if pred.get("risk_reward") else 1.0

        score = (
            abs(predicted_return) * 4.0 +
            confidence * 0.25 +
            model_agreement * 0.20 +
            min(risk_reward, 5) * 3.0
        )

        model_preds = pred.get("model_predictions", {})

        stocks.append({
            "ticker": ticker,
            "name": ticker_names.get(ticker, _clean_name(ticker)),
            "open_price": round(open_price, 2),
            "predicted_price": round(predicted_price, 2),
            "current_price": round(current_price, 2),
            "prev_close": round(prev_close, 2),
            "open_to_current_pct": round(open_to_current_pct, 3),
            "open_to_predicted_pct": round(open_to_predicted_pct, 3),
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
        })

    stocks.sort(key=lambda x: x["composite_score"], reverse=True)
    top_10 = stocks[:10]

    gainers = [s for s in stocks if s["change_pct"] > 0]
    losers = [s for s in stocks if s["change_pct"] < 0]

    return {
        "date": today_str,
        "total_stocks": len(stocks),
        "market_summary": {
            "gainers": len(gainers),
            "losers": len(losers),
            "unchanged": len(stocks) - len(gainers) - len(losers),
            "avg_change_pct": round(float(np.mean([s["change_pct"] for s in stocks])) if stocks else 0, 3),
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
            result = _build_daily_analysis()
            with _daily_analysis_lock:
                _daily_analysis_cache = result
                _daily_analysis_cache_time = time.time()
            log.info(f"Daily analysis cache refreshed — {result['total_stocks']} stocks")
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
    return jsonify({"error": "Analysis is being computed, please retry in ~15 seconds..."}), 202


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

    current_price = curr.get("price", 0)
    predicted_price = pred.get("predicted_price", 0)
    predicted_return = pred.get("predicted_return", 0)
    signal = pred.get("signal", "")
    confidence = pred.get("confidence", 0)

    open_to_current_pct = ((current_price - open_price) / open_price * 100) if open_price > 0 else 0
    open_to_predicted_pct = ((predicted_price - open_price) / open_price * 100) if open_price > 0 and predicted_price > 0 else 0

    return jsonify({
        "ticker": ticker,
        "name": ticker_names.get(ticker, _clean_name(ticker)),
        "open_price": round(open_price, 2),
        "predicted_price": round(predicted_price, 2),
        "current_price": round(current_price, 2),
        "prev_close": round(prev_close, 2),
        "open_to_current_pct": round(open_to_current_pct, 3),
        "open_to_predicted_pct": round(open_to_predicted_pct, 3),
        "predicted_return": round(predicted_return, 3),
        "signal": signal,
        "confidence": round(confidence, 1),
        "change": curr.get("change", 0),
        "change_pct": curr.get("change_pct", 0),
    })


@app.route("/api/news/<ticker>")
def api_news(ticker: str):
    """Get real-world news and sentiment for a specific stock."""
    try:
        stock_name = ticker_names.get(ticker, _clean_name(ticker))
        sentiment = get_news_sentiment(ticker, stock_name)
        return jsonify({
            "ticker": ticker,
            "name": stock_name,
            "sentiment": sentiment,
        })
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
        if hasattr(predictor, 'models') and predictor.models:
            for name, model in predictor.models.items():
                inner = getattr(model, 'model', model)
                if hasattr(inner, 'feature_importances_'):
                    fi = inner.feature_importances_
                    feat_names = getattr(inner, 'feature_names_in_', None) or getattr(model, 'feature_names', None)
                    if feat_names is not None:
                        # Pair names with importances, sort by importance desc
                        paired = sorted(zip(feat_names, fi), key=lambda x: x[1], reverse=True)[:20]
                        importances[name] = {str(n): round(float(v), 4) for n, v in paired}

        # Provide top indicators affecting price
        indicators = pred.get("indicators", {})
        fundamentals = pred.get("fundamentals", {})

        key_factors = []
        # RSI
        rsi = indicators.get("rsi")
        if rsi is not None:
            if rsi > 70:
                key_factors.append({"factor": "RSI (Overbought)", "value": f"{rsi:.1f}", "impact": "bearish", "weight": 0.15})
            elif rsi < 30:
                key_factors.append({"factor": "RSI (Oversold)", "value": f"{rsi:.1f}", "impact": "bullish", "weight": 0.15})
            else:
                key_factors.append({"factor": "RSI", "value": f"{rsi:.1f}", "impact": "neutral", "weight": 0.10})

        # MACD
        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                key_factors.append({"factor": "MACD Crossover", "value": f"{macd:.3f}", "impact": "bullish", "weight": 0.12})
            else:
                key_factors.append({"factor": "MACD Crossover", "value": f"{macd:.3f}", "impact": "bearish", "weight": 0.12})

        # Volume
        vol_ratio = indicators.get("volume_ratio")
        if vol_ratio is not None:
            if vol_ratio > 1.5:
                key_factors.append({"factor": "High Volume", "value": f"{vol_ratio:.2f}x", "impact": "bullish", "weight": 0.10})
            elif vol_ratio < 0.5:
                key_factors.append({"factor": "Low Volume", "value": f"{vol_ratio:.2f}x", "impact": "bearish", "weight": 0.08})

        # ADX
        adx = indicators.get("adx")
        if adx is not None:
            key_factors.append({"factor": "ADX (Trend Strength)", "value": f"{adx:.1f}", "impact": "bullish" if adx > 25 else "neutral", "weight": 0.08})

        # ATR
        atr = indicators.get("atr_pct")
        if atr is not None:
            key_factors.append({"factor": "Volatility (ATR%)", "value": f"{atr*100:.2f}%", "impact": "neutral", "weight": 0.07})

        # Fundamentals
        pe = fundamentals.get("pe_ratio")
        if pe is not None and pe > 0:
            key_factors.append({"factor": "P/E Ratio", "value": f"{pe:.1f}", "impact": "bearish" if pe > 40 else "neutral" if pe > 20 else "bullish", "weight": 0.10})

        # Sort by weight
        key_factors.sort(key=lambda x: x["weight"], reverse=True)

        return jsonify({
            "ticker": ticker,
            "key_factors": key_factors,
            "model_importances": importances,
        })
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
        sharpe_ratio as calc_sharpe, sortino_ratio as calc_sortino,
        max_drawdown as calc_mdd, max_drawdown_duration as calc_mdd_dur,
        value_at_risk as calc_var, conditional_var as calc_cvar,
        parametric_var as calc_pvar,
        return_skewness, return_kurtosis, tail_ratio as calc_tail,
        jarque_bera_test, monte_carlo_backtest,
        information_ratio as calc_ir,
    )

    try:
        # Use top picks or daily analysis stocks for portfolio analytics
        with _daily_analysis_lock:
            stocks = _daily_analysis_cache.get("all_stocks", [])

        if not stocks:
            return jsonify({"error": "Daily analysis not ready yet"}), 202

        # Get buy-signal stocks as the "portfolio" — use top composite-scored stocks
        portfolio_tickers = [s["ticker"] for s in stocks
                             if s.get("signal") in ("BUY", "STRONG_BUY")][:20]
        if len(portfolio_tickers) < 5:
            # Fall back to top 10 by composite score
            portfolio_tickers = [s["ticker"] for s in stocks[:10]]

        # Download 6 months of history for robust risk calcs
        ticker_str = " ".join(portfolio_tickers + ["^NSEI"])
        data = yf.download(ticker_str, period="6mo", interval="1d",
                          progress=False, auto_adjust=True, threads=True)

        if data is None or data.empty:
            return jsonify({"error": "Unable to fetch historical data"}), 500

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
                c: round(float(v), 3)
                for c, v in corr_matrix[col].items()
            }

        # --- Sector Exposure ---
        sector_exposure = {}
        for sec, tickers in tickers_by_sector.items():
            overlap = [t for t in portfolio_tickers if t in tickers]
            if overlap:
                sector_exposure[SECTOR_DISPLAY.get(sec, sec)] = {
                    "count": len(overlap),
                    "tickers": overlap,
                    "weight_pct": round(len(overlap) / len(portfolio_tickers) * 100, 1),
                }

        # --- Portfolio-level returns (equal-weighted) ---
        portfolio_cols = [c for c in returns.columns if c != "^NSEI" and c in surviving_tickers]
        if not portfolio_cols:
            return jsonify({"error": "Insufficient data for analytics"}), 500

        port_returns = returns[portfolio_cols].mean(axis=1)
        port_equity = (1 + port_returns).cumprod() * 100000

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
            "parametric_var_cf_95": _safe(calc_pvar, port_returns, 0.95, "cornish_fisher"),
            "skewness": _safe(return_skewness, port_returns),
            "excess_kurtosis": _safe(return_kurtosis, port_returns),
            "tail_ratio": _safe(calc_tail, port_returns, default=1.0),
            "annualized_volatility": round(float(port_returns.std() * np.sqrt(252)), 4),
        }

        # Add benchmark-relative metrics
        if benchmark_returns is not None and not benchmark_returns.empty:
            risk_metrics["information_ratio"] = _safe(calc_ir, port_returns, benchmark_returns)

        # --- Statistical Tests ---
        stat_tests = {
            "jarque_bera": jarque_bera_test(port_returns),
        }

        # --- Monte Carlo ---
        mc_result = monte_carlo_backtest(
            port_returns,
            n_simulations=500,
            initial_capital=100000,
        )

        # --- Equity Curve (for chart) ---
        equity_data = [
            {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
            for d, v in port_equity.items()
        ]

        return jsonify(_sanitize({
            "portfolio_tickers": surviving_tickers,
            "n_stocks": len(surviving_tickers),
            "period": f"{actual_period_days} days",
            "risk_metrics": risk_metrics,
            "correlation_matrix": corr_data,
            "sector_exposure": sector_exposure,
            "statistical_tests": stat_tests,
            "monte_carlo": mc_result,
            "equity_curve": equity_data,
        }))

    except Exception as e:
        log.error(f"Risk analytics error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/risk")
def risk_dashboard():
    """Portfolio risk analytics dashboard page."""
    return render_template("risk.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_tickers()
    log.info(f"Loaded {len(all_tickers)} tickers across {len(tickers_by_sector)} sectors")

    # Load models in background thread
    model_thread = threading.Thread(target=load_models_background, daemon=True)
    model_thread.start()

    # Start daily analysis background computation thread
    analysis_thread = threading.Thread(target=_daily_analysis_background_loop, daemon=True)
    analysis_thread.start()
    log.info("Daily analysis background thread started")

    port = int(os.environ.get("FLASK_PORT", 5001))
    log.info(f"Starting Flask server at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
