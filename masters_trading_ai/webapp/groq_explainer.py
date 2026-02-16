"""
Groq AI Explainer — provides contextual explanations for trading concepts.

Uses Groq's LLM API to explain fundamentals (PE ratio, etc.), Greeks,
technical indicators, and provide strategy suggestions based on current values.

This module does NOT influence trading predictions — it only explains
concepts and current values to help the user understand what they see.
"""

import os
import json
import time
import hashlib
import re
import warnings
import threading
import contextvars
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

try:
    from groq import Groq
except ImportError:
    Groq = None

# ── Config ──────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY_2", "")
GROQ_API_KEY_3 = os.environ.get("GROQ_API_KEY_3", "")
GROQ_API_KEYS = os.environ.get("GROQ_API_KEYS", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
GROQ_MODELS = os.environ.get("GROQ_MODELS", "")
GROQ_KEY_ROTATION_ENABLED = os.environ.get(
    "GROQ_KEY_ROTATION_ENABLED", "1"
).lower() not in (
    "0",
    "false",
    "no",
)


def _load_groq_keys() -> list[str]:
    candidates: list[str] = []
    for raw in (GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3):
        v = str(raw or "").strip()
        if v:
            candidates.append(v)
    if GROQ_API_KEYS:
        for raw in re.split(r"[,\n;]+", str(GROQ_API_KEYS)):
            v = str(raw or "").strip()
            if v:
                candidates.append(v)
    seen: set[str] = set()
    unique: list[str] = []
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


_groq_keys: list[str] = _load_groq_keys()
if not _groq_keys:
    warnings.warn(
        "No GROQ_API_KEY configured (.env). AI explanations will be unavailable. "
        "Set GROQ_API_KEY (and optional GROQ_API_KEY_2/GROQ_API_KEY_3).",
        stacklevel=2,
    )


def _load_groq_models() -> list[str]:
    candidates: list[str] = []
    if GROQ_MODELS:
        for raw in re.split(r"[,\n;]+", str(GROQ_MODELS)):
            v = str(raw or "").strip()
            if v:
                candidates.append(v)
    if GROQ_MODEL:
        candidates.insert(0, GROQ_MODEL)

    # Always keep one smaller fallback unless user explicitly provided it.
    candidates.append("llama-3.1-8b-instant")

    seen: set[str] = set()
    out: list[str] = []
    for model in candidates:
        if model in seen:
            continue
        seen.add(model)
        out.append(model)
    return out


_groq_models: list[str] = _load_groq_models()
CACHE_DIR = Path(__file__).parent.parent / "cache" / "groq_explanations"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Rate limit ──────────────────────────────────────
_last_call_time = 0.0
MIN_CALL_INTERVAL = float(os.environ.get("GROQ_MIN_CALL_INTERVAL_SEC", "1.0"))
GROQ_GLOBAL_MAX_PER_MIN = max(1, int(os.environ.get("GROQ_GLOBAL_MAX_PER_MIN", "45")))
GROQ_ENDPOINT_MAX_PER_MIN = max(
    1, int(os.environ.get("GROQ_ENDPOINT_MAX_PER_MIN", "12"))
)
GROQ_QUEUE_WAIT_SEC = float(os.environ.get("GROQ_QUEUE_WAIT_SEC", "8.0"))
GROQ_DEGRADED_COOLDOWN_SEC = max(
    30, int(os.environ.get("GROQ_DEGRADED_COOLDOWN_SEC", "120"))
)

_rate_limit_lock = threading.Lock()
_endpoint_locks: dict[str, threading.Lock] = {}
_global_call_ts: deque[float] = deque()
_endpoint_call_ts: dict[str, deque[float]] = defaultdict(deque)
_key_lock = threading.Lock()
_key_last_429_at: dict[int, float] = {}
_active_key_index = 0
_groq_endpoint_ctx = contextvars.ContextVar("groq_endpoint", default="global")
_degraded_until = 0.0
_degraded_reason = ""
_last_429_at = 0.0
_last_error = ""
_last_success_at = 0.0


def _get_client_for_key(api_key: str):
    """Get Groq client instance for a specific key."""
    if Groq is None:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def _next_key_order() -> list[int]:
    """Round-robin key order for each request (no cooldown gating)."""
    global _active_key_index
    if not _groq_keys:
        return []
    n = len(_groq_keys)
    if n == 1 or not GROQ_KEY_ROTATION_ENABLED:
        return [0]

    with _key_lock:
        start = _active_key_index % n
        _active_key_index = (start + 1) % n

    return [(start + offset) % n for offset in range(n)]


def _record_key_429(idx: int, now_ts: float) -> None:
    with _key_lock:
        _key_last_429_at[idx] = now_ts


def _set_active_key(idx: int) -> None:
    global _active_key_index
    with _key_lock:
        if _groq_keys:
            _active_key_index = idx % len(_groq_keys)


def _cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def _get_cached(key: str) -> Optional[str]:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            # Cache valid for 24 hours
            if time.time() - data.get("ts", 0) < 86400:
                return data["text"]
        except Exception:
            pass
    return None


def _set_cache(key: str, text: str):
    path = CACHE_DIR / f"{key}.json"
    try:
        path.write_text(json.dumps({"text": text, "ts": time.time()}))
    except Exception:
        pass


def set_groq_request_endpoint(endpoint: str):
    """Set current Groq endpoint scope for request-level rate limiting."""
    name = str(endpoint or "global")
    return _groq_endpoint_ctx.set(name)


def reset_groq_request_endpoint(token) -> None:
    """Restore previous endpoint scope token."""
    try:
        _groq_endpoint_ctx.reset(token)
    except Exception:
        pass


def _prune_rate_windows(now_ts: float) -> None:
    cutoff = now_ts - 60.0
    while _global_call_ts and _global_call_ts[0] <= cutoff:
        _global_call_ts.popleft()
    for dq in _endpoint_call_ts.values():
        while dq and dq[0] <= cutoff:
            dq.popleft()


def _reserve_rate_slot(
    endpoint: str, max_wait_sec: float = GROQ_QUEUE_WAIT_SEC
) -> bool:
    """
    Strict global + per-endpoint limiter.
    Returns True if a slot is reserved, False if exhausted for max_wait_sec.
    """
    start = time.time()
    while True:
        now_ts = time.time()
        with _rate_limit_lock:
            _prune_rate_windows(now_ts)
            ep_dq = _endpoint_call_ts[endpoint]
            global_ok = len(_global_call_ts) < GROQ_GLOBAL_MAX_PER_MIN
            endpoint_ok = len(ep_dq) < GROQ_ENDPOINT_MAX_PER_MIN
            if global_ok and endpoint_ok:
                _global_call_ts.append(now_ts)
                ep_dq.append(now_ts)
                return True

            global_wait = (
                max(0.0, 60.0 - (now_ts - _global_call_ts[0]))
                if _global_call_ts
                else 0.5
            )
            endpoint_wait = max(0.0, 60.0 - (now_ts - ep_dq[0])) if ep_dq else 0.5
            wait_for = min(max(global_wait, endpoint_wait), 1.0)

        if (time.time() - start) >= max_wait_sec:
            return False
        time.sleep(max(wait_for, 0.05))


def _mark_degraded(reason: str, *, error: str = "", is_429: bool = False) -> None:
    global _degraded_until, _degraded_reason, _last_429_at, _last_error
    now_ts = time.time()
    _degraded_until = max(_degraded_until, now_ts + GROQ_DEGRADED_COOLDOWN_SEC)
    _degraded_reason = str(reason or "temporary_unavailable")
    if is_429:
        _last_429_at = now_ts
    if error:
        _last_error = str(error)[:300]


def get_groq_system_status() -> dict:
    """Expose Groq queue/degraded status for UI and diagnostics."""
    now_ts = time.time()
    with _rate_limit_lock:
        _prune_rate_windows(now_ts)
        endpoint = _groq_endpoint_ctx.get() or "global"
        endpoint_used = len(_endpoint_call_ts.get(endpoint, deque()))
        global_used = len(_global_call_ts)
    with _key_lock:
        active_slot = (_active_key_index + 1) if _groq_keys else None
        key_last_429 = [
            {
                "slot": idx + 1,
                "last_429_iso": (
                    datetime.fromtimestamp(_key_last_429_at.get(idx, 0)).isoformat()
                    if _key_last_429_at.get(idx, 0)
                    else None
                ),
            }
            for idx in sorted(range(len(_groq_keys)))
        ]
    degraded = now_ts < _degraded_until
    return {
        "degraded_mode": degraded,
        "degraded_reason": _degraded_reason if degraded else "",
        "degraded_until_iso": (
            datetime.fromtimestamp(_degraded_until).isoformat() if degraded else None
        ),
        "last_429_iso": (
            datetime.fromtimestamp(_last_429_at).isoformat() if _last_429_at else None
        ),
        "last_success_iso": (
            datetime.fromtimestamp(_last_success_at).isoformat()
            if _last_success_at
            else None
        ),
        "last_error": _last_error,
        "global_limit_per_min": GROQ_GLOBAL_MAX_PER_MIN,
        "endpoint_limit_per_min": GROQ_ENDPOINT_MAX_PER_MIN,
        "global_used_last_min": global_used,
        "endpoint_used_last_min": endpoint_used,
        "key_rotation_enabled": bool(GROQ_KEY_ROTATION_ENABLED),
        "key_pool_size": len(_groq_keys),
        "active_key_slot": active_slot,
        "model_pool": list(_groq_models),
        "key_last_429": key_last_429,
    }


def _call_groq(prompt: str, max_tokens: int = 500) -> str:
    """Call Groq API with queueing, key-rotation failover, and cache fallback."""
    global _last_call_time, _last_success_at, _degraded_until, _degraded_reason

    key = _cache_key(prompt)
    cached = _get_cached(key)
    if cached:
        return cached

    endpoint = str(_groq_endpoint_ctx.get() or "global")

    if not _groq_keys:
        return "Groq API not available. Set GROQ_API_KEY in .env"

    with _rate_limit_lock:
        endpoint_lock = _endpoint_locks.setdefault(endpoint, threading.Lock())

    # Strict per-endpoint queue (single in-flight request per endpoint).
    with endpoint_lock:
        if not _reserve_rate_slot(endpoint):
            _mark_degraded("local_queue_limit")
            return (
                cached
                or "Groq request budget exhausted for this minute. Using cached-only mode."
            )

        elapsed = time.time() - _last_call_time
        if elapsed < MIN_CALL_INTERVAL:
            time.sleep(MIN_CALL_INTERVAL - elapsed)

        last_err = ""
        saw_429 = False
        for idx in _next_key_order():
            client = _get_client_for_key(_groq_keys[idx])
            if client is None:
                continue
            key_rate_limited = False
            for model_name in _groq_models:
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a concise financial analyst assistant for an Indian stock trading app. "
                                    "Give clear, actionable explanations in 3-5 sentences. "
                                    "Always mention what the current value suggests the investor should consider. "
                                    "Use simple language. Be specific about implications. "
                                    "Never give definitive buy/sell advice — always say 'suggests' or 'indicates'. "
                                    "Format: start with a one-line definition, then explain current value implications."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.3,
                    )
                    _last_call_time = time.time()
                    _last_success_at = _last_call_time
                    _degraded_until = 0.0
                    _degraded_reason = ""
                    _set_active_key((idx + 1) % len(_groq_keys))
                    text = resp.choices[0].message.content.strip()
                    _set_cache(key, text)
                    return text
                except Exception as e:
                    err = str(e)
                    last_err = err
                    lower_err = err.lower()
                    is_429 = "429" in err or "rate limit" in lower_err
                    if is_429:
                        saw_429 = True
                        _record_key_429(idx, time.time())
                        key_rate_limited = True
                        break
                    # Try next model for the same key before failing over keys.
                    continue
            if key_rate_limited and GROQ_KEY_ROTATION_ENABLED and len(_groq_keys) > 1:
                continue

        if saw_429:
            _mark_degraded("upstream_429", error=last_err, is_429=True)
        else:
            _mark_degraded("upstream_error", error=last_err, is_429=False)
        cached_fallback = _get_cached(key)
        if cached_fallback:
            return cached_fallback
        if saw_429:
            return (
                "Groq rate-limited (429) across available keys. "
                "Degraded mode enabled with cached-only fallback."
            )
        return f"Groq API error: {last_err or 'unknown error'}"


# ════════════════════════════════════════════════════
# Public API — Called from Flask routes
# ════════════════════════════════════════════════════


def explain_fundamental(
    metric_name: str, value, ticker: str = "", stock_name: str = ""
) -> str:
    """Explain a fundamental metric (PE ratio, ROE, etc.) with current value context."""
    stock_ctx = f" for {stock_name} ({ticker})" if ticker else ""
    prompt = (
        f"Explain '{metric_name}' as a stock fundamental metric{stock_ctx}. "
        f"The current value is {value}. "
        f"What does this specific value tell an investor? "
        f"Is this value generally considered good, average, or concerning for an Indian stock? "
        f"What should the investor watch out for?"
    )
    return _call_groq(prompt)


def explain_greek(
    greek_name: str,
    value: float,
    option_type: str = "call",
    ticker: str = "",
    stock_name: str = "",
) -> str:
    """Explain an option Greek and what its current value means for trading."""
    stock_ctx = f" for {stock_name} ({ticker})" if ticker else ""
    prompt = (
        f"Explain the option Greek '{greek_name}'{stock_ctx}. "
        f"The current {greek_name} value is {value} for a {option_type} option. "
        f"What does this Greek measure in simple terms? "
        f"What does this specific value of {value} suggest the trader should do or be aware of? "
        f"Give a practical implication for an options trader."
    )
    return _call_groq(prompt)


# ── Indicator Thresholds ───────────────────────────
INDICATOR_THRESHOLDS = {
    "RSI": {
        "buy_below": 30,
        "sell_above": 70,
        "desc": "RSI (Relative Strength Index) measures momentum on a 0-100 scale.",
        "interpretation": "Below 30 = oversold (potential buy), Above 70 = overbought (potential sell)",
    },
    "MACD": {
        "buy_condition": "MACD crosses above signal line",
        "sell_condition": "MACD crosses below signal line",
        "desc": "MACD shows momentum by comparing two moving averages.",
        "interpretation": "Positive = bullish momentum, Negative = bearish momentum",
    },
    "ADX": {
        "strong_trend": 25,
        "very_strong": 50,
        "desc": "ADX measures trend strength (not direction) on a 0-100 scale.",
        "interpretation": "Below 20 = weak/no trend, 20-25 = potential trend, Above 25 = strong trend",
    },
    "Stochastic": {
        "buy_below": 20,
        "sell_above": 80,
        "desc": "Stochastic compares closing price to price range over a period.",
        "interpretation": "Below 20 = oversold (potential buy), Above 80 = overbought (potential sell)",
    },
    "Bollinger Bands": {
        "buy_condition": "Price touches lower band",
        "sell_condition": "Price touches upper band",
        "desc": "Bollinger Bands show volatility with 2 standard deviations from moving average.",
        "interpretation": "Price at lower band = potential buy, Price at upper band = potential sell",
    },
    "ATR": {
        "desc": "ATR (Average True Range) measures volatility in absolute terms.",
        "interpretation": "Higher ATR = more volatile, use for stop-loss sizing (typically 1.5-2x ATR)",
    },
    "Volume Ratio": {
        "high_volume": 1.5,
        "low_volume": 0.5,
        "desc": "Volume Ratio compares current volume to average volume.",
        "interpretation": "Above 1.5 = high interest (confirms trend), Below 0.5 = low interest",
    },
    "RVOL": {
        "high_volume": 1.5,
        "desc": "RVOL (Relative Volume) compares current volume to historical average.",
        "interpretation": "Above 1.5 = unusual activity, important for breakout confirmation",
    },
    "OBV": {
        "desc": "OBV (On-Balance Volume) tracks cumulative volume flow.",
        "interpretation": "Rising OBV with rising price = bullish confirmation, Divergence = warning",
    },
}


def explain_indicator(
    indicator_name: str,
    app_value,
    actual_value=None,
    ticker: str = "",
    stock_name: str = "",
) -> str:
    """Explain a technical indicator with buy/sell thresholds and current value context."""
    stock_ctx = f" for {stock_name} ({ticker})" if ticker else ""
    value_ctx = f"The current value is {app_value}."
    if actual_value is not None:
        value_ctx += f" The actual/live market value is {actual_value}."

    # Get threshold info if available
    threshold_info = ""
    for key, thresholds in INDICATOR_THRESHOLDS.items():
        if key.lower() in indicator_name.lower():
            threshold_info = f"\n\nKey thresholds for {key}:\n"
            if "buy_below" in thresholds:
                threshold_info += f"• BUY signal: Below {thresholds['buy_below']}\n"
            if "sell_above" in thresholds:
                threshold_info += f"• SELL signal: Above {thresholds['sell_above']}\n"
            if "buy_condition" in thresholds:
                threshold_info += f"• BUY condition: {thresholds['buy_condition']}\n"
            if "sell_condition" in thresholds:
                threshold_info += f"• SELL condition: {thresholds['sell_condition']}\n"
            if "strong_trend" in thresholds:
                threshold_info += (
                    f"• Strong trend: Above {thresholds['strong_trend']}\n"
                )
            if "high_volume" in thresholds:
                threshold_info += f"• High volume: Above {thresholds['high_volume']}\n"
            threshold_info += (
                f"\nInterpretation: {thresholds.get('interpretation', '')}"
            )
            break

    prompt = (
        f"Explain the technical indicator '{indicator_name}'{stock_ctx}. "
        f"{value_ctx} "
        f"What does this indicator measure in simple terms? "
        f"Based on the current value of {app_value}, give a specific trading signal: "
        f"1) Is this a BUY, SELL, or HOLD signal? "
        f"2) What exact value would trigger a BUY? What value would trigger a SELL? "
        f"3) What should the trader do right now based on this value?"
        f"{threshold_info}"
    )
    return _call_groq(prompt, max_tokens=600)


def get_stock_overview(
    ticker: str,
    stock_name: str,
    fundamentals: dict = None,
    current_price: float = 0,
    prediction_signal: str = "",
) -> str:
    """
    Get a comprehensive stock overview including what the company does,
    recent sentiment, and investment thesis.
    """
    # Build fundamentals context
    fund_ctx = ""
    if fundamentals:
        fund_items = []
        if fundamentals.get("sector"):
            fund_items.append(f"Sector: {fundamentals['sector']}")
        if fundamentals.get("pe_ratio"):
            fund_items.append(f"P/E: {fundamentals['pe_ratio']:.1f}")
        if fundamentals.get("market_cap"):
            mc = fundamentals["market_cap"]
            mc_str = f"₹{mc/1e12:.2f}T" if mc > 1e12 else f"₹{mc/1e9:.2f}B"
            fund_items.append(f"Market Cap: {mc_str}")
        if fundamentals.get("revenue_growth"):
            fund_items.append(f"Revenue Growth: {fundamentals['revenue_growth']}%")
        if fundamentals.get("profit_margin"):
            fund_items.append(f"Profit Margin: {fundamentals['profit_margin']}%")
        if fundamentals.get("debt_to_equity"):
            fund_items.append(f"Debt/Equity: {fundamentals['debt_to_equity']:.1f}")
        if fundamentals.get("fifty_two_high") and fundamentals.get("fifty_two_low"):
            fund_items.append(
                f"52W Range: ₹{fundamentals['fifty_two_low']:.0f} - ₹{fundamentals['fifty_two_high']:.0f}"
            )
        if fundamentals.get("analyst_upside"):
            fund_items.append(
                f"Analyst Target Upside: {fundamentals['analyst_upside']:.1f}%"
            )
        if fund_items:
            fund_ctx = "Key Metrics: " + ", ".join(fund_items)

    prompt = (
        f"Provide a comprehensive overview of {stock_name} ({ticker}) on the Indian stock market. "
        f"Current price: ₹{current_price:.2f}. {fund_ctx}\n\n"
        f"Our ML model signal: {prediction_signal if prediction_signal else 'Not available'}\n\n"
        f"Please provide:\n"
        f"1. **What the company does**: A 2-sentence description of the company's business.\n"
        f"2. **Recent sentiment**: What is the general market sentiment around this stock? "
        f"Consider recent events, sector trends, and market conditions.\n"
        f"3. **Key strengths and risks**: 2 bullet points each.\n"
        f"4. **Investment thesis**: In 2-3 sentences, why should an investor consider this stock, "
        f"or why should they be cautious?\n"
        f"Be specific to {stock_name} and the Indian market context."
    )
    return _call_groq(prompt, max_tokens=700)


def get_news_sentiment(ticker: str, stock_name: str) -> str:
    """
    Get news-based sentiment analysis for a stock.
    Note: This uses Groq's knowledge - for real-time news, integrate a news API.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    prompt = (
        f"Analyze the recent news sentiment for {stock_name} ({ticker}) on the Indian stock market. "
        f"Today date: {today}. "
        f"Based on your knowledge of recent events and market conditions:\n\n"
        f"1. **Overall Sentiment**: Is the sentiment Bullish, Bearish, or Neutral?\n"
        f"2. **Key factors affecting sentiment**: List 3 recent factors or events.\n"
        f"3. **Sector outlook**: How is the sector performing?\n"
        f"4. **Risk events to watch**: Any upcoming events that could impact the stock?\n\n"
        f"If you don't have recent specific news, provide general sector and market context "
        f"that would affect this stock."
    )
    return _call_groq(prompt, max_tokens=500)


def get_groq_strategy(ticker: str, stock_name: str, prediction_data: dict) -> str:
    """
    Get Groq's independent strategy recommendation based on visible data.

    This does NOT influence the ML prediction — it provides a separate
    AI perspective using fundamentals, indicators, and market context.
    """
    # Build context from prediction data
    ctx_parts = []

    pred_return = prediction_data.get("predicted_return", 0)
    signal = prediction_data.get("signal", "HOLD")
    confidence = prediction_data.get("confidence", 50)
    ctx_parts.append(
        f"ML Prediction: {signal} (predicted return: {pred_return:.3f}%, confidence: {confidence:.0f}%)"
    )

    model_preds = prediction_data.get("model_predictions", {})
    if model_preds:
        ctx_parts.append(
            "Individual model predictions: "
            + ", ".join(f"{k}: {v:.3f}%" for k, v in model_preds.items())
        )

    fund = prediction_data.get("fundamentals", {})
    if fund:
        fund_items = []
        if fund.get("pe_ratio"):
            fund_items.append(f"P/E: {fund['pe_ratio']:.1f}")
        if fund.get("pb_ratio"):
            fund_items.append(f"P/B: {fund['pb_ratio']:.1f}")
        if fund.get("roe"):
            fund_items.append(f"ROE: {fund['roe']*100:.1f}%")
        if fund.get("debt_to_equity"):
            fund_items.append(f"D/E: {fund['debt_to_equity']:.1f}")
        if fund.get("dividend_yield"):
            fund_items.append(f"Div Yield: {fund['dividend_yield']*100:.2f}%")
        if fund.get("revenue_growth"):
            fund_items.append(f"Rev Growth: {fund['revenue_growth']*100:.1f}%")
        if fund.get("beta"):
            fund_items.append(f"Beta: {fund['beta']:.2f}")
        if fund_items:
            ctx_parts.append("Fundamentals: " + ", ".join(fund_items))

    current_price = prediction_data.get("current_price", 0)
    target = prediction_data.get("target_price", 0)
    sl = prediction_data.get("stop_loss", 0)
    ctx_parts.append(
        f"Current price: ₹{current_price:.2f}, Target: ₹{target:.2f}, Stop Loss: ₹{sl:.2f}"
    )

    atr = prediction_data.get("atr_pct", 0)
    vol_ratio = prediction_data.get("volume_ratio", 1)
    ctx_parts.append(f"ATR%: {atr:.2f}%, Volume Ratio: {vol_ratio:.2f}")

    context = "\n".join(ctx_parts)

    prompt = (
        f"You are analyzing {stock_name} ({ticker}) on the Indian stock market (NSE/BSE). "
        f"Here is the current data:\n\n{context}\n\n"
        f"Based on this data, provide your independent strategy assessment in 4-5 sentences. "
        f"Consider: Is the stock overvalued/undervalued? Is momentum favorable? "
        f"What risk factors should the investor watch? "
        f"What would be a prudent approach today? "
        f"Be concise and specific to THIS stock's current situation."
    )
    return _call_groq(prompt, max_tokens=400)


def get_combined_strategy(
    ticker: str, stock_name: str, prediction_data: dict, groq_strategy: str
) -> str:
    """
    Get a combined recommendation merging ML prediction + Groq analysis.

    Synthesizes both perspectives into a unified, balanced recommendation.
    """
    signal = prediction_data.get("signal", "HOLD")
    pred_return = prediction_data.get("predicted_return", 0)
    confidence = prediction_data.get("confidence", 50)

    prompt = (
        f"For {stock_name} ({ticker}), two analyses have been done:\n\n"
        f"**ML Model Prediction**: Signal={signal}, Expected Return={pred_return:.3f}%, "
        f"Confidence={confidence:.0f}%\n\n"
        f"**Groq AI Analysis**: {groq_strategy}\n\n"
        f"Synthesize both perspectives into a single, balanced recommendation in 3-4 sentences. "
        f"If they agree, emphasize the conviction. If they disagree, explain the conflict "
        f"and suggest a cautious approach. Be specific about entry/exit considerations. "
        f"End with a clear summary: Lean BUY, Lean SELL, or WAIT."
    )
    return _call_groq(prompt, max_tokens=350)


def explain_risk_term(term: str, context: str = "") -> str:
    """Explain a portfolio/risk analytics term in practical language."""
    prompt = (
        f"Explain the risk analytics term '{term}' for an Indian equity investor. "
        f"Give: 1) simple definition, 2) how to read high/low values, "
        f"3) one practical action point. "
        f"Keep it concise but useful. Context: {context or 'Portfolio risk dashboard'}."
    )
    return _call_groq(prompt, max_tokens=350)


def get_groq_price_forecast(
    ticker: str,
    stock_name: str,
    open_price: float,
    strategy_predicted_price: float,
    current_price: float,
    sentiment_text: str = "",
) -> dict:
    """
    Ask Groq for a JSON-only AI price forecast from current context.
    Returns dict with keys: ai_predicted_price, outlook, rationale.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    prompt = (
        f"You are analyzing {stock_name} ({ticker}) on NSE. "
        f"Today date: {today}. "
        f"Open price: {open_price:.2f}. "
        f"Strategy predicted price (before market): {strategy_predicted_price:.2f}. "
        f"Current price: {current_price:.2f}. "
        f"News/sentiment context: {sentiment_text[:1200]}. "
        f"Return ONLY valid JSON with keys: "
        f"ai_predicted_price (number), outlook (Bullish/Bearish/Neutral), rationale (string <= 90 words). "
        f"Keep ai_predicted_price realistic, within +/-8% of current price."
    )
    raw = _call_groq(prompt, max_tokens=350)

    payload = None
    try:
        payload = json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                payload = json.loads(m.group(0))
            except Exception:
                payload = None

    if not isinstance(payload, dict):
        return {
            "ai_predicted_price": None,
            "outlook": "Unavailable",
            "rationale": (
                raw[:240] if isinstance(raw, str) else "AI forecast unavailable"
            ),
            "source": "fallback_non_json",
        }

    ai_price = payload.get(
        "ai_predicted_price", strategy_predicted_price or current_price
    )
    try:
        ai_price = float(ai_price)
    except Exception:
        ai_price = 0.0

    if ai_price <= 0:
        return {
            "ai_predicted_price": None,
            "outlook": str(payload.get("outlook", "Unavailable"))[:20],
            "rationale": str(payload.get("rationale", ""))[:600],
            "source": "fallback_invalid_price",
        }

    base = current_price if current_price > 0 else strategy_predicted_price
    if base > 0:
        lo, hi = base * 0.92, base * 1.08
        ai_price = min(max(ai_price, lo), hi)

    return {
        "ai_predicted_price": round(ai_price, 2),
        "outlook": str(payload.get("outlook", "Neutral"))[:20],
        "rationale": str(payload.get("rationale", ""))[:600],
        "source": "groq_json",
    }


def explain_model(model_name: str) -> str:
    """Explain a model in practical trading terms."""
    prompt = (
        f"Explain the ML model '{model_name}' used in stock prediction. "
        f"Give 1) what it does, 2) strengths, 3) weaknesses, "
        f"4) when trader should trust it less. Keep it concise."
    )
    return _call_groq(prompt, max_tokens=320)


def stock_chat_response(
    ticker: str,
    stock_name: str,
    question: str,
    prediction_data: dict | None = None,
    indicator_snapshot: dict | None = None,
) -> str:
    """Answer contextual stock question with current model and indicator context."""
    prediction_data = prediction_data or {}
    indicator_snapshot = indicator_snapshot or {}
    prompt = (
        f"You are a trading assistant for {stock_name} ({ticker}). "
        f"User question: {question}\n\n"
        f"Prediction context: signal={prediction_data.get('signal')}, "
        f"predicted_return={prediction_data.get('predicted_return')}, "
        f"confidence={prediction_data.get('confidence')}, "
        f"model_agreement={prediction_data.get('model_agreement')}.\n"
        f"Indicators: {json.dumps(indicator_snapshot)[:1000]}.\n\n"
        f"Answer in short bullets: meaning now, risk, and one actionable next step. "
        f"Do not give guaranteed returns."
    )
    return _call_groq(prompt, max_tokens=450)


def portfolio_profit_suggestion(summary: dict) -> str:
    """Suggest practical improvement steps for portfolio profitability."""
    prompt = (
        "You are assisting with an Indian equity portfolio. "
        f"Summary: {json.dumps(summary)[:1400]}\n\n"
        "Give a concise response with:\n"
        "1) current health,\n"
        "2) top 3 improvements to increase risk-adjusted profit,\n"
        "3) what to avoid,\n"
        "4) one immediate action.\n"
        "No guaranteed claims."
    )
    return _call_groq(prompt, max_tokens=520)


def suggest_ticker_shortlist(candidates: list[dict]) -> str:
    """Suggest a concise ticker shortlist from model-ranked candidates."""
    prompt = (
        "You are reviewing model-ranked NSE tickers for short-term trading. "
        f"Candidates JSON: {json.dumps(candidates)[:2200]}\n\n"
        "Return short bullets:\n"
        "1) Top 3 tickers to prioritize and why,\n"
        "2) 2 tickers to avoid and why,\n"
        "3) One risk-control rule for all picks.\n"
        "No guaranteed claims."
    )
    return _call_groq(prompt, max_tokens=480)


def review_trade_plan(
    ticker: str,
    stock_name: str,
    entry_price: float,
    quantity: float,
    current_price: float,
    signal: str,
    predicted_return_pct: float,
    confidence: float,
    agreement: float,
    sentiment_text: str,
) -> str:
    """Review user-selected trade plan and provide corrective suggestions."""
    prompt = (
        f"Review this planned trade for {stock_name} ({ticker}) on NSE.\n"
        f"Planned entry price: {entry_price:.2f}, quantity: {quantity:.2f}.\n"
        f"Current market price: {current_price:.2f}.\n"
        f"Model signal: {signal}, predicted_return={predicted_return_pct:.3f}%, "
        f"confidence={confidence:.1f}%, agreement={agreement:.1f}%.\n"
        f"News sentiment context: {sentiment_text[:1000]}.\n\n"
        "Answer in sections:\n"
        "- Is this plan aligned or misaligned with current context?\n"
        "- Top 3 risks in this specific plan.\n"
        "- A safer alternative (entry zone, position sizing, invalidation).\n"
        "Do not promise profits."
    )
    return _call_groq(prompt, max_tokens=620)


def ai_risk_assessment(summary: dict) -> str:
    """Separate AI page narrative: portfolio risk interpretation."""
    prompt = (
        "You are generating an AI risk report for an Indian equity portfolio. "
        f"Portfolio summary JSON: {json.dumps(summary)[:1800]}\n\n"
        "Provide:\n"
        "1) Current risk posture,\n"
        "2) Concentration/liquidity concerns,\n"
        "3) Drawdown and scenario risks,\n"
        "4) 3 concrete actions to reduce downside without over-trading.\n"
        "Keep concise and practical. No guaranteed outcomes."
    )
    return _call_groq(prompt, max_tokens=650)
