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
if not GROQ_API_KEY:
    import warnings
    warnings.warn(
        "GROQ_API_KEY not set in .env — AI explanations will be unavailable. "
        "Get a key at https://console.groq.com/keys",
        stacklevel=2,
    )
GROQ_MODEL = "llama-3.3-70b-versatile"
CACHE_DIR = Path(__file__).parent.parent / "cache" / "groq_explanations"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Rate limit ──────────────────────────────────────
_last_call_time = 0.0
MIN_CALL_INTERVAL = 1.0  # seconds between API calls


def _get_client():
    """Get Groq client instance or None if unavailable."""
    if Groq is None:
        return None
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception:
        return None


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
    path.write_text(json.dumps({"text": text, "ts": time.time()}))


def _call_groq(prompt: str, max_tokens: int = 500) -> str:
    """Call Groq API with rate limiting and caching."""
    global _last_call_time

    key = _cache_key(prompt)
    cached = _get_cached(key)
    if cached:
        return cached

    client = _get_client()
    if client is None:
        return "Groq API not available. Install: pip install groq"

    # Rate limit
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - elapsed)

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
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
        text = resp.choices[0].message.content.strip()
        _set_cache(key, text)
        return text
    except Exception as e:
        return f"Groq API error: {str(e)}"


# ════════════════════════════════════════════════════
# Public API — Called from Flask routes
# ════════════════════════════════════════════════════

def explain_fundamental(metric_name: str, value, ticker: str = "", stock_name: str = "") -> str:
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


def explain_greek(greek_name: str, value: float, option_type: str = "call",
                  ticker: str = "", stock_name: str = "") -> str:
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
        "interpretation": "Below 30 = oversold (potential buy), Above 70 = overbought (potential sell)"
    },
    "MACD": {
        "buy_condition": "MACD crosses above signal line",
        "sell_condition": "MACD crosses below signal line",
        "desc": "MACD shows momentum by comparing two moving averages.",
        "interpretation": "Positive = bullish momentum, Negative = bearish momentum"
    },
    "ADX": {
        "strong_trend": 25,
        "very_strong": 50,
        "desc": "ADX measures trend strength (not direction) on a 0-100 scale.",
        "interpretation": "Below 20 = weak/no trend, 20-25 = potential trend, Above 25 = strong trend"
    },
    "Stochastic": {
        "buy_below": 20,
        "sell_above": 80,
        "desc": "Stochastic compares closing price to price range over a period.",
        "interpretation": "Below 20 = oversold (potential buy), Above 80 = overbought (potential sell)"
    },
    "Bollinger Bands": {
        "buy_condition": "Price touches lower band",
        "sell_condition": "Price touches upper band",
        "desc": "Bollinger Bands show volatility with 2 standard deviations from moving average.",
        "interpretation": "Price at lower band = potential buy, Price at upper band = potential sell"
    },
    "ATR": {
        "desc": "ATR (Average True Range) measures volatility in absolute terms.",
        "interpretation": "Higher ATR = more volatile, use for stop-loss sizing (typically 1.5-2x ATR)"
    },
    "Volume Ratio": {
        "high_volume": 1.5,
        "low_volume": 0.5,
        "desc": "Volume Ratio compares current volume to average volume.",
        "interpretation": "Above 1.5 = high interest (confirms trend), Below 0.5 = low interest"
    },
    "RVOL": {
        "high_volume": 1.5,
        "desc": "RVOL (Relative Volume) compares current volume to historical average.",
        "interpretation": "Above 1.5 = unusual activity, important for breakout confirmation"
    },
    "OBV": {
        "desc": "OBV (On-Balance Volume) tracks cumulative volume flow.",
        "interpretation": "Rising OBV with rising price = bullish confirmation, Divergence = warning"
    }
}


def explain_indicator(indicator_name: str, app_value, actual_value=None,
                      ticker: str = "", stock_name: str = "") -> str:
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
                threshold_info += f"• Strong trend: Above {thresholds['strong_trend']}\n"
            if "high_volume" in thresholds:
                threshold_info += f"• High volume: Above {thresholds['high_volume']}\n"
            threshold_info += f"\nInterpretation: {thresholds.get('interpretation', '')}"
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


def get_stock_overview(ticker: str, stock_name: str, fundamentals: dict = None,
                       current_price: float = 0, prediction_signal: str = "") -> str:
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
            mc = fundamentals['market_cap']
            mc_str = f"₹{mc/1e12:.2f}T" if mc > 1e12 else f"₹{mc/1e9:.2f}B"
            fund_items.append(f"Market Cap: {mc_str}")
        if fundamentals.get("revenue_growth"):
            fund_items.append(f"Revenue Growth: {fundamentals['revenue_growth']}%")
        if fundamentals.get("profit_margin"):
            fund_items.append(f"Profit Margin: {fundamentals['profit_margin']}%")
        if fundamentals.get("debt_to_equity"):
            fund_items.append(f"Debt/Equity: {fundamentals['debt_to_equity']:.1f}")
        if fundamentals.get("fifty_two_high") and fundamentals.get("fifty_two_low"):
            fund_items.append(f"52W Range: ₹{fundamentals['fifty_two_low']:.0f} - ₹{fundamentals['fifty_two_high']:.0f}")
        if fundamentals.get("analyst_upside"):
            fund_items.append(f"Analyst Target Upside: {fundamentals['analyst_upside']:.1f}%")
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
    ctx_parts.append(f"ML Prediction: {signal} (predicted return: {pred_return:.3f}%, confidence: {confidence:.0f}%)")

    model_preds = prediction_data.get("model_predictions", {})
    if model_preds:
        ctx_parts.append("Individual model predictions: " + ", ".join(
            f"{k}: {v:.3f}%" for k, v in model_preds.items()
        ))

    fund = prediction_data.get("fundamentals", {})
    if fund:
        fund_items = []
        if fund.get("pe_ratio"): fund_items.append(f"P/E: {fund['pe_ratio']:.1f}")
        if fund.get("pb_ratio"): fund_items.append(f"P/B: {fund['pb_ratio']:.1f}")
        if fund.get("roe"): fund_items.append(f"ROE: {fund['roe']*100:.1f}%")
        if fund.get("debt_to_equity"): fund_items.append(f"D/E: {fund['debt_to_equity']:.1f}")
        if fund.get("dividend_yield"): fund_items.append(f"Div Yield: {fund['dividend_yield']*100:.2f}%")
        if fund.get("revenue_growth"): fund_items.append(f"Rev Growth: {fund['revenue_growth']*100:.1f}%")
        if fund.get("beta"): fund_items.append(f"Beta: {fund['beta']:.2f}")
        if fund_items:
            ctx_parts.append("Fundamentals: " + ", ".join(fund_items))

    current_price = prediction_data.get("current_price", 0)
    target = prediction_data.get("target_price", 0)
    sl = prediction_data.get("stop_loss", 0)
    ctx_parts.append(f"Current price: ₹{current_price:.2f}, Target: ₹{target:.2f}, Stop Loss: ₹{sl:.2f}")

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


def get_combined_strategy(ticker: str, stock_name: str,
                          prediction_data: dict, groq_strategy: str) -> str:
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
            "rationale": raw[:240] if isinstance(raw, str) else "AI forecast unavailable",
            "source": "fallback_non_json",
        }

    ai_price = payload.get("ai_predicted_price", strategy_predicted_price or current_price)
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
