"""
Live Prediction Pipeline
=========================
Loads trained ML models and generates real-time predictions
for the Flask web application.

Workflow:
  1. Load saved XGBoost, LightGBM, LSTM, Transformer, Ridge ensemble
  2. Download 2 years of OHLCV data via yfinance
  3. Run FeaturePipeline.run_single() to produce feature matrix
  4. Generate base model predictions (last row only)
  5. Stack through Ridge meta-learner → final ensemble prediction
  6. Translate predicted return into actionable signal

The predicted value is the next-day return (%) from the regression models.
"""

import sys
import os
import json
import re
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch  # Initialize PyTorch early — prevents segfault when loading after statsmodels/arch

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from groq import Groq
except ImportError:
    Groq = None

# Project imports
from ..utils.constants import MODELS_DIR, PROJECT_ROOT
from ..models.xgboost_model import XGBoostModel
from ..models.lightgbm_model import LightGBMModel
from ..models.lstm_model import LSTMModel
from ..models.transformer_model import TransformerModel
from ..models.arima_model import ARIMAModel
from ..models.garch_model import GARCHModel
from ..models.ensemble import EnsembleModel
from ..features.pipeline import FeaturePipeline
from ..features.fundamentals import FundamentalAnalyzer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = 9  # 9:15 AM IST
MARKET_OPEN_MIN = 15
MARKET_CLOSE = 15  # 3:30 PM IST
MARKET_CLOSE_MIN = 30
CACHE_DIR = PROJECT_ROOT / "cache"
log = logging.getLogger(__name__)

SENTIMENT_WEIGHTS = {"day0": 0.35, "day1": 0.25, "day2": 0.15, "day3": 0.05}
ENSEMBLE_WEIGHT = float(os.environ.get("ENSEMBLE_WEIGHT", "0.80"))
SENTIMENT_WEIGHT = float(os.environ.get("SENTIMENT_WEIGHT", "0.20"))
SENTIMENT_RETURN_SCALE = float(os.environ.get("SENTIMENT_RETURN_SCALE", "0.05"))
SENTIMENT_LOOKBACK_DAYS = int(os.environ.get("SENTIMENT_LOOKBACK_DAYS", "3"))
SENTIMENT_CACHE_TTL_SECONDS = int(os.environ.get("SENTIMENT_CACHE_TTL_SECONDS", "1800"))
NEGATIVE_SENTIMENT_CUTOFF = float(os.environ.get("NEGATIVE_SENTIMENT_CUTOFF", "-0.45"))
GROQ_SENTIMENT_MODEL = os.environ.get("GROQ_SENTIMENT_MODEL", "llama-3.3-70b-versatile")
ENABLE_GROQ_NEWS_SENTIMENT = os.environ.get(
    "ENABLE_GROQ_NEWS_SENTIMENT", "1"
).strip().lower() not in {"0", "false", "no"}
GROQ_SENTIMENT_MAX_CALLS_PER_MINUTE = int(
    os.environ.get("GROQ_SENTIMENT_MAX_CALLS_PER_MINUTE", "8")
)


def _sanitize_predicted_return(value: float, base_preds: dict) -> float:
    """
    Normalize model output to a sane decimal return.

    Rules:
      - None/NaN/inf => 0.0
      - If |value| > 2.0, treat it as percent units and divide by 100
      - Cap to +/- 0.5 (50%)
      - Log warning when conversion/capping happens
    """
    if value is None:
        return 0.0

    try:
        raw_value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(raw_value):
        return 0.0

    sanitized = raw_value
    converted = False
    capped = False

    if abs(sanitized) > 2.0:
        sanitized = sanitized / 100.0
        converted = True

    clipped = float(np.clip(sanitized, -0.5, 0.5))
    if not np.isclose(clipped, sanitized):
        sanitized = clipped
        capped = True

    if converted or capped:
        log.warning(
            "Sanitized predicted_return raw=%s final=%s converted=%s capped=%s base_preds=%s",
            round(raw_value, 6),
            round(sanitized, 6),
            converted,
            capped,
            base_preds,
        )

    return sanitized


# ---------------------------------------------------------------------------
# Market status
# ---------------------------------------------------------------------------
def get_market_status() -> dict:
    """
    Determine current market status based on IST time.

    Returns
    -------
    dict with keys:
        status : str — 'pre_market', 'market_open', 'after_hours', 'weekend'
        ist_now : datetime
        next_open : str
        description : str
    """
    now = datetime.now(IST)
    weekday = now.weekday()  # 0=Mon, 6=Sun

    market_open_time = now.replace(
        hour=MARKET_OPEN, minute=MARKET_OPEN_MIN, second=0, microsecond=0
    )
    market_close_time = now.replace(
        hour=MARKET_CLOSE, minute=MARKET_CLOSE_MIN, second=0, microsecond=0
    )

    if weekday >= 5:  # Saturday or Sunday
        days_until_monday = 7 - weekday
        next_open = (now + timedelta(days=days_until_monday)).replace(
            hour=MARKET_OPEN, minute=MARKET_OPEN_MIN, second=0, microsecond=0
        )
        return {
            "status": "weekend",
            "ist_now": now,
            "next_open": next_open.strftime("%a %d %b %I:%M %p IST"),
            "description": "Market closed (weekend)",
        }

    if now < market_open_time:
        return {
            "status": "pre_market",
            "ist_now": now,
            "next_open": market_open_time.strftime("%I:%M %p IST"),
            "description": f"Market opens at {market_open_time.strftime('%I:%M %p')} IST",
        }

    if now > market_close_time:
        if weekday == 4:  # Friday
            next_open = (now + timedelta(days=3)).replace(
                hour=MARKET_OPEN, minute=MARKET_OPEN_MIN, second=0, microsecond=0
            )
        else:
            next_open = (now + timedelta(days=1)).replace(
                hour=MARKET_OPEN, minute=MARKET_OPEN_MIN, second=0, microsecond=0
            )
        return {
            "status": "after_hours",
            "ist_now": now,
            "next_open": next_open.strftime("%a %d %b %I:%M %p IST"),
            "description": "Market closed (after hours)",
        }

    return {
        "status": "market_open",
        "ist_now": now,
        "next_open": "Now",
        "description": "Market is OPEN",
    }


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
class PredictionCache:
    """Simple JSON-based disk cache for predictions."""

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_seconds: int | None = None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "predictions.json"
        env_ttl = os.environ.get("PREDICTION_CACHE_TTL_SECONDS")
        if ttl_seconds is None:
            try:
                ttl_seconds = int(env_ttl) if env_ttl is not None else 6 * 60 * 60
            except ValueError:
                ttl_seconds = 6 * 60 * 60
        self.ttl_seconds = max(0, int(ttl_seconds))
        self._data = self._load()

    def _load(self) -> dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self):
        with open(self.cache_file, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def get(self, ticker: str) -> dict | None:
        """Get cached prediction. Returns None if stale."""
        entry = self._data.get(ticker)
        if entry is None:
            return None
        ts = entry.get("timestamp")
        if not ts:
            return None
        try:
            cached_time = datetime.fromisoformat(ts)
        except Exception:
            return None
        now = datetime.now(IST)
        # Make cached_time offset-aware if needed
        if cached_time.tzinfo is None:
            cached_time = cached_time.replace(tzinfo=IST)
        if self.ttl_seconds > 0 and now - cached_time > timedelta(
            seconds=self.ttl_seconds
        ):
            # Drop stale entries from disk so stale/placeholder values do not linger.
            self._data.pop(ticker, None)
            self._save()
            return None
        return entry

    def set(self, ticker: str, prediction: dict):
        """Cache a prediction result."""
        prediction["timestamp"] = datetime.now(IST).isoformat()
        self._data[ticker] = prediction
        self._save()

    def invalidate(self, ticker: str | None = None):
        """Invalidate a specific ticker cache key or clear all cached predictions."""
        if ticker is None:
            self.clear()
            return
        if ticker in self._data:
            self._data.pop(ticker, None)
            self._save()

    def clear(self):
        self._data = {}
        self._save()


# ---------------------------------------------------------------------------
# Live Predictor
# ---------------------------------------------------------------------------
class LivePredictor:
    """
    Generates live ML predictions by loading trained models
    and running the feature pipeline on fresh data.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models = {}
        self.ensemble = None
        self.pipeline = FeaturePipeline()
        self.cache = PredictionCache()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self._loaded = False
        self._load_started_at: str | None = None
        self._load_completed_at: str | None = None
        self._load_steps: list[dict] = []
        self._recent_model_outputs: dict[str, list[float]] = {}
        self._stale_models: set[str] = set()
        self._groq_client = None
        self._groq_last_call = 0.0
        self._groq_sentiment_call_times: list[float] = []
        self._sentiment_cache_file = CACHE_DIR / "news_sentiment_cache.json"
        self._sentiment_cache: dict = self._load_sentiment_cache()

    def _load_sentiment_cache(self) -> dict:
        if not self._sentiment_cache_file.exists():
            return {}
        try:
            data = json.loads(self._sentiment_cache_file.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_sentiment_cache(self) -> None:
        try:
            self._sentiment_cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._sentiment_cache_file.write_text(
                json.dumps(self._sentiment_cache, indent=2, default=str)
            )
        except Exception:
            pass

    def _get_groq_client(self):
        if self._groq_client is not None:
            return self._groq_client
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key or Groq is None:
            return None
        try:
            self._groq_client = Groq(api_key=api_key)
        except Exception:
            self._groq_client = None
        return self._groq_client

    def _groq_sentiment_quota_available(self) -> bool:
        if GROQ_SENTIMENT_MAX_CALLS_PER_MINUTE <= 0:
            return False
        now = time.time()
        self._groq_sentiment_call_times = [
            t for t in self._groq_sentiment_call_times if (now - t) < 60.0
        ]
        return len(self._groq_sentiment_call_times) < GROQ_SENTIMENT_MAX_CALLS_PER_MINUTE

    @staticmethod
    def _sentiment_decay_weight(days_ago: int) -> float:
        if days_ago <= 0:
            return SENTIMENT_WEIGHTS["day0"]
        if days_ago == 1:
            return SENTIMENT_WEIGHTS["day1"]
        if days_ago == 2:
            return SENTIMENT_WEIGHTS["day2"]
        return SENTIMENT_WEIGHTS["day3"]

    @staticmethod
    def _parse_news_datetime(raw_value) -> datetime | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, (int, float)):
            try:
                return datetime.fromtimestamp(float(raw_value), tz=IST)
            except Exception:
                return None
        if isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                return None
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=IST)
                return parsed.astimezone(IST)
            except Exception:
                return None
        return None

    def _fetch_recent_news_articles(
        self,
        ticker: str,
        prediction_dt: datetime,
        max_days: int = SENTIMENT_LOOKBACK_DAYS,
        max_items: int = 12,
    ) -> list[dict]:
        if yf is None:
            return []
        try:
            news_items = yf.Ticker(ticker).news or []
        except Exception:
            return []

        articles: list[dict] = []
        for row in news_items:
            if not isinstance(row, dict):
                continue
            published_dt = self._parse_news_datetime(
                row.get("providerPublishTime")
                or row.get("published")
                or row.get("published_at")
            )
            if published_dt is None:
                continue
            days_ago = (prediction_dt.date() - published_dt.astimezone(IST).date()).days
            if days_ago < 0 or days_ago > max_days:
                continue
            title = str(row.get("title", "") or "").strip()
            if not title:
                continue
            articles.append(
                {
                    "published_at": published_dt.isoformat(),
                    "days_ago": int(days_ago),
                    "title": title,
                    "summary": str(row.get("summary", "") or "").strip(),
                    "publisher": str(
                        row.get("publisher")
                        or row.get("provider")
                        or row.get("source")
                        or "unknown"
                    ).strip(),
                    "link": str(row.get("link", "") or "").strip(),
                }
            )
            if len(articles) >= max_items:
                break
        return articles

    @staticmethod
    def _fallback_headline_sentiment(text: str) -> tuple[float, int]:
        t = str(text or "").lower()
        if not t:
            return 0.0, 0
        positive_tokens = [
            "beat",
            "growth",
            "surge",
            "upgrade",
            "profit",
            "expands",
            "bullish",
            "record",
            "strong",
            "wins",
            "gain",
        ]
        negative_tokens = [
            "miss",
            "fall",
            "loss",
            "downgrade",
            "lawsuit",
            "probe",
            "cuts",
            "weak",
            "drop",
            "bearish",
            "decline",
        ]
        pos_hits = sum(1 for tok in positive_tokens if tok in t)
        neg_hits = sum(1 for tok in negative_tokens if tok in t)
        net = pos_hits - neg_hits
        if net == 0:
            return 0.0, 0
        polarity = 1 if net > 0 else -1
        score = min(1.0, 0.2 + (abs(net) * 0.15))
        return score, polarity

    @staticmethod
    def _extract_json_object(text: str) -> dict:
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _score_news_articles_with_groq(
        self, ticker: str, articles: list[dict]
    ) -> list[dict]:
        if not articles:
            return []

        # Stable cache key per ticker + article window.
        key_basis = {
            "ticker": ticker,
            "articles": [
                {
                    "published_at": a.get("published_at"),
                    "title": a.get("title"),
                    "publisher": a.get("publisher"),
                }
                for a in articles
            ],
        }
        cache_key = hashlib.sha1(
            json.dumps(key_basis, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cached = self._sentiment_cache.get(cache_key)
        now_ts = time.time()
        if isinstance(cached, dict):
            ts = float(cached.get("ts", 0) or 0)
            if (now_ts - ts) < SENTIMENT_CACHE_TTL_SECONDS:
                items = cached.get("items", [])
                if isinstance(items, list):
                    return items

        client = self._get_groq_client()
        if (not ENABLE_GROQ_NEWS_SENTIMENT) or client is None or (not self._groq_sentiment_quota_available()):
            scored = []
            for idx, article in enumerate(articles):
                score, polarity = self._fallback_headline_sentiment(
                    f"{article.get('title', '')}. {article.get('summary', '')}"
                )
                scored.append(
                    {
                        "idx": idx,
                        "sentiment_score": round(float(score), 4),
                        "sentiment_polarity": int(polarity),
                    }
                )
            return scored

        prompt_rows = []
        for idx, article in enumerate(articles):
            prompt_rows.append(
                {
                    "idx": idx,
                    "date": article.get("published_at"),
                    "source": article.get("publisher"),
                    "title": article.get("title"),
                    "summary": article.get("summary", "")[:240],
                }
            )

        prompt = (
            "Score each article sentiment for Indian equity trading.\n"
            "Return strict JSON only with schema: "
            '{"items":[{"idx":0,"sentiment_score":0.0,"sentiment_polarity":0}]}.\n'
            "Rules:\n"
            "- sentiment_score is absolute sentiment intensity from 0 to 1.\n"
            "- sentiment_polarity must be 1 for positive, -1 for negative, 0 for neutral.\n"
            "- Use article headline/summary only, no extra commentary.\n"
            f"Ticker: {ticker}\n"
            f"Articles: {json.dumps(prompt_rows)}"
        )

        # Small safety delay to avoid 429s in fast loops.
        elapsed = time.time() - self._groq_last_call
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)

        scored: list[dict] = []
        try:
            self._groq_sentiment_call_times.append(time.time())
            resp = client.chat.completions.create(
                model=GROQ_SENTIMENT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON generator for financial sentiment scoring."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=800,
            )
            self._groq_last_call = time.time()
            text = str(resp.choices[0].message.content or "").strip()
            payload = self._extract_json_object(text)
            items = payload.get("items", []) if isinstance(payload, dict) else []
            for row in items if isinstance(items, list) else []:
                if not isinstance(row, dict):
                    continue
                idx = int(row.get("idx", -1))
                score = float(row.get("sentiment_score", 0) or 0)
                polarity = int(float(row.get("sentiment_polarity", 0) or 0))
                if idx < 0 or idx >= len(articles):
                    continue
                score = float(np.clip(abs(score), 0.0, 1.0))
                if polarity > 0:
                    polarity = 1
                elif polarity < 0:
                    polarity = -1
                else:
                    polarity = 0
                scored.append(
                    {
                        "idx": idx,
                        "sentiment_score": round(score, 4),
                        "sentiment_polarity": polarity,
                    }
                )
        except Exception as exc:
            log.debug("Groq sentiment scoring failed for %s: %s", ticker, exc)

        # Fallback per missing rows.
        by_idx = {int(s.get("idx", -1)): s for s in scored}
        for idx, article in enumerate(articles):
            if idx in by_idx:
                continue
            score, polarity = self._fallback_headline_sentiment(
                f"{article.get('title', '')}. {article.get('summary', '')}"
            )
            by_idx[idx] = {
                "idx": idx,
                "sentiment_score": round(float(score), 4),
                "sentiment_polarity": int(polarity),
            }

        final_rows = [by_idx[i] for i in sorted(by_idx)]
        self._sentiment_cache[cache_key] = {"ts": now_ts, "items": final_rows}
        self._save_sentiment_cache()
        return final_rows

    def _compute_weighted_sentiment_signal(
        self,
        ticker: str,
        prediction_dt: datetime,
    ) -> dict:
        """
        Compute time-decayed sentiment signal:
          weighted_sentiment = sum(day_avg_sentiment * day_weight) / sum(day_weights)
        where day_weight uses SENTIMENT_WEIGHTS and day_avg_sentiment is signed.
        """
        articles = self._fetch_recent_news_articles(
            ticker=ticker,
            prediction_dt=prediction_dt,
            max_days=SENTIMENT_LOOKBACK_DAYS,
        )
        if not articles:
            return {
                "weighted_sentiment_raw": 0.0,
                "weighted_sentiment_decimal": 0.0,
                "article_count": 0,
                "sources": [],
                "weights_used": {},
            }

        scored = self._score_news_articles_with_groq(ticker, articles)
        by_idx = {int(r.get("idx", -1)): r for r in scored}

        # Edge case: multiple articles on same day are averaged before day-weighting.
        day_scores: dict[int, list[float]] = defaultdict(list)
        source_set = set()
        for idx, article in enumerate(articles):
            srow = by_idx.get(idx, {})
            score = float(srow.get("sentiment_score", 0) or 0)
            polarity = int(float(srow.get("sentiment_polarity", 0) or 0))
            score = float(np.clip(abs(score), 0.0, 1.0))
            if polarity > 0:
                polarity = 1
            elif polarity < 0:
                polarity = -1
            else:
                polarity = 0
            day = int(article.get("days_ago", 0) or 0)
            if day < 0 or day > SENTIMENT_LOOKBACK_DAYS:
                continue
            day_scores[day].append(score * polarity)
            src = str(article.get("publisher", "") or "").strip()
            if src:
                source_set.add(src)

        if not day_scores:
            return {
                "weighted_sentiment_raw": 0.0,
                "weighted_sentiment_decimal": 0.0,
                "article_count": 0,
                "sources": sorted(source_set),
                "weights_used": {},
            }

        numerator = 0.0
        denominator = 0.0
        weights_used = {}
        for day, values in day_scores.items():
            if not values:
                continue
            day_avg = float(np.mean(values))
            day_weight = self._sentiment_decay_weight(day)
            numerator += day_avg * day_weight
            denominator += day_weight
            weights_used[f"day{day}"] = round(day_weight, 4)

        if denominator <= 0:
            weighted_raw = 0.0
        else:
            weighted_raw = numerator / denominator

        weighted_raw = float(np.clip(weighted_raw, -1.0, 1.0))
        weighted_decimal = float(
            np.clip(weighted_raw * SENTIMENT_RETURN_SCALE, -0.5, 0.5)
        )
        return {
            "weighted_sentiment_raw": round(weighted_raw, 6),
            "weighted_sentiment_decimal": round(weighted_decimal, 6),
            "article_count": sum(len(v) for v in day_scores.values()),
            "sources": sorted(source_set),
            "weights_used": weights_used,
        }

    def _record_model_output(self, model_name: str, value: float) -> None:
        """Track recent model outputs and flag near-constant models as stale."""
        hist = self._recent_model_outputs.setdefault(model_name, [])
        hist.append(float(value))
        if len(hist) > 80:
            del hist[:-80]

        if len(hist) < 20:
            return

        arr = np.array(hist[-40:], dtype=float)
        if arr.size < 20:
            return

        rounded = np.round(arr, 6)
        mode_count = max(np.unique(rounded, return_counts=True)[1])
        mode_ratio = mode_count / len(rounded)
        std_val = float(np.std(arr))

        if mode_ratio >= 0.85 or std_val < 1e-5:
            if model_name not in self._stale_models:
                self._stale_models.add(model_name)
                log.warning(
                    "Detected near-constant output for model=%s (std=%.8f mode_ratio=%.2f); excluding it from live ensemble inputs",
                    model_name,
                    std_val,
                    mode_ratio,
                )

    def _set_step(self, name: str, status: str, error: str | None = None) -> None:
        for step in self._load_steps:
            if step.get("name") == name:
                step["status"] = status
                if error:
                    step["error"] = error
                return
        entry = {"name": name, "status": status}
        if error:
            entry["error"] = error
        self._load_steps.append(entry)

    def get_load_status(self) -> dict:
        total = len(self._load_steps)
        loaded = sum(1 for step in self._load_steps if step.get("status") == "loaded")
        failed = sum(1 for step in self._load_steps if step.get("status") == "failed")
        in_progress = next(
            (
                step.get("name")
                for step in self._load_steps
                if step.get("status") == "loading"
            ),
            None,
        )
        return {
            "started_at": self._load_started_at,
            "completed_at": self._load_completed_at,
            "total_steps": total,
            "loaded_steps": loaded,
            "failed_steps": failed,
            "in_progress": in_progress,
            "steps": self._load_steps,
        }

    def load_models(self, force_reload: bool = False) -> dict:
        """
        Load all trained models from disk.

        Loads XGBoost, LightGBM (joblib) and LSTM, Transformer (PyTorch).
        PyTorch models are loaded first to avoid C-extension conflicts
        with statsmodels/arch library initialisation.

        The ensemble Ridge meta-learner expects all 4 base model columns:
        [xgboost, lightgbm, lstm, transformer].

        Returns
        -------
        dict : model_name → model object
        """
        if force_reload:
            self.models = {}
            self.ensemble = None
            self._loaded = False

        if self._loaded:
            return self.models

        self._load_started_at = datetime.now(IST).isoformat()
        self._load_completed_at = None
        self._load_steps = [
            {"name": "lstm", "status": "pending"},
            {"name": "transformer", "status": "pending"},
            {"name": "xgboost", "status": "pending"},
            {"name": "lightgbm", "status": "pending"},
            {"name": "arima", "status": "pending"},
            {"name": "garch", "status": "pending"},
            {"name": "ensemble", "status": "pending"},
        ]

        loaded = {}
        canonical_order = [
            "arima",
            "garch",
            "xgboost",
            "lightgbm",
            "lstm",
            "transformer",
        ]

        # ── LSTM (load PyTorch models first to avoid C-extension conflicts) ──
        lstm_path = self.models_dir / "lstm_model.pt"
        lstm_joblib = self.models_dir / "lstm_model.joblib"
        self._set_step("lstm", "loading")
        if lstm_path.exists() or lstm_joblib.exists():
            try:
                model = LSTMModel()
                model.load(lstm_path)
                loaded["lstm"] = model
                self._set_step("lstm", "loaded")
            except Exception as e:
                print(f"  ⚠ LSTM load failed: {e}")
                self._set_step("lstm", "failed", str(e))
        else:
            self._set_step("lstm", "failed", "model file not found")

        # ── Transformer ──
        transformer_path = self.models_dir / "transformer_model.pt"
        transformer_joblib = self.models_dir / "transformer_model.joblib"
        self._set_step("transformer", "loading")
        if transformer_path.exists() or transformer_joblib.exists():
            try:
                model = TransformerModel()
                model.load(transformer_path)
                loaded["transformer"] = model
                self._set_step("transformer", "loaded")
            except Exception as e:
                print(f"  ⚠ Transformer load failed: {e}")
                self._set_step("transformer", "failed", str(e))
        else:
            self._set_step("transformer", "failed", "model file not found")

        # ── XGBoost (joblib) ──
        xgb_path = self.models_dir / "xgboost_model.joblib"
        self._set_step("xgboost", "loading")
        if xgb_path.exists():
            try:
                model = XGBoostModel()
                model.load(xgb_path)
                loaded["xgboost"] = model
                print(f"  Loaded xgboost from {xgb_path}")
                self._set_step("xgboost", "loaded")
            except Exception as e:
                print(f"  ⚠ XGBoost load failed: {e}")
                self._set_step("xgboost", "failed", str(e))
        else:
            self._set_step("xgboost", "failed", "model file not found")

        # ── LightGBM (joblib) ──
        lgb_path = self.models_dir / "lightgbm_model.joblib"
        self._set_step("lightgbm", "loading")
        if lgb_path.exists():
            try:
                model = LightGBMModel()
                model.load(lgb_path)
                loaded["lightgbm"] = model
                print(f"  Loaded lightgbm from {lgb_path}")
                self._set_step("lightgbm", "loaded")
            except Exception as e:
                print(f"  ⚠ LightGBM load failed: {e}")
                self._set_step("lightgbm", "failed", str(e))
        else:
            self._set_step("lightgbm", "failed", "model file not found")

        # ── ARIMA (joblib — uses smooth() for instant restore) ──
        arima_path = self.models_dir / "arima_model.joblib"
        self._set_step("arima", "loading")
        if arima_path.exists():
            try:
                model = ARIMAModel()
                model.load(arima_path)
                loaded["arima"] = model
                self._set_step("arima", "loaded")
            except Exception as e:
                print(f"  ⚠ ARIMA load failed: {e}")
                self._set_step("arima", "failed", str(e))
        else:
            self._set_step("arima", "failed", "model file not found")

        # ── GARCH (joblib — uses fix() for instant restore) ──
        garch_path = self.models_dir / "garch_model.joblib"
        self._set_step("garch", "loading")
        if garch_path.exists():
            try:
                model = GARCHModel()
                model.load(garch_path)
                loaded["garch"] = model
                self._set_step("garch", "loaded")
            except Exception as e:
                print(f"  ⚠ GARCH load failed: {e}")
                self._set_step("garch", "failed", str(e))
        else:
            self._set_step("garch", "failed", "model file not found")

        # ── Ensemble meta-learner ──
        ens_path = self.models_dir / "ensemble_ridge_meta.joblib"
        self._set_step("ensemble", "loading")
        if ens_path.exists():
            try:
                ens = EnsembleModel()
                ens.load(ens_path)  # Custom load handles all formats

                # Ensure model_names covers canonical order
                if not ens.model_names:
                    ens.model_names = canonical_order

                # Extend weights if new models were added
                if ens.learned_weights:
                    for name in canonical_order:
                        if name not in ens.learned_weights:
                            ens.learned_weights[name] = 0.0
                    print(f"  Using saved direction-accuracy weights")
                else:
                    # Fall back: equal weight across loaded models
                    ens.learned_weights = {
                        n: 1.0 / max(len(loaded), 1) if n in loaded else 0.0
                        for n in canonical_order
                    }

                self.ensemble = ens
                print(f"  Loaded ensemble from {ens_path}")
                print(f"  Ensemble weights: {ens.learned_weights}")
                self._set_step("ensemble", "loaded")
            except Exception as e:
                print(f"  ⚠ Ensemble load failed: {e}")
                self._set_step("ensemble", "failed", str(e))
        else:
            self._set_step("ensemble", "failed", "model file not found")

        self.models = loaded
        self._loaded = True
        self._load_completed_at = datetime.now(IST).isoformat()
        # Model refresh must invalidate stale cached predictions.
        self.cache.clear()
        print(f"  ✅ All models loaded: {list(loaded.keys())}")
        return loaded

    def generate_features(self, ticker: str, period: str = "2y") -> pd.DataFrame | None:
        """
        Download data and run the feature pipeline for a ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker (e.g., "RELIANCE.NS")
        period : str
            yfinance period string

        Returns
        -------
        pd.DataFrame or None
            Feature-engineered DataFrame, or None on failure
        """
        if yf is None:
            return None

        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df is None or len(df) < 300:
                return None

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Ensure required columns
            required = ["Open", "High", "Low", "Close", "Volume"]
            for col in required:
                if col not in df.columns:
                    return None

            df = df[required].copy()
            df = df.dropna()

            # Add Returns column (needed by pipeline)
            df["Returns"] = df["Close"].pct_change()

            # Run feature pipeline (skip correlated drop — models need all features)
            feat_df = self.pipeline.run_single(df.copy(), ticker, drop_correlated=False)
            return feat_df

        except Exception as e:
            print(f"Feature generation failed for {ticker}: {e}")
            return None

    def predict_single(
        self,
        ticker: str,
        use_cache: bool = True,
    ) -> dict | None:
        """
        Generate a full ML prediction for a single ticker.

        Returns
        -------
        dict with keys:
            ticker, predicted_return, predicted_price, current_price,
            previous_close, model_predictions, ensemble_weight,
            signal, confidence, entry_price, stop_loss, target_price,
            atr_pct, volume_ratio, rvol, timestamp
        """
        # Force-refresh path should invalidate stale cached entry immediately.
        if not use_cache:
            self.cache.invalidate(ticker)

        # Check cache first
        if use_cache:
            cached = self.cache.get(ticker)
            if cached is not None:
                live = self.get_live_price(ticker)
                if live:
                    current_price = float(live.get("price", 0) or 0)
                    cached["current_price"] = round(current_price, 2)
                    cached["previous_close"] = float(
                        live.get(
                            "previous_close",
                            cached.get("previous_close", current_price),
                        )
                        or current_price
                    )
                else:
                    current_price = float(cached.get("current_price", 0) or 0)
                if current_price <= 0:
                    return None

                cached_return_pct = float(cached.get("predicted_return", 0) or 0)
                cached_return_decimal = _sanitize_predicted_return(
                    cached_return_pct / 100.0, {}
                )
                atr_pct = float(cached.get("atr_pct", 2.0) or 2.0) / 100.0
                volume_ratio = float(cached.get("volume_ratio", 1.0) or 1.0)
                rvol = float(cached.get("rvol", 1.0) or 1.0)
                model_agreement = (
                    float(cached.get("model_agreement", 50.0) or 50.0) / 100.0
                )
                signal, confidence = self._generate_signal(
                    cached_return_decimal,
                    atr_pct=atr_pct,
                    volume_ratio=volume_ratio,
                    rvol=rvol,
                    model_agreement=model_agreement,
                )
                cached["predicted_return"] = round(cached_return_decimal * 100, 4)
                cached["predicted_price"] = round(
                    current_price * (1 + cached_return_decimal), 2
                )
                assert cached["predicted_price"] == round(
                    current_price * (1 + cached_return_decimal), 2
                )
                cached["target_price"] = cached["predicted_price"]
                cached["signal"] = signal
                cached["confidence"] = round(confidence, 1)
                return cached

        # Ensure models are loaded
        if not self._loaded:
            self.load_models()

        if not self.models:
            return None

        # Generate features
        feat_df = self.generate_features(ticker)
        if feat_df is None or len(feat_df) < 60:
            return None

        # Use true live market price (prevents split-adjusted drift from long-window feature data).
        live = self.get_live_price(ticker)
        if live:
            current_price = float(live.get("price", 0) or 0)
            previous_close = float(
                live.get("previous_close", current_price) or current_price
            )
            current_volume = float(live.get("volume", 0) or 0)
        else:
            try:
                if "Close" in feat_df.columns:
                    current_price = float(feat_df["Close"].iloc[-1])
                    previous_close = (
                        float(feat_df["Close"].iloc[-2])
                        if len(feat_df) >= 2
                        else current_price
                    )
                else:
                    current_price = 0
                    previous_close = 0
                current_volume = (
                    float(feat_df["Volume"].iloc[-1])
                    if "Volume" in feat_df.columns
                    else 0
                )
            except Exception:
                current_price = 0
                previous_close = current_price
                current_volume = 0

        if current_price <= 0:
            return None

        avg_volume_30d = 0.0
        try:
            if "Volume" in feat_df.columns:
                vol_series = pd.to_numeric(feat_df["Volume"], errors="coerce").dropna()
                if not vol_series.empty:
                    avg_volume_30d = float(vol_series.tail(30).mean())
            if avg_volume_30d <= 0 and current_volume > 0:
                avg_volume_30d = float(current_volume)
        except Exception:
            avg_volume_30d = float(current_volume or 0)

        # --- Base model predictions ---
        base_preds = {}

        # ARIMA (per-ticker forecast)
        if "arima" in self.models:
            try:
                arima_model = self.models["arima"]
                fc = arima_model.forecast_ticker(ticker, arima_model.forecast_horizon)
                base_preds["arima"] = float(fc)
            except Exception as e:
                print(f"ARIMA prediction failed: {e}")

        # GARCH (per-ticker mean forecast)
        if "garch" in self.models:
            try:
                garch_model = self.models["garch"]
                mean_fc, vol_fc = garch_model.forecast_ticker(
                    ticker, garch_model.forecast_horizon
                )
                base_preds["garch"] = float(mean_fc)
            except Exception as e:
                print(f"GARCH prediction failed: {e}")

        # XGBoost (uses last row)
        if "xgboost" in self.models:
            try:
                xgb_model = self.models["xgboost"]
                # Align features to what model expects
                available = [f for f in xgb_model.feature_names if f in feat_df.columns]
                missing = [
                    f for f in xgb_model.feature_names if f not in feat_df.columns
                ]
                pred_df = feat_df[available].copy()
                for m in missing:
                    pred_df[m] = 0.0
                pred_df = pred_df[xgb_model.feature_names]
                preds = xgb_model.predict(pred_df)
                xgb_pred = float(preds[-1])
                self._record_model_output("xgboost", xgb_pred)
                if "xgboost" not in self._stale_models:
                    base_preds["xgboost"] = xgb_pred
            except Exception as e:
                print(f"XGBoost prediction failed: {e}")

        # LightGBM (uses last row)
        if "lightgbm" in self.models:
            try:
                lgb_model = self.models["lightgbm"]
                available = [f for f in lgb_model.feature_names if f in feat_df.columns]
                missing = [
                    f for f in lgb_model.feature_names if f not in feat_df.columns
                ]
                pred_df = feat_df[available].copy()
                for m in missing:
                    pred_df[m] = 0.0
                pred_df = pred_df[lgb_model.feature_names]
                preds = lgb_model.predict(pred_df)
                lgb_pred = float(preds[-1])
                self._record_model_output("lightgbm", lgb_pred)
                if "lightgbm" not in self._stale_models:
                    base_preds["lightgbm"] = lgb_pred
            except Exception as e:
                print(f"LightGBM prediction failed: {e}")

        # LSTM (uses sequence of last seq_len rows)
        if "lstm" in self.models:
            try:
                lstm_model = self.models["lstm"]
                available = [
                    f for f in lstm_model.feature_names if f in feat_df.columns
                ]
                missing = [
                    f for f in lstm_model.feature_names if f not in feat_df.columns
                ]
                pred_df = feat_df[available].copy()
                for m in missing:
                    pred_df[m] = 0.0
                pred_df = pred_df[lstm_model.feature_names]
                # LSTM needs enough rows for sequence
                if len(pred_df) >= lstm_model.seq_len:
                    preds = lstm_model.predict(pred_df)
                    if len(preds) > 0:
                        lstm_pred = float(preds[-1])
                        self._record_model_output("lstm", lstm_pred)
                        if "lstm" not in self._stale_models:
                            base_preds["lstm"] = lstm_pred
            except Exception as e:
                print(f"LSTM prediction failed: {e}")

        # Transformer (uses sequence of last seq_len rows)
        if "transformer" in self.models:
            try:
                tf_model = self.models["transformer"]
                available = [f for f in tf_model.feature_names if f in feat_df.columns]
                missing = [
                    f for f in tf_model.feature_names if f not in feat_df.columns
                ]
                pred_df = feat_df[available].copy()
                for m in missing:
                    pred_df[m] = 0.0
                pred_df = pred_df[tf_model.feature_names]
                if len(pred_df) >= tf_model.seq_len:
                    preds = tf_model.predict(pred_df)
                    if len(preds) > 0:
                        tf_pred = float(np.ravel(preds[-1])[0])
                        self._record_model_output("transformer", tf_pred)
                        if "transformer" not in self._stale_models:
                            base_preds["transformer"] = tf_pred
            except Exception as e:
                print(f"Transformer prediction failed: {e}")

        if not base_preds:
            return None

        # --- Ensemble prediction ---
        predicted_return = None
        ensemble_weights = {}

        if self.ensemble is not None and len(base_preds) >= 2:
            try:
                # Build prediction DataFrame matching ensemble's expected columns
                ens_cols = (
                    self.ensemble.model_names
                    if hasattr(self.ensemble, "model_names")
                    else list(base_preds.keys())
                )
                ens_data = {}
                for col in ens_cols:
                    ens_data[col] = [base_preds.get(col, 0.0)]
                ens_df = pd.DataFrame(ens_data)
                predicted_return = float(self.ensemble.predict(ens_df)[0])

                # Build effective weights only for models that produced predictions
                raw_weights = getattr(self.ensemble, "learned_weights", {})
                if raw_weights:
                    # Re-normalise over actually-available models
                    available_wts = {k: raw_weights.get(k, 0.0) for k in base_preds}
                    total = sum(available_wts.values())
                    if total > 0:
                        ensemble_weights = {
                            k: v / total for k, v in available_wts.items()
                        }
                    else:
                        ensemble_weights = {
                            k: 1.0 / len(base_preds) for k in base_preds
                        }
                else:
                    ensemble_weights = {k: 1.0 / len(base_preds) for k in base_preds}

                # Get direction probability if dual-learner is available
                if hasattr(self.ensemble, "predict_direction_probability"):
                    try:
                        dir_prob = float(
                            self.ensemble.predict_direction_probability(ens_df)[0]
                        )
                    except Exception:
                        dir_prob = None
                else:
                    dir_prob = None
            except Exception as e:
                print(f"Ensemble prediction failed: {e}")
                # Fallback: use direction-accuracy weighted average if weights available
                raw_weights = getattr(self.ensemble, "learned_weights", {})
                if raw_weights:
                    available_wts = {k: raw_weights.get(k, 0) for k in base_preds}
                    total = sum(available_wts.values())
                    if total > 0:
                        predicted_return = sum(
                            v * available_wts[k] / total for k, v in base_preds.items()
                        )
                        ensemble_weights = {
                            k: v / total for k, v in available_wts.items()
                        }
                    else:
                        predicted_return = float(np.mean(list(base_preds.values())))
                else:
                    predicted_return = float(np.mean(list(base_preds.values())))
                dir_prob = None
        else:
            predicted_return = float(np.mean(list(base_preds.values())))
            dir_prob = None

        # --- Blend technical prediction with time-decayed news sentiment ---
        technical_prediction = float(predicted_return or 0.0)
        sentiment_meta = self._compute_weighted_sentiment_signal(
            ticker=ticker,
            prediction_dt=datetime.now(IST),
        )
        weighted_sentiment = float(
            sentiment_meta.get("weighted_sentiment_decimal", 0.0) or 0.0
        )
        sentiment_article_count = int(sentiment_meta.get("article_count", 0) or 0)

        if sentiment_article_count > 0:
            ensemble_weight_used = float(np.clip(ENSEMBLE_WEIGHT, 0.0, 1.0))
            sentiment_weight_used = float(np.clip(SENTIMENT_WEIGHT, 0.0, 1.0))
            if sentiment_meta.get("weighted_sentiment_raw", 0.0) <= NEGATIVE_SENTIMENT_CUTOFF:
                # Very poor sentiment: ignore sentiment contribution and trust model output.
                ensemble_weight_used = 1.0
                sentiment_weight_used = 0.0
        else:
            # No fresh news in lookback window => neutral sentiment contribution.
            ensemble_weight_used = 1.0
            sentiment_weight_used = 0.0
            weighted_sentiment = 0.0

        predicted_return = (
            ensemble_weight_used * technical_prediction
            + sentiment_weight_used * weighted_sentiment
        )

        # --- Compute derived quantities ---
        predicted_return = _sanitize_predicted_return(predicted_return, base_preds)
        predicted_price = round(current_price * (1 + predicted_return), 2)
        assert predicted_price == round(current_price * (1 + predicted_return), 2)

        # Get ATR% from features if available
        atr_pct = (
            float(feat_df["ATR_pct"].iloc[-1]) if "ATR_pct" in feat_df.columns else 0.02
        )
        volume_ratio = (
            float(feat_df["Volume_SMA_ratio"].iloc[-1])
            if "Volume_SMA_ratio" in feat_df.columns
            else 1.0
        )
        rvol = float(feat_df["RVOL"].iloc[-1]) if "RVOL" in feat_df.columns else 1.0

        # Model agreement combines directional consistency + magnitude alignment.
        if base_preds:
            if ensemble_weights:
                total_w = sum(abs(float(ensemble_weights.get(k, 0) or 0)) for k in base_preds)
                if total_w > 0:
                    weighted_sign = sum(
                        (1 if float(v) > 0 else -1)
                        * (float(ensemble_weights.get(k, 0) or 0) / total_w)
                        for k, v in base_preds.items()
                    )
                    directional_agreement = float(np.clip(abs(weighted_sign), 0.0, 1.0))
                    ref = abs(technical_prediction) + max(atr_pct * 0.5, 1e-6)
                    magnitude_alignment = sum(
                        (float(ensemble_weights.get(k, 0) or 0) / total_w)
                        * max(0.0, 1.0 - (abs(float(v) - technical_prediction) / (2.0 * ref)))
                        for k, v in base_preds.items()
                    )
                    model_agreement = float(
                        np.clip(
                            0.7 * directional_agreement + 0.3 * magnitude_alignment,
                            0.0,
                            1.0,
                        )
                    )
                else:
                    signs = [1 if v > 0 else -1 for v in base_preds.values()]
                    model_agreement = abs(sum(signs)) / len(signs)
            else:
                signs = [1 if v > 0 else -1 for v in base_preds.values()]
                model_agreement = abs(sum(signs)) / len(signs)
        else:
            model_agreement = 0.0

        # --- Signal generation ---
        signal, confidence = self._generate_signal(
            predicted_return,
            atr_pct,
            volume_ratio,
            rvol,
            model_agreement=model_agreement,
        )

        # --- Entry/exit levels ---
        atr_value = current_price * atr_pct
        if signal in ("STRONG_BUY", "BUY"):
            entry_price = current_price * 0.998  # Slightly below
            stop_loss = current_price - 2.0 * atr_value
            target_price = predicted_price
        elif signal in ("STRONG_SELL", "SELL"):
            entry_price = current_price * 1.002  # Slightly above
            stop_loss = current_price + 2.0 * atr_value
            target_price = predicted_price
        else:
            entry_price = current_price
            stop_loss = current_price - 2.0 * atr_value
            target_price = predicted_price

        risk_reward = abs(predicted_price - entry_price) / (
            abs(entry_price - stop_loss) + 1e-10
        )

        result = {
            "ticker": ticker,
            "predicted_return": round(predicted_return * 100, 4),  # in %
            "technical_predicted_return": round(technical_prediction * 100, 4),
            "weighted_sentiment_raw": float(
                sentiment_meta.get("weighted_sentiment_raw", 0.0) or 0.0
            ),
            "weighted_sentiment_decimal": float(
                sentiment_meta.get("weighted_sentiment_decimal", 0.0) or 0.0
            ),
            "sentiment_articles": int(sentiment_meta.get("article_count", 0) or 0),
            "sentiment_sources": sentiment_meta.get("sources", []),
            "sentiment_decay_weights": sentiment_meta.get("weights_used", {}),
            "ensemble_weight_used": round(ensemble_weight_used, 4),
            "sentiment_weight_used": round(sentiment_weight_used, 4),
            "predicted_price": predicted_price,
            "current_price": round(current_price, 2),
            "previous_close": round(previous_close, 2),
            "model_predictions": {k: round(v * 100, 4) for k, v in base_preds.items()},
            "ensemble_weights": {k: round(v, 4) for k, v in ensemble_weights.items()},
            "ensemble_strategy": (
                getattr(self.ensemble, "best_strategy", "simple_average")
                if self.ensemble
                else "fallback_average"
            ),
            "excluded_models": sorted(self._stale_models) if self._stale_models else [],
            "signal": signal,
            "confidence": round(confidence, 1),
            "direction_probability": (
                round(dir_prob * 100, 1) if dir_prob is not None else None
            ),
            "model_agreement": round(model_agreement * 100, 1),
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "target_price": round(target_price, 2),
            "risk_reward": round(risk_reward, 2),
            "atr_pct": round(atr_pct * 100, 2),
            "volume_ratio": round(volume_ratio, 2),
            "rvol": round(rvol, 2),
            "avg_volume_30d": round(avg_volume_30d, 2) if avg_volume_30d > 0 else 0.0,
            "liquidity_factor": 1.0,
        }

        # --- Add technical indicators ---
        try:
            indicators = {}
            indicator_cols = {
                "RSI_14": "rsi",
                "MACD": "macd",
                "MACD_Signal": "macd_signal",
                "SMA_20": "sma_20",
                "SMA_50": "sma_50",
                "EMA_12": "ema_12",
                "EMA_26": "ema_26",
                "BB_Upper": "bb_upper",
                "BB_Lower": "bb_lower",
                "ATR_pct": "atr_pct",
                "Volume_SMA_ratio": "volume_ratio",
                "ADX": "adx",
                "BB_Middle": "bb_middle",
            }
            for col, key in indicator_cols.items():
                if col in feat_df.columns:
                    val = feat_df[col].iloc[-1]
                    if pd.notna(val):
                        indicators[key] = float(val)
            if indicators:
                result["indicators"] = indicators
        except Exception:
            pass

        # --- Add options greeks (if available) ---
        try:
            from ..options.greeks import BlackScholesGreeks

            bs = BlackScholesGreeks()
            # Calculate ATM call greeks using historical volatility
            if current_price > 0:
                hist_vol = (
                    float(feat_df["Returns"].std() * np.sqrt(252))
                    if "Returns" in feat_df.columns
                    else 0.3
                )
                T = 30 / 365  # 30 days to expiry (approximate)
                K = round(current_price / 50) * 50  # Round to nearest 50 for strike
                greeks_data = bs.all_greeks(current_price, K, T, hist_vol)
                if greeks_data:
                    greeks_data["iv"] = round(hist_vol, 4)
                    result["greeks"] = greeks_data
        except Exception:
            pass

        # --- Add fundamental data ---
        try:
            fund_data = self.fundamental_analyzer.get_recommendation(ticker)
            if "error" not in fund_data:
                result["fundamentals"] = {
                    "name": fund_data.get("name", ""),
                    "sector": fund_data.get("sector", ""),
                    "pe_ratio": fund_data.get("pe_ratio"),
                    "pb_ratio": fund_data.get("pb_ratio"),
                    "dividend_yield": fund_data.get("dividend_yield"),
                    "roe": fund_data.get("roe"),
                    "debt_to_equity": fund_data.get("debt_to_equity"),
                    "revenue_growth": fund_data.get("revenue_growth"),
                    "earnings_growth": fund_data.get("earnings_growth"),
                    "profit_margin": fund_data.get("profit_margin"),
                    "analyst_upside": fund_data.get("analyst_upside"),
                    "value_score": fund_data.get("value_score"),
                    "quality_score": fund_data.get("quality_score"),
                    "growth_score": fund_data.get("growth_score"),
                    "fundamental_score": fund_data.get("fundamental_score"),
                    "fundamental_recommendation": fund_data.get("recommendation"),
                    "market_cap": fund_data.get("market_cap"),
                    "target_price_analyst": fund_data.get("target_price"),
                    "fifty_two_high": fund_data.get("fifty_two_high"),
                    "fifty_two_low": fund_data.get("fifty_two_low"),
                    "beta": fund_data.get("beta"),
                }
        except Exception as e:
            print(f"Fundamental data fetch failed for {ticker}: {e}")

        # Add IST timestamp
        result["generated_at"] = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")

        # Cache the result
        self.cache.set(ticker, result)
        return result

    def get_strategy_price(self, ticker: str, use_cache: bool = True) -> dict | None:
        """
        Strategy-only price payload derived from ensemble model output.
        This is intentionally separate from Groq/AI narrative forecasts.
        """
        pred = self.predict_single(ticker, use_cache=use_cache)
        if pred is None:
            return None

        live = self.get_live_price(ticker) or {}
        current_price = float(pred.get("current_price", 0) or live.get("price", 0) or 0)
        open_price = float(live.get("open", current_price) or current_price)
        if open_price <= 0:
            open_price = current_price
        if current_price <= 0 or open_price <= 0:
            return None

        predicted_return_decimal = _sanitize_predicted_return(
            float(pred.get("predicted_return", 0) or 0) / 100.0,
            {},
        )
        strategy_price = round(open_price * (1 + predicted_return_decimal), 2)
        assert strategy_price == round(open_price * (1 + predicted_return_decimal), 2)

        generated_at = str(pred.get("timestamp") or datetime.now(IST).isoformat())
        return {
            "ticker": ticker,
            "strategy_price": strategy_price,
            "rr_ratio": float(pred.get("risk_reward", 0) or 0),
            "strategy_generated_at": generated_at,
            "predicted_return_decimal": predicted_return_decimal,
            "source": "strategy_engine",
            "open_price": round(open_price, 2),
            "current_price": round(current_price, 2),
        }

    def _generate_signal(
        self,
        predicted_return: float,
        atr_pct: float,
        volume_ratio: float,
        rvol: float,
        model_agreement: float = 0.0,
    ) -> tuple[str, float]:
        """
        Convert ML prediction into actionable signal.

        Uses predicted return magnitude relative to ATR,
        plus volume confirmation.

        Returns
        -------
        (signal, confidence)
        """
        # Base confidence from prediction magnitude vs ATR
        if atr_pct > 0:
            magnitude = abs(predicted_return) / atr_pct
        else:
            magnitude = abs(predicted_return) / 0.02

        # Start with base confidence
        confidence = 50.0

        # Prediction direction and magnitude
        if predicted_return > 0:
            direction = 1
            confidence += min(magnitude * 15, 30)  # Up to +30 from magnitude
        else:
            direction = -1
            confidence += min(magnitude * 15, 30)

        # Volume confirmation bonus
        if volume_ratio > 1.5:
            confidence += 5  # High volume confirms move
        elif volume_ratio < 0.5:
            confidence -= 2  # Low volume = weak conviction, but not an automatic HOLD

        if rvol > 2.0:
            confidence += 3  # Very high relative volume
        elif rvol < 0.5:
            confidence -= 1.5

        # Model agreement bonus/penalty
        if model_agreement >= 0.8:
            confidence += 8  # Strong agreement — high conviction
        elif model_agreement >= 0.6:
            confidence += 3
        elif model_agreement < 0.4:
            confidence -= 5  # Models disagree — lower conviction

        confidence = max(0, min(100, confidence))
        move_pct = abs(predicted_return) * 100.0

        # Map to signal
        if direction > 0:
            if confidence >= 68 and move_pct >= 0.45:
                signal = "STRONG_BUY"
            elif confidence >= 54 and move_pct >= 0.18:
                signal = "BUY"
            else:
                signal = "HOLD"
        else:
            if confidence >= 68 and move_pct >= 0.45:
                signal = "STRONG_SELL"
            elif confidence >= 54 and move_pct >= 0.18:
                signal = "SELL"
            else:
                signal = "HOLD"

        return signal, confidence

    def _resolve_top_pick_tickers(
        self,
        tickers: list[str] | None = None,
        sectors: list[str] | None = None,
    ) -> list[str]:
        if tickers is not None:
            return sorted(set(tickers))

        import yaml

        config_path = PROJECT_ROOT / "config" / "tickers.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        resolved = []
        target_sectors = sectors or ["large_cap", "banking"]
        for sec in target_sectors:
            syms = data.get(sec, [])
            if isinstance(syms, list):
                resolved.extend(syms)
        return sorted(set(resolved))

    @staticmethod
    def _score_prediction(pred: dict) -> float:
        """
        Composite ranking score for top picks.

        score = |predicted_return_decimal| × (confidence/100) ×
                (model_agreement/100) × liquidity_factor
        """
        pred_ret_decimal = abs(float(pred.get("predicted_return", 0) or 0)) / 100.0
        conf = max(0.0, min(100.0, float(pred.get("confidence", 50) or 50))) / 100.0
        agreement = (
            max(0.0, min(100.0, float(pred.get("model_agreement", 50) or 50))) / 100.0
        )
        liq = float(pred.get("liquidity_factor", 1.0) or 1.0)
        if not np.isfinite(liq) or liq <= 0:
            liq = 1.0
        return pred_ret_decimal * conf * agreement * liq

    @staticmethod
    def _compute_liquidity_factor(avg_volume_30d: float, median_volume_30d: float) -> float:
        """
        Liquidity normalization based on 30-day average volume.
        Clipped to avoid overweighting outliers.
        """
        avg_vol = float(avg_volume_30d or 0)
        med_vol = float(median_volume_30d or 0)
        if not np.isfinite(avg_vol) or avg_vol <= 0:
            return 0.6
        if not np.isfinite(med_vol) or med_vol <= 0:
            return 1.0
        return float(np.clip(avg_vol / med_vol, 0.6, 1.5))

    def predict_top_picks_grouped(
        self,
        tickers: list[str] | None = None,
        top_n: int = 10,
        sectors: list[str] | None = None,
    ) -> dict[str, list[dict]]:
        top_n = max(1, min(int(top_n), 50))
        if not self._loaded:
            self.load_models()

        scan_tickers = self._resolve_top_pick_tickers(tickers=tickers, sectors=sectors)
        results = []

        for ticker in scan_tickers:
            try:
                pred = self.predict_single(ticker, use_cache=True)
                if pred is None:
                    continue
                if pred.get("current_price", 0) <= 0:
                    continue
                results.append(pred)
            except Exception:
                continue

        volume_samples = [
            float(r.get("avg_volume_30d", 0) or 0)
            for r in results
            if float(r.get("avg_volume_30d", 0) or 0) > 0
        ]
        median_volume_30d = (
            float(np.median(volume_samples))
            if volume_samples
            else 0.0
        )
        for pred in results:
            liq_factor = self._compute_liquidity_factor(
                avg_volume_30d=float(pred.get("avg_volume_30d", 0) or 0),
                median_volume_30d=median_volume_30d,
            )
            pred["liquidity_factor"] = round(liq_factor, 4)
            pred["median_volume_30d"] = (
                round(median_volume_30d, 2) if median_volume_30d > 0 else None
            )
            pred["_score"] = self._score_prediction(pred)

        buy_signals = {"BUY", "STRONG_BUY"}
        sell_signals = {"SELL", "STRONG_SELL"}

        buys = sorted(
            [r for r in results if str(r.get("signal", "")).upper() in buy_signals],
            key=lambda x: (
                x.get("confidence", 0),
                x.get("_score", 0),
                x.get("predicted_return", 0),
            ),
            reverse=True,
        )[:top_n]
        sells = sorted(
            [r for r in results if str(r.get("signal", "")).upper() in sell_signals],
            key=lambda x: (
                x.get("confidence", 0),
                x.get("_score", 0),
            ),
            reverse=True,
        )[:top_n]
        holds = sorted(
            [r for r in results if str(r.get("signal", "")).upper() == "HOLD"],
            key=lambda x: (
                x.get("confidence", 0),
                x.get("_score", 0),
            ),
            reverse=True,
        )[:top_n]

        for r in buys:
            r["pick_type"] = "BUY"
        for r in sells:
            r["pick_type"] = "SELL"
        for r in holds:
            r["pick_type"] = "HOLD"

        return {"top_buy": buys, "top_sell": sells, "top_hold": holds}

    def predict_top_picks(
        self,
        tickers: list[str] | None = None,
        top_n: int = 5,
        n: int | None = None,
        sectors: list[str] | None = None,
    ) -> list[dict]:
        """
        Scan a list of tickers and return the top N actionable picks.

        Uses a fast scan: for each ticker, runs predict_single and ranks
        by score = |predicted_return| × confidence × model_agreement.

        Parameters
        ----------
        tickers : list[str] or None
            If None, loads from config/tickers.yaml.
        top_n : int
            Number of top picks to return.
        sectors : list[str] or None
            If given, only scan tickers in these sectors.

        Returns
        -------
        list[dict] — top_n results sorted by score (best first).
        """
        top_n = max(1, min(int(top_n), 50))
        if n is not None:
            top_n = max(1, min(int(n), 50))

        grouped = self.predict_top_picks_grouped(
            tickers=tickers,
            top_n=top_n,
            sectors=sectors,
        )
        actionable = grouped["top_buy"] + grouped["top_sell"]
        actionable = sorted(
            actionable,
            key=lambda x: (
                x.get("confidence", 0),
                x.get("_score", 0),
            ),
            reverse=True,
        )
        return actionable[:top_n]

    def get_after_hours_review(self) -> dict:
        """
        After market close: compare today's predictions vs actual closing prices.

        Returns
        -------
        dict with keys: predictions, actuals, hit_rate, alpha, results
        """
        import json

        log_dir = PROJECT_ROOT / "cache" / "prediction_log"
        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        log_file = log_dir / f"{today_str}.json"

        if not log_file.exists():
            return {"error": "No predictions logged today"}

        with open(log_file) as f:
            predictions = json.load(f)

        if not predictions:
            return {"error": "Empty prediction log"}

        # Fetch actual closing prices
        results = []
        for ticker, pred in predictions.items():
            try:
                live = self.get_live_price(ticker)
                if live is None:
                    continue

                actual_close = live["price"]
                pred_price = pred.get("current_price", 0)
                pred_return = pred.get("predicted_return", 0) / 100
                actual_return = (
                    (actual_close - pred_price) / pred_price if pred_price > 0 else 0
                )

                pred_dir = "UP" if pred_return > 0 else "DOWN"
                actual_dir = "UP" if actual_return > 0 else "DOWN"
                correct = pred_dir == actual_dir

                alpha = actual_return if correct else -abs(actual_return)

                results.append(
                    {
                        "ticker": ticker,
                        "signal": pred.get("signal", "N/A"),
                        "predicted_return": pred.get("predicted_return", 0),
                        "predicted_price": pred.get("predicted_price", 0),
                        "actual_close": actual_close,
                        "actual_return_pct": round(actual_return * 100, 4),
                        "direction_correct": correct,
                        "alpha_pct": round(alpha * 100, 4),
                    }
                )
            except Exception:
                continue

        if not results:
            return {"error": "Could not fetch actuals"}

        hit_rate = (
            sum(1 for r in results if r["direction_correct"]) / len(results) * 100
        )
        avg_alpha = np.mean([r["alpha_pct"] for r in results])
        total_alpha = sum(r["alpha_pct"] for r in results)

        return {
            "date": today_str,
            "total_predictions": len(results),
            "hit_rate": round(hit_rate, 1),
            "avg_alpha": round(avg_alpha, 3),
            "total_alpha": round(total_alpha, 3),
            "results": results,
        }

    def get_premarket_outlook(self, tickers: list[str] | None = None) -> list[dict]:
        """
        Pre-market: generate next-day price estimates for key stocks.

        Returns list of dicts with strategy/open and AI/current comparison fields.
        """
        if tickers is None:
            # Default to top large caps for pre-market view
            tickers = [
                "RELIANCE.NS",
                "TCS.NS",
                "HDFCBANK.NS",
                "INFY.NS",
                "ICICIBANK.NS",
                "ITC.NS",
                "SBIN.NS",
                "BHARTIARTL.NS",
                "BAJFINANCE.NS",
                "LT.NS",
            ]

        if not self._loaded:
            self.load_models()

        outlook = []
        for ticker in tickers:
            try:
                pred = self.predict_single(ticker, use_cache=True)
                if pred is None:
                    continue

                # Get recent 5-day OHLCV for mini candle chart
                live = self.get_live_price(ticker)
                current_price = float(pred.get("current_price", 0) or 0)
                if current_price <= 0:
                    continue
                open_price = float(
                    live.get("open", current_price) if live else current_price
                )
                predicted_return_pct = float(pred.get("predicted_return", 0) or 0)
                strategy_price_at_open = (
                    round(open_price * (1 + predicted_return_pct / 100.0), 2)
                    if open_price > 0
                    else float(pred.get("predicted_price", 0) or 0)
                )
                ai_predicted_price = float(pred.get("predicted_price", 0) or 0)
                strategy_direction = (
                    "UP"
                    if strategy_price_at_open > open_price
                    else "DOWN" if strategy_price_at_open < open_price else "FLAT"
                )
                ai_direction = (
                    "UP"
                    if ai_predicted_price > current_price
                    else "DOWN" if ai_predicted_price < current_price else "FLAT"
                )

                entry = {
                    "ticker": ticker,
                    "name": ticker.replace(".NS", ""),
                    "current_price": round(current_price, 2),
                    "open_price": round(open_price, 2) if open_price > 0 else None,
                    "strategy_price_at_open": (
                        strategy_price_at_open if strategy_price_at_open > 0 else None
                    ),
                    "ai_predicted_price": (
                        round(ai_predicted_price, 2) if ai_predicted_price > 0 else None
                    ),
                    "strategy_direction": strategy_direction,
                    "ai_direction": ai_direction,
                    "captured_at": datetime.now(IST).isoformat(),
                    "predicted_price": pred.get("predicted_price", 0),
                    "predicted_return": pred.get("predicted_return", 0),
                    "signal": pred.get("signal", "HOLD"),
                    "confidence": pred.get("confidence", 50),
                    "entry_price": pred.get("entry_price", 0),
                    "stop_loss": pred.get("stop_loss", 0),
                    "target_price": pred.get("target_price", 0),
                    "risk_reward": pred.get("risk_reward", 0),
                    "model_predictions": pred.get("model_predictions", {}),
                    "model_agreement": pred.get("model_agreement", 0),
                }

                if live:
                    entry["live_price"] = live["price"]
                    entry["day_change"] = live["day_change"]
                    entry["volume"] = live["volume"]

                outlook.append(entry)
            except Exception:
                continue

        return outlook

    def get_live_price(self, ticker: str) -> dict | None:
        """
        Get the latest available price data for a ticker.

        Returns current price, day change, and volume info.
        """
        if yf is None:
            return None
        try:
            # Use unadjusted candles for live-trading price parity with broker/UI quotes.
            data = yf.download(
                ticker, period="2d", interval="1d", progress=False, auto_adjust=False
            )
            if data is None or data.empty:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            current = float(data["Close"].iloc[-1])
            prev_close = float(data["Close"].iloc[-2]) if len(data) >= 2 else current
            day_change = (current - prev_close) / prev_close
            volume = float(data["Volume"].iloc[-1])

            return {
                "price": round(current, 2),
                "previous_close": round(prev_close, 2),
                "day_change": round(day_change * 100, 2),
                "volume": int(volume),
                "high": round(float(data["High"].iloc[-1]), 2),
                "low": round(float(data["Low"].iloc[-1]), 2),
                "open": round(float(data["Open"].iloc[-1]), 2),
            }
        except Exception:
            return None

    def get_intraday_data(
        self, ticker: str, period: str = "5d", interval: str = "15m"
    ) -> pd.DataFrame | None:
        """Download intraday data (instance method)."""
        return get_intraday_data(ticker, period, interval)


def get_intraday_data(
    ticker: str, period: str = "1d", interval: str = "5m"
) -> pd.DataFrame | None:
    """
    Download intraday data for a ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker
    period : str
        Period of intraday data
    interval : str
        Candle interval (1m, 5m, 15m)

    Returns
    -------
    pd.DataFrame or None
    """
    if yf is None:
        return None
    try:
        # Keep intraday chart prices in raw market units (not split-adjusted).
        df = yf.download(
            ticker, period=period, interval=interval, progress=False, auto_adjust=False
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None
