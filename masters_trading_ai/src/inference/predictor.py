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
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch  # Initialize PyTorch early — prevents segfault when loading after statsmodels/arch

try:
    import yfinance as yf
except ImportError:
    yf = None

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
MARKET_OPEN = 9   # 9:15 AM IST
MARKET_OPEN_MIN = 15
MARKET_CLOSE = 15  # 3:30 PM IST
MARKET_CLOSE_MIN = 30
CACHE_DIR = PROJECT_ROOT / "cache"


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

    market_open_time = now.replace(hour=MARKET_OPEN, minute=MARKET_OPEN_MIN, second=0, microsecond=0)
    market_close_time = now.replace(hour=MARKET_CLOSE, minute=MARKET_CLOSE_MIN, second=0, microsecond=0)

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

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "predictions.json"
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
        """Get cached prediction. Returns None if stale (>6 hours)."""
        entry = self._data.get(ticker)
        if entry is None:
            return None
        cached_time = datetime.fromisoformat(entry["timestamp"])
        now = datetime.now(IST)
        # Make cached_time offset-aware if needed
        if cached_time.tzinfo is None:
            cached_time = cached_time.replace(tzinfo=IST)
        if now - cached_time > timedelta(hours=6):
            return None  # Stale
        return entry

    def set(self, ticker: str, prediction: dict):
        """Cache a prediction result."""
        prediction["timestamp"] = datetime.now(IST).isoformat()
        self._data[ticker] = prediction
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

    def load_models(self) -> dict:
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
        if self._loaded:
            return self.models

        loaded = {}
        canonical_order = ["arima", "garch", "xgboost", "lightgbm", "lstm", "transformer"]

        # ── LSTM (load PyTorch models first to avoid C-extension conflicts) ──
        lstm_path = self.models_dir / "lstm_model.pt"
        lstm_joblib = self.models_dir / "lstm_model.joblib"
        if lstm_path.exists() or lstm_joblib.exists():
            try:
                model = LSTMModel()
                model.load(lstm_path)
                loaded["lstm"] = model
            except Exception as e:
                print(f"  ⚠ LSTM load failed: {e}")

        # ── Transformer ──
        transformer_path = self.models_dir / "transformer_model.pt"
        transformer_joblib = self.models_dir / "transformer_model.joblib"
        if transformer_path.exists() or transformer_joblib.exists():
            try:
                model = TransformerModel()
                model.load(transformer_path)
                loaded["transformer"] = model
            except Exception as e:
                print(f"  ⚠ Transformer load failed: {e}")

        # ── XGBoost (joblib) ──
        xgb_path = self.models_dir / "xgboost_model.joblib"
        if xgb_path.exists():
            try:
                model = XGBoostModel()
                model.load(xgb_path)
                loaded["xgboost"] = model
                print(f"  Loaded xgboost from {xgb_path}")
            except Exception as e:
                print(f"  ⚠ XGBoost load failed: {e}")

        # ── LightGBM (joblib) ──
        lgb_path = self.models_dir / "lightgbm_model.joblib"
        if lgb_path.exists():
            try:
                model = LightGBMModel()
                model.load(lgb_path)
                loaded["lightgbm"] = model
                print(f"  Loaded lightgbm from {lgb_path}")
            except Exception as e:
                print(f"  ⚠ LightGBM load failed: {e}")

        # ── ARIMA (joblib — uses smooth() for instant restore) ──
        arima_path = self.models_dir / "arima_model.joblib"
        if arima_path.exists():
            try:
                model = ARIMAModel()
                model.load(arima_path)
                loaded["arima"] = model
            except Exception as e:
                print(f"  ⚠ ARIMA load failed: {e}")

        # ── GARCH (joblib — uses fix() for instant restore) ──
        garch_path = self.models_dir / "garch_model.joblib"
        if garch_path.exists():
            try:
                model = GARCHModel()
                model.load(garch_path)
                loaded["garch"] = model
            except Exception as e:
                print(f"  ⚠ GARCH load failed: {e}")

        # ── Ensemble meta-learner ──
        ens_path = self.models_dir / "ensemble_ridge_meta.joblib"
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
            except Exception as e:
                print(f"  ⚠ Ensemble load failed: {e}")

        self.models = loaded
        self._loaded = True
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
        # Check cache first
        if use_cache:
            cached = self.cache.get(ticker)
            if cached is not None:
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

        # Get current price info from the feature DataFrame (avoid redundant yfinance call)
        try:
            if "Close" in feat_df.columns:
                current_price = float(feat_df["Close"].iloc[-1])
                previous_close = float(feat_df["Close"].iloc[-2]) if len(feat_df) >= 2 else current_price
            else:
                current_price = 0
                previous_close = 0
            current_volume = float(feat_df["Volume"].iloc[-1]) if "Volume" in feat_df.columns else 0
        except Exception:
            current_price = 0
            previous_close = current_price
            current_volume = 0

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
                mean_fc, vol_fc = garch_model.forecast_ticker(ticker, garch_model.forecast_horizon)
                base_preds["garch"] = float(mean_fc)
            except Exception as e:
                print(f"GARCH prediction failed: {e}")

        # XGBoost (uses last row)
        if "xgboost" in self.models:
            try:
                xgb_model = self.models["xgboost"]
                # Align features to what model expects
                available = [f for f in xgb_model.feature_names if f in feat_df.columns]
                missing = [f for f in xgb_model.feature_names if f not in feat_df.columns]
                pred_df = feat_df[available].copy()
                for m in missing:
                    pred_df[m] = 0.0
                pred_df = pred_df[xgb_model.feature_names]
                preds = xgb_model.predict(pred_df)
                base_preds["xgboost"] = float(preds[-1])
            except Exception as e:
                print(f"XGBoost prediction failed: {e}")

        # LightGBM (uses last row)
        if "lightgbm" in self.models:
            try:
                lgb_model = self.models["lightgbm"]
                available = [f for f in lgb_model.feature_names if f in feat_df.columns]
                missing = [f for f in lgb_model.feature_names if f not in feat_df.columns]
                pred_df = feat_df[available].copy()
                for m in missing:
                    pred_df[m] = 0.0
                pred_df = pred_df[lgb_model.feature_names]
                preds = lgb_model.predict(pred_df)
                base_preds["lightgbm"] = float(preds[-1])
            except Exception as e:
                print(f"LightGBM prediction failed: {e}")

        # LSTM (uses sequence of last seq_len rows)
        if "lstm" in self.models:
            try:
                lstm_model = self.models["lstm"]
                available = [f for f in lstm_model.feature_names if f in feat_df.columns]
                missing = [f for f in lstm_model.feature_names if f not in feat_df.columns]
                pred_df = feat_df[available].copy()
                for m in missing:
                    pred_df[m] = 0.0
                pred_df = pred_df[lstm_model.feature_names]
                # LSTM needs enough rows for sequence
                if len(pred_df) >= lstm_model.seq_len:
                    preds = lstm_model.predict(pred_df)
                    if len(preds) > 0:
                        base_preds["lstm"] = float(preds[-1])
            except Exception as e:
                print(f"LSTM prediction failed: {e}")

        # Transformer (uses sequence of last seq_len rows)
        if "transformer" in self.models:
            try:
                tf_model = self.models["transformer"]
                available = [f for f in tf_model.feature_names if f in feat_df.columns]
                missing = [f for f in tf_model.feature_names if f not in feat_df.columns]
                pred_df = feat_df[available].copy()
                for m in missing:
                    pred_df[m] = 0.0
                pred_df = pred_df[tf_model.feature_names]
                if len(pred_df) >= tf_model.seq_len:
                    preds = tf_model.predict(pred_df)
                    if len(preds) > 0:
                        base_preds["transformer"] = float(np.ravel(preds[-1])[0])
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
                ens_cols = self.ensemble.model_names if hasattr(self.ensemble, "model_names") else list(base_preds.keys())
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
                        ensemble_weights = {k: v / total for k, v in available_wts.items()}
                    else:
                        ensemble_weights = {k: 1.0 / len(base_preds) for k in base_preds}
                else:
                    ensemble_weights = {k: 1.0 / len(base_preds) for k in base_preds}

                # Get direction probability if dual-learner is available
                if hasattr(self.ensemble, "predict_direction_probability"):
                    try:
                        dir_prob = float(self.ensemble.predict_direction_probability(ens_df)[0])
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
                        ensemble_weights = {k: v / total for k, v in available_wts.items()}
                    else:
                        predicted_return = float(np.mean(list(base_preds.values())))
                else:
                    predicted_return = float(np.mean(list(base_preds.values())))
                dir_prob = None
        else:
            predicted_return = float(np.mean(list(base_preds.values())))
            dir_prob = None

        # --- Compute derived quantities ---
        predicted_price = current_price * (1 + predicted_return)

        # Get ATR% from features if available
        atr_pct = float(feat_df["ATR_pct"].iloc[-1]) if "ATR_pct" in feat_df.columns else 0.02
        volume_ratio = float(feat_df["Volume_SMA_ratio"].iloc[-1]) if "Volume_SMA_ratio" in feat_df.columns else 1.0
        rvol = float(feat_df["RVOL"].iloc[-1]) if "RVOL" in feat_df.columns else 1.0

        # Model agreement (what fraction of base models agree on direction)
        if base_preds:
            signs = [1 if v > 0 else -1 for v in base_preds.values()]
            model_agreement = abs(sum(signs)) / len(signs)
        else:
            model_agreement = 0.0

        # --- Signal generation ---
        signal, confidence = self._generate_signal(
            predicted_return, atr_pct, volume_ratio, rvol,
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

        risk_reward = abs(predicted_price - entry_price) / (abs(entry_price - stop_loss) + 1e-10)

        result = {
            "ticker": ticker,
            "predicted_return": round(predicted_return * 100, 4),  # in %
            "predicted_price": round(predicted_price, 2),
            "current_price": round(current_price, 2),
            "previous_close": round(previous_close, 2),
            "model_predictions": {k: round(v * 100, 4) for k, v in base_preds.items()},
            "ensemble_weights": {k: round(v, 4) for k, v in ensemble_weights.items()},
            "ensemble_strategy": getattr(self.ensemble, "best_strategy", "simple_average") if self.ensemble else "fallback_average",
            "signal": signal,
            "confidence": round(confidence, 1),
            "direction_probability": round(dir_prob * 100, 1) if dir_prob is not None else None,
            "model_agreement": round(model_agreement * 100, 1),
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "target_price": round(target_price, 2),
            "risk_reward": round(risk_reward, 2),
            "atr_pct": round(atr_pct * 100, 2),
            "volume_ratio": round(volume_ratio, 2),
            "rvol": round(rvol, 2),
        }

        # --- Add technical indicators ---
        try:
            indicators = {}
            indicator_cols = {
                "RSI_14": "rsi", "MACD": "macd", "MACD_Signal": "macd_signal",
                "SMA_20": "sma_20", "SMA_50": "sma_50",
                "EMA_12": "ema_12", "EMA_26": "ema_26",
                "BB_Upper": "bb_upper", "BB_Lower": "bb_lower",
                "ATR_pct": "atr_pct", "Volume_SMA_ratio": "volume_ratio",
                "ADX": "adx", "BB_Middle": "bb_middle",
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
                hist_vol = float(feat_df["Returns"].std() * np.sqrt(252)) if "Returns" in feat_df.columns else 0.3
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
            confidence -= 5  # Low volume = weak conviction

        if rvol > 2.0:
            confidence += 3  # Very high relative volume
        elif rvol < 0.5:
            confidence -= 3

        # Model agreement bonus/penalty
        if model_agreement >= 0.8:
            confidence += 8   # Strong agreement — high conviction
        elif model_agreement >= 0.6:
            confidence += 3
        elif model_agreement < 0.4:
            confidence -= 5   # Models disagree — lower conviction

        confidence = max(0, min(100, confidence))

        # Map to signal
        if direction > 0:
            if confidence >= 75:
                signal = "STRONG_BUY"
            elif confidence >= 60:
                signal = "BUY"
            else:
                signal = "HOLD"
        else:
            if confidence >= 75:
                signal = "STRONG_SELL"
            elif confidence >= 60:
                signal = "SELL"
            else:
                signal = "HOLD"

        return signal, confidence

    def predict_top_picks(
        self,
        tickers: list[str] | None = None,
        top_n: int = 5,
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
        if tickers is None:
            import yaml
            config_path = PROJECT_ROOT / "config" / "tickers.yaml"
            with open(config_path) as f:
                data = yaml.safe_load(f)
            tickers = []
            target_sectors = sectors or ["large_cap", "banking"]
            for sec in target_sectors:
                syms = data.get(sec, [])
                if isinstance(syms, list):
                    tickers.extend(syms)
            tickers = sorted(set(tickers))

        if not self._loaded:
            self.load_models()

        results = []
        for ticker in tickers:
            try:
                pred = self.predict_single(ticker, use_cache=True)
                if pred is None:
                    continue
                # Compute composite score for ranking
                pred_ret = abs(pred.get("predicted_return", 0))
                conf = pred.get("confidence", 50)
                agreement = pred.get("model_agreement", 50)
                rr = max(pred.get("risk_reward", 0), 0)
                # Score: higher is better opportunity  
                score = pred_ret * (conf / 100) * (agreement / 100) * (1 + rr * 0.1)
                pred["_score"] = score
                results.append(pred)
            except Exception:
                continue

        # Separate buys and sells, sort by score
        buys = sorted(
            [r for r in results if r.get("predicted_return", 0) > 0],
            key=lambda x: x["_score"],
            reverse=True,
        )
        sells = sorted(
            [r for r in results if r.get("predicted_return", 0) < 0],
            key=lambda x: x["_score"],
            reverse=True,
        )

        # Return top_n buys (primary picks) + top sells as warnings
        top_buys = buys[:top_n]
        top_sells = sells[:min(3, len(sells))]

        # Mark pick type
        for r in top_buys:
            r["pick_type"] = "BUY"
        for r in top_sells:
            r["pick_type"] = "SELL"

        return top_buys + top_sells

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
                actual_return = (actual_close - pred_price) / pred_price if pred_price > 0 else 0

                pred_dir = "UP" if pred_return > 0 else "DOWN"
                actual_dir = "UP" if actual_return > 0 else "DOWN"
                correct = pred_dir == actual_dir

                alpha = actual_return if correct else -abs(actual_return)

                results.append({
                    "ticker": ticker,
                    "signal": pred.get("signal", "N/A"),
                    "predicted_return": pred.get("predicted_return", 0),
                    "predicted_price": pred.get("predicted_price", 0),
                    "actual_close": actual_close,
                    "actual_return_pct": round(actual_return * 100, 4),
                    "direction_correct": correct,
                    "alpha_pct": round(alpha * 100, 4),
                })
            except Exception:
                continue

        if not results:
            return {"error": "Could not fetch actuals"}

        hit_rate = sum(1 for r in results if r["direction_correct"]) / len(results) * 100
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

        Returns list of dicts with predicted prices and recent candle data.
        """
        if tickers is None:
            # Default to top large caps for pre-market view
            tickers = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
                "ICICIBANK.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS",
                "BAJFINANCE.NS", "LT.NS",
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

                entry = {
                    "ticker": ticker,
                    "name": ticker.replace(".NS", ""),
                    "current_price": pred.get("current_price", 0),
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
            data = yf.download(ticker, period="2d", interval="1d", progress=False, auto_adjust=True)
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


    def get_intraday_data(self, ticker: str, period: str = "5d", interval: str = "15m") -> pd.DataFrame | None:
        """Download intraday data (instance method)."""
        return get_intraday_data(ticker, period, interval)


def get_intraday_data(ticker: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame | None:
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
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None
