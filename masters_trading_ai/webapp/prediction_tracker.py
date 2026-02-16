"""
Prediction Tracker — Records daily prediction outcomes (hit/miss).

Stores daily results in JSON files and provides monthly accuracy reports
that can be used to retrain/improve future models.

Structure:
  cache/prediction_tracking/
    daily/YYYY-MM-DD.json       — daily prediction + outcomes
    monthly/YYYY-MM.json        — monthly aggregated accuracy
    accuracy_history.json       — running history for dashboards
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = PROJECT_ROOT / "cache" / "prediction_tracking"
DAILY_DIR = TRACKING_DIR / "daily"
MONTHLY_DIR = TRACKING_DIR / "monthly"
ACCURACY_FILE = TRACKING_DIR / "accuracy_history.json"

for d in [TRACKING_DIR, DAILY_DIR, MONTHLY_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class PredictionTracker:
    """Track whether each predicted price threshold was hit during the day."""

    SCHEMA_VERSION = 4

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            out = float(value)
            if np.isfinite(out):
                return out
        except Exception:
            pass
        return default

    @staticmethod
    def _direction(from_price: float, to_price: float, eps: float = 1e-9) -> str:
        if not np.isfinite(from_price) or not np.isfinite(to_price):
            return "FLAT"
        delta = to_price - from_price
        if delta > eps:
            return "UP"
        if delta < -eps:
            return "DOWN"
        return "FLAT"

    @staticmethod
    def record_prediction(
        ticker: str,
        prediction_data: dict,
        snapshot_type: Optional[str] = None,
    ):
        """
        Record a new prediction for today.

        Stores: predicted_return, predicted_price, current_price at prediction time,
        threshold, signal, confidence, model_predictions, timestamp.
        """
        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        file_path = DAILY_DIR / f"{today_str}.json"

        existing = {}
        if file_path.exists():
            try:
                existing = json.loads(file_path.read_text())
            except (json.JSONDecodeError, OSError):
                existing = {}

        pred_return = PredictionTracker._to_float(
            prediction_data.get("predicted_return", 0)
        )  # in %
        current_price = PredictionTracker._to_float(
            prediction_data.get("current_price", 0)
        )
        predicted_price = PredictionTracker._to_float(
            prediction_data.get("predicted_price", 0)
        )
        open_price = PredictionTracker._to_float(
            prediction_data.get(
                "open_price", prediction_data.get("price_at_prediction", current_price)
            ),
            default=current_price,
        )
        strategy_price_at_open = PredictionTracker._to_float(
            prediction_data.get(
                "strategy_price_at_open",
                prediction_data.get("strategy_predicted_price", predicted_price),
            ),
            default=predicted_price,
        )
        ai_last_prediction = PredictionTracker._to_float(
            prediction_data.get("ai_last_prediction", 0),
            default=0.0,
        )
        strategy_predicted_at_open = str(
            prediction_data.get("strategy_predicted_at_open", "")
            or prediction_data.get("timestamp", "")
            or datetime.now(IST).isoformat()
        )
        ai_predicted_at_open = str(
            prediction_data.get("ai_predicted_at_open", "")
            or prediction_data.get("timestamp", "")
            or datetime.now(IST).isoformat()
        )
        ai_last_prediction_at = str(
            prediction_data.get("ai_last_prediction_at", "")
            or prediction_data.get("timestamp", "")
            or datetime.now(IST).isoformat()
        )
        snapshot_type_value = (
            str(snapshot_type or prediction_data.get("snapshot_type") or "intraday")
            .strip()
            .lower()
        )
        if snapshot_type_value not in {
            "market_open",
            "near_open_fallback",
            "premarket_preview",
            "intraday",
        }:
            snapshot_type_value = "intraday"

        strategy_direction = PredictionTracker._direction(
            open_price, strategy_price_at_open
        )
        ai_direction = (
            PredictionTracker._direction(current_price, ai_last_prediction)
            if ai_last_prediction > 0 and current_price > 0
            else "N/A"
        )
        ai_source = str(prediction_data.get("ai_source", "none") or "none")
        strategy_vs_ai_direction = prediction_data.get("strategy_vs_ai_direction")
        if (
            strategy_vs_ai_direction is None
            and ai_last_prediction > 0
            and current_price > 0
        ):
            strategy_vs_ai_direction = strategy_direction == ai_direction
        if strategy_vs_ai_direction not in (True, False):
            strategy_vs_ai_direction = None

        # direction_comparison is reserved for strategy-vs-actual and is populated
        # when outcomes are checked after market data is available.
        direction_comparison = prediction_data.get("direction_comparison")
        if direction_comparison not in (True, False):
            direction_comparison = None

        # Threshold: the predicted price level that needs to be hit
        # For BUY: stock needs to reach at or above predicted_price
        # For SELL: stock needs to reach at or below predicted_price
        is_bullish = pred_return > 0

        existing[ticker] = {
            "predicted_return_pct": round(pred_return, 4),
            "predicted_price": round(predicted_price, 2),
            "price_at_prediction": round(current_price, 2),
            "open_price": round(open_price, 2),
            "strategy_price_at_open": round(strategy_price_at_open, 2),
            "ai_last_prediction": round(ai_last_prediction, 2),
            "ai_source": ai_source if ai_last_prediction > 0 else "none",
            "strategy_predicted_at_open": strategy_predicted_at_open,
            "ai_predicted_at_open": ai_predicted_at_open,
            "ai_last_prediction_at": ai_last_prediction_at,
            "strategy_direction_at_open": strategy_direction,
            "ai_direction_last": ai_direction,
            "strategy_vs_ai_direction": strategy_vs_ai_direction,
            "direction_comparison": direction_comparison,
            "signal": prediction_data.get("signal", "HOLD"),
            "confidence": round(prediction_data.get("confidence", 50), 1),
            "is_bullish": is_bullish,
            "threshold_price": round(predicted_price, 2),
            "model_predictions": prediction_data.get("model_predictions", {}),
            "ensemble_weights": prediction_data.get("ensemble_weights", {}),
            "schema_version": PredictionTracker.SCHEMA_VERSION,
            "timestamp": datetime.now(IST).isoformat(),
            # Outcome fields — filled later
            "outcome": None,  # "HIT" or "MISS"
            "actual_close": None,
            "actual_high": None,
            "actual_low": None,
            "actual_return_pct": None,
            "actual_direction": None,
            "direction_correct": None,
            "checked_at": None,
        }

        file_path.write_text(json.dumps(existing, indent=2, default=str))

    @staticmethod
    def check_outcomes(date_str: Optional[str] = None) -> dict:
        """
        Check if predictions for a given date hit their threshold.

        Uses end-of-day close and compares strategy-open direction vs actual direction.

        Returns dict of {ticker: outcome_data}
        """
        import yfinance as yf

        if date_str is None:
            date_str = datetime.now(IST).strftime("%Y-%m-%d")

        file_path = DAILY_DIR / f"{date_str}.json"
        if not file_path.exists():
            return {"error": f"No predictions for {date_str}"}

        predictions = json.loads(file_path.read_text())
        if not predictions:
            return {"error": "Empty predictions"}

        tickers = sorted(predictions.keys())
        results = {}
        updated = False

        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            start = target_date.strftime("%Y-%m-%d")
            end = (target_date + timedelta(days=7)).strftime("%Y-%m-%d")

            # Fetch daily data once for all tickers and read the first trading bar on/after date.
            data = yf.download(
                tickers,
                start=start,
                end=end,
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=True,
            )
            if data is None or data.empty:
                return {"error": "Could not fetch actual prices"}

            for ticker in tickers:
                pred = predictions[ticker]
                try:
                    if len(tickers) == 1:
                        close_vals = data["Close"].dropna()
                        high_vals = data["High"].dropna()
                        low_vals = data["Low"].dropna()
                    else:
                        close_vals = data["Close"][ticker].dropna()
                        high_vals = data["High"][ticker].dropna()
                        low_vals = data["Low"][ticker].dropna()

                    if close_vals.empty:
                        continue

                    if isinstance(close_vals.index, pd.DatetimeIndex):
                        close_vals = close_vals[close_vals.index.date >= target_date]
                        high_vals = high_vals[high_vals.index.date >= target_date]
                        low_vals = low_vals[low_vals.index.date >= target_date]
                    if close_vals.empty:
                        continue

                    actual_close = float(close_vals.iloc[0])
                    actual_high = (
                        float(high_vals.iloc[0])
                        if not high_vals.empty
                        else actual_close
                    )
                    actual_low = (
                        float(low_vals.iloc[0]) if not low_vals.empty else actual_close
                    )

                    price_at_pred = PredictionTracker._to_float(
                        pred.get("price_at_prediction", 0)
                    )
                    if price_at_pred <= 0:
                        continue

                    strategy_price_at_open = PredictionTracker._to_float(
                        pred.get(
                            "strategy_price_at_open",
                            pred.get("predicted_price", price_at_pred),
                        ),
                        default=price_at_pred,
                    )
                    open_price = PredictionTracker._to_float(
                        pred.get("open_price", price_at_pred),
                        default=price_at_pred,
                    )
                    ai_last_prediction = PredictionTracker._to_float(
                        pred.get(
                            "ai_last_prediction",
                            pred.get("predicted_price", price_at_pred),
                        ),
                        default=price_at_pred,
                    )
                    strategy_predicted_at_open = str(
                        pred.get("strategy_predicted_at_open", "")
                        or pred.get("timestamp", "")
                    )
                    ai_predicted_at_open = str(
                        pred.get("ai_predicted_at_open", "")
                        or pred.get("timestamp", "")
                    )
                    ai_last_prediction_at = str(
                        pred.get("ai_last_prediction_at", "")
                        or pred.get("timestamp", "")
                    )
                    strategy_direction = pred.get(
                        "strategy_direction_at_open"
                    ) or PredictionTracker._direction(
                        open_price, strategy_price_at_open
                    )
                    ai_direction = pred.get(
                        "ai_direction_last"
                    ) or PredictionTracker._direction(price_at_pred, ai_last_prediction)
                    actual_direction = PredictionTracker._direction(
                        open_price, actual_close
                    )
                    actual_return = (actual_close - price_at_pred) / price_at_pred * 100
                    direction_comparison = strategy_direction == actual_direction
                    direction_vs_actual = ai_direction == actual_direction

                    outcome = "HIT" if direction_comparison else "MISS"

                    # Update prediction record
                    pred["outcome"] = outcome
                    pred["actual_close"] = round(actual_close, 2)
                    pred["actual_high"] = round(actual_high, 2)
                    pred["actual_low"] = round(actual_low, 2)
                    pred["actual_return_pct"] = round(actual_return, 4)
                    pred["actual_direction"] = actual_direction
                    pred["strategy_price_at_open"] = round(strategy_price_at_open, 2)
                    pred["open_price"] = round(open_price, 2)
                    pred["ai_last_prediction"] = round(ai_last_prediction, 2)
                    pred["strategy_predicted_at_open"] = strategy_predicted_at_open
                    pred["ai_predicted_at_open"] = ai_predicted_at_open
                    pred["ai_last_prediction_at"] = ai_last_prediction_at
                    pred["strategy_direction_at_open"] = strategy_direction
                    pred["ai_direction_last"] = ai_direction
                    pred["direction_comparison"] = bool(direction_comparison)
                    pred["direction_correct"] = bool(direction_comparison)
                    pred["ai_direction_vs_actual"] = bool(direction_vs_actual)
                    pred["snapshot_type"] = str(
                        pred.get("snapshot_type", "intraday")
                    ).lower()
                    pred["schema_version"] = PredictionTracker.SCHEMA_VERSION
                    pred["checked_at"] = datetime.now(IST).isoformat()
                    updated = True

                    results[ticker] = {
                        "ticker": ticker,
                        "outcome": outcome,
                        "predicted_return_pct": pred["predicted_return_pct"],
                        "actual_return_pct": round(actual_return, 4),
                        "predicted_price": pred["predicted_price"],
                        "strategy_price_at_open": round(strategy_price_at_open, 2),
                        "ai_last_prediction": round(ai_last_prediction, 2),
                        "open_price": round(open_price, 2),
                        "actual_close": round(actual_close, 2),
                        "actual_high": round(actual_high, 2),
                        "actual_low": round(actual_low, 2),
                        "strategy_predicted_at_open": strategy_predicted_at_open,
                        "ai_predicted_at_open": ai_predicted_at_open,
                        "ai_last_prediction_at": ai_last_prediction_at,
                        "strategy_direction_at_open": strategy_direction,
                        "ai_direction_last": ai_direction,
                        "actual_direction": actual_direction,
                        "direction_comparison": bool(direction_comparison),
                        "direction_correct": bool(direction_comparison),
                        "ai_direction_vs_actual": bool(direction_vs_actual),
                        "snapshot_type": pred.get("snapshot_type", "intraday"),
                        "signal": pred["signal"],
                        "confidence": pred["confidence"],
                        "is_bullish": bool(pred.get("is_bullish", False)),
                    }
                except Exception as e:
                    results[ticker] = {
                        "ticker": ticker,
                        "outcome": "ERROR",
                        "error": str(e),
                    }
        except Exception as e:
            return {"error": f"Price fetch failed: {str(e)}"}

        if updated:
            file_path.write_text(json.dumps(predictions, indent=2, default=str))

        return results

    @staticmethod
    def get_monthly_report(
        year: Optional[int] = None, month: Optional[int] = None
    ) -> dict:
        """
        Aggregate daily outcomes into monthly accuracy report.

        Returns:
            total, hits, misses, accuracy %, breakdown by signal type,
            avg predicted vs actual return, best/worst predictions
        """
        now = datetime.now(IST)
        if year is None:
            year = now.year
        if month is None:
            month = now.month

        month_prefix = f"{year}-{month:02d}"
        daily_files = sorted(DAILY_DIR.glob(f"{month_prefix}-*.json"))

        if not daily_files:
            return {
                "period": month_prefix,
                "error": "No prediction data for this month",
                "total": 0,
            }

        all_preds = []
        for f in daily_files:
            try:
                day_data = json.loads(f.read_text())
                for ticker, pred in day_data.items():
                    pred["_date"] = f.stem
                    pred["_ticker"] = ticker
                    all_preds.append(pred)
            except Exception:
                continue

        total = len(all_preds)
        evaluated = [p for p in all_preds if p.get("outcome") in ("HIT", "MISS")]
        hits = sum(1 for p in evaluated if p["outcome"] == "HIT")
        misses = sum(1 for p in evaluated if p["outcome"] == "MISS")
        pending = total - len(evaluated)

        accuracy = (hits / len(evaluated) * 100) if evaluated else 0

        # Breakdown by signal
        signal_breakdown: dict[str, dict[str, float]] = {}
        for p in evaluated:
            sig = p.get("signal", "HOLD")
            if sig not in signal_breakdown:
                signal_breakdown[sig] = {"total": 0.0, "hits": 0.0, "misses": 0.0}
            signal_breakdown[sig]["total"] += 1.0
            if p["outcome"] == "HIT":
                signal_breakdown[sig]["hits"] += 1.0
            else:
                signal_breakdown[sig]["misses"] += 1.0

        for sig in signal_breakdown:
            s = signal_breakdown[sig]
            s["accuracy"] = (
                round(s["hits"] / s["total"] * 100, 1) if s["total"] > 0 else 0
            )

        # Average predicted vs actual return
        pred_returns = [
            p["predicted_return_pct"]
            for p in evaluated
            if p.get("predicted_return_pct") is not None
        ]
        actual_returns = [
            p["actual_return_pct"]
            for p in evaluated
            if p.get("actual_return_pct") is not None
        ]

        report = {
            "period": month_prefix,
            "total_predictions": total,
            "evaluated": len(evaluated),
            "pending": pending,
            "hits": hits,
            "misses": misses,
            "accuracy_pct": round(accuracy, 1),
            "signal_breakdown": signal_breakdown,
            "avg_predicted_return": (
                round(np.mean(pred_returns), 4) if pred_returns else 0
            ),
            "avg_actual_return": (
                round(np.mean(actual_returns), 4) if actual_returns else 0
            ),
            "days_with_data": len(daily_files),
        }

        # Save monthly report
        monthly_file = MONTHLY_DIR / f"{month_prefix}.json"
        monthly_file.write_text(json.dumps(report, indent=2, default=str))

        # Update running accuracy history
        PredictionTracker._update_accuracy_history(month_prefix, report)

        return report

    @staticmethod
    def _update_accuracy_history(period: str, report: dict):
        """Append to running accuracy history for model improvement tracking."""
        history = []
        if ACCURACY_FILE.exists():
            try:
                history = json.loads(ACCURACY_FILE.read_text())
            except Exception:
                history = []

        # Update or append
        existing_idx = next(
            (i for i, h in enumerate(history) if h["period"] == period), None
        )
        entry = {
            "period": period,
            "accuracy_pct": report["accuracy_pct"],
            "total": report["total_predictions"],
            "evaluated": report["evaluated"],
            "hits": report["hits"],
            "misses": report["misses"],
            "avg_predicted_return": report["avg_predicted_return"],
            "avg_actual_return": report["avg_actual_return"],
        }
        if existing_idx is not None:
            history[existing_idx] = entry
        else:
            history.append(entry)

        history.sort(key=lambda x: x["period"])
        ACCURACY_FILE.write_text(json.dumps(history, indent=2))

    @staticmethod
    def get_accuracy_history() -> list:
        """Get running monthly accuracy history for model improvement."""
        if ACCURACY_FILE.exists():
            try:
                return json.loads(ACCURACY_FILE.read_text())
            except Exception:
                return []
        return []

    @staticmethod
    def get_training_feedback_data() -> dict:
        """
        Export prediction outcomes in a format suitable for model retraining.

        Returns a dict with:
        - all_outcomes: list of (date, ticker, predicted, actual, hit/miss)
        - model_specific: per-model accuracy breakdown
        - confidence_calibration: accuracy by confidence bucket

        This data can be fed into the next training cycle to improve models.
        """
        all_files = sorted(DAILY_DIR.glob("*.json"))
        outcomes = []
        model_correct = {}  # model_name -> [correct_count, total_count]

        for f in all_files:
            try:
                day_data = json.loads(f.read_text())
                for ticker, pred in day_data.items():
                    if pred.get("outcome") not in ("HIT", "MISS"):
                        continue

                    strategy_price_at_open = PredictionTracker._to_float(
                        pred.get(
                            "strategy_price_at_open", pred.get("predicted_price", 0)
                        )
                    )
                    ai_last_prediction = PredictionTracker._to_float(
                        pred.get("ai_last_prediction", pred.get("predicted_price", 0))
                    )
                    actual_close = PredictionTracker._to_float(
                        pred.get("actual_close", 0)
                    )
                    direction_comparison = bool(pred.get("direction_comparison", False))

                    outcomes.append(
                        {
                            "date": f.stem,
                            "ticker": ticker,
                            "predicted_return": pred["predicted_return_pct"],
                            "actual_return": pred.get("actual_return_pct", 0),
                            "outcome": pred["outcome"],
                            "signal": pred.get("signal", "HOLD"),
                            "confidence": pred.get("confidence", 50),
                            "open_price": round(
                                PredictionTracker._to_float(pred.get("open_price", 0)),
                                2,
                            ),
                            "strategy_price_at_open": round(strategy_price_at_open, 2),
                            "ai_last_prediction": round(ai_last_prediction, 2),
                            "strategy_predicted_at_open": pred.get(
                                "strategy_predicted_at_open"
                            ),
                            "ai_predicted_at_open": pred.get("ai_predicted_at_open"),
                            "ai_last_prediction_at": pred.get("ai_last_prediction_at"),
                            "actual_close": round(actual_close, 2),
                            "direction_comparison": direction_comparison,
                            "direction_correct": direction_comparison,
                            "strategy_direction_at_open": pred.get(
                                "strategy_direction_at_open", "FLAT"
                            ),
                            "ai_direction_last": pred.get("ai_direction_last", "FLAT"),
                            "actual_direction": pred.get("actual_direction", "FLAT"),
                            "snapshot_type": pred.get("snapshot_type", "intraday"),
                            "checked_at": pred.get("checked_at"),
                        }
                    )

                    # Track per-model direction accuracy
                    actual_ret = pred.get("actual_return_pct", 0)
                    for model_name, model_pred in pred.get(
                        "model_predictions", {}
                    ).items():
                        if model_name not in model_correct:
                            model_correct[model_name] = [0, 0]
                        model_correct[model_name][1] += 1
                        # Check if model's direction was correct
                        if (model_pred > 0 and actual_ret > 0) or (
                            model_pred <= 0 and actual_ret <= 0
                        ):
                            model_correct[model_name][0] += 1
            except Exception:
                continue

        # Model accuracy breakdown
        model_accuracy = {}
        for name, (correct, total) in model_correct.items():
            model_accuracy[name] = {
                "correct": correct,
                "total": total,
                "accuracy_pct": round(correct / total * 100, 1) if total > 0 else 0,
            }

        # Confidence calibration
        conf_buckets = {
            "0-40": [0, 0],
            "40-55": [0, 0],
            "55-70": [0, 0],
            "70-85": [0, 0],
            "85-100": [0, 0],
        }
        for o in outcomes:
            conf = o["confidence"]
            if conf < 40:
                bucket = "0-40"
            elif conf < 55:
                bucket = "40-55"
            elif conf < 70:
                bucket = "55-70"
            elif conf < 85:
                bucket = "70-85"
            else:
                bucket = "85-100"
            conf_buckets[bucket][1] += 1
            if o["outcome"] == "HIT":
                conf_buckets[bucket][0] += 1

        conf_calibration = {}
        for bucket, (hits, total) in conf_buckets.items():
            conf_calibration[bucket] = {
                "hits": hits,
                "total": total,
                "accuracy_pct": round(hits / total * 100, 1) if total > 0 else 0,
            }

        return {
            "schema_version": PredictionTracker.SCHEMA_VERSION,
            "exported_at": datetime.now(IST).isoformat(),
            "total_outcomes": len(outcomes),
            "outcomes": outcomes,
            "model_accuracy": model_accuracy,
            "confidence_calibration": conf_calibration,
        }

    @staticmethod
    def get_daily_summary(date_str: Optional[str] = None) -> dict:
        """Get a summary of predictions and outcomes for a specific day."""
        if date_str is None:
            date_str = datetime.now(IST).strftime("%Y-%m-%d")

        file_path = DAILY_DIR / f"{date_str}.json"
        if not file_path.exists():
            return {"date": date_str, "predictions": [], "total": 0}

        try:
            data = json.loads(file_path.read_text())
        except Exception:
            return {"date": date_str, "predictions": [], "total": 0}

        predictions = []
        for ticker, pred in data.items():
            predictions.append(
                {
                    "ticker": ticker,
                    "predicted_return_pct": pred.get("predicted_return_pct", 0),
                    "predicted_price": pred.get("predicted_price", 0),
                    "price_at_prediction": pred.get("price_at_prediction", 0),
                    "open_price": pred.get("open_price"),
                    "strategy_price_at_open": pred.get("strategy_price_at_open"),
                    "ai_last_prediction": pred.get("ai_last_prediction"),
                    "strategy_predicted_at_open": pred.get(
                        "strategy_predicted_at_open"
                    ),
                    "ai_predicted_at_open": pred.get("ai_predicted_at_open"),
                    "ai_last_prediction_at": pred.get("ai_last_prediction_at"),
                    "strategy_direction_at_open": pred.get(
                        "strategy_direction_at_open"
                    ),
                    "ai_direction_last": pred.get("ai_direction_last"),
                    "direction_comparison": pred.get("direction_comparison"),
                    "snapshot_type": pred.get("snapshot_type", "intraday"),
                    "signal": pred.get("signal", "HOLD"),
                    "confidence": pred.get("confidence", 50),
                    "outcome": pred.get("outcome"),  # None, "HIT", "MISS"
                    "actual_close": pred.get("actual_close"),
                    "actual_return_pct": pred.get("actual_return_pct"),
                }
            )

        total = len(predictions)
        evaluated = [p for p in predictions if p["outcome"] in ("HIT", "MISS")]
        hits = sum(1 for p in evaluated if p["outcome"] == "HIT")

        return {
            "date": date_str,
            "total": total,
            "evaluated": len(evaluated),
            "hits": hits,
            "misses": len(evaluated) - hits,
            "accuracy_pct": (
                round(hits / len(evaluated) * 100, 1) if evaluated else None
            ),
            "predictions": predictions,
        }
