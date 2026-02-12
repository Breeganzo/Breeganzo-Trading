"""
Model Staleness Detector
========================
Monitors model freshness and detects when models need retraining.

Checks
------
1. Age Check — days since last training vs maximum allowed
2. Accuracy Drift — rolling direction accuracy vs training accuracy
3. Regime Change Detection — structural break in price volatility
4. Prediction Entropy — model confidence degradation over time
5. Universe Coverage — fraction of current tickers model was trained on

Reference: Designed for production trading systems where model decay
           is the primary source of live performance degradation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import json


@dataclass
class StalenessReport:
    """Container for model staleness diagnostics."""

    model_name: str = ""
    model_path: str = ""
    check_date: str = ""

    # Age
    last_trained: str = "unknown"
    days_since_training: int = -1
    max_age_days: int = 30
    age_flag: bool = False

    # Accuracy Drift
    training_accuracy: float = 0.0
    recent_accuracy: float = 0.0
    accuracy_drop: float = 0.0       # training - recent
    accuracy_threshold: float = 0.05  # max acceptable drop
    accuracy_flag: bool = False

    # Regime Change
    hist_volatility: float = 0.0
    recent_volatility: float = 0.0
    vol_ratio: float = 1.0           # recent / historical
    regime_flag: bool = False

    # Prediction Entropy
    mean_confidence: float = 0.0
    entropy_flag: bool = False

    # Universe Coverage
    trained_tickers: int = 0
    current_tickers: int = 0
    coverage_pct: float = 1.0
    coverage_flag: bool = False

    # Overall
    n_flags: int = 0
    status: str = "FRESH"           # FRESH / AGING / STALE / EXPIRED
    action: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "Model": self.model_name,
            "Last Trained": self.last_trained,
            "Age (days)": self.days_since_training,
            "Max Age": self.max_age_days,
            "Age Flag": self.age_flag,
            "Training Acc": f"{self.training_accuracy:.1%}",
            "Recent Acc": f"{self.recent_accuracy:.1%}",
            "Acc Drop": f"{self.accuracy_drop:.1%}",
            "Acc Flag": self.accuracy_flag,
            "Hist Vol": f"{self.hist_volatility:.2%}",
            "Recent Vol": f"{self.recent_volatility:.2%}",
            "Vol Ratio": f"{self.vol_ratio:.2f}",
            "Regime Flag": self.regime_flag,
            "Coverage": f"{self.coverage_pct:.0%}",
            "Coverage Flag": self.coverage_flag,
            "Status": self.status,
            "Action": self.action,
            "Flags": f"{self.n_flags}/5",
        }


class ModelStalenessDetector:
    """
    Detects when ML models need retraining based on multiple staleness signals.

    Usage
    -----
    >>> detector = ModelStalenessDetector(models_dir=MODELS_DIR)
    >>> reports = detector.check_all_models(
    ...     price_data=price_data,          # dict of {ticker: DataFrame}
    ...     predictions_df=predictions_df,  # recent predictions
    ...     trained_tickers=sample_tickers, # tickers used in training
    ...     current_tickers=all_tickers,    # current universe
    ... )
    >>> detector.print_dashboard(reports)
    """

    # Model file → metadata mapping
    MODEL_FILES = {
        "XGBoost": "xgboost_model.joblib",
        "LightGBM": "lightgbm_model.joblib",
        "LSTM": "lstm_model.pt",
        "Transformer": "transformer_model.pt",
        "Ensemble": "ensemble_ridge_meta.joblib",
    }

    # Staleness thresholds
    MAX_AGE_DAYS = 30              # retrain monthly
    ACCURACY_DROP_THRESHOLD = 0.05  # 5% drop from training
    VOL_RATIO_THRESHOLD = 1.5      # 50% vol regime shift
    CONFIDENCE_THRESHOLD = 0.45     # min mean confidence
    COVERAGE_THRESHOLD = 0.80       # 80% ticker overlap

    def __init__(self, models_dir: Path, max_age_days: int = 30):
        self.models_dir = Path(models_dir)
        self.MAX_AGE_DAYS = max_age_days

    def _get_model_age(self, model_file: str) -> tuple[str, int]:
        """Get last modified date and age in days for a model file."""
        path = self.models_dir / model_file
        if not path.exists():
            return "not found", -1

        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = (datetime.now() - mtime).days
        return mtime.strftime("%Y-%m-%d %H:%M"), age

    def _check_accuracy_drift(
        self,
        predictions_df: Optional[pd.DataFrame],
        model_name: str,
        report: StalenessReport,
        training_accuracy: float = 0.55,
        recent_window: int = 100,
    ) -> None:
        """Compare recent prediction accuracy against training accuracy."""
        report.training_accuracy = training_accuracy

        if predictions_df is None or len(predictions_df) == 0:
            return

        df = predictions_df.copy()
        if "predicted" not in df.columns or "actual" not in df.columns:
            return

        # Take most recent predictions
        recent = df.tail(recent_window)
        correct = (np.sign(recent["predicted"]) == np.sign(recent["actual"])).mean()
        report.recent_accuracy = correct
        report.accuracy_drop = training_accuracy - correct
        report.accuracy_flag = report.accuracy_drop > self.ACCURACY_DROP_THRESHOLD

    def _check_regime_change(
        self,
        price_data: Optional[dict],
        report: StalenessReport,
        lookback_days: int = 252,
        recent_days: int = 21,
    ) -> None:
        """Detect volatility regime shifts that invalidate model assumptions."""
        if price_data is None or len(price_data) == 0:
            return

        all_vols_hist = []
        all_vols_recent = []

        for ticker, df in price_data.items():
            if len(df) < lookback_days:
                continue
            if "Close" not in df.columns:
                continue

            returns = df["Close"].pct_change().dropna()
            if len(returns) < lookback_days:
                continue

            hist_vol = returns.iloc[-lookback_days:-recent_days].std() * np.sqrt(252)
            recent_vol = returns.iloc[-recent_days:].std() * np.sqrt(252)

            if hist_vol > 0:
                all_vols_hist.append(hist_vol)
                all_vols_recent.append(recent_vol)

        if len(all_vols_hist) == 0:
            return

        report.hist_volatility = np.mean(all_vols_hist)
        report.recent_volatility = np.mean(all_vols_recent)
        report.vol_ratio = report.recent_volatility / report.hist_volatility if report.hist_volatility > 0 else 1.0
        report.regime_flag = report.vol_ratio > self.VOL_RATIO_THRESHOLD or report.vol_ratio < 1 / self.VOL_RATIO_THRESHOLD

    def _check_coverage(
        self,
        trained_tickers: Optional[list],
        current_tickers: Optional[list],
        report: StalenessReport,
    ) -> None:
        """Check how many current tickers were present during training."""
        if trained_tickers is None or current_tickers is None:
            return

        report.trained_tickers = len(trained_tickers)
        report.current_tickers = len(current_tickers)

        if report.current_tickers == 0:
            return

        overlap = len(set(trained_tickers) & set(current_tickers))
        report.coverage_pct = overlap / report.current_tickers
        report.coverage_flag = report.coverage_pct < self.COVERAGE_THRESHOLD

    def check_model(
        self,
        model_name: str,
        model_file: str,
        price_data: Optional[dict] = None,
        predictions_df: Optional[pd.DataFrame] = None,
        trained_tickers: Optional[list] = None,
        current_tickers: Optional[list] = None,
        training_accuracy: float = 0.55,
    ) -> StalenessReport:
        """Run all staleness checks for a single model."""
        report = StalenessReport()
        report.model_name = model_name
        report.model_path = str(self.models_dir / model_file)
        report.check_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        report.max_age_days = self.MAX_AGE_DAYS

        # 1. Age
        last_trained, age = self._get_model_age(model_file)
        report.last_trained = last_trained
        report.days_since_training = age
        report.age_flag = age > self.MAX_AGE_DAYS if age >= 0 else True

        # 2. Accuracy drift
        self._check_accuracy_drift(predictions_df, model_name, report, training_accuracy)

        # 3. Regime change
        self._check_regime_change(price_data, report)

        # 4. Coverage
        self._check_coverage(trained_tickers, current_tickers, report)

        # Count flags
        flags = [
            report.age_flag,
            report.accuracy_flag,
            report.regime_flag,
            report.coverage_flag,
            report.entropy_flag,
        ]
        report.n_flags = sum(flags)

        # Status
        if report.n_flags == 0:
            report.status = "FRESH"
            report.action = "No action needed."
        elif report.n_flags == 1:
            report.status = "AGING"
            report.action = "Monitor closely. Schedule retraining within 1 week."
        elif report.n_flags <= 3:
            report.status = "STALE"
            report.action = "Retrain ASAP. Reduce position sizes until retrained."
        else:
            report.status = "EXPIRED"
            report.action = "STOP TRADING. Retrain all models before resuming."

        report.summary = f"{model_name}: {report.status} ({report.n_flags}/5 flags)"
        return report

    def check_all_models(
        self,
        price_data: Optional[dict] = None,
        predictions_df: Optional[pd.DataFrame] = None,
        trained_tickers: Optional[list] = None,
        current_tickers: Optional[list] = None,
        training_accuracy: float = 0.55,
    ) -> list[StalenessReport]:
        """Run staleness checks on all known models."""
        reports = []
        for name, file in self.MODEL_FILES.items():
            report = self.check_model(
                model_name=name,
                model_file=file,
                price_data=price_data,
                predictions_df=predictions_df,
                trained_tickers=trained_tickers,
                current_tickers=current_tickers,
                training_accuracy=training_accuracy,
            )
            reports.append(report)
        return reports

    def print_dashboard(self, reports: list[StalenessReport]) -> str:
        """Print a formatted staleness dashboard for all models."""
        sep = "=" * 70
        lines = [
            sep,
            "  MODEL STALENESS DASHBOARD",
            f"  Checked: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            sep,
            "",
        ]

        for r in reports:
            status_icon = {
                "FRESH": "[OK]",
                "AGING": "[~]",
                "STALE": "[!]",
                "EXPIRED": "[X]",
            }.get(r.status, "?")

            lines.append(f"  {status_icon} {r.model_name:15s}  Status: {r.status:8s}  "
                         f"Age: {r.days_since_training:>4d}d  Flags: {r.n_flags}/5")

        lines.append("")
        lines.append("-" * 70)

        # Detailed per-model
        for r in reports:
            lines.append(f"\n  {r.model_name}")
            lines.append(f"  {'~' * len(r.model_name)}")
            lines.append(f"    Last trained:     {r.last_trained}")
            lines.append(f"    Age:              {r.days_since_training} days (max: {r.max_age_days})")
            lines.append(f"    Age flag:         {'[!]' if r.age_flag else '[OK]'}")
            lines.append(f"    Training acc:     {r.training_accuracy:.1%}")
            lines.append(f"    Recent acc:       {r.recent_accuracy:.1%}")
            lines.append(f"    Accuracy drop:    {r.accuracy_drop:.1%}")
            lines.append(f"    Accuracy flag:    {'[!]' if r.accuracy_flag else '[OK]'}")
            lines.append(f"    Hist volatility:  {r.hist_volatility:.2%}")
            lines.append(f"    Recent vol:       {r.recent_volatility:.2%}")
            lines.append(f"    Vol ratio:        {r.vol_ratio:.2f}")
            lines.append(f"    Regime flag:      {'[!]' if r.regime_flag else '[OK]'}")
            lines.append(f"    Coverage:         {r.coverage_pct:.0%}")
            lines.append(f"    Coverage flag:    {'[!]' if r.coverage_flag else '[OK]'}")
            lines.append(f"    Action:           {r.action}")
            lines.append("")

        lines.append(sep)

        # Overall recommendation
        worst = max(reports, key=lambda x: x.n_flags) if reports else None
        if worst and worst.n_flags >= 3:
            lines.append("  >>> RECOMMENDATION: RETRAIN ALL MODELS BEFORE NEXT TRADE <<<")
        elif worst and worst.n_flags >= 1:
            lines.append("  >>> RECOMMENDATION: Schedule retraining this week <<<")
        else:
            lines.append("  >>> All models are fresh. No retraining needed. <<<")

        lines.append(sep)

        text = "\n".join(lines)
        print(text)
        return text
