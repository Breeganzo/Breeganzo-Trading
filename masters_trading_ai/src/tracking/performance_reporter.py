"""
Performance Reporter — computes accuracy, precision, recall, F1 from prediction logs.

Generates human-readable daily and cumulative performance reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta


class PerformanceReporter:
    """Compute and format trading prediction performance metrics."""

    def __init__(self, log_dir: str | Path = None):
        if log_dir is None:
            from ..utils.constants import PROJECT_ROOT
            log_dir = PROJECT_ROOT / "data" / "predictions"
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / "prediction_log.csv"

    def _load(self) -> pd.DataFrame:
        if self.log_file.exists():
            try:
                df = pd.read_csv(self.log_file)
                df["Date"] = pd.to_datetime(df["Date"])
                return df
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
        return pd.DataFrame()

    def compute_metrics(self, df: pd.DataFrame = None,
                        days: int = None) -> dict:
        """
        Compute accuracy, precision, recall, F1 for BUY signals.

        Parameters
        ----------
        df : DataFrame, optional
            Prediction log. If None, loads from disk.
        days : int, optional
            Restrict to last N days.

        Returns
        -------
        dict with keys: total_predictions, evaluated, accuracy,
              precision_buy, recall_buy, f1_buy, avg_return,
              win_rate, avg_win, avg_loss, profit_factor
        """
        if df is None:
            df = self._load()
        if len(df) == 0:
            return self._empty_metrics()

        if days:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
            df = df[df["Date"] >= cutoff]

        total = len(df)

        # Only evaluate rows where actual returns are filled
        evaluated = df.dropna(subset=["Actual_Return"])
        n_eval = len(evaluated)

        if n_eval == 0:
            metrics = self._empty_metrics()
            metrics["total_predictions"] = total
            return metrics

        # Direction accuracy: predicted direction matches actual direction
        pred_dir = (evaluated["Predicted_Return"] > 0).astype(int)
        actual_dir = (evaluated["Actual_Return"] > 0).astype(int)
        accuracy = (pred_dir == actual_dir).mean()

        # BUY signal precision/recall
        buy_signals = evaluated["Signal"] == "BUY"
        actual_positive = evaluated["Actual_Return"] > 0

        tp = (buy_signals & actual_positive).sum()
        fp = (buy_signals & ~actual_positive).sum()
        fn = (~buy_signals & actual_positive).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        # Return-based metrics
        buy_returns = evaluated.loc[buy_signals, "Actual_Return"]
        avg_return = buy_returns.mean() if len(buy_returns) > 0 else 0.0
        wins = buy_returns[buy_returns > 0]
        losses = buy_returns[buy_returns <= 0]
        win_rate = len(wins) / len(buy_returns) if len(buy_returns) > 0 else 0.0
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
        profit_factor = (wins.sum() / abs(losses.sum())
                         if len(losses) > 0 and losses.sum() != 0 else float('inf'))

        # Hit rate
        hit_target = evaluated.get("Hit_Target", pd.Series(dtype=float))
        hit_stop = evaluated.get("Hit_Stop", pd.Series(dtype=float))
        target_hit_rate = (hit_target == True).mean() if len(hit_target.dropna()) > 0 else np.nan
        stop_hit_rate = (hit_stop == True).mean() if len(hit_stop.dropna()) > 0 else np.nan

        return {
            "total_predictions": total,
            "evaluated": n_eval,
            "direction_accuracy": round(accuracy, 4),
            "precision_buy": round(precision, 4),
            "recall_buy": round(recall, 4),
            "f1_buy": round(f1, 4),
            "avg_return_buy": round(avg_return, 6),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 6),
            "avg_loss": round(avg_loss, 6),
            "profit_factor": round(profit_factor, 4),
            "target_hit_rate": round(target_hit_rate, 4) if not np.isnan(target_hit_rate) else None,
            "stop_hit_rate": round(stop_hit_rate, 4) if not np.isnan(stop_hit_rate) else None,
        }

    def generate_report(self, days: int = None) -> str:
        """Generate a human-readable performance report string."""
        m = self.compute_metrics(days=days)
        period = f"Last {days} days" if days else "All time"

        report = []
        report.append("=" * 60)
        report.append(f"  TRADING PREDICTION PERFORMANCE REPORT")
        report.append(f"  Period: {period}")
        report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
        report.append("=" * 60)
        report.append("")
        report.append(f"  Total Predictions:    {m['total_predictions']}")
        report.append(f"  Evaluated (w/ actuals): {m['evaluated']}")
        report.append("")
        report.append("  --- Signal Quality ---")
        report.append(f"  Direction Accuracy:   {m['direction_accuracy']:.1%}")
        report.append(f"  BUY Precision:        {m['precision_buy']:.1%}")
        report.append(f"  BUY Recall:           {m['recall_buy']:.1%}")
        report.append(f"  BUY F1 Score:         {m['f1_buy']:.1%}")
        report.append("")
        report.append("  --- Return Metrics ---")
        report.append(f"  Avg Return (BUY):     {m['avg_return_buy']:.4%}")
        report.append(f"  Win Rate:             {m['win_rate']:.1%}")
        report.append(f"  Avg Win:              {m['avg_win']:.4%}")
        report.append(f"  Avg Loss:             {m['avg_loss']:.4%}")
        report.append(f"  Profit Factor:        {m['profit_factor']:.2f}")
        if m.get("target_hit_rate") is not None:
            report.append(f"  Target Hit Rate:      {m['target_hit_rate']:.1%}")
        if m.get("stop_hit_rate") is not None:
            report.append(f"  Stop Hit Rate:        {m['stop_hit_rate']:.1%}")
        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def daily_summary(self, date_str: str = None) -> dict:
        """Get summary for a specific day."""
        df = self._load()
        if len(df) == 0:
            return {"date": date_str, "buys": 0, "sells": 0, "holds": 0}

        if date_str is None:
            date_str = date.today().isoformat()

        day_df = df[df["Date"].dt.strftime("%Y-%m-%d") == date_str]

        return {
            "date": date_str,
            "total": len(day_df),
            "buys": (day_df["Signal"] == "BUY").sum(),
            "sells": (day_df["Signal"] == "SELL").sum(),
            "holds": (day_df["Signal"] == "HOLD").sum(),
            "avg_predicted_return": day_df["Predicted_Return"].mean(),
            "avg_confidence": day_df["Confidence"].mean(),
            "avg_model_agreement": day_df["Model_Agreement"].mean(),
        }

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "total_predictions": 0, "evaluated": 0,
            "direction_accuracy": 0.0, "precision_buy": 0.0,
            "recall_buy": 0.0, "f1_buy": 0.0, "avg_return_buy": 0.0,
            "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "target_hit_rate": None,
            "stop_hit_rate": None,
        }
