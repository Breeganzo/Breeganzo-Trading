"""
Prediction Logger — logs daily predictions to CSV for accuracy tracking.

Logs: Date, Ticker, Predicted_Return, Signal, Confidence, Model_Agreement,
      Actual_Close (filled next day), Actual_Return (filled next day).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import json


class PredictionLogger:
    """Append-only CSV logger for daily ML predictions."""

    COLUMNS = [
        "Date", "Ticker", "Predicted_Return", "Signal", "Confidence",
        "Model_Agreement", "Entry_Price", "Target_Price", "Stop_Loss",
        "XGB_Pred", "LGB_Pred", "LSTM_Pred", "TF_Pred",
        "Actual_Close", "Actual_Return", "Hit_Target", "Hit_Stop",
    ]

    def __init__(self, log_dir: str | Path = None):
        if log_dir is None:
            from ..utils.constants import PROJECT_ROOT
            log_dir = PROJECT_ROOT / "data" / "predictions"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "prediction_log.csv"

        # Create file with headers if it doesn't exist
        if not self.log_file.exists():
            pd.DataFrame(columns=self.COLUMNS).to_csv(self.log_file, index=False)

    def log_predictions(self, predictions: list[dict]) -> int:
        """
        Log a batch of predictions.

        Parameters
        ----------
        predictions : list[dict]
            Each dict should have keys matching COLUMNS (at minimum:
            Date, Ticker, Predicted_Return, Signal).

        Returns
        -------
        int : number of predictions logged
        """
        if not predictions:
            return 0

        today = date.today().isoformat()
        rows = []
        for pred in predictions:
            row = {col: np.nan for col in self.COLUMNS}
            row["Date"] = pred.get("Date", today)
            row["Ticker"] = pred.get("Ticker", "")
            row["Predicted_Return"] = pred.get("Predicted_Return", np.nan)
            row["Signal"] = pred.get("Signal", "HOLD")
            row["Confidence"] = pred.get("Confidence", np.nan)
            row["Model_Agreement"] = pred.get("Model_Agreement", np.nan)
            row["Entry_Price"] = pred.get("Entry_Price", np.nan)
            row["Target_Price"] = pred.get("Target_Price", np.nan)
            row["Stop_Loss"] = pred.get("Stop_Loss", np.nan)
            row["XGB_Pred"] = pred.get("XGB_Pred", np.nan)
            row["LGB_Pred"] = pred.get("LGB_Pred", np.nan)
            row["LSTM_Pred"] = pred.get("LSTM_Pred", np.nan)
            row["TF_Pred"] = pred.get("TF_Pred", np.nan)
            rows.append(row)

        df = pd.DataFrame(rows, columns=self.COLUMNS)

        # Remove any existing entries for the same date+ticker to avoid dupes
        existing = self._load()
        if len(existing) > 0:
            existing = existing[
                ~((existing["Date"] == today) &
                  (existing["Ticker"].isin(df["Ticker"].values)))
            ]
            df = pd.concat([existing, df], ignore_index=True)

        df.to_csv(self.log_file, index=False)
        return len(rows)

    def update_actuals(self, actuals: dict[str, dict]) -> int:
        """
        Fill in actual prices/returns for past predictions.

        Parameters
        ----------
        actuals : dict
            {date_str: {ticker: {"Actual_Close": float, "Actual_Return": float,
                                  "Hit_Target": bool, "Hit_Stop": bool}}}

        Returns
        -------
        int : number of rows updated
        """
        df = self._load()
        if len(df) == 0:
            return 0

        updated = 0
        for date_str, tickers in actuals.items():
            for ticker, vals in tickers.items():
                mask = (df["Date"] == date_str) & (df["Ticker"] == ticker)
                if mask.any():
                    for key, val in vals.items():
                        if key in df.columns:
                            df.loc[mask, key] = val
                    updated += mask.sum()

        df.to_csv(self.log_file, index=False)
        return updated

    def get_predictions(self, date_str: str = None,
                        ticker: str = None) -> pd.DataFrame:
        """Get predictions, optionally filtered by date and/or ticker."""
        df = self._load()
        if date_str:
            df = df[df["Date"] == date_str]
        if ticker:
            df = df[df["Ticker"] == ticker]
        return df

    def get_recent(self, days: int = 30) -> pd.DataFrame:
        """Get predictions from the last N days."""
        df = self._load()
        if len(df) == 0:
            return df
        df["Date"] = pd.to_datetime(df["Date"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        return df[df["Date"] >= cutoff].sort_values("Date", ascending=False)

    def _load(self) -> pd.DataFrame:
        """Load the prediction log CSV."""
        if self.log_file.exists():
            try:
                return pd.read_csv(self.log_file)
            except pd.errors.EmptyDataError:
                return pd.DataFrame(columns=self.COLUMNS)
        return pd.DataFrame(columns=self.COLUMNS)
