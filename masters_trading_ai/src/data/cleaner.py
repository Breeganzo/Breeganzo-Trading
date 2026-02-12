"""
Data Cleaner
=============
Handles missing data, stock splits, outliers, and data quality checks.
Produces clean, analysis-ready DataFrames saved to data/interim/.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from ..utils.constants import INTERIM_DATA_DIR


class DataCleaner:
    """
    Clean raw OHLCV data: forward-fill gaps, remove outliers,
    check for stock splits, and filter low-quality tickers.

    Usage
    -----
    >>> cleaner = DataCleaner()
    >>> clean_data = cleaner.clean_all(raw_data_dict)
    """

    def __init__(
        self,
        interim_dir: Optional[Path] = None,
        max_missing_pct: float = 0.20,
        forward_fill_limit: int = 3,
        outlier_sigma: float = 5.0,
        min_trading_days: int = 200,
    ):
        self.interim_dir = interim_dir or INTERIM_DATA_DIR
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.max_missing_pct = max_missing_pct
        self.forward_fill_limit = forward_fill_limit
        self.outlier_sigma = outlier_sigma
        self.min_trading_days = min_trading_days

    def clean_single(self, df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """
        Clean a single ticker's OHLCV data.

        Steps
        -----
        1. Use Adj Close for adjusted prices (accounts for splits/dividends)
        2. Forward-fill small gaps (≤3 consecutive days)
        3. Drop rows where all OHLCV are NaN
        4. Cap outlier returns at ±5σ
        5. Check minimum data length
        6. Add daily returns column

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame with DatetimeIndex
        ticker : str
            Ticker symbol for logging

        Returns
        -------
        pd.DataFrame or None
            Cleaned DataFrame, or None if data quality is insufficient.
        """
        if df is None or len(df) == 0:
            return None

        df = df.copy()

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort by date
        df = df.sort_index()

        # Use Adj Close if available (adjusts for splits and dividends)
        if "Adj Close" in df.columns:
            # Calculate adjustment ratio
            adj_ratio = df["Adj Close"] / df["Close"]
            # Apply to OHLC
            for col in ["Open", "High", "Low", "Close"]:
                if col in df.columns:
                    df[col] = df[col] * adj_ratio
            df.drop(columns=["Adj Close"], inplace=True, errors="ignore")

        # Check missing data percentage
        missing_pct = df[["Open", "High", "Low", "Close"]].isnull().mean().mean()
        if missing_pct > self.max_missing_pct:
            print(f"  ⚠ {ticker}: {missing_pct:.1%} missing data — SKIPPED")
            return None

        # Forward-fill small gaps
        df = df.ffill(limit=self.forward_fill_limit)

        # Drop any remaining rows with NaN in OHLC
        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        # Fill volume NaN with 0
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].fillna(0)

        # Check minimum length
        if len(df) < self.min_trading_days:
            print(f"  ⚠ {ticker}: only {len(df)} days — SKIPPED (need {self.min_trading_days})")
            return None

        # Cap outlier returns
        df["Returns"] = df["Close"].pct_change()
        mean_ret = df["Returns"].mean()
        std_ret = df["Returns"].std()
        upper = mean_ret + self.outlier_sigma * std_ret
        lower = mean_ret - self.outlier_sigma * std_ret
        n_outliers = ((df["Returns"] > upper) | (df["Returns"] < lower)).sum()
        if n_outliers > 0:
            df["Returns"] = df["Returns"].clip(lower=lower, upper=upper)
            # Reconstruct Close from capped returns
            returns_for_recon = df["Returns"].copy()
            returns_for_recon.iloc[0] = 0.0  # First row has NaN from pct_change
            df["Close"] = df["Close"].iloc[0] * (1 + returns_for_recon).cumprod()

        # Ensure High >= Low and High >= Close, Low <= Close
        df["High"] = df[["High", "Close", "Open"]].max(axis=1)
        df["Low"] = df[["Low", "Close", "Open"]].min(axis=1)

        # Add log returns
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))

        return df

    def clean_all(self, raw_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Clean all tickers and save to interim directory.

        Parameters
        ----------
        raw_data : dict
            {ticker: raw_dataframe}

        Returns
        -------
        dict
            {ticker: cleaned_dataframe} — only tickers that passed quality checks.
        """
        clean_data = {}
        dropped = []

        print(f"Cleaning {len(raw_data)} tickers...")
        for ticker, df in tqdm(raw_data.items(), desc="Cleaning"):
            cleaned = self.clean_single(df, ticker)
            if cleaned is not None:
                clean_data[ticker] = cleaned
                # Save to interim
                safe_name = ticker.replace("^", "IDX_").replace("=", "_")
                cleaned.to_parquet(self.interim_dir / f"{safe_name}.parquet")
            else:
                dropped.append(ticker)

        print(f"\nCleaning complete: {len(clean_data)} passed, {len(dropped)} dropped")
        if dropped:
            print(f"Dropped: {dropped}")

        return clean_data

    def quality_report(self, raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate a data quality report for all tickers."""
        rows = []
        for ticker, df in raw_data.items():
            if df is None or len(df) == 0:
                rows.append({"Ticker": ticker, "Rows": 0, "Missing%": 1.0,
                             "Status": "Empty"})
                continue

            missing = df[["Open", "High", "Low", "Close"]].isnull().mean().mean()
            rows.append({
                "Ticker": ticker,
                "Rows": len(df),
                "Start": df.index[0].strftime("%Y-%m-%d") if len(df) > 0 else "",
                "End": df.index[-1].strftime("%Y-%m-%d") if len(df) > 0 else "",
                "Missing%": f"{missing:.2%}",
                "Avg_Volume": f"{df['Volume'].mean():,.0f}" if "Volume" in df.columns else "N/A",
                "Status": "OK" if missing <= self.max_missing_pct and len(df) >= self.min_trading_days else "FAIL",
            })

        return pd.DataFrame(rows).sort_values("Status", ascending=False)
