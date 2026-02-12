"""
Cross-Asset Feature Engineering
================================
Adds global/macro features: India VIX, Nifty returns, crude oil,
gold, USD/INR changes. These capture macro regime and risk sentiment.

All features use only past data (no look-ahead bias).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from ..utils.constants import EXTERNAL_DATA_DIR


class CrossAssetFeatures:
    """
    Merge cross-asset features into each stock's feature DataFrame.

    These features are the same for all stocks on a given day,
    capturing market-wide sentiment and macro conditions.
    """

    def __init__(self, external_dir: Optional[Path] = None):
        self.external_dir = external_dir or EXTERNAL_DATA_DIR
        self._load_external_data()

    def _load_external_data(self):
        """Load pre-downloaded external data."""
        self.external = {}
        mappings = {
            "IDX_NSEI": "Nifty",
            "IDX_NSEBANK": "BankNifty",
            "IDX_INDIAVIX": "VIX",
            "USDINR_X": "USDINR",
            "GC_F": "Gold",
            "CL_F": "Crude",
        }
        for filename, label in mappings.items():
            path = self.external_dir / f"{filename}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                if "Close" in df.columns:
                    self.external[label] = df["Close"]
                elif "Adj Close" in df.columns:
                    self.external[label] = df["Adj Close"]

    def compute(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-asset features to a stock DataFrame.

        Features Added
        --------------
        - Nifty_Return_1d: Nifty 50 daily return
        - Nifty_Return_5d: Nifty 50 5-day return
        - BankNifty_Return_1d: Bank Nifty daily return
        - VIX_Level: India VIX level
        - VIX_Change: India VIX daily change
        - VIX_SMA5_ratio: VIX relative to its 5-day SMA
        - USDINR_Change: USD/INR daily change
        - Gold_Return_1d: Gold daily return
        - Crude_Return_1d: Crude oil daily return
        - Crude_Return_5d: Crude oil 5-day return

        Parameters
        ----------
        stock_df : pd.DataFrame
            Stock's OHLCV + technical features DataFrame

        Returns
        -------
        pd.DataFrame
            With cross-asset features added
        """
        df = stock_df.copy()

        for label, series in self.external.items():
            # Align to stock's dates via forward-fill (use last known value)
            aligned = series.reindex(df.index, method="ffill")

            if label == "VIX":
                # VIX is a level, not a price — use directly
                df["VIX_Level"] = aligned
                df["VIX_Change"] = aligned.pct_change()
                vix_sma5 = aligned.rolling(5, min_periods=5).mean()
                df["VIX_SMA5_ratio"] = aligned / (vix_sma5 + 1e-10)
                # VIX percentile (rolling 252-day)
                df["VIX_Percentile"] = aligned.rolling(252, min_periods=60).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )
            else:
                # Compute returns
                ret_1d = aligned.pct_change()
                ret_5d = aligned.pct_change(5)
                df[f"{label}_Return_1d"] = ret_1d
                df[f"{label}_Return_5d"] = ret_5d

        return df


def compute_fii_dii_features(fii_dii_df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add FII/DII flow features if data is available.

    Parameters
    ----------
    fii_dii_df : pd.DataFrame
        Must have columns: Date, FII_Net, DII_Net (in ₹ crores)
    stock_df : pd.DataFrame
        Stock DataFrame to merge into

    Returns
    -------
    pd.DataFrame
    """
    df = stock_df.copy()
    if fii_dii_df is not None and len(fii_dii_df) > 0:
        fii_dii = fii_dii_df.set_index("Date") if "Date" in fii_dii_df.columns else fii_dii_df
        fii_dii = fii_dii.reindex(df.index, method="ffill")

        if "FII_Net" in fii_dii.columns:
            df["FII_Net_Flow"] = fii_dii["FII_Net"]
            df["FII_5d_Flow"] = fii_dii["FII_Net"].rolling(5, min_periods=1).sum()
        if "DII_Net" in fii_dii.columns:
            df["DII_Net_Flow"] = fii_dii["DII_Net"]
            df["DII_5d_Flow"] = fii_dii["DII_Net"].rolling(5, min_periods=1).sum()

    return df
