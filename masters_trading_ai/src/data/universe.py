"""
Ticker Universe Manager
========================
Loads tickers from config/tickers.yaml, organises them into buckets,
and provides utility methods for filtering and validation.
"""

import yaml
import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from ..utils.constants import CONFIG_DIR


class TickerUniverse:
    """Manage the stock universe from tickers.yaml."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_DIR / "tickers.yaml"
        self._load()

    def _load(self):
        """Load tickers from YAML config."""
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Trading universe (all buckets except external/options)
        self.buckets = {
            "large_cap": self.config.get("large_cap", []),
            "banking": self.config.get("banking", []),
            "mid_cap": self.config.get("mid_cap", []),
            "high_volatility": self.config.get("high_volatility", []),
            "commodities": self.config.get("commodities", []),
        }

        # External tickers for cross-asset features (not for trading)
        self.external = self.config.get("external", [])

        # Options underlyings
        self.options_underlyings = self.config.get("options_underlyings", [])

        # Build flat list and ticker-to-bucket mapping
        self.all_tickers = []
        self.ticker_to_bucket = {}
        for bucket_name, tickers in self.buckets.items():
            for t in tickers:
                self.all_tickers.append(t)
                self.ticker_to_bucket[t] = bucket_name

    def get_tickers(self, bucket: Optional[str] = None) -> list[str]:
        """
        Get tickers, optionally filtered by bucket.

        Parameters
        ----------
        bucket : str, optional
            One of: large_cap, banking, mid_cap, high_volatility, commodities.
            If None, returns all trading tickers.

        Returns
        -------
        list[str]
            List of ticker symbols.
        """
        if bucket is None:
            return self.all_tickers.copy()
        if bucket not in self.buckets:
            raise ValueError(
                f"Unknown bucket '{bucket}'. Valid: {list(self.buckets.keys())}"
            )
        return self.buckets[bucket].copy()

    def get_bucket(self, ticker: str) -> str:
        """Return the bucket name for a given ticker."""
        return self.ticker_to_bucket.get(ticker, "unknown")

    def get_all_download_tickers(self) -> list[str]:
        """Return all tickers including external (for download)."""
        return self.all_tickers + self.external

    def validate_tickers(self, sample_period: str = "5d") -> dict:
        """
        Check which tickers are valid by attempting a small download.

        Returns
        -------
        dict
            {"valid": [...], "invalid": [...]}
        """
        valid, invalid = [], []
        for ticker in tqdm(self.all_tickers, desc="Validating tickers"):
            try:
                df = yf.download(ticker, period=sample_period, progress=False)
                if df is not None and len(df) > 0:
                    valid.append(ticker)
                else:
                    invalid.append(ticker)
            except Exception:
                invalid.append(ticker)
        return {"valid": valid, "invalid": invalid}

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of the universe."""
        rows = []
        for bucket_name, tickers in self.buckets.items():
            for t in tickers:
                rows.append({"Ticker": t, "Bucket": bucket_name})
        df = pd.DataFrame(rows)
        return df

    def __len__(self) -> int:
        return len(self.all_tickers)

    def __repr__(self) -> str:
        counts = {k: len(v) for k, v in self.buckets.items()}
        return f"TickerUniverse({len(self)} tickers: {counts})"
