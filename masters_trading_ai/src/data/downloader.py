"""
Data Downloader
================
Downloads OHLCV data from yfinance for all tickers in the universe.
Saves raw data as parquet files. Includes retry logic and rate limiting.
"""

import time
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from tqdm import tqdm

from ..utils.constants import RAW_DATA_DIR, EXTERNAL_DATA_DIR
from .universe import TickerUniverse


class DataDownloader:
    """
    Download historical OHLCV data for all tickers.

    Usage
    -----
    >>> dl = DataDownloader(universe)
    >>> results = dl.download_all(years=5)
    >>> print(results["summary"])
    """

    def __init__(
        self,
        universe: TickerUniverse,
        raw_dir: Optional[Path] = None,
        external_dir: Optional[Path] = None,
    ):
        self.universe = universe
        self.raw_dir = raw_dir or RAW_DATA_DIR
        self.external_dir = external_dir or EXTERNAL_DATA_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)

    def download_ticker(
        self,
        ticker: str,
        start: str,
        end: str,
        max_retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        Download OHLCV data for a single ticker with retry logic.

        Parameters
        ----------
        ticker : str
            yfinance ticker symbol (e.g., "RELIANCE.NS")
        start : str
            Start date "YYYY-MM-DD"
        end : str
            End date "YYYY-MM-DD"
        max_retries : int
            Number of retry attempts on failure

        Returns
        -------
        pd.DataFrame or None
            OHLCV DataFrame, or None if download failed.
        """
        for attempt in range(max_retries):
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                )
                if df is not None and len(df) > 0:
                    # Flatten multi-level columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.index.name = "Date"
                    return df
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"  ✗ Failed to download {ticker}: {e}")
        return None

    def download_all(
        self,
        years: int = 10,
        end_date: Optional[str] = None,
        batch_sleep: float = 0.5,
    ) -> dict:
        """
        Download all tickers in the universe.

        Parameters
        ----------
        years : int
            Number of years of historical data.
        end_date : str, optional
            End date. Defaults to today.
        batch_sleep : float
            Seconds to sleep between downloads (rate limiting).

        Returns
        -------
        dict
            {
                "success": [list of tickers],
                "failed": [list of tickers],
                "summary": pd.DataFrame with ticker, rows, date_range
            }
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=years * 365)
        ).strftime("%Y-%m-%d")

        success, failed = [], []
        summaries = []

        # Download trading tickers
        print(f"Downloading {len(self.universe)} trading tickers...")
        print(f"Period: {start_date} to {end_date}")
        print("-" * 60)

        for ticker in tqdm(self.universe.all_tickers, desc="Trading tickers"):
            df = self.download_ticker(ticker, start_date, end_date)
            if df is not None and len(df) > 0:
                # Save as parquet
                safe_name = ticker.replace("^", "IDX_").replace("=", "_")
                save_path = self.raw_dir / f"{safe_name}.parquet"
                df.to_parquet(save_path)
                success.append(ticker)
                summaries.append({
                    "Ticker": ticker,
                    "Bucket": self.universe.get_bucket(ticker),
                    "Rows": len(df),
                    "Start": df.index[0].strftime("%Y-%m-%d"),
                    "End": df.index[-1].strftime("%Y-%m-%d"),
                    "Status": "✓",
                })
            else:
                failed.append(ticker)
                summaries.append({
                    "Ticker": ticker,
                    "Bucket": self.universe.get_bucket(ticker),
                    "Rows": 0,
                    "Start": "",
                    "End": "",
                    "Status": "✗",
                })
            time.sleep(batch_sleep)

        # Download external tickers
        print(f"\nDownloading {len(self.universe.external)} external tickers...")
        for ticker in tqdm(self.universe.external, desc="External tickers"):
            df = self.download_ticker(ticker, start_date, end_date)
            if df is not None and len(df) > 0:
                safe_name = ticker.replace("^", "IDX_").replace("=", "_")
                save_path = self.external_dir / f"{safe_name}.parquet"
                df.to_parquet(save_path)
                success.append(ticker)
            else:
                failed.append(ticker)
            time.sleep(batch_sleep)

        summary_df = pd.DataFrame(summaries)
        print(f"\n{'='*60}")
        print(f"Download complete: {len(success)} success, {len(failed)} failed")
        if failed:
            print(f"Failed tickers: {failed}")

        return {
            "success": success,
            "failed": failed,
            "summary": summary_df,
        }

    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load a previously downloaded ticker from parquet."""
        safe_name = ticker.replace("^", "IDX_").replace("=", "_")
        # Check both raw and external dirs
        for dir_path in [self.raw_dir, self.external_dir]:
            path = dir_path / f"{safe_name}.parquet"
            if path.exists():
                return pd.read_parquet(path)
        return None

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all downloaded trading tickers as a dict."""
        data = {}
        for ticker in self.universe.all_tickers:
            df = self.load_ticker(ticker)
            if df is not None:
                data[ticker] = df
        return data
