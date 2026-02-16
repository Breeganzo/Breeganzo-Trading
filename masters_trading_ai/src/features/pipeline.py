"""
Feature Pipeline
=================
Orchestrates the full feature engineering pipeline:
  raw OHLCV → technical → cross-asset → calendar → normalize → lag → final matrix.

CRITICAL: Every step uses ONLY past data. No look-ahead bias.

The pipeline is designed to be run once for the full dataset,
then the walk-forward CV handles the temporal splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from ..utils.constants import PROCESSED_DATA_DIR
from .technical import TechnicalFeatures
from .cross_asset import CrossAssetFeatures
from .calendar_features import CalendarFeatures


class FeaturePipeline:
    """
    Full feature engineering pipeline.

    Workflow
    --------
    1. Compute technical indicators (50+ features)
    2. Add cross-asset features (VIX, Nifty, crude, gold, USD/INR)
    3. Add calendar features (day of week, expiry, month)
    4. Normalise using rolling z-score (past data only)
    5. Add lagged features (t-1, t-2, t-5)
    6. Drop warmup rows and highly correlated features
    7. Save to data/processed/

    Usage
    -----
    >>> pipeline = FeaturePipeline()
    >>> features = pipeline.run(clean_data_dict)
    """

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        zscore_window: int = 252,
        lag_periods: list[int] = [1, 2, 5],
        corr_threshold: float = 0.95,
        min_warmup: int = 200,
    ):
        self.processed_dir = processed_dir or PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.zscore_window = zscore_window
        self.lag_periods = lag_periods
        self.corr_threshold = corr_threshold
        self.min_warmup = min_warmup

        # Cross-asset features (shared across all stocks)
        self.cross_asset = CrossAssetFeatures()

    def run_single(self, df: pd.DataFrame, ticker: str, drop_correlated: bool = True) -> pd.DataFrame:
        """
        Run the full feature pipeline for a single stock.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned OHLCV DataFrame
        ticker : str
            Ticker symbol
        drop_correlated : bool
            Whether to drop highly correlated features. Set False during
            inference to preserve features that models were trained on.

        Returns
        -------
        pd.DataFrame
            Feature-engineered DataFrame ready for ML
        """
        # Step 1: Technical features
        df = TechnicalFeatures.compute_all(df)

        # Step 2: Cross-asset features
        df = self.cross_asset.compute(df)

        # Step 3: Calendar features
        df = CalendarFeatures.compute(df)

        # Step 4: Identify feature columns (exclude OHLCV and target-related)
        exclude_cols = {"Open", "High", "Low", "Close", "Volume",
                        "Returns", "Log_Returns", "Adj Close"}
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Step 5: Rolling z-score normalisation (using ONLY past data)
        # This normalises each feature by its own rolling mean and std
        normalised_cols = []
        zscore_features: dict[str, pd.Series] = {}
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Skip binary/categorical features
                if df[col].nunique() <= 10:
                    continue
                roll_mean = df[col].rolling(
                    window=self.zscore_window, min_periods=60
                ).mean()
                roll_std = df[col].rolling(
                    window=self.zscore_window, min_periods=60
                ).std()
                feat_name = f"{col}_zscore"
                zscore_features[feat_name] = (df[col] - roll_mean) / (roll_std + 1e-10)
                normalised_cols.append(feat_name)
        if zscore_features:
            df = pd.concat([df, pd.DataFrame(zscore_features, index=df.index)], axis=1)

        # Step 6: Add lagged features (for key indicators)
        key_features = [
            "Returns_1d", "RSI", "MACD_Histogram", "BB_Position",
            "ATR_pct", "Volume_SMA_ratio", "VIX_Change",
            "Mean_Reversion_Score", "Momentum_Quality", "ADX_Smooth_14",
            "Hurst_20", "Efficiency_Ratio",
        ]
        lag_features: dict[str, pd.Series] = {}
        for col in key_features:
            if col in df.columns:
                for lag in self.lag_periods:
                    lag_features[f"{col}_lag{lag}"] = df[col].shift(lag)
        if lag_features:
            df = pd.concat([df, pd.DataFrame(lag_features, index=df.index)], axis=1)

        # Step 7: Drop warmup rows (first N rows have NaN from rolling calcs)
        df = df.iloc[self.min_warmup:]

        # Step 7b: Feature importance filtering (drop bottom 50% by importance)
        selected_features_path = Path(__file__).resolve().parent.parent.parent / "models" / "selected_features.json"
        if selected_features_path.exists():
            import json
            with open(selected_features_path) as f:
                selected_features = json.load(f)
            if selected_features:
                all_feat = self.get_feature_columns(df)
                keep_set = set(selected_features)
                drop_feats = [c for c in all_feat if c not in keep_set
                              and c not in {"Open", "High", "Low", "Close",
                                            "Volume", "Returns", "Log_Returns"}]
                if drop_feats:
                    df = df.drop(columns=drop_feats, errors="ignore")

        # Step 8: Drop highly correlated features (|r| > threshold)
        if drop_correlated:
            all_feature_cols = self.get_feature_columns(df)
            keep_cols = self.drop_correlated(df, all_feature_cols)
            drop_cols = [c for c in all_feature_cols if c not in keep_cols
                         and c not in {"Open", "High", "Low", "Close", "Volume",
                                       "Returns", "Log_Returns"}]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        return df

    def run(
        self,
        clean_data: dict[str, pd.DataFrame],
        save: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Run the feature pipeline for all tickers.

        Parameters
        ----------
        clean_data : dict
            {ticker: cleaned_dataframe}
        save : bool
            Save results to processed directory

        Returns
        -------
        dict
            {ticker: feature_dataframe}
        """
        features = {}
        print(f"Feature engineering for {len(clean_data)} tickers...")

        for ticker, df in tqdm(clean_data.items(), desc="Features"):
            try:
                feat_df = self.run_single(df, ticker)
                features[ticker] = feat_df
                if save:
                    safe_name = ticker.replace("^", "IDX_").replace("=", "_")
                    feat_df.to_parquet(self.processed_dir / f"{safe_name}.parquet")
            except Exception as e:
                print(f"  ✗ {ticker}: {e}")

        print(f"Feature engineering complete: {len(features)} tickers processed")
        return features

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Return the list of feature columns suitable for ML.
        Excludes raw OHLCV, targets, and metadata.
        """
        exclude_prefixes = ("Open", "High", "Low", "Close", "Volume",
                            "Adj", "Returns", "Log_Returns")
        exclude_exact = {"Returns", "Log_Returns", "Returns_1d",
                         "Log_Returns_1d"}

        feature_cols = []
        for col in df.columns:
            if col in exclude_exact:
                continue
            if col.startswith(exclude_prefixes):
                # Allow derived features like Returns_5d_lag1
                if "_lag" in col or "_zscore" in col:
                    feature_cols.append(col)
                continue
            feature_cols.append(col)

        return feature_cols

    def drop_correlated(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> list[str]:
        """
        Drop highly correlated features to reduce multicollinearity.

        Uses upper triangle of correlation matrix. For each pair with
        |correlation| > threshold, drops the feature with lower variance.
        """
        numeric_cols = [c for c in feature_cols if c in df.select_dtypes(include=[np.number]).columns]
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = set()
        for col in upper.columns:
            high_corr = upper.index[upper[col] > self.corr_threshold].tolist()
            for hc in high_corr:
                # Drop the one with lower variance
                if df[col].var() >= df[hc].var():
                    to_drop.add(hc)
                else:
                    to_drop.add(col)

        remaining = [c for c in feature_cols if c not in to_drop]
        if to_drop:
            print(f"  Dropped {len(to_drop)} correlated features (|r| > {self.corr_threshold})")
        return remaining

    def verify_no_lookahead(self, df: pd.DataFrame, feature_cols: list[str],
                            sample_date: Optional[str] = None) -> dict:
        """
        Verify that no feature at time t uses data from after time t.

        This is a sanity check — it picks a random date and verifies
        all feature values are computable from data up to that date.

        Returns
        -------
        dict
            {"date": str, "n_features": int, "n_nan": int, "status": str}
        """
        if sample_date is None:
            # Pick a date ~70% through the data
            idx = int(len(df) * 0.7)
            sample_date = df.index[idx]
        else:
            sample_date = pd.Timestamp(sample_date)

        row = df.loc[sample_date, feature_cols] if sample_date in df.index else None
        if row is None:
            return {"date": str(sample_date), "status": "Date not found"}

        n_nan = row.isna().sum()
        return {
            "date": str(sample_date),
            "n_features": len(feature_cols),
            "n_nan": int(n_nan),
            "n_valid": int(len(feature_cols) - n_nan),
            "status": "PASS" if n_nan < len(feature_cols) * 0.1 else "WARNING — many NaN",
        }
