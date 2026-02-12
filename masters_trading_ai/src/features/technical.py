"""
Technical Feature Engineering
==============================
Computes 50+ technical indicators using only past data (no look-ahead bias).

IMPORTANT: Every rolling window and shift operation uses ONLY past data.
- We never use .shift(-n) (negative shift = future data).
- All rolling windows use min_periods to avoid NaN contamination.
- All indicators are lagged by at least 1 day to ensure the feature
  at time t uses only data available at or before market close on day t.
"""

import pandas as pd
import numpy as np
from typing import Optional

from ..utils.constants import (
    SMA_PERIODS, EMA_PERIODS, RSI_PERIOD, ATR_PERIOD,
    BOLLINGER_PERIOD, BOLLINGER_STD, STOCHASTIC_PERIOD,
    CCI_PERIOD, ADX_PERIOD, WILLIAMS_R_PERIOD, RETURN_PERIODS,
    VOLATILITY_WINDOWS,
)


class TechnicalFeatures:
    """
    Compute technical analysis features for a single stock.

    All features are computed using ONLY past data (no look-ahead bias).
    The features at row t use data up to and including day t.
    When used as model input for predicting day t+1,
    these features are known at market close on day t.
    """

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical features on an OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: Open, High, Low, Close, Volume
            Index must be DatetimeIndex

        Returns
        -------
        pd.DataFrame
            Original columns + all technical features
        """
        df = df.copy()

        # =====================================================================
        # 1. TREND INDICATORS
        # =====================================================================

        # --- Simple Moving Averages ---
        for period in SMA_PERIODS:
            df[f"SMA_{period}"] = df["Close"].rolling(window=period, min_periods=period).mean()
            # Price relative to SMA (ratio) — more stationary than raw SMA
            df[f"Close_SMA_{period}_ratio"] = df["Close"] / df[f"SMA_{period}"]

        # --- Exponential Moving Averages ---
        for period in EMA_PERIODS:
            df[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()

        # --- MACD (Moving Average Convergence Divergence) ---
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

        # --- ADX (Average Directional Index) ---
        df = TechnicalFeatures._compute_adx(df, period=ADX_PERIOD)

        # =====================================================================
        # 2. MOMENTUM INDICATORS
        # =====================================================================

        # --- RSI (Relative Strength Index) ---
        df["RSI"] = TechnicalFeatures._compute_rsi(df["Close"], period=RSI_PERIOD)

        # --- Stochastic Oscillator ---
        low_min = df["Low"].rolling(window=STOCHASTIC_PERIOD, min_periods=STOCHASTIC_PERIOD).min()
        high_max = df["High"].rolling(window=STOCHASTIC_PERIOD, min_periods=STOCHASTIC_PERIOD).max()
        df["Stochastic_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-10)
        df["Stochastic_D"] = df["Stochastic_K"].rolling(window=3, min_periods=3).mean()

        # --- Williams %R ---
        high_max_w = df["High"].rolling(window=WILLIAMS_R_PERIOD, min_periods=WILLIAMS_R_PERIOD).max()
        low_min_w = df["Low"].rolling(window=WILLIAMS_R_PERIOD, min_periods=WILLIAMS_R_PERIOD).min()
        df["Williams_R"] = -100 * (high_max_w - df["Close"]) / (high_max_w - low_min_w + 1e-10)

        # --- CCI (Commodity Channel Index) ---
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        tp_sma = typical_price.rolling(window=CCI_PERIOD, min_periods=CCI_PERIOD).mean()
        tp_mad = typical_price.rolling(window=CCI_PERIOD, min_periods=CCI_PERIOD).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        df["CCI"] = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-10)

        # --- Rate of Change (ROC) ---
        for period in RETURN_PERIODS:
            df[f"ROC_{period}"] = df["Close"].pct_change(periods=period)

        # --- Money Flow Index (MFI) ---
        df["MFI"] = TechnicalFeatures._compute_mfi(df, period=14)

        # =====================================================================
        # 3. VOLATILITY INDICATORS
        # =====================================================================

        # --- Bollinger Bands ---
        bb_sma = df["Close"].rolling(window=BOLLINGER_PERIOD, min_periods=BOLLINGER_PERIOD).mean()
        bb_std = df["Close"].rolling(window=BOLLINGER_PERIOD, min_periods=BOLLINGER_PERIOD).std()
        df["BB_Upper"] = bb_sma + BOLLINGER_STD * bb_std
        df["BB_Lower"] = bb_sma - BOLLINGER_STD * bb_std
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_sma  # Normalized
        df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-10)

        # --- ATR (Average True Range) ---
        df["ATR"] = TechnicalFeatures._compute_atr(df, period=ATR_PERIOD)
        df["ATR_pct"] = df["ATR"] / df["Close"]  # Normalized ATR

        # --- Historical Volatility (rolling) ---
        for window in VOLATILITY_WINDOWS:
            df[f"Volatility_{window}d"] = df["Returns"].rolling(
                window=window, min_periods=window
            ).std() * np.sqrt(252)  # Annualised

        # --- Garman-Klass Volatility ---
        df["GK_Volatility"] = TechnicalFeatures._garman_klass(df, window=21)

        # --- Parkinson Volatility ---
        df["Parkinson_Vol"] = TechnicalFeatures._parkinson_volatility(df, window=21)

        # --- High-Low Range (normalised) ---
        df["HL_Range_pct"] = (df["High"] - df["Low"]) / df["Close"]

        # --- Close-Open Range ---
        df["CO_Range_pct"] = (df["Close"] - df["Open"]) / df["Open"]

        # --- Gap (Overnight Return) ---
        df["Gap_pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

        # =====================================================================
        # 4. VOLUME INDICATORS
        # =====================================================================

        # --- OBV (On-Balance Volume) ---
        df["OBV"] = TechnicalFeatures._compute_obv(df)

        # --- Volume SMA Ratio ---
        vol_sma = df["Volume"].rolling(window=20, min_periods=20).mean()
        df["Volume_SMA_ratio"] = df["Volume"] / (vol_sma + 1e-10)

        # --- VWAP Proxy (rolling intraday proxy) ---
        # True VWAP requires intraday data; this is a daily approximation
        df["VWAP_proxy"] = (
            (df["High"] + df["Low"] + df["Close"]) / 3 * df["Volume"]
        ).rolling(window=20, min_periods=20).sum() / (
            df["Volume"].rolling(window=20, min_periods=20).sum() + 1e-10
        )
        df["Close_VWAP_ratio"] = df["Close"] / (df["VWAP_proxy"] + 1e-10)

        # --- Force Index ---
        df["Force_Index"] = df["Close"].diff(1) * df["Volume"]
        df["Force_Index_13"] = df["Force_Index"].ewm(span=13, adjust=False).mean()

        # --- Accumulation/Distribution Line ---
        clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-10)
        df["AD_Line"] = (clv * df["Volume"]).cumsum()

        # --- Chaikin Money Flow (21-period) ---
        mf_volume = clv * df["Volume"]
        df["CMF_21"] = (
            mf_volume.rolling(window=21, min_periods=21).sum()
            / (df["Volume"].rolling(window=21, min_periods=21).sum() + 1e-10)
        )

        # --- Volume Rate of Change (5-day) ---
        df["Volume_ROC_5"] = df["Volume"].pct_change(periods=5)

        # --- Relative Volume (RVOL) vs 20-day average ---
        df["RVOL"] = df["Volume"] / (df["Volume"].rolling(window=20, min_periods=20).mean() + 1e-10)

        # =====================================================================
        # 4b. REGIME & QUALITY FEATURES (Industry-standard additions)
        # =====================================================================

        # --- ADX Smoothed Trend Strength (helps avoid ranging markets) ---
        df["ADX_Smooth_14"] = df["ADX"].rolling(window=14, min_periods=7).mean()

        # --- Mean Reversion Score (z-score of close vs 20-day SMA) ---
        sma_20 = df["Close"].rolling(window=20, min_periods=20).mean()
        sma_20_std = df["Close"].rolling(window=20, min_periods=20).std()
        df["Mean_Reversion_Score"] = (df["Close"] - sma_20) / (sma_20_std + 1e-10)

        # --- Momentum Quality (RSI divergence from price trend) ---
        # When price makes new highs but RSI doesn't → exhaustion
        price_pctile = df["Close"].rolling(window=20, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        rsi_pctile = df["RSI"].rolling(window=20, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        df["Momentum_Quality"] = rsi_pctile - price_pctile  # Positive = healthy momentum

        # --- Hurst Exponent Approximation (regime detection) ---
        # H > 0.5 = trending, H < 0.5 = mean-reverting, H ≈ 0.5 = random walk
        def _approx_hurst(series, window=20):
            """Rolling Hurst exponent approximation using R/S analysis."""
            result = pd.Series(np.nan, index=series.index)
            data = series.values
            for i in range(window, len(data)):
                seg = data[i-window:i]
                if np.std(seg) == 0:
                    result.iloc[i] = 0.5
                    continue
                mean_seg = np.mean(seg)
                dev = np.cumsum(seg - mean_seg)
                r = np.max(dev) - np.min(dev)
                s = np.std(seg, ddof=1)
                if s > 0 and r > 0:
                    # H ≈ log(R/S) / log(n)
                    result.iloc[i] = np.log(r / s) / np.log(window)
                else:
                    result.iloc[i] = 0.5
            return result

        df["Hurst_20"] = _approx_hurst(df["Close"].pct_change().fillna(0), window=20)

        # --- Regime Classification (derived from Hurst) ---
        # 1 = trending, 0 = random, -1 = mean-reverting
        df["Regime_Trending"] = (df["Hurst_20"] > 0.55).astype(float)
        df["Regime_MeanRev"] = (df["Hurst_20"] < 0.45).astype(float)

        # --- Efficiency Ratio (Kaufman) — trend quality ---
        er_period = 10
        direction = (df["Close"] - df["Close"].shift(er_period)).abs()
        volatility = df["Close"].diff().abs().rolling(window=er_period, min_periods=er_period).sum()
        df["Efficiency_Ratio"] = direction / (volatility + 1e-10)

        # =====================================================================
        # 5. PRICE PATTERN FEATURES
        # =====================================================================

        # --- Returns at multiple horizons ---
        df["Returns_1d"] = df["Close"].pct_change(1)
        df["Returns_5d"] = df["Close"].pct_change(5)
        df["Returns_10d"] = df["Close"].pct_change(10)
        df["Returns_21d"] = df["Close"].pct_change(21)

        # --- Log returns ---
        df["Log_Returns_1d"] = np.log(df["Close"] / df["Close"].shift(1))

        # --- Return momentum (acceleration) ---
        df["Return_Momentum"] = df["Returns_1d"] - df["Returns_1d"].shift(1)

        # --- Distance from 52-week high/low ---
        df["Dist_52w_High"] = df["Close"] / df["High"].rolling(252, min_periods=60).max() - 1
        df["Dist_52w_Low"] = df["Close"] / df["Low"].rolling(252, min_periods=60).min() - 1

        return df

    # =========================================================================
    # Helper Methods (all use only past data)
    # =========================================================================

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index using Wilder's smoothing."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range."""
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average Directional Index."""
        df = df.copy()
        # +DM and -DM
        up_move = df["High"] - df["High"].shift(1)
        down_move = df["Low"].shift(1) - df["Low"]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        atr = TechnicalFeatures._compute_atr(df, period)
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
            alpha=1/period, min_periods=period, adjust=False
        ).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
            alpha=1/period, min_periods=period, adjust=False
        ).mean() / (atr + 1e-10)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df["ADX"] = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        df["Plus_DI"] = plus_di
        df["Minus_DI"] = minus_di
        return df

    @staticmethod
    def _compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Money Flow Index."""
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        money_flow = typical_price * df["Volume"]
        tp_diff = typical_price.diff()
        positive_flow = money_flow.where(tp_diff > 0, 0.0).rolling(period, min_periods=period).sum()
        negative_flow = money_flow.where(tp_diff <= 0, 0.0).rolling(period, min_periods=period).sum()
        mfr = positive_flow / (negative_flow + 1e-10)
        return 100 - (100 / (1 + mfr))

    @staticmethod
    def _compute_obv(df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume."""
        obv = pd.Series(0.0, index=df.index)
        obv = np.where(df["Close"] > df["Close"].shift(1), df["Volume"],
               np.where(df["Close"] < df["Close"].shift(1), -df["Volume"], 0))
        return pd.Series(obv, index=df.index).cumsum()

    @staticmethod
    def _garman_klass(df: pd.DataFrame, window: int = 21) -> pd.Series:
        """
        Garman-Klass volatility estimator.
        Uses OHLC data — more efficient than close-to-close volatility.
        """
        log_hl = np.log(df["High"] / df["Low"]) ** 2
        log_co = np.log(df["Close"] / df["Open"]) ** 2
        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return np.sqrt(gk.rolling(window=window, min_periods=window).mean() * 252)

    @staticmethod
    def _parkinson_volatility(df: pd.DataFrame, window: int = 21) -> pd.Series:
        """Parkinson volatility estimator (uses high-low range)."""
        log_hl_sq = np.log(df["High"] / df["Low"]) ** 2
        factor = 1 / (4 * np.log(2))
        return np.sqrt(
            factor * log_hl_sq.rolling(window=window, min_periods=window).mean() * 252
        )
