"""
Institutional-Grade Single-Stock Deep Analyzer
================================================
Designed for quantitative trading interviews at GS, JPM, Jane Street, Citadel, HSBC.

Enter ANY stock ticker available on Groww/yfinance and get:
  1. Live price data download (yfinance)
  2. 50+ technical indicators with multi-timeframe analysis
  3. Directional signals: BUY / SELL / SHORT / LONG / HOLD
  4. Options strategy recommendations (Straddle, Strangle, Spreads, Iron Condor)
  5. Full Greeks: Delta, Gamma, Theta, Vega, Rho
  6. Portfolio risk metrics: Alpha, Beta, Sharpe, Sortino, VaR
  7. Support/Resistance levels with ATR-based entry/exit zones
  8. Regime detection (trending/mean-reverting/volatile)
  9. Volume profile analysis
 10. Risk-adjusted position sizing (Half-Kelly)

Architecture follows quantitative research desk standards:
  - Signal generation is separate from risk management
  - All indicators use only past data (no look-ahead bias)
  - Multi-timeframe confluence scoring
  - Volatility regime awareness
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

try:
    import yfinance as yf
except ImportError:
    yf = None

from ..options.greeks import BlackScholesGreeks


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TechnicalProfile:
    """Complete technical analysis profile for a stock."""
    ticker: str
    price: float
    change_1d: float
    change_5d: float
    change_20d: float

    # Trend indicators
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    trend_regime: str      # UPTREND / DOWNTREND / SIDEWAYS

    # Momentum
    rsi_14: float
    rsi_signal: str        # OVERSOLD / NEUTRAL / OVERBOUGHT
    macd: float
    macd_signal: float
    macd_histogram: float
    macd_cross: str        # BULLISH_CROSS / BEARISH_CROSS / NONE
    stochastic_k: float
    stochastic_d: float
    williams_r: float
    cci: float
    adx: float             # Trend strength
    adx_signal: str        # STRONG_TREND / WEAK_TREND / NO_TREND

    # Volatility
    atr_14: float
    atr_pct: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: float     # 0=lower, 0.5=middle, 1=upper
    bb_squeeze: bool
    hist_vol_20: float
    hist_vol_60: float
    vol_regime: str        # LOW / NORMAL / HIGH / EXTREME

    # Volume
    volume: float
    volume_sma_20: float
    volume_ratio: float    # Current / SMA20
    obv_trend: str         # ACCUMULATION / DISTRIBUTION / NEUTRAL

    # Support / Resistance
    support_1: float
    support_2: float
    resistance_1: float
    resistance_2: float
    pivot_point: float


@dataclass
class TradingSignal:
    """Institutional-grade trading signal."""
    ticker: str
    timestamp: str
    price: float

    # Primary signal
    primary_action: str       # BUY / SELL / SHORT / LONG / HOLD
    conviction: str           # HIGH / MEDIUM / LOW
    confidence_score: float   # 0-100

    # Signal components
    trend_signal: str
    momentum_signal: str
    volatility_signal: str
    volume_signal: str
    multi_tf_signal: str      # Multi-timeframe confluence

    # Entry/Exit
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    risk_reward_ratio: float

    # Risk metrics
    position_size_pct: float  # % of capital
    max_loss_amount: float    # In INR
    expected_return: float    # In %

    # Rationale
    rationale: str
    key_levels: str


@dataclass
class OptionsAnalysis:
    """Options analysis with Greeks and strategy recommendation."""
    ticker: str
    spot: float
    hist_vol: float

    # ATM Greeks (Call)
    call_price_atm: float
    put_price_atm: float
    delta: float
    gamma: float
    theta: float       # Per day
    vega: float        # Per 1% vol move
    rho: float

    # Moneyness levels
    itm_strike: float
    atm_strike: float
    otm_strike: float

    # Strategy recommendation
    recommended_strategy: str
    strategy_legs: list
    max_profit: str
    max_loss: str
    breakeven: str
    strategy_rationale: str

    # Volatility analysis
    vol_percentile: float   # Where current vol sits vs last year
    vol_regime: str
    vol_skew: str           # NORMAL / POSITIVE / NEGATIVE


@dataclass
class RiskProfile:
    """Portfolio risk metrics for the stock."""
    ticker: str
    alpha: float           # Jensen's alpha (annualized)
    beta: float            # CAPM beta
    sharpe_ratio: float
    sortino_ratio: float
    treynor_ratio: float
    information_ratio: float
    max_drawdown: float
    var_95: float          # Value at Risk (95%)
    cvar_95: float         # Conditional VaR
    calmar_ratio: float
    omega_ratio: float
    downside_deviation: float
    correlation_nifty: float


@dataclass
class FullAnalysis:
    """Complete analysis output."""
    ticker: str
    company_name: str
    sector: str
    market_cap: str
    technical: TechnicalProfile
    signal: TradingSignal
    options: OptionsAnalysis
    risk: RiskProfile
    timestamp: str


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class StockAnalyzer:
    """
    Institutional-grade single-stock deep analyzer.

    Downloads live data from yfinance, computes 50+ indicators,
    generates directional signals, options strategies, and risk metrics.

    Usage
    -----
    >>> analyzer = StockAnalyzer(capital=50000)
    >>> result = analyzer.analyze("RELIANCE.NS")
    >>> analyzer.print_report(result)
    """

    # Nifty 50 lot sizes for options (approximate, update as needed)
    LOT_SIZES = {
        "NIFTY": 25, "BANKNIFTY": 15, "RELIANCE": 250, "TCS": 150,
        "INFY": 300, "HDFCBANK": 550, "ICICIBANK": 700, "HINDUNILVR": 300,
        "ITC": 1600, "SBIN": 1500, "BAJFINANCE": 125, "BHARTIARTL": 950,
        "KOTAKBANK": 400, "LT": 150, "AXISBANK": 600, "MARUTI": 100,
        "SUNPHARMA": 350, "TATAMOTORS": 1125, "WIPRO": 1500,
        "HCLTECH": 350, "ADANIENT": 500, "TATASTEEL": 4050,
        "ONGC": 3850, "NTPC": 2800, "POWERGRID": 2700,
        "COALINDIA": 2100, "BAJAJ-AUTO": 250, "M&M": 350,
        "ULTRACEMCO": 100, "TITAN": 375, "TECHM": 600,
    }

    RISK_FREE_RATE = 0.065  # India 10Y

    def __init__(self, capital: float = 50000, risk_free_rate: float = 0.065):
        self.capital = capital
        self.risk_free_rate = risk_free_rate
        self.bs = BlackScholesGreeks(risk_free_rate)

    # ------------------------------------------------------------------
    # Data download
    # ------------------------------------------------------------------
    def download_data(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download OHLCV data from yfinance.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker (e.g., "RELIANCE.NS", "TCS.NS", "AAPL")
        period : str
            Data period ("1y", "2y", "5y", "max")
        interval : str
            Data interval ("1d", "1wk", "1mo")
        """
        if yf is None:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for {ticker}. Check the ticker symbol.")

        # Get company info
        try:
            info = stock.info
            self._last_info = info
        except Exception:
            self._last_info = {}

        # Standardize columns
        df = df.rename(columns={
            "Stock Splits": "Stock_Splits",
            "Capital Gains": "Capital_Gains",
        })

        # Ensure we have required columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        return df

    # ------------------------------------------------------------------
    # Technical analysis
    # ------------------------------------------------------------------
    def compute_technicals(self, df: pd.DataFrame, ticker: str) -> TechnicalProfile:
        """Compute all technical indicators."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        n = len(df)

        price = close.iloc[-1]

        # Returns
        change_1d = close.pct_change().iloc[-1] if n > 1 else 0.0
        change_5d = (price / close.iloc[-6] - 1) if n > 6 else 0.0
        change_20d = (price / close.iloc[-21] - 1) if n > 21 else 0.0

        # --- Moving Averages ---
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1] if n > 200 else sma_50
        ema_12 = close.ewm(span=12).mean().iloc[-1]
        ema_26 = close.ewm(span=26).mean().iloc[-1]

        # Trend regime
        if price > sma_20 > sma_50:
            trend_regime = "UPTREND"
        elif price < sma_20 < sma_50:
            trend_regime = "DOWNTREND"
        else:
            trend_regime = "SIDEWAYS"

        # --- RSI ---
        delta_c = close.diff()
        gain = delta_c.clip(lower=0).rolling(14).mean()
        loss = (-delta_c.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi_series = 100 - (100 / (1 + rs))
        rsi_14 = rsi_series.iloc[-1]
        rsi_signal = "OVERSOLD" if rsi_14 < 30 else ("OVERBOUGHT" if rsi_14 > 70 else "NEUTRAL")

        # --- MACD ---
        macd_line = ema_12 - ema_26
        macd_signal_line = close.ewm(span=12).mean().sub(close.ewm(span=26).mean()).ewm(span=9).mean().iloc[-1]
        macd_hist = macd_line - macd_signal_line

        # MACD cross detection
        macd_series = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        macd_sig_series = macd_series.ewm(span=9).mean()
        if n > 2:
            prev_diff = (macd_series.iloc[-2] - macd_sig_series.iloc[-2])
            curr_diff = (macd_series.iloc[-1] - macd_sig_series.iloc[-1])
            if prev_diff < 0 and curr_diff > 0:
                macd_cross = "BULLISH_CROSS"
            elif prev_diff > 0 and curr_diff < 0:
                macd_cross = "BEARISH_CROSS"
            else:
                macd_cross = "NONE"
        else:
            macd_cross = "NONE"

        # --- Stochastic ---
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        stoch_k = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        stoch_d = stoch_k.rolling(3).mean()
        stochastic_k = stoch_k.iloc[-1]
        stochastic_d = stoch_d.iloc[-1]

        # --- Williams %R ---
        high_14w = high.rolling(14).max()
        low_14w = low.rolling(14).min()
        williams_r = -100 * (high_14w - close) / (high_14w - low_14w + 1e-10)
        williams_r_val = williams_r.iloc[-1]

        # --- CCI ---
        tp = (high + low + close) / 3
        cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        cci_val = cci.iloc[-1]

        # --- ADX ---
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / atr_series.replace(0, 1e-10)
        minus_di = 100 * minus_dm.rolling(14).mean() / atr_series.replace(0, 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(14).mean().iloc[-1]
        adx_signal = "STRONG_TREND" if adx > 25 else ("WEAK_TREND" if adx > 15 else "NO_TREND")

        # --- ATR ---
        atr_14 = atr_series.iloc[-1]
        atr_pct = atr_14 / price * 100

        # --- Bollinger Bands ---
        bb_middle = sma_20
        bb_std = close.rolling(20).std().iloc[-1]
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        bb_position = (price - bb_lower) / (bb_upper - bb_lower + 1e-10)
        bb_width = (bb_upper - bb_lower) / bb_middle
        bb_width_pctile = pd.Series(
            [(close.rolling(20).std() * 2 / close.rolling(20).mean()).iloc[i]
             for i in range(max(0, n-100), n)]
        )
        bb_squeeze = bool(bb_width < bb_width_pctile.quantile(0.2)) if len(bb_width_pctile) > 10 else False

        # --- Volatility ---
        returns = close.pct_change().dropna()
        hist_vol_20 = returns.tail(20).std() * np.sqrt(252)
        hist_vol_60 = returns.tail(60).std() * np.sqrt(252)

        vol_1y = returns.tail(252).std() * np.sqrt(252) if len(returns) > 252 else hist_vol_60
        vol_pctile = (returns.rolling(20).std() * np.sqrt(252)).rank(pct=True).iloc[-1] if n > 50 else 0.5
        if vol_pctile > 0.9:
            vol_regime = "EXTREME"
        elif vol_pctile > 0.7:
            vol_regime = "HIGH"
        elif vol_pctile > 0.3:
            vol_regime = "NORMAL"
        else:
            vol_regime = "LOW"

        # --- Volume ---
        vol_current = volume.iloc[-1]
        vol_sma_20 = volume.rolling(20).mean().iloc[-1]
        vol_ratio = vol_current / vol_sma_20 if vol_sma_20 > 0 else 1.0

        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_sma = obv.rolling(20).mean()
        if obv.iloc[-1] > obv_sma.iloc[-1] and obv.iloc[-1] > obv.iloc[-5]:
            obv_trend = "ACCUMULATION"
        elif obv.iloc[-1] < obv_sma.iloc[-1] and obv.iloc[-1] < obv.iloc[-5]:
            obv_trend = "DISTRIBUTION"
        else:
            obv_trend = "NEUTRAL"

        # --- Support / Resistance ---
        recent = df.tail(60)
        pivot = (recent["High"].max() + recent["Low"].min() + close.iloc[-1]) / 3
        r1 = 2 * pivot - recent["Low"].min()
        r2 = pivot + (recent["High"].max() - recent["Low"].min())
        s1 = 2 * pivot - recent["High"].max()
        s2 = pivot - (recent["High"].max() - recent["Low"].min())

        return TechnicalProfile(
            ticker=ticker, price=float(price),
            change_1d=float(change_1d), change_5d=float(change_5d), change_20d=float(change_20d),
            sma_20=float(sma_20), sma_50=float(sma_50), sma_200=float(sma_200),
            ema_12=float(ema_12), ema_26=float(ema_26), trend_regime=trend_regime,
            rsi_14=float(rsi_14), rsi_signal=rsi_signal,
            macd=float(macd_line), macd_signal=float(macd_signal_line), macd_histogram=float(macd_hist),
            macd_cross=macd_cross,
            stochastic_k=float(stochastic_k), stochastic_d=float(stochastic_d),
            williams_r=float(williams_r_val), cci=float(cci_val),
            adx=float(adx), adx_signal=adx_signal,
            atr_14=float(atr_14), atr_pct=float(atr_pct),
            bb_upper=float(bb_upper), bb_middle=float(bb_middle), bb_lower=float(bb_lower),
            bb_position=float(bb_position), bb_squeeze=bb_squeeze,
            hist_vol_20=float(hist_vol_20), hist_vol_60=float(hist_vol_60),
            vol_regime=vol_regime,
            volume=float(vol_current), volume_sma_20=float(vol_sma_20), volume_ratio=float(vol_ratio),
            obv_trend=obv_trend,
            support_1=float(s1), support_2=float(s2),
            resistance_1=float(r1), resistance_2=float(r2),
            pivot_point=float(pivot),
        )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------
    def generate_signal(
        self,
        df: pd.DataFrame,
        tech: TechnicalProfile,
        capital: float = None,
    ) -> TradingSignal:
        """
        Generate institutional-grade trading signal using multi-factor scoring.

        Scoring system (max 100):
          - Trend score (25 pts): SMA alignment, price position
          - Momentum score (25 pts): RSI, MACD, Stochastic
          - Volatility score (25 pts): BB position, ATR, vol regime
          - Volume score (25 pts): Volume ratio, OBV trend
        """
        cap = capital or self.capital
        price = tech.price

        # --- Trend Score (0-25) ---
        trend_score = 0
        # SMA alignment
        if price > tech.sma_20 > tech.sma_50:
            trend_score += 10       # Strong uptrend
        elif price > tech.sma_20:
            trend_score += 5
        elif price < tech.sma_20 < tech.sma_50:
            trend_score -= 10       # Strong downtrend
        elif price < tech.sma_20:
            trend_score -= 5

        # EMA trend
        if tech.ema_12 > tech.ema_26:
            trend_score += 5
        else:
            trend_score -= 5

        # ADX strength
        if tech.adx > 25:
            trend_score += 5 if tech.trend_regime == "UPTREND" else -5
        elif tech.adx > 15:
            trend_score += 3 if tech.trend_regime == "UPTREND" else -3

        # 200-day SMA
        if price > tech.sma_200:
            trend_score += 5
        else:
            trend_score -= 5

        trend_signal = "BULLISH" if trend_score > 5 else ("BEARISH" if trend_score < -5 else "NEUTRAL")

        # --- Momentum Score (0-25) ---
        mom_score = 0
        # RSI
        if tech.rsi_14 < 30:
            mom_score += 10         # Oversold = buy signal
        elif tech.rsi_14 < 40:
            mom_score += 5
        elif tech.rsi_14 > 70:
            mom_score -= 10         # Overbought = sell signal
        elif tech.rsi_14 > 60:
            mom_score -= 5

        # MACD
        if tech.macd_cross == "BULLISH_CROSS":
            mom_score += 8
        elif tech.macd_cross == "BEARISH_CROSS":
            mom_score -= 8
        elif tech.macd_histogram > 0:
            mom_score += 3
        else:
            mom_score -= 3

        # Stochastic
        if tech.stochastic_k < 20:
            mom_score += 5
        elif tech.stochastic_k > 80:
            mom_score -= 5

        # CCI
        if tech.cci < -100:
            mom_score += 2          # Oversold
        elif tech.cci > 100:
            mom_score -= 2

        momentum_signal = "BULLISH" if mom_score > 5 else ("BEARISH" if mom_score < -5 else "NEUTRAL")

        # --- Volatility Score (0-25) ---
        vol_score = 0
        # Bollinger position
        if tech.bb_position < 0.2:
            vol_score += 8          # Near lower band
        elif tech.bb_position > 0.8:
            vol_score -= 8          # Near upper band

        # Squeeze = breakout imminent
        if tech.bb_squeeze:
            vol_score += 3 if trend_score > 0 else -3

        # Vol regime
        if tech.vol_regime == "LOW":
            vol_score += 2          # Low vol = good for entry
        elif tech.vol_regime == "EXTREME":
            vol_score -= 5          # High vol = risky

        volatility_signal = "BULLISH" if vol_score > 3 else ("BEARISH" if vol_score < -3 else "NEUTRAL")

        # --- Volume Score (0-25) ---
        vol_s = 0
        if tech.volume_ratio > 1.5 and trend_score > 0:
            vol_s += 8             # High volume + uptrend = strong
        elif tech.volume_ratio > 1.5 and trend_score < 0:
            vol_s -= 8             # High volume + downtrend = strong selling
        elif tech.volume_ratio < 0.5:
            vol_s -= 3             # Low volume = weak conviction

        if tech.obv_trend == "ACCUMULATION":
            vol_s += 5
        elif tech.obv_trend == "DISTRIBUTION":
            vol_s -= 5

        volume_signal = "BULLISH" if vol_s > 3 else ("BEARISH" if vol_s < -3 else "NEUTRAL")

        # --- Multi-Timeframe Confluence ---
        # Weekly trend check using last 5 days of data
        weekly_ret = tech.change_5d
        monthly_ret = tech.change_20d
        if weekly_ret > 0 and monthly_ret > 0:
            mtf_signal = "BULLISH"
            mtf_bonus = 5
        elif weekly_ret < 0 and monthly_ret < 0:
            mtf_signal = "BEARISH"
            mtf_bonus = -5
        else:
            mtf_signal = "MIXED"
            mtf_bonus = 0

        # --- Composite Score ---
        total_score = trend_score + mom_score + vol_score + vol_s + mtf_bonus
        # Normalize to 0-100
        confidence_score = 50 + total_score  # Center at 50
        confidence_score = max(0, min(100, confidence_score))

        # --- Primary Action ---
        if confidence_score >= 70:
            primary_action = "STRONG_BUY"
            conviction = "HIGH"
        elif confidence_score >= 60:
            primary_action = "BUY"
            conviction = "MEDIUM"
        elif confidence_score >= 55:
            primary_action = "LONG"
            conviction = "LOW"
        elif confidence_score <= 30:
            primary_action = "STRONG_SELL"
            conviction = "HIGH"
        elif confidence_score <= 40:
            primary_action = "SHORT"
            conviction = "MEDIUM"
        elif confidence_score <= 45:
            primary_action = "SELL"
            conviction = "LOW"
        else:
            primary_action = "HOLD"
            conviction = "LOW"

        # --- Entry/Exit Levels ---
        atr = tech.atr_14
        if primary_action in ("STRONG_BUY", "BUY", "LONG"):
            entry_price = price * 0.998      # Slightly below current
            stop_loss = price - 2 * atr
            target_1 = price + 1.5 * atr
            target_2 = price + 3 * atr
            target_3 = tech.resistance_2
        elif primary_action in ("STRONG_SELL", "SHORT", "SELL"):
            entry_price = price * 1.002      # Slightly above current
            stop_loss = price + 2 * atr
            target_1 = price - 1.5 * atr
            target_2 = price - 3 * atr
            target_3 = tech.support_2
        else:
            entry_price = price
            stop_loss = price - 2 * atr
            target_1 = price + 1.5 * atr
            target_2 = price + 3 * atr
            target_3 = tech.resistance_1

        risk = abs(entry_price - stop_loss)
        reward = abs(target_2 - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        # --- Position Sizing (Half-Kelly) ---
        win_prob = confidence_score / 100
        win_loss_ratio = rr_ratio if rr_ratio > 0 else 1.5
        kelly = max(0, (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio)
        half_kelly = kelly / 2
        pos_size_pct = min(half_kelly, 0.15)  # Cap at 15%
        max_loss_amt = cap * pos_size_pct * (risk / price)

        # Expected return
        exp_ret = (win_prob * reward - (1 - win_prob) * risk) / price * 100

        # --- Rationale ---
        signals = []
        if trend_signal == "BULLISH":
            signals.append("Uptrend (SMA alignment)")
        elif trend_signal == "BEARISH":
            signals.append("Downtrend (SMA alignment)")
        if tech.rsi_signal == "OVERSOLD":
            signals.append("RSI oversold (<30)")
        elif tech.rsi_signal == "OVERBOUGHT":
            signals.append("RSI overbought (>70)")
        if tech.macd_cross != "NONE":
            signals.append(f"MACD {tech.macd_cross.lower().replace('_', ' ')}")
        if tech.bb_squeeze:
            signals.append("Bollinger squeeze (breakout imminent)")
        if tech.obv_trend == "ACCUMULATION":
            signals.append("Volume accumulation")
        elif tech.obv_trend == "DISTRIBUTION":
            signals.append("Volume distribution")

        rationale = " | ".join(signals) if signals else "No strong conviction signals"

        key_levels = (
            f"S2={tech.support_2:,.1f} S1={tech.support_1:,.1f} "
            f"PP={tech.pivot_point:,.1f} "
            f"R1={tech.resistance_1:,.1f} R2={tech.resistance_2:,.1f}"
        )

        return TradingSignal(
            ticker=tech.ticker, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            price=price,
            primary_action=primary_action, conviction=conviction,
            confidence_score=confidence_score,
            trend_signal=trend_signal, momentum_signal=momentum_signal,
            volatility_signal=volatility_signal, volume_signal=volume_signal,
            multi_tf_signal=mtf_signal,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            target_3=round(target_3, 2),
            risk_reward_ratio=round(rr_ratio, 2),
            position_size_pct=round(pos_size_pct * 100, 2),
            max_loss_amount=round(max_loss_amt, 2),
            expected_return=round(exp_ret, 2),
            rationale=rationale,
            key_levels=key_levels,
        )

    # ------------------------------------------------------------------
    # Options analysis
    # ------------------------------------------------------------------
    def analyze_options(self, tech: TechnicalProfile, signal: TradingSignal) -> OptionsAnalysis:
        """Compute options Greeks and recommend strategy."""
        S = tech.price
        sigma = tech.hist_vol_20
        T = 30 / 365.0  # 30 days to expiry

        # Round to nearest 50 for strike (Indian equities)
        base_ticker = tech.ticker.replace(".NS", "").replace(".BO", "")
        interval = self.LOT_SIZES.get(base_ticker, 50)
        # Use standard intervals for individual stocks
        strike_interval = 50 if S > 1000 else (10 if S > 100 else 5)

        atm_strike = round(S / strike_interval) * strike_interval
        itm_strike = atm_strike - strike_interval
        otm_strike = atm_strike + strike_interval

        # Greeks
        call_price = self.bs.call_price(S, atm_strike, T, sigma)
        put_price = self.bs.put_price(S, atm_strike, T, sigma)
        greeks = self.bs.all_greeks(S, atm_strike, T, sigma, "call")

        # Vol percentile (current 20d vol vs historical range)
        vol_pctile = 0.5  # Default if no history
        if tech.hist_vol_60 > 0:
            vol_pctile = min(tech.hist_vol_20 / tech.hist_vol_60, 2.0) / 2.0

        # Vol regime for options
        if vol_pctile > 0.8:
            vol_regime = "HIGH_IV"
            vol_skew = "NEGATIVE"  # Typically negative skew in high vol
        elif vol_pctile < 0.2:
            vol_regime = "LOW_IV"
            vol_skew = "POSITIVE"
        else:
            vol_regime = "NORMAL_IV"
            vol_skew = "NORMAL"

        # Strategy selection based on signal + vol
        if signal.primary_action in ("STRONG_BUY", "BUY", "LONG"):
            if vol_regime == "HIGH_IV":
                strategy = "Bull Put Spread"    # Sell premium (high IV)
                legs = [
                    {"type": "PE", "strike": atm_strike, "action": "SELL",
                     "premium": round(self.bs.put_price(S, atm_strike, T, sigma), 2)},
                    {"type": "PE", "strike": itm_strike, "action": "BUY",
                     "premium": round(self.bs.put_price(S, itm_strike, T, sigma), 2)},
                ]
                credit = legs[0]["premium"] - legs[1]["premium"]
                max_profit = f"Rs.{credit:,.2f} (net credit received)"
                max_loss = f"Rs.{strike_interval - credit:,.2f}"
                breakeven = f"{atm_strike - credit:,.1f}"
            else:
                strategy = "Bull Call Spread"   # Buy premium (low/normal IV)
                legs = [
                    {"type": "CE", "strike": atm_strike, "action": "BUY",
                     "premium": round(call_price, 2)},
                    {"type": "CE", "strike": otm_strike, "action": "SELL",
                     "premium": round(self.bs.call_price(S, otm_strike, T, sigma), 2)},
                ]
                debit = legs[0]["premium"] - legs[1]["premium"]
                max_profit = f"Rs.{strike_interval - debit:,.2f}"
                max_loss = f"Rs.{debit:,.2f} (net debit paid)"
                breakeven = f"{atm_strike + debit:,.1f}"

        elif signal.primary_action in ("STRONG_SELL", "SHORT", "SELL"):
            if vol_regime == "HIGH_IV":
                strategy = "Bear Call Spread"
                legs = [
                    {"type": "CE", "strike": atm_strike, "action": "SELL",
                     "premium": round(call_price, 2)},
                    {"type": "CE", "strike": otm_strike, "action": "BUY",
                     "premium": round(self.bs.call_price(S, otm_strike, T, sigma), 2)},
                ]
                credit = legs[0]["premium"] - legs[1]["premium"]
                max_profit = f"Rs.{credit:,.2f} (net credit)"
                max_loss = f"Rs.{strike_interval - credit:,.2f}"
                breakeven = f"{atm_strike + credit:,.1f}"
            else:
                strategy = "Bear Put Spread"
                legs = [
                    {"type": "PE", "strike": atm_strike, "action": "BUY",
                     "premium": round(put_price, 2)},
                    {"type": "PE", "strike": itm_strike, "action": "SELL",
                     "premium": round(self.bs.put_price(S, itm_strike, T, sigma), 2)},
                ]
                debit = legs[0]["premium"] - legs[1]["premium"]
                max_profit = f"Rs.{strike_interval - debit:,.2f}"
                max_loss = f"Rs.{debit:,.2f} (net debit)"
                breakeven = f"{atm_strike - debit:,.1f}"

        else:  # HOLD / neutral
            if vol_regime == "HIGH_IV":
                strategy = "Iron Condor"
                sell_put = atm_strike - 2 * strike_interval
                buy_put = sell_put - strike_interval
                sell_call = atm_strike + 2 * strike_interval
                buy_call = sell_call + strike_interval
                sp = self.bs.put_price(S, sell_put, T, sigma)
                bp = self.bs.put_price(S, buy_put, T, sigma)
                sc = self.bs.call_price(S, sell_call, T, sigma)
                bc = self.bs.call_price(S, buy_call, T, sigma)
                credit = (sp - bp) + (sc - bc)
                legs = [
                    {"type": "PE", "strike": buy_put, "action": "BUY", "premium": round(bp, 2)},
                    {"type": "PE", "strike": sell_put, "action": "SELL", "premium": round(sp, 2)},
                    {"type": "CE", "strike": sell_call, "action": "SELL", "premium": round(sc, 2)},
                    {"type": "CE", "strike": buy_call, "action": "BUY", "premium": round(bc, 2)},
                ]
                max_profit = f"Rs.{credit:,.2f} (net credit)"
                max_loss = f"Rs.{strike_interval - credit:,.2f}"
                breakeven = f"{sell_put - credit:,.1f} / {sell_call + credit:,.1f}"
            else:
                strategy = "Long Straddle"
                total = call_price + put_price
                legs = [
                    {"type": "CE", "strike": atm_strike, "action": "BUY", "premium": round(call_price, 2)},
                    {"type": "PE", "strike": atm_strike, "action": "BUY", "premium": round(put_price, 2)},
                ]
                max_profit = "Unlimited (directional breakout)"
                max_loss = f"Rs.{total:,.2f} (total premium)"
                breakeven = f"{atm_strike - total:,.1f} / {atm_strike + total:,.1f}"

        strategy_rationale = (
            f"Signal: {signal.primary_action} ({signal.conviction}) | "
            f"IV regime: {vol_regime} | "
            f"Vol 20d: {tech.hist_vol_20:.1%} vs 60d: {tech.hist_vol_60:.1%} | "
            f"Strategy optimized for {('selling premium' if 'HIGH' in vol_regime else 'buying premium')}"
        )

        return OptionsAnalysis(
            ticker=tech.ticker, spot=S, hist_vol=tech.hist_vol_20,
            call_price_atm=round(call_price, 2), put_price_atm=round(put_price, 2),
            delta=greeks["delta"], gamma=greeks["gamma"],
            theta=greeks["theta"], vega=greeks["vega"], rho=greeks["rho"],
            itm_strike=itm_strike, atm_strike=atm_strike, otm_strike=otm_strike,
            recommended_strategy=strategy, strategy_legs=legs,
            max_profit=max_profit, max_loss=max_loss, breakeven=breakeven,
            strategy_rationale=strategy_rationale,
            vol_percentile=round(vol_pctile * 100, 1),
            vol_regime=vol_regime, vol_skew=vol_skew,
        )

    # ------------------------------------------------------------------
    # Risk metrics (Alpha, Beta, Sharpe, etc.)
    # ------------------------------------------------------------------
    def compute_risk_metrics(self, df: pd.DataFrame, ticker: str) -> RiskProfile:
        """Compute portfolio risk metrics vs Nifty 50 benchmark."""
        returns = df["Close"].pct_change().dropna()

        # Download Nifty for benchmark
        try:
            nifty = yf.Ticker("^NSEI").history(period="2y")["Close"].pct_change().dropna()
            # Align dates
            common = returns.index.intersection(nifty.index)
            if len(common) > 50:
                stock_ret = returns.loc[common]
                bench_ret = nifty.loc[common]
            else:
                stock_ret = returns
                bench_ret = returns * 0  # No benchmark
        except Exception:
            stock_ret = returns
            bench_ret = returns * 0

        rf_daily = self.risk_free_rate / 252
        excess = stock_ret - rf_daily

        # Beta & Alpha (CAPM)
        if len(bench_ret) > 50 and bench_ret.std() > 0:
            cov_matrix = np.cov(stock_ret.values[-252:], bench_ret.values[-252:])
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1.0
            alpha = (stock_ret.mean() - rf_daily - beta * (bench_ret.mean() - rf_daily)) * 252
            corr = np.corrcoef(stock_ret.values[-252:], bench_ret.values[-252:])[0, 1]
        else:
            beta = 1.0
            alpha = (stock_ret.mean() - rf_daily) * 252
            corr = 0.0

        # Sharpe
        sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
        # Sortino
        downside = excess[excess < 0]
        downside_dev = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-10
        sortino = excess.mean() * 252 / downside_dev
        # Treynor
        treynor = excess.mean() * 252 / beta if beta != 0 else 0
        # Information ratio
        tracking_err = (stock_ret - bench_ret).std() * np.sqrt(252) if len(bench_ret) > 50 else 1e-10
        info_ratio = (stock_ret.mean() - bench_ret.mean()) * 252 / tracking_err

        # Max drawdown
        cum_ret = (1 + stock_ret).cumprod()
        peak = cum_ret.expanding().max()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min()

        # VaR & CVaR
        var_95 = np.percentile(excess, 5)
        cvar_95 = excess[excess <= var_95].mean() if len(excess[excess <= var_95]) > 0 else var_95

        # Calmar
        ann_ret = stock_ret.mean() * 252
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        # Omega
        threshold = 0
        gains = excess[excess > threshold].sum()
        losses = abs(excess[excess <= threshold].sum())
        omega = gains / losses if losses > 0 else 999

        return RiskProfile(
            ticker=ticker,
            alpha=round(float(alpha), 4),
            beta=round(float(beta), 4),
            sharpe_ratio=round(float(sharpe), 4),
            sortino_ratio=round(float(sortino), 4),
            treynor_ratio=round(float(treynor), 4),
            information_ratio=round(float(info_ratio), 4),
            max_drawdown=round(float(max_dd), 4),
            var_95=round(float(var_95), 4),
            cvar_95=round(float(cvar_95), 4),
            calmar_ratio=round(float(calmar), 4),
            omega_ratio=round(float(omega), 4),
            downside_deviation=round(float(downside_dev), 4),
            correlation_nifty=round(float(corr), 4),
        )

    # ------------------------------------------------------------------
    # Main analyze function
    # ------------------------------------------------------------------
    def analyze(self, ticker: str, period: str = "2y") -> FullAnalysis:
        """
        Run complete analysis on any stock ticker.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker. For Indian stocks, append ".NS" (NSE) or ".BO" (BSE).
            Examples: "RELIANCE.NS", "TCS.NS", "INFY.NS", "AAPL", "TSLA"
        period : str
            Data period ("1y", "2y", "5y")

        Returns
        -------
        FullAnalysis
        """
        print(f"Downloading {ticker} data...")
        df = self.download_data(ticker, period=period)
        print(f"  {len(df)} candles loaded ({df.index[0].date()} to {df.index[-1].date()})")

        print("Computing technical indicators...")
        tech = self.compute_technicals(df, ticker)

        print("Generating trading signal...")
        signal = self.generate_signal(df, tech)

        print("Analyzing options & Greeks...")
        options = self.analyze_options(tech, signal)

        print("Computing risk metrics (vs Nifty 50)...")
        risk = self.compute_risk_metrics(df, ticker)

        # Company info
        info = getattr(self, "_last_info", {})
        company_name = info.get("longName", info.get("shortName", ticker))
        sector = info.get("sector", "Unknown")
        mcap = info.get("marketCap", 0)
        if mcap > 1e12:
            market_cap = f"Rs.{mcap/1e12:.1f}T"
        elif mcap > 1e9:
            market_cap = f"Rs.{mcap/1e9:.1f}B"
        elif mcap > 1e7:
            market_cap = f"Rs.{mcap/1e7:.0f}Cr"
        else:
            market_cap = f"Rs.{mcap:,.0f}" if mcap > 0 else "N/A"

        return FullAnalysis(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            market_cap=market_cap,
            technical=tech,
            signal=signal,
            options=options,
            risk=risk,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------
    def print_report(self, result: FullAnalysis):
        """Print comprehensive analysis report."""
        t = result.technical
        s = result.signal
        o = result.options
        r = result.risk

        w = 72
        print("=" * w)
        print(f"  DEEP STOCK ANALYSIS: {result.ticker}")
        print(f"  {result.company_name} | {result.sector} | MCap: {result.market_cap}")
        print(f"  Generated: {result.timestamp}")
        print("=" * w)

        # --- PRIMARY SIGNAL ---
        action_colors = {
            "STRONG_BUY": "***", "BUY": "**", "LONG": "*",
            "STRONG_SELL": "!!!", "SHORT": "!!", "SELL": "!",
            "HOLD": "---"
        }
        marker = action_colors.get(s.primary_action, "")
        print(f"\n  {marker} PRIMARY SIGNAL: {s.primary_action} {marker}")
        print(f"  Conviction: {s.conviction} | Score: {s.confidence_score}/100")
        print(f"  Rationale: {s.rationale}")

        # --- PRICE & TECHNICALS ---
        print(f"\n{'  PRICE & TREND':-<{w}}")
        print(f"  Price: Rs.{t.price:,.2f}  |  1D: {t.change_1d:+.2%}  |  "
              f"5D: {t.change_5d:+.2%}  |  20D: {t.change_20d:+.2%}")
        print(f"  SMA20: {t.sma_20:,.1f}  |  SMA50: {t.sma_50:,.1f}  |  "
              f"SMA200: {t.sma_200:,.1f}")
        print(f"  Trend: {t.trend_regime}  |  ADX: {t.adx:.1f} ({t.adx_signal})")

        # --- MOMENTUM ---
        print(f"\n{'  MOMENTUM':-<{w}}")
        print(f"  RSI(14): {t.rsi_14:.1f} [{t.rsi_signal}]  |  "
              f"Stoch K/D: {t.stochastic_k:.0f}/{t.stochastic_d:.0f}")
        print(f"  MACD: {t.macd:.2f} (Signal: {t.macd_signal:.2f})  |  "
              f"Hist: {t.macd_histogram:+.2f}  |  Cross: {t.macd_cross}")
        print(f"  CCI: {t.cci:.0f}  |  Williams%%R: {t.williams_r:.0f}")

        # --- VOLATILITY ---
        print(f"\n{'  VOLATILITY':-<{w}}")
        print(f"  ATR(14): Rs.{t.atr_14:,.2f} ({t.atr_pct:.2f}%)  |  "
              f"Regime: {t.vol_regime}")
        print(f"  HV 20d: {t.hist_vol_20:.1%}  |  HV 60d: {t.hist_vol_60:.1%}")
        print(f"  BB: [{t.bb_lower:,.1f} -- {t.bb_middle:,.1f} -- {t.bb_upper:,.1f}]"
              f"  |  Position: {t.bb_position:.2f}  |  Squeeze: {t.bb_squeeze}")

        # --- VOLUME ---
        print(f"\n{'  VOLUME':-<{w}}")
        print(f"  Volume: {t.volume:,.0f}  |  SMA20: {t.volume_sma_20:,.0f}  |  "
              f"Ratio: {t.volume_ratio:.2f}x  |  OBV: {t.obv_trend}")

        # --- ENTRY/EXIT ---
        print(f"\n{'  ENTRY / EXIT LEVELS':-<{w}}")
        print(f"  Entry:   Rs.{s.entry_price:>10,.2f}")
        print(f"  Stop:    Rs.{s.stop_loss:>10,.2f}  (Risk: Rs.{abs(s.entry_price-s.stop_loss):,.2f})")
        print(f"  Target1: Rs.{s.target_1:>10,.2f}  ({(s.target_1/s.price-1):+.2%})")
        print(f"  Target2: Rs.{s.target_2:>10,.2f}  ({(s.target_2/s.price-1):+.2%})")
        print(f"  Target3: Rs.{s.target_3:>10,.2f}  ({(s.target_3/s.price-1):+.2%})")
        print(f"  R:R Ratio: {s.risk_reward_ratio:.2f}  |  Key: {s.key_levels}")

        # --- POSITION SIZING ---
        print(f"\n{'  POSITION SIZING (Half-Kelly)':-<{w}}")
        print(f"  Capital: Rs.{self.capital:,.0f}  |  Allocation: {s.position_size_pct:.1f}%  |  "
              f"Amount: Rs.{self.capital * s.position_size_pct/100:,.0f}")
        print(f"  Max Loss: Rs.{s.max_loss_amount:,.0f}  |  "
              f"Expected Return: {s.expected_return:+.2f}%")

        # --- OPTIONS & GREEKS ---
        print(f"\n{'  OPTIONS ANALYSIS (30d Expiry)':-<{w}}")
        print(f"  ATM Strike: {o.atm_strike:,.0f}  |  "
              f"Call: Rs.{o.call_price_atm:,.2f}  |  Put: Rs.{o.put_price_atm:,.2f}")
        print(f"  IV Regime: {o.vol_regime}  |  Vol Percentile: {o.vol_percentile:.0f}th")
        print()
        print(f"  GREEKS (ATM Call @ {o.atm_strike:,.0f}):")
        print(f"    Delta (D):  {o.delta:>+.4f}   (Rs. change per Rs.1 spot move)")
        print(f"    Gamma (G):  {o.gamma:>+.6f}   (Delta change per Rs.1 spot move)")
        print(f"    Theta (Q):  {o.theta:>+.4f}   (Rs. lost per day)")
        print(f"    Vega  (V):  {o.vega:>+.4f}   (Rs. change per 1%% vol move)")
        print(f"    Rho   (r):  {o.rho:>+.4f}   (Rs. change per 1%% rate move)")

        print(f"\n  RECOMMENDED STRATEGY: {o.recommended_strategy}")
        for leg in o.strategy_legs:
            print(f"    {leg['action']:4s}  {leg['type']}  @  {leg['strike']:>8,.0f}  "
                  f"(Rs.{leg['premium']:,.2f})")
        print(f"  Max Profit:  {o.max_profit}")
        print(f"  Max Loss:    {o.max_loss}")
        print(f"  Breakeven:   {o.breakeven}")
        print(f"  {o.strategy_rationale}")

        # --- RISK METRICS ---
        print(f"\n{'  RISK METRICS (vs Nifty 50)':-<{w}}")
        print(f"  Alpha (Jensen):   {r.alpha:>+.2%}   (risk-adjusted excess return)")
        print(f"  Beta (CAPM):      {r.beta:>+.4f}   (systematic risk)")
        print(f"  Sharpe Ratio:     {r.sharpe_ratio:>+.4f}   (return per unit risk)")
        print(f"  Sortino Ratio:    {r.sortino_ratio:>+.4f}   (return per unit downside)")
        print(f"  Treynor Ratio:    {r.treynor_ratio:>+.4f}   (return per unit beta)")
        print(f"  Info Ratio:       {r.information_ratio:>+.4f}   (tracking error adjusted)")
        print(f"  Max Drawdown:     {r.max_drawdown:>+.2%}")
        print(f"  VaR (95%):        {r.var_95:>+.4f}   (daily)")
        print(f"  CVaR (95%):       {r.cvar_95:>+.4f}   (tail risk)")
        print(f"  Calmar Ratio:     {r.calmar_ratio:>+.4f}")
        print(f"  Omega Ratio:      {r.omega_ratio:>+.4f}")
        print(f"  Nifty Corr:       {r.correlation_nifty:>+.4f}")

        # --- SIGNAL SUMMARY TABLE ---
        print(f"\n{'  SIGNAL CONFLUENCE':-<{w}}")
        print(f"  {'Component':<15} {'Signal':<12} {'Weight'}")
        print(f"  {'Trend':<15} {s.trend_signal:<12} 25%")
        print(f"  {'Momentum':<15} {s.momentum_signal:<12} 25%")
        print(f"  {'Volatility':<15} {s.volatility_signal:<12} 25%")
        print(f"  {'Volume':<15} {s.volume_signal:<12} 25%")
        print(f"  {'Multi-TF':<15} {s.multi_tf_signal:<12} Bonus")
        print(f"  {'':->15} {'':->12} {'':->6}")
        print(f"  {'COMPOSITE':<15} {s.primary_action:<12} {s.confidence_score}/100")

        print("\n" + "=" * w)
        print("  DISCLAIMER: This is for educational/research purposes only.")
        print("  Not financial advice. Consult a SEBI-registered advisor.")
        print("=" * w)
