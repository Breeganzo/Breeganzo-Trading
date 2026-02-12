"""
Backtest Engine
================
Vectorised backtester that simulates trading with:
- Multiple tickers
- Transaction costs (Groww fee model)
- Slippage
- Position sizing
- Stop-loss and take-profit

Takes model signals → outputs equity curve, trade log, and metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

from .costs import GrowwCostCalculator
from .metrics import compute_all_metrics
from ..utils.constants import RISK_FREE_RATE


@dataclass
class Trade:
    """Record of a single trade."""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    position_size: float       # In ₹
    shares: int
    direction: str             # "LONG" or "SHORT"
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float
    hold_days: int
    signal_confidence: float


class BacktestEngine:
    """
    Vectorised backtester with transaction costs.

    Usage
    -----
    >>> engine = BacktestEngine(initial_capital=100000)
    >>> results = engine.run(signals_df, price_data)
    >>> print(results["metrics"])
    >>> engine.plot_results()

    Parameters
    ----------
    initial_capital : float
        Starting capital in ₹
    max_positions : int
        Maximum concurrent positions
    max_position_pct : float
        Maximum % of capital per position
    slippage_pct : float
        Slippage per trade (default: 0.05%)
    stop_loss_pct : float
        Stop-loss threshold (default: 5%)
    take_profit_pct : float
        Take-profit threshold (default: 10%)
    trade_type : str
        Groww trade type for cost calculation
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        max_positions: int = 15,
        max_position_pct: float = 0.10,
        slippage_pct: float = 0.0005,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.10,
        trade_type: str = "equity_delivery",
        min_signal_threshold: float = 0.015,          # Raised: 1.5% (was 0.5%)
        min_hold_days: int = 3,
        require_model_agreement: bool = True,           # NEW: both models must agree
        require_volume_confirmation: bool = True,       # NEW: volume > 1.2x avg
        use_atr_stop: bool = True,                      # NEW: ATR-based dynamic stop
        atr_stop_multiplier: float = 1.5,               # NEW: 1.5 × ATR
        rebalance_frequency: str = "weekly",             # NEW: weekly/daily
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.slippage_pct = slippage_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trade_type = trade_type
        self.min_signal_threshold = min_signal_threshold
        self.min_hold_days = min_hold_days
        self.require_model_agreement = require_model_agreement
        self.require_volume_confirmation = require_volume_confirmation
        self.use_atr_stop = use_atr_stop
        self.atr_stop_multiplier = atr_stop_multiplier
        self.rebalance_frequency = rebalance_frequency
        self.cost_calculator = GrowwCostCalculator()

    def run(
        self,
        signals: pd.DataFrame,
        price_data: dict[str, pd.DataFrame],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> dict:
        """
        Run the backtest.

        Parameters
        ----------
        signals : pd.DataFrame
            Must have columns: Date, Ticker, Signal, Confidence, Predicted_Range
            Signal: 1 = BUY, -1 = SELL, 0 = HOLD
        price_data : dict
            {ticker: DataFrame with Close column}
        benchmark_returns : pd.Series, optional
            For alpha/beta calculation

        Returns
        -------
        dict
            {
                "equity_curve": pd.Series,
                "daily_returns": pd.Series,
                "trade_log": list[Trade],
                "metrics": dict,
                "cost_analysis": dict,
            }
        """
        # Sort signals by date
        signals = signals.sort_values("Date").reset_index(drop=True)
        all_dates = sorted(signals["Date"].unique())

        # State tracking
        capital = self.initial_capital
        positions = {}  # ticker -> {shares, entry_price, entry_date, confidence}
        equity_curve = []
        daily_returns = []
        trade_log = []
        total_costs = 0.0

        prev_equity = self.initial_capital

        for date in all_dates:
            day_signals = signals[signals["Date"] == date]

            # --- Check stop-loss / take-profit on existing positions ---
            tickers_to_close = []
            for ticker, pos in positions.items():
                if ticker in price_data and date in price_data[ticker].index:
                    current_price = price_data[ticker].loc[date, "Close"]
                    ret = (current_price - pos["entry_price"]) / pos["entry_price"]

                    # Dynamic ATR-based stop-loss (adapts per stock volatility)
                    if self.use_atr_stop and "atr_pct" in pos:
                        dynamic_stop = self.atr_stop_multiplier * pos["atr_pct"]
                    else:
                        dynamic_stop = self.stop_loss_pct

                    if ret <= -dynamic_stop:
                        tickers_to_close.append((ticker, "stop_loss"))
                    elif ret >= self.take_profit_pct:
                        tickers_to_close.append((ticker, "take_profit"))

            # Close positions that hit stop-loss or take-profit
            for ticker, reason in tickers_to_close:
                if ticker in price_data and date in price_data[ticker].index:
                    pos = positions[ticker]
                    # Enforce minimum hold period (except stop-loss)
                    hold_days = (pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days
                    if reason == "take_profit" and hold_days < self.min_hold_days:
                        continue  # Don't exit too early for profit-taking

                    exit_price = price_data[ticker].loc[date, "Close"]
                    exit_price *= (1 - self.slippage_pct)  # Slippage on exit

                    pos = positions[ticker]
                    sell_value = pos["shares"] * exit_price
                    buy_value = pos["shares"] * pos["entry_price"]
                    cost = self.cost_calculator.round_trip_cost(
                        buy_value, sell_value, self.trade_type
                    )

                    gross_pnl = sell_value - buy_value
                    net_pnl = gross_pnl - cost.total
                    capital += sell_value - cost.total
                    total_costs += cost.total

                    trade_log.append(Trade(
                        ticker=ticker,
                        entry_date=pos["entry_date"],
                        exit_date=str(date)[:10],
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        position_size=buy_value,
                        shares=pos["shares"],
                        direction="LONG",
                        gross_pnl=gross_pnl,
                        costs=cost.total,
                        net_pnl=net_pnl,
                        return_pct=net_pnl / buy_value,
                        hold_days=(pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days,
                        signal_confidence=pos["confidence"],
                    ))
                    del positions[ticker]

            # --- Process new signals ---
            # SELL signals first (free up capital)
            sell_signals = day_signals[day_signals["Signal"] == -1]
            for _, row in sell_signals.iterrows():
                ticker = row["Ticker"]
                if ticker in positions and ticker in price_data:
                    # Enforce minimum hold period for sells
                    pos = positions[ticker]
                    hold_days = (pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days
                    if hold_days < self.min_hold_days:
                        continue  # Don't sell too early

                    if date in price_data[ticker].index:
                        exit_price = price_data[ticker].loc[date, "Close"]
                        exit_price *= (1 - self.slippage_pct)

                        pos = positions[ticker]
                        sell_value = pos["shares"] * exit_price
                        buy_value = pos["shares"] * pos["entry_price"]
                        cost = self.cost_calculator.round_trip_cost(
                            buy_value, sell_value, self.trade_type
                        )

                        gross_pnl = sell_value - buy_value
                        net_pnl = gross_pnl - cost.total
                        capital += sell_value - cost.total
                        total_costs += cost.total

                        trade_log.append(Trade(
                            ticker=ticker,
                            entry_date=pos["entry_date"],
                            exit_date=str(date)[:10],
                            entry_price=pos["entry_price"],
                            exit_price=exit_price,
                            position_size=buy_value,
                            shares=pos["shares"],
                            direction="LONG",
                            gross_pnl=gross_pnl,
                            costs=cost.total,
                            net_pnl=net_pnl,
                            return_pct=net_pnl / buy_value if buy_value > 0 else 0,
                            hold_days=(pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days,
                            signal_confidence=pos["confidence"],
                        ))
                        del positions[ticker]

            # BUY signals — filter by minimum signal threshold
            buy_signals = day_signals[day_signals["Signal"] == 1].copy()

            # Weekly rebalancing: only enter new positions on Monday
            if self.rebalance_frequency == "weekly":
                if pd.Timestamp(date).dayofweek != 0:  # 0 = Monday
                    buy_signals = buy_signals.iloc[0:0]  # Empty DataFrame

            # Only trade when predicted return exceeds threshold (reduces noise trades)
            if "Predicted_Return" in buy_signals.columns:
                buy_signals = buy_signals[
                    buy_signals["Predicted_Return"].abs() >= self.min_signal_threshold
                ]
            elif "Predicted_Range" in buy_signals.columns:
                buy_signals = buy_signals[
                    buy_signals["Predicted_Range"].abs() >= self.min_signal_threshold
                ]

            # Model agreement filter: both XGB and LGB must agree on direction
            if self.require_model_agreement and len(buy_signals) > 0:
                if "Model_Agreement" in buy_signals.columns:
                    buy_signals = buy_signals[buy_signals["Model_Agreement"] >= 1.0]

            # Volume confirmation: only enter on above-average volume
            if self.require_volume_confirmation and len(buy_signals) > 0:
                if "Volume_Ratio" in buy_signals.columns:
                    buy_signals = buy_signals[buy_signals["Volume_Ratio"] >= 1.2]

            buy_signals = buy_signals.sort_values(
                "Confidence", ascending=False
            )
            for _, row in buy_signals.iterrows():
                ticker = row["Ticker"]
                if ticker in positions:
                    continue  # Already in position
                if len(positions) >= self.max_positions:
                    break  # Max positions reached
                if ticker not in price_data:
                    continue
                if date not in price_data[ticker].index:
                    continue

                # Position sizing: min(max_position_pct * total_equity, available_capital)
                total_equity = capital + sum(
                    pos["shares"] * price_data[t].loc[date, "Close"]
                    for t, pos in positions.items()
                    if t in price_data and date in price_data[t].index
                )
                position_value = min(
                    self.max_position_pct * total_equity,
                    capital * 0.95,  # Keep 5% cash buffer
                )
                if position_value < 500:  # Minimum trade size
                    continue

                entry_price = price_data[ticker].loc[date, "Close"]
                entry_price *= (1 + self.slippage_pct)  # Slippage on entry

                shares = int(position_value / entry_price)
                if shares == 0:
                    continue

                actual_cost = shares * entry_price
                buy_cost = self.cost_calculator.buy_cost(actual_cost, self.trade_type)
                capital -= actual_cost + buy_cost.total
                total_costs += buy_cost.total

                # Store ATR% for dynamic stop-loss
                atr_pct_val = row.get("ATR_pct", self.stop_loss_pct)
                if isinstance(atr_pct_val, str):
                    try:
                        atr_pct_val = float(atr_pct_val)
                    except (ValueError, TypeError):
                        atr_pct_val = self.stop_loss_pct

                positions[ticker] = {
                    "shares": shares,
                    "entry_price": entry_price,
                    "entry_date": str(date)[:10],
                    "confidence": row.get("Confidence", 0.5),
                    "atr_pct": atr_pct_val,
                }

            # --- Calculate daily equity ---
            position_value = 0
            for ticker, pos in positions.items():
                if ticker in price_data and date in price_data[ticker].index:
                    position_value += pos["shares"] * price_data[ticker].loc[date, "Close"]

            total_equity = capital + position_value
            equity_curve.append({"Date": date, "Equity": total_equity})

            daily_ret = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            daily_returns.append({"Date": date, "Return": daily_ret})
            prev_equity = total_equity

        # --- Build output ---
        equity_df = pd.DataFrame(equity_curve).set_index("Date")["Equity"]
        returns_df = pd.DataFrame(daily_returns).set_index("Date")["Return"]

        # Compute metrics
        metrics = compute_all_metrics(
            portfolio_returns=returns_df,
            equity_curve=equity_df,
            benchmark_returns=benchmark_returns,
            risk_free_rate=RISK_FREE_RATE,
        )

        # Cost analysis
        cost_analysis = {
            "Total Costs (₹)": total_costs,
            "Avg Cost per Trade (₹)": total_costs / max(len(trade_log), 1),
            "Costs as % of Initial": total_costs / self.initial_capital * 100,
            "Total Trades": len(trade_log),
            "Winning Trades": sum(1 for t in trade_log if t.net_pnl > 0),
            "Losing Trades": sum(1 for t in trade_log if t.net_pnl <= 0),
        }

        return {
            "equity_curve": equity_df,
            "daily_returns": returns_df,
            "trade_log": trade_log,
            "metrics": metrics,
            "cost_analysis": cost_analysis,
        }

    def trade_log_to_df(self, trade_log: list[Trade]) -> pd.DataFrame:
        """Convert trade log to DataFrame for analysis."""
        if not trade_log:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "Ticker": t.ticker,
                "Entry Date": t.entry_date,
                "Exit Date": t.exit_date,
                "Entry Price": f"₹{t.entry_price:.2f}",
                "Exit Price": f"₹{t.exit_price:.2f}",
                "Shares": t.shares,
                "Position (₹)": f"₹{t.position_size:,.0f}",
                "Gross P&L": f"₹{t.gross_pnl:,.2f}",
                "Costs": f"₹{t.costs:.2f}",
                "Net P&L": f"₹{t.net_pnl:,.2f}",
                "Return %": f"{t.return_pct:.2%}",
                "Days Held": t.hold_days,
                "Confidence": f"{t.signal_confidence:.2f}",
            }
            for t in trade_log
        ])
