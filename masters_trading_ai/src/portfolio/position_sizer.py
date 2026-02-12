"""
Position Sizing
================
Determines how much capital to allocate per trade.

Methods:
- Half-Kelly: Conservative version of Kelly Criterion
- Equal Risk: Size inversely proportional to volatility
- Equal Weight: Simple equal allocation
"""

import numpy as np
import pandas as pd


class PositionSizer:
    """
    Position sizing calculator.

    The Kelly Criterion formula:
        f* = (p × b - q) / b

    Where:
        f* = fraction of capital to bet
        p = probability of winning
        q = 1 - p (probability of losing)
        b = odds ratio (avg win / avg loss)

    Half-Kelly uses f*/2 for safety — full Kelly is too aggressive
    and leads to excessive drawdowns in practice.
    """

    def __init__(self, method: str = "half_kelly"):
        self.method = method

    def kelly_fraction(
        self,
        win_prob: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate Kelly fraction.

        Parameters
        ----------
        win_prob : float
            Probability of winning trade (0-1)
        avg_win : float
            Average winning trade return (positive)
        avg_loss : float
            Average losing trade return (positive, expressed as absolute)

        Returns
        -------
        float
            Fraction of capital to allocate (0-1)
        """
        if avg_loss == 0 or avg_win == 0:
            return 0.0

        b = avg_win / avg_loss  # Odds ratio
        q = 1 - win_prob
        kelly = (win_prob * b - q) / b

        if self.method == "half_kelly":
            kelly *= 0.5

        # Clamp to [0, 0.25] — never bet more than 25%
        return max(0.0, min(kelly, 0.25))

    def equal_risk_weights(
        self,
        volatilities: pd.Series,
        target_vol: float = 0.15,
    ) -> pd.Series:
        """
        Risk-parity weighting: allocate inversely to volatility.

        Each position contributes equally to portfolio risk.

        Parameters
        ----------
        volatilities : pd.Series
            Ticker → annualised volatility
        target_vol : float
            Target portfolio volatility

        Returns
        -------
        pd.Series
            Ticker → portfolio weight (sums to 1)
        """
        inv_vol = 1.0 / (volatilities + 1e-10)
        weights = inv_vol / inv_vol.sum()
        return weights

    def equal_weights(self, n_positions: int) -> float:
        """Simple equal allocation."""
        if n_positions == 0:
            return 0.0
        return 1.0 / n_positions

    def size_position(
        self,
        capital: float,
        win_prob: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.015,
        max_pct: float = 0.10,
    ) -> float:
        """
        Calculate position size in ₹.

        Parameters
        ----------
        capital : float
            Total available capital
        win_prob : float
            Model's predicted win probability
        avg_win, avg_loss : float
            Historical average win/loss sizes
        max_pct : float
            Maximum position as % of capital

        Returns
        -------
        float
            Position size in ₹
        """
        if self.method == "equal_weight":
            fraction = max_pct
        elif self.method in ("kelly", "half_kelly"):
            fraction = self.kelly_fraction(win_prob, avg_win, avg_loss)
        else:
            fraction = max_pct

        fraction = min(fraction, max_pct)
        return capital * fraction
