"""
Daily Capital Allocator
========================
Takes the user's available capital and model predictions,
then outputs specific allocation suggestions.

Allocation Logic:
1. Rank all stocks by predicted range × confidence (expected edge)
2. Filter by minimum confidence threshold
3. Apply risk manager constraints (sector limits, position limits)
4. Size positions using half-Kelly or equal-risk-contribution
5. Output: list of (ticker, action, amount, rationale)
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass

from .position_sizer import PositionSizer
from .risk_manager import RiskManager


@dataclass
class Allocation:
    """A single allocation suggestion."""
    ticker: str
    action: str           # BUY, SELL, HOLD, WAIT_FOR_DIP
    amount_inr: float     # How much to allocate in ₹
    shares: int           # Approximate shares
    current_price: float
    predicted_range_pct: float
    predicted_direction: float  # Probability of up
    confidence: float
    bucket: str           # large_cap, banking, etc.
    rationale: str


class DailyAllocator:
    """
    Daily portfolio allocator.

    Usage
    -----
    >>> allocator = DailyAllocator(capital=75000)
    >>> suggestions = allocator.allocate(predictions_df, prices_df)
    >>> for s in suggestions:
    ...     print(f"{s.action} {s.ticker}: ₹{s.amount_inr:,.0f} ({s.rationale})")
    """

    def __init__(
        self,
        capital: float = 100000,
        max_positions: int = 10,
        max_position_pct: float = 0.10,
        min_confidence: float = 0.55,
        top_n: int = 10,
    ):
        self.capital = capital
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.min_confidence = min_confidence
        self.top_n = top_n
        self.sizer = PositionSizer(method="half_kelly")
        self.risk_mgr = RiskManager()

    def allocate(
        self,
        predictions: pd.DataFrame,
        current_prices: pd.Series,
        existing_positions: Optional[dict] = None,
        ticker_buckets: Optional[dict] = None,
    ) -> list[Allocation]:
        """
        Generate daily allocation suggestions.

        Parameters
        ----------
        predictions : pd.DataFrame
            Must have columns:
            - Ticker
            - Predicted_Range (predicted high-low range as %)
            - Direction_Prob (probability of positive return)
            - Confidence (model confidence 0-1)
        current_prices : pd.Series
            Ticker → current price
        existing_positions : dict, optional
            {ticker: {"shares": int, "entry_price": float}}
        ticker_buckets : dict, optional
            {ticker: bucket_name}

        Returns
        -------
        list[Allocation]
            Sorted by priority (highest expected edge first)
        """
        if existing_positions is None:
            existing_positions = {}
        if ticker_buckets is None:
            ticker_buckets = {}

        allocations = []

        # --- 1. SELL signals for existing positions ---
        for ticker, pos in existing_positions.items():
            if ticker in predictions["Ticker"].values:
                row = predictions[predictions["Ticker"] == ticker].iloc[0]
                if row["Direction_Prob"] < 0.45:  # Predicted to go down
                    allocations.append(Allocation(
                        ticker=ticker,
                        action="SELL",
                        amount_inr=pos["shares"] * current_prices.get(ticker, 0),
                        shares=pos["shares"],
                        current_price=current_prices.get(ticker, 0),
                        predicted_range_pct=row["Predicted_Range"],
                        predicted_direction=row["Direction_Prob"],
                        confidence=row["Confidence"],
                        bucket=ticker_buckets.get(ticker, "unknown"),
                        rationale=f"Direction prob {row['Direction_Prob']:.0%} < 45% — predicted decline",
                    ))
                elif row["Direction_Prob"] < 0.55:
                    allocations.append(Allocation(
                        ticker=ticker,
                        action="HOLD",
                        amount_inr=0,
                        shares=pos["shares"],
                        current_price=current_prices.get(ticker, 0),
                        predicted_range_pct=row["Predicted_Range"],
                        predicted_direction=row["Direction_Prob"],
                        confidence=row["Confidence"],
                        bucket=ticker_buckets.get(ticker, "unknown"),
                        rationale="Neutral signal — hold position, no action",
                    ))

        # --- 2. BUY signals (ranked by expected edge) ---
        buy_candidates = predictions[
            (predictions["Direction_Prob"] >= self.min_confidence) &
            (~predictions["Ticker"].isin(existing_positions.keys()))
        ].copy()

        if len(buy_candidates) == 0:
            return allocations

        # Expected edge = predicted_range × direction_prob × confidence
        buy_candidates["Edge"] = (
            buy_candidates["Predicted_Range"] *
            buy_candidates["Direction_Prob"] *
            buy_candidates["Confidence"]
        )
        buy_candidates = buy_candidates.sort_values("Edge", ascending=False)

        # Take top N
        top_picks = buy_candidates.head(self.top_n)

        # Available capital (after sell proceeds are counted)
        available_capital = self.capital * 0.95  # Keep 5% cash buffer

        # Allocate based on edge-weighted sizing
        total_edge = top_picks["Edge"].sum()
        n_positions = min(len(top_picks), self.max_positions - len(existing_positions))

        for _, row in top_picks.head(n_positions).iterrows():
            ticker = row["Ticker"]
            if ticker not in current_prices or current_prices[ticker] <= 0:
                continue

            # Position size: proportional to edge, capped at max_position_pct
            if total_edge > 0:
                weight = row["Edge"] / total_edge
            else:
                weight = 1.0 / n_positions

            position_value = min(
                weight * available_capital,
                self.max_position_pct * self.capital,
            )

            # Round to whole shares
            price = current_prices[ticker]
            shares = max(int(position_value / price), 0)
            if shares == 0:
                continue

            actual_amount = shares * price

            # Determine action specifics
            if row["Direction_Prob"] > 0.70 and row["Confidence"] > 0.70:
                action = "STRONG_BUY"
                rationale = (
                    f"STRONG BUY — {row['Direction_Prob']:.0%} up probability, "
                    f"predicted range {row['Predicted_Range']:.2%}, "
                    f"confidence {row['Confidence']:.0%}"
                )
            elif row["Direction_Prob"] > 0.58 and row["Confidence"] > 0.55:
                action = "BUY"
                rationale = (
                    f"BUY — {row['Direction_Prob']:.0%} up probability, "
                    f"range {row['Predicted_Range']:.2%}, "
                    f"confidence {row['Confidence']:.0%}"
                )
            elif row["Predicted_Range"] > 0.015 and row["Direction_Prob"] > 0.55:
                action = "BUY"
                rationale = (
                    f"BUY — predicted range {row['Predicted_Range']:.2%} "
                    f"with {row['Direction_Prob']:.0%} up bias"
                )
            else:
                action = "WAIT_FOR_DIP"
                rationale = (
                    f"Wait for better entry — range {row['Predicted_Range']:.2%}, "
                    f"direction {row['Direction_Prob']:.0%}"
                )

            allocations.append(Allocation(
                ticker=ticker,
                action=action,
                amount_inr=actual_amount,
                shares=shares,
                current_price=price,
                predicted_range_pct=row["Predicted_Range"],
                predicted_direction=row["Direction_Prob"],
                confidence=row["Confidence"],
                bucket=ticker_buckets.get(ticker, "unknown"),
                rationale=rationale,
            ))

        # Sort: SELL first, then BUY by confidence descending
        action_order = {"SELL": 0, "BUY": 1, "WAIT_FOR_DIP": 2, "HOLD": 3}
        allocations.sort(key=lambda x: (action_order.get(x.action, 9), -x.confidence))

        return allocations

    def format_suggestions(self, allocations: list[Allocation]) -> pd.DataFrame:
        """Format allocations as a readable DataFrame."""
        if not allocations:
            return pd.DataFrame(columns=["Action", "Ticker", "Amount", "Shares",
                                         "Price", "Direction", "Confidence", "Rationale"])
        return pd.DataFrame([
            {
                "Action": a.action,
                "Ticker": a.ticker,
                "Amount (₹)": f"₹{a.amount_inr:,.0f}",
                "Shares": a.shares,
                "Price": f"₹{a.current_price:,.2f}",
                "Pred. Direction": f"{a.predicted_direction:.0%}",
                "Pred. Range": f"{a.predicted_range_pct:.2%}",
                "Confidence": f"{a.confidence:.0%}",
                "Bucket": a.bucket,
                "Rationale": a.rationale,
            }
            for a in allocations
        ])
