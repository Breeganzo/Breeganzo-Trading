"""
Options Signal Generator
=========================
Generates basic options trading signals for Nifty and BankNifty.

Strategy Signals
----------------
1. **Covered Call**: Own underlying + sell OTM call
   → Bullish bias, earn premium, cap upside
2. **Protective Put**: Own underlying + buy OTM put
   → Bullish bias with downside protection
3. **Bull Call Spread**: Buy ATM call + sell OTM call
   → Moderately bullish, limited risk & reward
4. **Bear Put Spread**: Buy ATM put + sell OTM put
   → Moderately bearish, limited risk & reward
5. **Iron Condor**: Sell OTM put + buy further OTM put + sell OTM call + buy further OTM call
   → Neutral / range-bound, earn premium
6. **Long Straddle**: Buy ATM call + ATM put
   → Expecting big move, direction unknown

Signal Logic
------------
- Uses predicted direction + confidence from model
- Uses predicted volatility vs implied volatility
- Selects strategy based on market regime
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

from .greeks import BlackScholesGreeks


@dataclass
class OptionsSignal:
    """A single options trading signal."""
    underlying: str
    strategy: str
    direction: str          # BULLISH, BEARISH, NEUTRAL
    confidence: float
    legs: List[dict]        # Each leg: {type, strike, expiry_days, action}
    max_profit: str         # Description
    max_loss: str           # Description
    breakeven: str          # Description
    rationale: str
    greeks_summary: dict


class OptionsSignalGenerator:
    """
    Generate options signals for Nifty / BankNifty.

    Uses model predictions + volatility analysis to suggest
    appropriate options strategies.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate
    lot_sizes : dict
        Underlying → lot size mapping
    """

    # Standard lot sizes for NSE index options
    DEFAULT_LOT_SIZES = {
        "NIFTY": 25,       # Nifty 50 lot = 25
        "BANKNIFTY": 15,   # BankNifty lot = 15
    }

    # Strike interval for selecting OTM strikes
    STRIKE_INTERVALS = {
        "NIFTY": 50,
        "BANKNIFTY": 100,
    }

    def __init__(
        self,
        risk_free_rate: float = 0.065,
        lot_sizes: Optional[dict] = None,
    ):
        self.bs = BlackScholesGreeks(risk_free_rate)
        self.lot_sizes = lot_sizes or self.DEFAULT_LOT_SIZES

    def _round_strike(self, price: float, underlying: str, direction: str = "nearest") -> float:
        """Round to nearest valid strike price."""
        interval = self.STRIKE_INTERVALS.get(underlying, 50)
        if direction == "up":
            return np.ceil(price / interval) * interval
        elif direction == "down":
            return np.floor(price / interval) * interval
        return round(price / interval) * interval

    def _select_strategy(
        self,
        pred_direction: float,
        confidence: float,
        vol_ratio: float,
    ) -> str:
        """
        Select options strategy based on:
        - pred_direction: Predicted return (positive = bullish)
        - confidence: Model confidence (0-1)
        - vol_ratio: predicted_vol / implied_vol
          > 1 means we think vol will be higher than market expects

        Decision matrix:
        ┌────────────┬───────────┬─────────────────────┐
        │ Direction  │ Vol View  │ Strategy            │
        ├────────────┼───────────┼─────────────────────┤
        │ Bullish    │ Low conf  │ Covered Call         │
        │ Bullish    │ High conf │ Bull Call Spread     │
        │ Bearish    │ Low conf  │ Protective Put       │
        │ Bearish    │ High conf │ Bear Put Spread      │
        │ Neutral    │ Low vol   │ Iron Condor          │
        │ Neutral    │ High vol  │ Long Straddle        │
        └────────────┴───────────┴─────────────────────┘
        """
        # Determine direction regime
        if abs(pred_direction) < 0.005:
            regime = "neutral"
        elif pred_direction > 0:
            regime = "bullish"
        else:
            regime = "bearish"

        if regime == "neutral":
            if vol_ratio > 1.1:
                return "long_straddle"
            else:
                return "iron_condor"
        elif regime == "bullish":
            if confidence >= 0.65:
                return "bull_call_spread"
            else:
                return "covered_call"
        else:  # bearish
            if confidence >= 0.65:
                return "bear_put_spread"
            else:
                return "protective_put"

    def _build_bull_call_spread(
        self, spot: float, T: float, sigma: float, underlying: str
    ) -> dict:
        """Build bull call spread legs."""
        atm = self._round_strike(spot, underlying)
        otm = atm + 2 * self.STRIKE_INTERVALS[underlying]

        buy_price = self.bs.call_price(spot, atm, T, sigma)
        sell_price = self.bs.call_price(spot, otm, T, sigma)
        net_debit = buy_price - sell_price

        legs = [
            {"type": "CE", "strike": atm, "action": "BUY", "premium": round(buy_price, 2)},
            {"type": "CE", "strike": otm, "action": "SELL", "premium": round(sell_price, 2)},
        ]
        return {
            "legs": legs,
            "max_profit": f"₹{(otm - atm - net_debit) * self.lot_sizes[underlying]:,.0f} "
                          f"({otm - atm - net_debit:.0f} per lot unit)",
            "max_loss": f"₹{net_debit * self.lot_sizes[underlying]:,.0f} "
                        f"(net debit {net_debit:.0f})",
            "breakeven": f"{atm + net_debit:.0f}",
        }

    def _build_bear_put_spread(
        self, spot: float, T: float, sigma: float, underlying: str
    ) -> dict:
        """Build bear put spread legs."""
        atm = self._round_strike(spot, underlying)
        otm = atm - 2 * self.STRIKE_INTERVALS[underlying]

        buy_price = self.bs.put_price(spot, atm, T, sigma)
        sell_price = self.bs.put_price(spot, otm, T, sigma)
        net_debit = buy_price - sell_price

        legs = [
            {"type": "PE", "strike": atm, "action": "BUY", "premium": round(buy_price, 2)},
            {"type": "PE", "strike": otm, "action": "SELL", "premium": round(sell_price, 2)},
        ]
        return {
            "legs": legs,
            "max_profit": f"₹{(atm - otm - net_debit) * self.lot_sizes[underlying]:,.0f}",
            "max_loss": f"₹{net_debit * self.lot_sizes[underlying]:,.0f}",
            "breakeven": f"{atm - net_debit:.0f}",
        }

    def _build_iron_condor(
        self, spot: float, T: float, sigma: float, underlying: str
    ) -> dict:
        """Build iron condor legs."""
        interval = self.STRIKE_INTERVALS[underlying]
        atm = self._round_strike(spot, underlying)

        sell_put = atm - 2 * interval
        buy_put = sell_put - interval
        sell_call = atm + 2 * interval
        buy_call = sell_call + interval

        sp = self.bs.put_price(spot, sell_put, T, sigma)
        bp = self.bs.put_price(spot, buy_put, T, sigma)
        sc = self.bs.call_price(spot, sell_call, T, sigma)
        bc = self.bs.call_price(spot, buy_call, T, sigma)

        net_credit = (sp - bp) + (sc - bc)
        lot = self.lot_sizes[underlying]

        legs = [
            {"type": "PE", "strike": buy_put, "action": "BUY", "premium": round(bp, 2)},
            {"type": "PE", "strike": sell_put, "action": "SELL", "premium": round(sp, 2)},
            {"type": "CE", "strike": sell_call, "action": "SELL", "premium": round(sc, 2)},
            {"type": "CE", "strike": buy_call, "action": "BUY", "premium": round(bc, 2)},
        ]
        return {
            "legs": legs,
            "max_profit": f"₹{net_credit * lot:,.0f} (net credit)",
            "max_loss": f"₹{(interval - net_credit) * lot:,.0f}",
            "breakeven": f"{sell_put - net_credit:.0f} / {sell_call + net_credit:.0f}",
        }

    def _build_long_straddle(
        self, spot: float, T: float, sigma: float, underlying: str
    ) -> dict:
        """Build long straddle legs."""
        atm = self._round_strike(spot, underlying)
        call_price = self.bs.call_price(spot, atm, T, sigma)
        put_price = self.bs.put_price(spot, atm, T, sigma)
        total = call_price + put_price
        lot = self.lot_sizes[underlying]

        legs = [
            {"type": "CE", "strike": atm, "action": "BUY", "premium": round(call_price, 2)},
            {"type": "PE", "strike": atm, "action": "BUY", "premium": round(put_price, 2)},
        ]
        return {
            "legs": legs,
            "max_profit": "Unlimited (theoretically)",
            "max_loss": f"₹{total * lot:,.0f} (total premium)",
            "breakeven": f"{atm - total:.0f} / {atm + total:.0f}",
        }

    def _build_covered_call(
        self, spot: float, T: float, sigma: float, underlying: str
    ) -> dict:
        """Build covered call legs (own underlying + sell call)."""
        otm = self._round_strike(spot * 1.02, underlying, "up")
        call_price = self.bs.call_price(spot, otm, T, sigma)
        lot = self.lot_sizes[underlying]

        legs = [
            {"type": "EQ", "strike": spot, "action": "BUY", "premium": round(spot, 2)},
            {"type": "CE", "strike": otm, "action": "SELL", "premium": round(call_price, 2)},
        ]
        return {
            "legs": legs,
            "max_profit": f"₹{(otm - spot + call_price) * lot:,.0f}",
            "max_loss": f"₹{(spot - call_price) * lot:,.0f} (if falls to 0)",
            "breakeven": f"{spot - call_price:.0f}",
        }

    def _build_protective_put(
        self, spot: float, T: float, sigma: float, underlying: str
    ) -> dict:
        """Build protective put legs (own underlying + buy put)."""
        otm = self._round_strike(spot * 0.98, underlying, "down")
        put_price = self.bs.put_price(spot, otm, T, sigma)
        lot = self.lot_sizes[underlying]

        legs = [
            {"type": "EQ", "strike": spot, "action": "BUY", "premium": round(spot, 2)},
            {"type": "PE", "strike": otm, "action": "BUY", "premium": round(put_price, 2)},
        ]
        return {
            "legs": legs,
            "max_profit": "Unlimited upside − put premium",
            "max_loss": f"₹{(spot - otm + put_price) * lot:,.0f}",
            "breakeven": f"{spot + put_price:.0f}",
        }

    def generate_signal(
        self,
        underlying: str,
        spot: float,
        pred_return: float,
        confidence: float,
        hist_vol: float,
        implied_vol: float = None,
        days_to_expiry: int = 30,
    ) -> OptionsSignal:
        """
        Generate an options trading signal.

        Parameters
        ----------
        underlying : str
            NIFTY or BANKNIFTY
        spot : float
            Current spot price
        pred_return : float
            Model-predicted return (e.g., 0.02 = +2%)
        confidence : float
            Model confidence (0-1)
        hist_vol : float
            Historical annualised volatility
        implied_vol : float, optional
            Market implied vol (if available). Defaults to hist_vol.
        days_to_expiry : int
            Days until expiry

        Returns
        -------
        OptionsSignal
        """
        if implied_vol is None:
            implied_vol = hist_vol

        T = days_to_expiry / 365.0
        vol_ratio = hist_vol / implied_vol if implied_vol > 0 else 1.0
        sigma = hist_vol

        # Select strategy
        strategy = self._select_strategy(pred_return, confidence, vol_ratio)

        # Build legs and profit/loss info
        builders = {
            "bull_call_spread": self._build_bull_call_spread,
            "bear_put_spread": self._build_bear_put_spread,
            "iron_condor": self._build_iron_condor,
            "long_straddle": self._build_long_straddle,
            "covered_call": self._build_covered_call,
            "protective_put": self._build_protective_put,
        }
        build_fn = builders[strategy]
        result = build_fn(spot, T, sigma, underlying)

        # Direction
        if pred_return > 0.005:
            direction = "BULLISH"
        elif pred_return < -0.005:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Greeks for ATM option
        atm = self._round_strike(spot, underlying)
        greeks = self.bs.all_greeks(spot, atm, T, sigma, "call")

        # Build rationale
        rationale = (
            f"Model predicts {pred_return:+.2%} move with {confidence:.0%} confidence. "
            f"Historical vol {hist_vol:.1%} vs implied {implied_vol:.1%} "
            f"(ratio {vol_ratio:.2f}). "
            f"Selected {strategy.replace('_', ' ').title()} strategy."
        )

        return OptionsSignal(
            underlying=underlying,
            strategy=strategy.replace("_", " ").title(),
            direction=direction,
            confidence=confidence,
            legs=result["legs"],
            max_profit=result["max_profit"],
            max_loss=result["max_loss"],
            breakeven=result["breakeven"],
            rationale=rationale,
            greeks_summary=greeks,
        )

    def format_signal(self, signal: OptionsSignal) -> str:
        """Pretty-print an options signal."""
        lines = [
            f"╔══════════════ Options Signal: {signal.underlying} ══════════════╗",
            f"  Strategy:   {signal.strategy}",
            f"  Direction:  {signal.direction}",
            f"  Confidence: {signal.confidence:.1%}",
            "",
            "  Legs:",
        ]
        for leg in signal.legs:
            lines.append(
                f"    {leg['action']:4s} {leg['type']} @ {leg['strike']:,.0f}  "
                f"(₹{leg['premium']:,.2f})"
            )
        lines.extend([
            "",
            f"  Max Profit:  {signal.max_profit}",
            f"  Max Loss:    {signal.max_loss}",
            f"  Breakeven:   {signal.breakeven}",
            "",
            f"  ATM Greeks:  Δ={signal.greeks_summary['delta']:.3f}  "
            f"Γ={signal.greeks_summary['gamma']:.5f}  "
            f"Θ={signal.greeks_summary['theta']:.3f}  "
            f"V={signal.greeks_summary['vega']:.3f}",
            "",
            f"  Rationale: {signal.rationale}",
            "╚═══════════════════════════════════════════════════════╝",
        ])
        return "\n".join(lines)
