"""
Transaction Cost Engine for Indian Equity Delivery Trades (Groww Broker).

Calculates all statutory and broker-specific charges applicable to equity
delivery trades on Indian exchanges (NSE/BSE) through Groww, including
STT, exchange charges, GST, SEBI charges, stamp duty, and slippage.

All monetary values are in INR. Rates are sourced from app.core.config.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.core.config import get_settings


class TradeType(str, Enum):
    """Direction of a trade leg."""

    BUY = "BUY"
    SELL = "SELL"


# Slippage guard-rails (absolute percentages expressed as decimals).
_MIN_SLIPPAGE: float = 0.001   # 0.1%
_MAX_SLIPPAGE: float = 0.003   # 0.3%


@dataclass(frozen=True, slots=True)
class TradeCosts:
    """Itemised breakdown of costs for a single trade leg."""

    turnover: float
    brokerage: float
    stt: float
    exchange_charges: float
    gst: float
    sebi_charges: float
    stamp_duty: float
    slippage: float
    total_cost: float
    net_amount: float  # BUY: turnover + total_cost, SELL: turnover - total_cost


@dataclass(frozen=True, slots=True)
class RoundTripReturn:
    """Net return summary for a complete buy-then-sell round trip."""

    buy_turnover: float
    sell_turnover: float
    gross_return: float
    buy_costs: float
    sell_costs: float
    total_costs: float
    net_return: float
    net_return_pct: float  # percentage relative to capital deployed (buy side)


class TransactionCostEngine:
    """Pure-computation engine for Groww equity delivery transaction costs.

    Instantiate once and reuse -- the object holds only rate configuration
    read from ``get_settings()`` at construction time.

    Parameters
    ----------
    brokerage_rate : float, optional
        Override the brokerage rate from settings.
    stt_buy_rate : float, optional
        Override the STT rate for buy trades.
    stt_sell_rate : float, optional
        Override the STT rate for sell trades.
    exchange_charge_rate : float, optional
        Override the exchange transaction charge rate.
    gst_rate : float, optional
        Override the GST rate.
    sebi_charge_rate : float, optional
        Override the SEBI turnover charge rate.
    stamp_duty_rate : float, optional
        Override the stamp duty rate.
    default_slippage : float, optional
        Override the default slippage assumption.
    """

    def __init__(
        self,
        *,
        brokerage_rate: Optional[float] = None,
        stt_buy_rate: Optional[float] = None,
        stt_sell_rate: Optional[float] = None,
        exchange_charge_rate: Optional[float] = None,
        gst_rate: Optional[float] = None,
        sebi_charge_rate: Optional[float] = None,
        stamp_duty_rate: Optional[float] = None,
        default_slippage: Optional[float] = None,
    ) -> None:
        settings = get_settings()

        self.brokerage_rate: float = (
            brokerage_rate if brokerage_rate is not None else settings.BROKERAGE_RATE
        )
        self.stt_buy_rate: float = (
            stt_buy_rate if stt_buy_rate is not None else settings.STT_BUY_RATE
        )
        self.stt_sell_rate: float = (
            stt_sell_rate if stt_sell_rate is not None else settings.STT_SELL_RATE
        )
        self.exchange_charge_rate: float = (
            exchange_charge_rate
            if exchange_charge_rate is not None
            else settings.EXCHANGE_CHARGE_RATE
        )
        self.gst_rate: float = (
            gst_rate if gst_rate is not None else settings.GST_RATE
        )
        self.sebi_charge_rate: float = (
            sebi_charge_rate
            if sebi_charge_rate is not None
            else settings.SEBI_CHARGE_RATE
        )
        self.stamp_duty_rate: float = (
            stamp_duty_rate
            if stamp_duty_rate is not None
            else settings.STAMP_DUTY_RATE
        )
        self.default_slippage: float = (
            default_slippage
            if default_slippage is not None
            else settings.DEFAULT_SLIPPAGE
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clamp_slippage(self, slippage_pct: Optional[float]) -> float:
        """Return slippage rate clamped to the allowed 0.1%-0.3% band."""
        raw = slippage_pct if slippage_pct is not None else self.default_slippage
        return max(_MIN_SLIPPAGE, min(_MAX_SLIPPAGE, raw))

    @staticmethod
    def _round2(value: float) -> float:
        """Round to 2 decimal places (paisa precision)."""
        return round(value, 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_costs(
        self,
        price: float,
        quantity: int,
        trade_type: str | TradeType,
        slippage_pct: Optional[float] = None,
    ) -> dict:
        """Calculate all transaction cost components for a single trade leg.

        Parameters
        ----------
        price : float
            Per-share price in INR.
        quantity : int
            Number of shares.
        trade_type : str or TradeType
            ``"BUY"`` or ``"SELL"`` (case-insensitive).
        slippage_pct : float, optional
            Slippage as a decimal (e.g. 0.002 for 0.2%).  Clamped to
            [0.1%, 0.3%].  Falls back to ``DEFAULT_SLIPPAGE`` from settings.

        Returns
        -------
        dict
            Itemised cost breakdown with keys matching :class:`TradeCosts`
            fields.

        Raises
        ------
        ValueError
            If *price* or *quantity* is non-positive, or *trade_type* is
            unrecognised.
        """
        # --- Validate inputs ---
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        side = TradeType(str(trade_type).upper())
        is_buy = side is TradeType.BUY

        turnover = price * quantity
        slippage_rate = self._clamp_slippage(slippage_pct)

        # 1. Brokerage (zero for Groww delivery)
        brokerage = self._round2(turnover * self.brokerage_rate)

        # 2. STT -- applies to both buy and sell for delivery
        stt_rate = self.stt_buy_rate if is_buy else self.stt_sell_rate
        stt = self._round2(turnover * stt_rate)

        # 3. Exchange transaction charges
        exchange_charges = self._round2(turnover * self.exchange_charge_rate)

        # 4. GST @ 18% on (brokerage + exchange charges)
        gst = self._round2((brokerage + exchange_charges) * self.gst_rate)

        # 5. SEBI turnover charges
        sebi_charges = self._round2(turnover * self.sebi_charge_rate)

        # 6. Stamp duty -- only on buy side
        stamp_duty = self._round2(turnover * self.stamp_duty_rate) if is_buy else 0.0

        # 7. Slippage (market-impact estimate)
        slippage = self._round2(turnover * slippage_rate)

        total_cost = self._round2(
            brokerage + stt + exchange_charges + gst + sebi_charges + stamp_duty + slippage
        )

        # Net amount: what actually leaves (BUY) or arrives (SELL) in wallet.
        if is_buy:
            net_amount = self._round2(turnover + total_cost)
        else:
            net_amount = self._round2(turnover - total_cost)

        costs = TradeCosts(
            turnover=self._round2(turnover),
            brokerage=brokerage,
            stt=stt,
            exchange_charges=exchange_charges,
            gst=gst,
            sebi_charges=sebi_charges,
            stamp_duty=stamp_duty,
            slippage=slippage,
            total_cost=total_cost,
            net_amount=net_amount,
        )
        # Return as plain dict for JSON-serialisability.
        return {
            "turnover": costs.turnover,
            "brokerage": costs.brokerage,
            "stt": costs.stt,
            "exchange_charges": costs.exchange_charges,
            "gst": costs.gst,
            "sebi_charges": costs.sebi_charges,
            "stamp_duty": costs.stamp_duty,
            "slippage": costs.slippage,
            "total_cost": costs.total_cost,
            "net_amount": costs.net_amount,
        }

    def calculate_net_return(
        self,
        buy_price: float,
        sell_price: float,
        quantity: int,
        slippage_pct: Optional[float] = None,
    ) -> dict:
        """Calculate net return for a complete buy-then-sell round trip.

        Parameters
        ----------
        buy_price : float
            Per-share purchase price in INR.
        sell_price : float
            Per-share sale price in INR.
        quantity : int
            Number of shares traded.
        slippage_pct : float, optional
            Slippage rate for both legs.

        Returns
        -------
        dict
            Keys: ``gross_return``, ``buy_costs``, ``sell_costs``,
            ``total_costs``, ``net_return``, ``net_return_pct``.
        """
        buy_costs = self.calculate_costs(buy_price, quantity, TradeType.BUY, slippage_pct)
        sell_costs = self.calculate_costs(sell_price, quantity, TradeType.SELL, slippage_pct)

        buy_turnover = self._round2(buy_price * quantity)
        sell_turnover = self._round2(sell_price * quantity)
        gross_return = self._round2(sell_turnover - buy_turnover)

        total_costs = self._round2(buy_costs["total_cost"] + sell_costs["total_cost"])
        net_return = self._round2(gross_return - total_costs)

        # Return percentage relative to total capital deployed on the buy side
        # (turnover + buy-side costs).
        capital_deployed = buy_costs["net_amount"]
        net_return_pct = self._round2((net_return / capital_deployed) * 100) if capital_deployed else 0.0

        result = RoundTripReturn(
            buy_turnover=buy_turnover,
            sell_turnover=sell_turnover,
            gross_return=gross_return,
            buy_costs=buy_costs["total_cost"],
            sell_costs=sell_costs["total_cost"],
            total_costs=total_costs,
            net_return=net_return,
            net_return_pct=net_return_pct,
        )
        return {
            "buy_turnover": result.buy_turnover,
            "sell_turnover": result.sell_turnover,
            "gross_return": result.gross_return,
            "buy_costs": result.buy_costs,
            "sell_costs": result.sell_costs,
            "total_costs": result.total_costs,
            "net_return": result.net_return,
            "net_return_pct": result.net_return_pct,
        }

    def estimate_breakeven_sell_price(
        self,
        buy_price: float,
        quantity: int,
        slippage_pct: Optional[float] = None,
    ) -> float:
        """Estimate the minimum sell price needed to break even after all costs.

        Uses iterative refinement because sell-side costs depend on the
        (unknown) sell price.  Convergence is typically reached in fewer
        than 20 iterations for any realistic price.

        Parameters
        ----------
        buy_price : float
            Per-share purchase price in INR.
        quantity : int
            Number of shares.
        slippage_pct : float, optional
            Slippage rate for both legs.

        Returns
        -------
        float
            Break-even per-share sell price rounded to 2 decimal places
            (tick-size precision for Indian equities >= INR 1).
        """
        if buy_price <= 0:
            raise ValueError(f"Buy price must be positive, got {buy_price}")
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        buy_costs = self.calculate_costs(buy_price, quantity, TradeType.BUY, slippage_pct)
        total_buy_outflow = buy_costs["net_amount"]  # turnover + buy costs

        # The sell price must be high enough that:
        #   sell_price * quantity - sell_costs(sell_price) >= total_buy_outflow
        #
        # We solve iteratively: guess sell_price, compute sell costs,
        # derive required sell price, repeat.

        slippage_rate = self._clamp_slippage(slippage_pct)

        # Effective sell-side cost rate (everything that reduces sell proceeds).
        # sell_cost = turnover * (stt_sell + exchange + sebi + slippage)
        #           + (brokerage + exchange_charges) * gst
        # Since brokerage is 0 for Groww delivery, GST = exchange_charges * gst_rate.
        # Combining: effective_rate = stt_sell + exchange*(1+gst) + sebi + slippage
        effective_sell_rate = (
            self.stt_sell_rate
            + self.exchange_charge_rate * (1.0 + self.gst_rate)
            + self.sebi_charge_rate
            + slippage_rate
            + self.brokerage_rate * (1.0 + self.gst_rate)
        )

        # Analytical first estimate: sell_turnover * (1 - rate) = total_buy_outflow
        # => sell_turnover = total_buy_outflow / (1 - rate)
        estimated_sell_turnover = total_buy_outflow / (1.0 - effective_sell_rate)
        candidate = estimated_sell_turnover / quantity

        # Iterative refinement to account for rounding at paisa level.
        for _ in range(30):
            candidate = self._round2(candidate)
            sell_costs = self.calculate_costs(candidate, quantity, TradeType.SELL, slippage_pct)
            net_sell_inflow = sell_costs["net_amount"]  # turnover - sell costs

            gap = total_buy_outflow - net_sell_inflow
            if abs(gap) < 0.01:
                break
            # Adjust candidate proportionally.
            candidate += gap / quantity

        # Final nudge: ensure we truly break even (round up if fractionally short).
        candidate = self._round2(candidate)
        sell_costs = self.calculate_costs(candidate, quantity, TradeType.SELL, slippage_pct)
        if sell_costs["net_amount"] < total_buy_outflow:
            candidate = self._round2(candidate + 0.01)

        return candidate
