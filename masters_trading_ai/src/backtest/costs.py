"""
Groww Transaction Cost Calculator
==================================
Models all transaction costs for trades executed through Groww:
brokerage, STT, exchange fees, GST, stamp duty, and DP charges.

Source: Groww pricing page + SEBI true-to-label regulations (2024-25)
"""

import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ..utils.constants import CONFIG_DIR


@dataclass
class TradeCost:
    """Breakdown of costs for a single trade (buy or sell side)."""
    brokerage: float = 0.0
    stt: float = 0.0
    transaction_charges: float = 0.0
    gst: float = 0.0
    sebi_fee: float = 0.0
    stamp_duty: float = 0.0
    dp_charges: float = 0.0

    @property
    def total(self) -> float:
        return (self.brokerage + self.stt + self.transaction_charges +
                self.gst + self.sebi_fee + self.stamp_duty + self.dp_charges)

    def __repr__(self) -> str:
        return (
            f"TradeCost(brokerage=₹{self.brokerage:.2f}, stt=₹{self.stt:.2f}, "
            f"txn=₹{self.transaction_charges:.2f}, gst=₹{self.gst:.2f}, "
            f"sebi=₹{self.sebi_fee:.2f}, stamp=₹{self.stamp_duty:.2f}, "
            f"dp=₹{self.dp_charges:.2f}) = ₹{self.total:.2f}"
        )


class GrowwCostCalculator:
    """
    Calculate transaction costs for trades on Groww.

    Supports: equity_delivery, equity_intraday, futures, options

    Usage
    -----
    >>> calc = GrowwCostCalculator()
    >>> cost = calc.round_trip_cost(buy_value=50000, sell_value=52000, trade_type="equity_delivery")
    >>> print(f"Total cost: ₹{cost.total:.2f}")
    >>> print(f"Cost as % of trade: {cost.total / 50000 * 100:.3f}%")
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_DIR / "groww_fees.yaml"
        with open(self.config_path, "r") as f:
            self.fees = yaml.safe_load(f)

    def buy_cost(self, value: float, trade_type: str = "equity_delivery") -> TradeCost:
        """
        Calculate costs for the BUY side of a trade.

        Parameters
        ----------
        value : float
            Total buy value in ₹
        trade_type : str
            One of: equity_delivery, equity_intraday, futures, options
        """
        fee = self.fees[trade_type]
        cost = TradeCost()

        # Brokerage: min(₹20, 0.05% of value)
        if "brokerage_pct" in fee:
            cost.brokerage = min(fee["brokerage_per_order"],
                                 value * fee["brokerage_pct"])
        else:
            cost.brokerage = fee["brokerage_per_order"]

        # STT on buy (only for equity delivery)
        if "stt_buy_pct" in fee:
            cost.stt = value * fee["stt_buy_pct"]

        # Exchange transaction charges
        cost.transaction_charges = value * fee["nse_transaction_pct"]

        # GST: 18% on (brokerage + transaction charges)
        cost.gst = fee["gst_pct"] * (cost.brokerage + cost.transaction_charges)

        # SEBI turnover fee
        cost.sebi_fee = value * fee["sebi_fee_per_crore"] / 1e7

        # Stamp duty (buy side)
        if "stamp_duty_buy_pct" in fee:
            cost.stamp_duty = value * fee["stamp_duty_buy_pct"]

        return cost

    def sell_cost(self, value: float, trade_type: str = "equity_delivery") -> TradeCost:
        """
        Calculate costs for the SELL side of a trade.

        Parameters
        ----------
        value : float
            Total sell value in ₹
        trade_type : str
            One of: equity_delivery, equity_intraday, futures, options
        """
        fee = self.fees[trade_type]
        cost = TradeCost()

        # Brokerage
        if "brokerage_pct" in fee:
            cost.brokerage = min(fee["brokerage_per_order"],
                                 value * fee["brokerage_pct"])
        else:
            cost.brokerage = fee["brokerage_per_order"]

        # STT on sell
        if "stt_sell_pct" in fee:
            cost.stt = value * fee["stt_sell_pct"]

        # Exchange transaction charges
        cost.transaction_charges = value * fee["nse_transaction_pct"]

        # GST
        cost.gst = fee["gst_pct"] * (cost.brokerage + cost.transaction_charges)

        # SEBI fee
        cost.sebi_fee = value * fee["sebi_fee_per_crore"] / 1e7

        # DP charges (equity delivery sell only)
        if trade_type == "equity_delivery":
            cost.dp_charges = fee.get("dp_charges_per_scrip", 15.93)

        return cost

    def round_trip_cost(
        self,
        buy_value: float,
        sell_value: float,
        trade_type: str = "equity_delivery",
    ) -> TradeCost:
        """
        Calculate total round-trip cost (buy + sell).

        Parameters
        ----------
        buy_value : float
            Total buy value in ₹
        sell_value : float
            Total sell value in ₹
        trade_type : str
            Trade type

        Returns
        -------
        TradeCost
            Combined buy + sell costs
        """
        buy = self.buy_cost(buy_value, trade_type)
        sell = self.sell_cost(sell_value, trade_type)

        return TradeCost(
            brokerage=buy.brokerage + sell.brokerage,
            stt=buy.stt + sell.stt,
            transaction_charges=buy.transaction_charges + sell.transaction_charges,
            gst=buy.gst + sell.gst,
            sebi_fee=buy.sebi_fee + sell.sebi_fee,
            stamp_duty=buy.stamp_duty + sell.stamp_duty,
            dp_charges=buy.dp_charges + sell.dp_charges,
        )

    def cost_pct(self, buy_value: float, sell_value: float,
                 trade_type: str = "equity_delivery") -> float:
        """Return round-trip cost as percentage of buy value."""
        cost = self.round_trip_cost(buy_value, sell_value, trade_type)
        return cost.total / buy_value if buy_value > 0 else 0.0

    def summary_table(self, trade_value: float = 50000) -> dict:
        """
        Generate a summary of costs for each trade type at a given value.

        Parameters
        ----------
        trade_value : float
            Example trade value in ₹

        Returns
        -------
        dict
            {trade_type: {total_cost, cost_pct, breakdown}}
        """
        summary = {}
        for trade_type in ["equity_delivery", "equity_intraday", "futures", "options"]:
            try:
                cost = self.round_trip_cost(trade_value, trade_value, trade_type)
                summary[trade_type] = {
                    "Total Cost (₹)": f"₹{cost.total:.2f}",
                    "Cost %": f"{cost.total / trade_value * 100:.3f}%",
                    "Brokerage": f"₹{cost.brokerage:.2f}",
                    "STT": f"₹{cost.stt:.2f}",
                    "GST": f"₹{cost.gst:.2f}",
                    "Stamp Duty": f"₹{cost.stamp_duty:.2f}",
                    "DP Charges": f"₹{cost.dp_charges:.2f}",
                }
            except KeyError:
                pass
        return summary
