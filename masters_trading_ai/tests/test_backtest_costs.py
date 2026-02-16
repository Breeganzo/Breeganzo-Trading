import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.costs import TradeCost


class ZeroCostCalculator:
    def buy_cost(self, value: float, trade_type: str = "equity_delivery") -> TradeCost:
        return TradeCost()

    def round_trip_cost(
        self,
        buy_value: float,
        sell_value: float,
        trade_type: str = "equity_delivery",
    ) -> TradeCost:
        return TradeCost()


def _sample_signals_and_prices():
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])  # Monday, Tuesday
    signals = pd.DataFrame(
        [
            {
                "Date": dates[0],
                "Ticker": "ABC.NS",
                "Signal": 1,
                "Confidence": 0.9,
                "Predicted_Return": 0.03,
            },
            {
                "Date": dates[1],
                "Ticker": "ABC.NS",
                "Signal": -1,
                "Confidence": 0.9,
                "Predicted_Return": 0.00,
            },
        ]
    )
    price_data = {
        "ABC.NS": pd.DataFrame(
            {"Close": [100.0, 110.0]},
            index=dates,
        )
    }
    return signals, price_data


def test_slippage_reduces_pnl():
    signals, price_data = _sample_signals_and_prices()

    no_slip = BacktestEngine(
        slippage_pct=0.0,
        min_hold_days=0,
        rebalance_frequency="daily",
        require_model_agreement=False,
        require_volume_confirmation=False,
        use_atr_stop=False,
    )
    no_slip.cost_calculator = ZeroCostCalculator()
    no_slip_result = no_slip.run(signals, price_data)

    with_slip = BacktestEngine(
        slippage_pct=0.01,
        min_hold_days=0,
        rebalance_frequency="daily",
        require_model_agreement=False,
        require_volume_confirmation=False,
        use_atr_stop=False,
    )
    with_slip.cost_calculator = ZeroCostCalculator()
    with_slip_result = with_slip.run(signals, price_data)

    assert (
        with_slip_result["equity_curve"].iloc[-1]
        < no_slip_result["equity_curve"].iloc[-1]
    )


def test_transaction_costs_reduce_pnl():
    signals, price_data = _sample_signals_and_prices()

    zero_cost = BacktestEngine(
        slippage_pct=0.0,
        min_hold_days=0,
        rebalance_frequency="daily",
        require_model_agreement=False,
        require_volume_confirmation=False,
        use_atr_stop=False,
    )
    zero_cost.cost_calculator = ZeroCostCalculator()
    zero_result = zero_cost.run(signals, price_data)

    with_cost = BacktestEngine(
        slippage_pct=0.0,
        min_hold_days=0,
        rebalance_frequency="daily",
        require_model_agreement=False,
        require_volume_confirmation=False,
        use_atr_stop=False,
    )
    with_cost_result = with_cost.run(signals, price_data)

    assert with_cost_result["cost_analysis"]["Total Costs (₹)"] > 0
    assert (
        with_cost_result["equity_curve"].iloc[-1] < zero_result["equity_curve"].iloc[-1]
    )
