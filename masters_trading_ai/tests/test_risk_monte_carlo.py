import numpy as np
import pandas as pd

from src.backtest.metrics import monte_carlo_backtest
from webapp import server


def test_monte_carlo_uses_equity_curve_source():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.001, 0.01, size=80))
    equity_curve = 100000 * (1 + returns).cumprod()

    result = monte_carlo_backtest(
        returns,
        n_simulations=200,
        equity_curve=equity_curve,
        min_history=30,
    )

    assert "error" not in result
    assert result["simulation_source"] == "equity_curve"
    assert result["n_simulations"] == 200
    assert result["terminal_wealth"]["p5"] <= result["terminal_wealth"]["p95"]


def test_monte_carlo_rejects_short_history():
    returns = pd.Series([0.01, -0.02, 0.005])
    equity_curve = pd.Series([100000.0, 101000.0, 98980.0])

    result = monte_carlo_backtest(
        returns,
        equity_curve=equity_curve,
        min_history=10,
    )

    assert "error" in result


def test_api_risk_analytics_ignores_non_portfolio_tickers(monkeypatch):
    monkeypatch.setattr(server, "_sanitize_portfolio_storage", lambda: {})
    monkeypatch.setattr(
        server,
        "_read_portfolio",
        lambda: [
            {
                "ticker": "ABC.NS",
                "name": "ABC",
                "quantity": 10,
                "entry_price": 100.0,
            }
        ],
    )
    monkeypatch.setattr(server, "tickers_by_sector", {"large_cap": ["ABC.NS"]})

    dates = pd.date_range("2025-10-01", periods=45, freq="B")
    close = pd.DataFrame(
        {
            "ABC.NS": np.linspace(100, 112, len(dates)),
            "^NSEI": np.linspace(20000, 20500, len(dates)),
        },
        index=dates,
    )
    fake_download_df = pd.concat({"Close": close}, axis=1)

    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: fake_download_df)

    with server.app.test_client() as client:
        resp = client.get("/api/risk-analytics?tickers=ABC.NS,XYZ.NS")
        assert resp.status_code == 200
        payload = resp.get_json()

    assert payload["portfolio_tickers"] == ["ABC.NS"]
    assert "XYZ.NS" in payload["ignored_tickers"]
    assert payload["monte_carlo"]["simulation_source"] == "equity_curve"
