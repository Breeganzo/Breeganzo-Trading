import numpy as np
import pandas as pd

from webapp import server


def _mock_download(*args, **kwargs):
    idx = pd.date_range("2025-11-01", periods=40, freq="D")
    close = pd.DataFrame(
        {
            "AAA.NS": np.linspace(100.0, 114.0, len(idx)),
            "^NSEI": np.linspace(19850.0, 20120.0, len(idx)),
        },
        index=idx,
    )
    return pd.concat({"Close": close}, axis=1)


def test_api_risk_analytics_uses_portfolio_initial_capital(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(
        server, "_sanitize_portfolio_storage", lambda: {"changed": False}
    )
    monkeypatch.setattr(
        server,
        "_read_portfolio",
        lambda: [
            {
                "ticker": "AAA.NS",
                "name": "AAA",
                "quantity": 10,
                "entry_price": 123.0,
            }
        ],
    )
    monkeypatch.setattr(server, "tickers_by_sector", {"large_cap": ["AAA.NS"]})
    monkeypatch.setattr(yf, "download", _mock_download)

    with server.app.test_client() as client:
        resp = client.get("/api/risk-analytics")
        assert resp.status_code == 200
        payload = resp.get_json()

    assert payload["portfolio_tickers"] == ["AAA.NS"]
    assert payload["initial_capital"] == 1230.0
    assert payload["monte_carlo"]["initial_capital"] == 1230.0
    assert payload["equity_curve"]


def test_api_risk_analytics_rejects_non_portfolio_tickers(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(
        server, "_sanitize_portfolio_storage", lambda: {"changed": False}
    )
    monkeypatch.setattr(
        server,
        "_read_portfolio",
        lambda: [
            {"ticker": "AAA.NS", "name": "AAA", "quantity": 2, "entry_price": 100.0}
        ],
    )
    monkeypatch.setattr(server, "tickers_by_sector", {"large_cap": ["AAA.NS"]})
    monkeypatch.setattr(yf, "download", _mock_download)

    with server.app.test_client() as client:
        resp = client.get("/api/risk-analytics?tickers=ZZZ.NS")
        assert resp.status_code == 400
        payload = resp.get_json()

    assert "not in portfolio" in payload["error"]


def test_api_risk_analytics_returns_graceful_payload_when_history_missing(monkeypatch):
    import pandas as pd
    import yfinance as yf

    monkeypatch.setattr(
        server, "_sanitize_portfolio_storage", lambda: {"changed": False}
    )
    monkeypatch.setattr(
        server,
        "_read_portfolio",
        lambda: [
            {"ticker": "AAA.NS", "name": "AAA", "quantity": 2, "entry_price": 100.0}
        ],
    )
    monkeypatch.setattr(server, "tickers_by_sector", {"large_cap": ["AAA.NS"]})
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: pd.DataFrame())

    with server.app.test_client() as client:
        resp = client.get("/api/risk-analytics")
        assert resp.status_code == 200
        payload = resp.get_json()

    assert payload["portfolio_tickers"] == ["AAA.NS"]
    assert payload["warning"] == "Unable to fetch historical data"
    assert payload["monte_carlo"]["error"] == "Insufficient data for analytics"
