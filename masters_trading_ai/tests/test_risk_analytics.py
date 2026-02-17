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


def test_api_risk_analytics_sector_exposure_not_over_100(monkeypatch):
    import yfinance as yf

    def _mock_download_multi(*args, **kwargs):
        idx = pd.date_range("2025-11-01", periods=45, freq="D")
        close = pd.DataFrame(
            {
                "AAA.NS": np.linspace(100.0, 120.0, len(idx)),
                "BBB.NS": np.linspace(200.0, 210.0, len(idx)),
                "^NSEI": np.linspace(19850.0, 20120.0, len(idx)),
            },
            index=idx,
        )
        return pd.concat({"Close": close}, axis=1)

    monkeypatch.setattr(
        server, "_sanitize_portfolio_storage", lambda: {"changed": False}
    )
    monkeypatch.setattr(
        server,
        "_read_portfolio",
        lambda: [
            {"ticker": "AAA.NS", "name": "AAA", "quantity": 2, "entry_price": 100.0},
            {"ticker": "BBB.NS", "name": "BBB", "quantity": 1, "entry_price": 200.0},
        ],
    )
    monkeypatch.setattr(
        server, "_read_portfolio_sim_state", lambda: {"open_positions": {}}
    )
    monkeypatch.setattr(
        server,
        "tickers_by_sector",
        {
            "large_cap": ["AAA.NS", "BBB.NS"],
            "banking": ["AAA.NS"],
            "commodities": ["BBB.NS"],
        },
    )
    monkeypatch.setattr(
        server,
        "ticker_to_sector",
        {"AAA.NS": "banking", "BBB.NS": "commodities"},
    )
    monkeypatch.setattr(yf, "download", _mock_download_multi)

    with server.app.test_client() as client:
        resp = client.get("/api/risk-analytics")
        assert resp.status_code == 200
        payload = resp.get_json()

    sector = payload.get("sector_exposure", {})
    total = sum(float(v.get("weight_pct", 0) or 0) for v in sector.values())
    assert 99.0 <= total <= 101.0
    assert all(float(v.get("weight_pct", 0) or 0) <= 100.0 for v in sector.values())


def test_api_risk_analytics_uses_sim_positions_when_manual_empty(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(
        server, "_sanitize_portfolio_storage", lambda: {"changed": False}
    )
    monkeypatch.setattr(server, "_read_portfolio", lambda: [])
    monkeypatch.setattr(
        server,
        "_read_portfolio_sim_state",
        lambda: {
            "open_positions": {
                "AAA.NS": {
                    "quantity": 3,
                    "avg_entry_price": 150.0,
                }
            }
        },
    )
    monkeypatch.setattr(server, "tickers_by_sector", {"large_cap": ["AAA.NS"]})
    monkeypatch.setattr(server, "ticker_to_sector", {"AAA.NS": "large_cap"})
    monkeypatch.setattr(yf, "download", _mock_download)

    with server.app.test_client() as client:
        resp = client.get("/api/risk-analytics")
        assert resp.status_code == 200
        payload = resp.get_json()

    assert payload.get("portfolio_source") == "simulation"
    assert payload.get("portfolio_tickers") == ["AAA.NS"]


def test_api_risk_analytics_recovers_missing_batch_ticker_history(monkeypatch):
    idx = pd.date_range("2025-11-01", periods=45, freq="D")

    def _safe_download(*args, **kwargs):
        ticker_arg = str(args[0])
        if ticker_arg.strip() == "BBB.NS":
            single = pd.DataFrame(
                {
                    "Close": np.linspace(210.0, 220.0, len(idx)),
                },
                index=idx,
            )
            return single
        close = pd.DataFrame(
            {
                "AAA.NS": np.linspace(100.0, 115.0, len(idx)),
                "^NSEI": np.linspace(19850.0, 20120.0, len(idx)),
            },
            index=idx,
        )
        return pd.concat({"Close": close}, axis=1)

    monkeypatch.setattr(
        server, "_sanitize_portfolio_storage", lambda: {"changed": False}
    )
    monkeypatch.setattr(
        server,
        "_read_portfolio",
        lambda: [
            {"ticker": "AAA.NS", "name": "AAA", "quantity": 2, "entry_price": 100.0},
            {"ticker": "BBB.NS", "name": "BBB", "quantity": 1, "entry_price": 210.0},
        ],
    )
    monkeypatch.setattr(
        server, "_read_portfolio_sim_state", lambda: {"open_positions": {}}
    )
    monkeypatch.setattr(server, "_safe_yf_download", _safe_download)
    monkeypatch.setattr(
        server,
        "ticker_to_sector",
        {"AAA.NS": "large_cap", "BBB.NS": "banking"},
    )
    monkeypatch.setattr(
        server,
        "tickers_by_sector",
        {"large_cap": ["AAA.NS"], "banking": ["BBB.NS"]},
    )

    with server.app.test_client() as client:
        resp = client.get("/api/risk-analytics")
        assert resp.status_code == 200
        payload = resp.get_json()

    assert set(payload.get("portfolio_tickers", [])) == {"AAA.NS", "BBB.NS"}
