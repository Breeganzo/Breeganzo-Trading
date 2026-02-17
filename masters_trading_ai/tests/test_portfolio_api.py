from pathlib import Path
from datetime import datetime

from webapp import server


def test_portfolio_add_merge_and_fetch(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "PORTFOLIO_FILE", tmp_path / "portfolio.json")
    monkeypatch.setattr(
        server, "PORTFOLIO_TRADES_FILE", tmp_path / "portfolio_trades.json"
    )
    monkeypatch.setattr(server, "ticker_names", {"ABC.NS": "ABC"})

    with server.app.test_client() as client:
        r1 = client.post(
            "/api/portfolio",
            json={"ticker": "ABC.NS", "quantity": 2, "entry_price": 100},
        )
        assert r1.status_code == 200
        p1 = r1.get_json()
        assert p1["count"] == 1
        assert p1["holdings"][0]["quantity"] == 2
        assert p1["holdings"][0]["avg_buy_price"] == 100

        # Merge with weighted average entry
        r2 = client.post(
            "/api/portfolio",
            json={"ticker": "ABC.NS", "quantity": 1, "entry_price": 130},
        )
        assert r2.status_code == 200
        p2 = r2.get_json()
        assert p2["count"] == 1
        assert p2["holdings"][0]["quantity"] == 3
        assert p2["holdings"][0]["avg_buy_price"] == 110.0

        r3 = client.get("/api/portfolio?ticker=ABC.NS")
        assert r3.status_code == 200
        p3 = r3.get_json()
        assert p3["count"] == 1
        assert p3["holdings"][0]["ticker"] == "ABC.NS"


def test_portfolio_delete(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "PORTFOLIO_FILE", tmp_path / "portfolio.json")
    monkeypatch.setattr(
        server, "PORTFOLIO_TRADES_FILE", tmp_path / "portfolio_trades.json"
    )

    with server.app.test_client() as client:
        client.post(
            "/api/portfolio", json={"ticker": "A.NS", "quantity": 1, "entry_price": 10}
        )
        client.post(
            "/api/portfolio", json={"ticker": "B.NS", "quantity": 1, "entry_price": 20}
        )
        out = client.delete("/api/portfolio?ticker=A.NS")
        assert out.status_code == 200
        payload = out.get_json()
        tickers = {h["ticker"] for h in payload["holdings"]}
        assert tickers == {"B.NS"}


def test_portfolio_avg_buy_only_for_holdings_and_price_tracker_fields(monkeypatch):
    monkeypatch.setattr(server, "models_loaded", True)
    monkeypatch.setattr(server, "ticker_names", {"ABC.NS": "ABC"})

    class _StubPredictor:
        def predict_single(self, ticker, use_cache=True):
            return {
                "ticker": ticker,
                "current_price": 100.0,
                "predicted_return": 1.5,
                "predicted_price": 101.5,
                "signal": "BUY",
                "confidence": 75.0,
                "timestamp": datetime.now(server.IST).isoformat(),
            }

    monkeypatch.setattr(server, "predictor", _StubPredictor())
    monkeypatch.setattr(
        server,
        "_get_live_prices_batch",
        lambda tickers: {
            "ABC.NS": {
                "price": 100.0,
                "open": 99.0,
                "prev_close": 98.5,
                "high": 101.0,
                "low": 98.7,
                "volume": 100000,
                "change": 1.0,
                "change_pct": 1.01,
            }
        },
    )
    monkeypatch.setattr(
        server,
        "_get_premarket_row_for_ticker",
        lambda ticker, date_str=None: {
            "strategy_price_at_open": 101.2,
            "ai_predicted_price": 101.5,
            "strategy_predicted_at_open": datetime.now(server.IST).isoformat(),
            "ai_predicted_at_open": datetime.now(server.IST).isoformat(),
            "captured_at": datetime.now(server.IST).isoformat(),
            "ai_source": "none",
        },
    )
    monkeypatch.setattr(
        server,
        "_resolve_ai_forecast_price",
        lambda *args, **kwargs: {
            "available": False,
            "price": 0.0,
            "source": "none",
            "generated_at_iso": datetime.now(server.IST).isoformat(),
        },
    )
    monkeypatch.setattr(
        server,
        "_read_portfolio_sim_state",
        lambda: {
            "open_positions": {
                "ABC.NS": {
                    "entry_range_low": 99.8,
                    "entry_range_high": 100.2,
                }
            }
        },
    )
    monkeypatch.setattr(
        server, "_prediction_window_type", lambda *_: "market_open_locked"
    )
    monkeypatch.setattr(server, "_is_next_day_prediction_window", lambda *_: False)
    monkeypatch.setattr(server, "get_market_status", lambda: {"status": "market_open"})

    with server.app.test_client() as client:
        tracker = client.get("/api/price-tracker/ABC.NS")
        assert tracker.status_code == 200
        payload = tracker.get_json()
        assert payload["market_open_price"] > 0
        assert payload["strategy_price_at_open"] > 0
        assert payload["entry_range_low"] > 0
        assert payload["entry_range_high"] > payload["entry_range_low"]
        assert payload["market_open_price"] != 0
        assert payload["strategy_price_at_open"] != 0
