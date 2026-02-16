from datetime import datetime

from webapp import server


def test_premarket_cutoff_uses_buffer(monkeypatch):
    monkeypatch.setattr(server, "PREMARKET_MAX_BUFFER_MINUTES", 30)
    now = datetime(2026, 2, 16, 8, 0, tzinfo=server.IST)
    cutoff = server._premarket_cutoff_dt(now)
    assert cutoff.hour == 8
    assert cutoff.minute == 45


def test_api_premarket_outlook_shape(tmp_path, monkeypatch):
    class StubPredictor:
        def predict_single(self, ticker, use_cache=True):
            return {
                "ticker": ticker,
                "predicted_return": 2.0,
                "predicted_price": 204.0,
                "current_price": 200.0,
                "signal": "BUY",
                "confidence": 66.0,
            }

    monkeypatch.setattr(server, "predictor", StubPredictor())
    monkeypatch.setattr(server, "models_loaded", True)
    monkeypatch.setattr(server, "tickers_by_sector", {"large_cap": ["ABC.NS"]})
    monkeypatch.setattr(server, "ticker_names", {"ABC.NS": "ABC"})
    monkeypatch.setattr(server, "PREMARKET_DEFAULT_TICKERS", 1)
    monkeypatch.setattr(server, "_premarket_snapshot", {})
    monkeypatch.setattr(
        server, "PREMARKET_OUTLOOK_FILE", tmp_path / "premarket_outlook.json"
    )
    monkeypatch.setattr(
        server,
        "_get_live_prices_batch",
        lambda tickers: {
            "ABC.NS": {
                "price": 200.0,
                "open": 198.0,
                "prev_close": 197.0,
            }
        },
    )
    monkeypatch.setattr(
        server.PredictionTracker,
        "record_prediction",
        staticmethod(lambda *args, **kwargs: None),
    )

    with server.app.test_client() as client:
        resp = client.get("/api/premarket-outlook?force=true")
        assert resp.status_code == 200
        payload = resp.get_json()

    assert "items" in payload
    assert payload["items"]
    item = payload["items"][0]
    for key in (
        "ticker",
        "current_price",
        "strategy_price_at_open",
        "ai_predicted_price",
        "strategy_direction",
        "ai_direction",
        "captured_at",
    ):
        assert key in item
    assert item["ticker"] == "ABC.NS"
    assert item["current_price"] > 0
    assert item["strategy_price_at_open"] > 0
