from webapp import server


class _StubCache:
    def __init__(self, entry=None):
        self._entry = entry or {}

    def get(self, _ticker):
        return self._entry

    def invalidate(self, _ticker=None):
        return None


def test_api_strategy_price_returns_strategy_fields(monkeypatch):
    class StubPredictor:
        def __init__(self):
            self.cache = _StubCache()

        def get_strategy_price(self, ticker, use_cache=True):
            return {
                "ticker": ticker,
                "strategy_price": 123.45,
                "rr_ratio": 1.8,
                "strategy_generated_at": "2026-02-16T09:30:00+05:30",
                "predicted_return_decimal": 0.012,
                "source": "strategy_engine",
                "open_price": 122.0,
                "current_price": 122.5,
            }

    monkeypatch.setattr(server, "predictor", StubPredictor())
    monkeypatch.setattr(server, "models_loaded", True)
    monkeypatch.setattr(
        server, "_prediction_window_type", lambda *_: "after_hours_live"
    )

    with server.app.test_client() as client:
        resp = client.get("/api/strategy-price/ABC.NS")
        assert resp.status_code == 200
        payload = resp.get_json()

    assert payload["ticker"] == "ABC.NS"
    assert payload["strategy_price"] == 123.45
    assert payload["rr_ratio"] == 1.8
    assert payload["source"] == "strategy_engine"
    assert payload["snapshot_type"] == "after_hours_live"


def test_api_debug_prediction_status_shape(monkeypatch):
    cache_entry = {
        "current_price": 100.0,
        "predicted_return": 2.0,
        "predicted_price": 102.0,
    }

    class StubPredictor:
        def __init__(self):
            self.cache = _StubCache(cache_entry)

    monkeypatch.setattr(server, "predictor", StubPredictor())
    monkeypatch.setattr(
        server, "_prediction_window_type", lambda *_: "market_open_locked"
    )
    monkeypatch.setattr(
        server,
        "_get_prediction_snapshot",
        lambda **_: {
            "snapshot_type": "market_open_locked",
            "captured_at": "2026-02-16T09:30:00+05:30",
            "items": [{"ticker": "ABC.NS", "strategy_price_at_open": 101.5}],
        },
    )
    monkeypatch.setattr(
        server,
        "_get_live_prices_batch",
        lambda tickers: {tickers[0]: {"price": 101.0, "open": 100.0}},
    )
    monkeypatch.setattr(
        server,
        "get_market_status",
        lambda: {"status": "market_open"},
    )

    with server.app.test_client() as client:
        resp = client.get("/api/debug/prediction-status/ABC.NS")
        assert resp.status_code == 200
        payload = resp.get_json()

    assert payload["ticker"] == "ABC.NS"
    assert payload["cache_hit"] is True
    assert payload["prediction_window"] == "market_open_locked"
    assert payload["snapshot_type"] == "market_open_locked"
    assert payload["predicted_price_formula_ok"] is True
