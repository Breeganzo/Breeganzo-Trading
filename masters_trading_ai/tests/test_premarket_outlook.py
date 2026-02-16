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
        "strategy_predicted_at_open",
        "ai_predicted_at_open",
    ):
        assert key in item
    assert item["ticker"] == "ABC.NS"
    assert item["current_price"] > 0
    assert item["strategy_price_at_open"] > 0


def test_api_price_tracker_includes_times_and_close_label(tmp_path, monkeypatch):
    class StubPredictor:
        def predict_single(self, ticker, use_cache=True):
            return {"ticker": ticker}

    monkeypatch.setattr(server, "predictor", StubPredictor())
    monkeypatch.setattr(server, "models_loaded", True)
    monkeypatch.setattr(
        server,
        "_price_cache",
        {
            "ABC.NS": {
                "price": 202.0,
                "open": 198.0,
                "prev_close": 197.0,
                "high": 204.0,
                "low": 196.0,
                "volume": 12345,
                "change": 5.0,
                "change_pct": 2.54,
            }
        },
    )
    monkeypatch.setattr(
        server,
        "_opening_prices",
        {
            "ABC.NS": {
                "open": 198.0,
                "prev_close": 197.0,
                "captured_at": "2026-02-16T09:15:05+05:30",
            }
        },
    )
    monkeypatch.setattr(
        server,
        "_premarket_snapshot",
        {
            "date": datetime.now(server.IST).strftime("%Y-%m-%d"),
            "items": [
                {
                    "ticker": "ABC.NS",
                    "strategy_price_at_open": 201.96,
                    "ai_predicted_price": 203.4,
                    "strategy_predicted_at_open": "2026-02-16T09:00:00+05:30",
                    "ai_predicted_at_open": "2026-02-16T09:00:00+05:30",
                    "captured_at": "2026-02-16T09:00:00+05:30",
                }
            ],
        },
    )
    monkeypatch.setattr(
        server,
        "get_market_status",
        lambda: {"status": "after_hours", "description": "", "next_open": "", "ist_now": datetime.now(server.IST)},
    )
    monkeypatch.setattr(server, "ticker_names", {"ABC.NS": "ABC"})

    logs_dir = tmp_path / "prediction_log"
    logs_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(server.IST).strftime("%Y-%m-%d")
    (logs_dir / f"{today}.json").write_text(
        '{"ABC.NS":{"predicted_return":2.0,"predicted_price":204.0,"current_price":202.0,"signal":"BUY","confidence":70.0,"timestamp":"2026-02-16T10:15:00+05:30"}}'
    )
    monkeypatch.setattr(server, "PREDICTION_LOG_DIR", logs_dir)

    with server.app.test_client() as client:
        resp = client.get("/api/price-tracker/ABC.NS")
        assert resp.status_code == 200
        payload = resp.get_json()

    assert payload["display_price_label"] == "Close Price"
    assert payload["strategy_predicted_at_open"] is not None
    assert payload["current_strategy_predicted_at"] is not None
    assert payload["current_ai_predicted_at"] is not None
    assert payload["close_price"] == 202.0
    strategy_ts = datetime.fromisoformat(payload["strategy_predicted_at_open"])
    ai_ts = datetime.fromisoformat(payload["ai_predicted_at_open"])
    assert strategy_ts.hour == 9 and 15 <= strategy_ts.minute <= 30
    assert ai_ts.hour == 9 and 15 <= ai_ts.minute <= 30


def test_normalize_premarket_snapshot_legacy_fields():
    legacy = {
        "date": "2026-02-16",
        "captured_at": "2026-02-16T16:22:14+05:30",
        "items": [
            {
                "ticker": "INFY.NS",
                "current_price": 1365.6,
                "strategy_price_at_open": 1380.5,
                "ai_predicted_price": None,
                "strategy_direction": "UP",
            }
        ],
    }

    out = server._normalize_premarket_snapshot(legacy)
    assert out["snapshot_type"] in {"market_open_live", "market_open_backfilled"}
    assert out["capture_note"]
    cap = datetime.fromisoformat(out["captured_at"])
    assert cap.hour == 9 and 15 <= cap.minute <= 30

    row = out["items"][0]
    assert row["strategy_source"] == "ensemble_models"
    assert row["ai_source"] == "none"
    assert row["ai_direction"] == "N/A"
