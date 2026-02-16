from datetime import datetime

from webapp import server


class _StubPredictor:
    def predict_single(self, ticker, use_cache=True):
        return {
            "ticker": ticker,
            "predicted_return": 2.0,
            "predicted_price": 204.0,
            "current_price": 200.0,
            "signal": "BUY",
            "confidence": 66.0,
        }


def _patch_now(monkeypatch, fixed_now: datetime):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    monkeypatch.setattr(server, "datetime", FixedDateTime)


def test_capture_market_open_snapshot_marks_market_open(tmp_path, monkeypatch):
    fixed_now = datetime(2026, 2, 16, 9, 18, tzinfo=server.IST)
    _patch_now(monkeypatch, fixed_now)

    monkeypatch.setattr(server, "predictor", _StubPredictor())
    monkeypatch.setattr(server, "models_loaded", True)
    monkeypatch.setattr(server, "tickers_by_sector", {"large_cap": ["ABC.NS"]})
    monkeypatch.setattr(server, "ticker_names", {"ABC.NS": "ABC"})
    monkeypatch.setattr(server, "PREMARKET_DEFAULT_TICKERS", 1)
    monkeypatch.setattr(server, "_premarket_snapshot", {})
    monkeypatch.setattr(
        server,
        "PREMARKET_OUTLOOK_FILE",
        tmp_path / "premarket_outlook.json",
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

    seen_snapshot_types = []

    def _record(_ticker, _payload, snapshot_type=None):
        seen_snapshot_types.append(snapshot_type)

    monkeypatch.setattr(
        server.PredictionTracker,
        "record_prediction",
        staticmethod(_record),
    )

    snapshot = server._capture_premarket_snapshot_if_due(force=True)
    assert snapshot["snapshot_type"] == "market_open"
    assert snapshot["items"]

    row = snapshot["items"][0]
    assert row["snapshot_type"] == "market_open"
    ts = datetime.fromisoformat(row["strategy_predicted_at_open"])
    assert ts.hour == 9 and 15 <= ts.minute <= 30
    assert row["strategy_price_at_open"] > 0
    assert seen_snapshot_types and seen_snapshot_types[0] == "market_open"


def test_capture_snapshot_before_open_returns_pending(tmp_path, monkeypatch):
    fixed_now = datetime(2026, 2, 16, 8, 40, tzinfo=server.IST)
    _patch_now(monkeypatch, fixed_now)

    monkeypatch.setattr(server, "_premarket_snapshot", {})
    monkeypatch.setattr(
        server,
        "PREMARKET_OUTLOOK_FILE",
        tmp_path / "premarket_outlook.json",
    )

    snapshot = server._capture_premarket_snapshot_if_due(force=False)
    assert snapshot["snapshot_type"] == "pending_market_open"
    assert snapshot["items"] == []
