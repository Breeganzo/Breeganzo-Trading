from datetime import datetime

from webapp import server


def _today_str() -> str:
    return datetime.now(server.IST).strftime("%Y-%m-%d")


def test_market_open_snapshot_builds_locked_from_open(monkeypatch):
    today = _today_str()
    disk_store = {
        "date": today,
        "snapshots": {
            "premarket_open": {
                "snapshot_type": "premarket_open",
                "captured_at": f"{today}T09:15:00+05:30",
                "items": [{"ticker": "AAA.NS", "strategy_price_at_open": 100.0}],
            }
        },
    }

    built_calls = []

    def _build(snapshot_type: str, force_live: bool = False):
        built_calls.append((snapshot_type, force_live))
        return {
            "snapshot_type": snapshot_type,
            "captured_at": f"{today}T09:30:00+05:30",
            "items": [{"ticker": "AAA.NS", "strategy_price_at_open": 101.5}],
        }

    monkeypatch.setattr(server, "_prediction_window_type", lambda *_: "market_open_locked")
    monkeypatch.setattr(server, "_load_prediction_snapshots_from_disk", lambda: disk_store)
    monkeypatch.setattr(server, "_save_prediction_snapshots_to_disk", lambda payload: None)
    monkeypatch.setattr(server, "_build_prediction_snapshot", _build)

    out = server._get_prediction_snapshot(force=False, use_latest_stored=False)
    assert out["snapshot_type"] == "market_open_locked"
    assert built_calls and built_calls[0][0] == "market_open_locked"


def test_premarket_snapshot_remains_frozen_without_force(monkeypatch):
    today = _today_str()
    frozen = {
        "date": today,
        "snapshots": {
            "premarket_open": {
                "snapshot_type": "premarket_open",
                "captured_at": f"{today}T09:15:00+05:30",
                "items": [{"ticker": "AAA.NS", "strategy_price_at_open": 100.0}],
            }
        },
    }

    monkeypatch.setattr(server, "_prediction_window_type", lambda *_: "premarket_open")
    monkeypatch.setattr(server, "_load_prediction_snapshots_from_disk", lambda: frozen)
    monkeypatch.setattr(server, "_save_prediction_snapshots_to_disk", lambda payload: None)

    def _should_not_build(*args, **kwargs):
        raise AssertionError("premarket snapshot should not rebuild when frozen")

    monkeypatch.setattr(server, "_build_prediction_snapshot", _should_not_build)

    out = server._get_prediction_snapshot(force=False, use_latest_stored=False)
    assert out["snapshot_type"] == "premarket_open"
    assert out["captured_at"].startswith(f"{today}T09:15")
