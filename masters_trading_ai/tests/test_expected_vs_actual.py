import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from webapp import prediction_tracker as tracker_module
from webapp.prediction_tracker import PredictionTracker
from webapp import server


def _setup_tracker_paths(tmp_path: Path, monkeypatch):
    tracking_dir = tmp_path / "tracking"
    daily_dir = tracking_dir / "daily"
    monthly_dir = tracking_dir / "monthly"
    accuracy_file = tracking_dir / "accuracy_history.json"
    daily_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(tracker_module, "TRACKING_DIR", tracking_dir)
    monkeypatch.setattr(tracker_module, "DAILY_DIR", daily_dir)
    monkeypatch.setattr(tracker_module, "MONTHLY_DIR", monthly_dir)
    monkeypatch.setattr(tracker_module, "ACCURACY_FILE", accuracy_file)
    return daily_dir


def test_prediction_tracker_persists_direction_fields(tmp_path: Path, monkeypatch):
    daily_dir = _setup_tracker_paths(tmp_path, monkeypatch)

    PredictionTracker.record_prediction(
        "ABC.NS",
        {
            "predicted_return": 2.0,
            "predicted_price": 104.0,
            "current_price": 100.0,
            "open_price": 99.0,
            "strategy_price_at_open": 101.0,
            "ai_last_prediction": 104.0,
            "signal": "BUY",
            "confidence": 65.0,
        },
    )

    date_str = datetime.now(tracker_module.IST).strftime("%Y-%m-%d")

    def fake_download(*args, **kwargs):
        idx = pd.to_datetime([date_str])
        return pd.DataFrame(
            {
                "Close": [102.0],
                "High": [103.0],
                "Low": [98.0],
            },
            index=idx,
        )

    monkeypatch.setattr("yfinance.download", fake_download)
    results = PredictionTracker.check_outcomes(date_str)
    assert "ABC.NS" in results
    assert results["ABC.NS"]["actual_close"] == 102.0
    assert "direction_comparison" in results["ABC.NS"]

    day_file = daily_dir / f"{date_str}.json"
    saved = json.loads(day_file.read_text())
    row = saved["ABC.NS"]
    assert row["actual_close"] == 102.0
    assert "strategy_price_at_open" in row
    assert "ai_last_prediction" in row
    assert "direction_comparison" in row


def test_api_expected_vs_actual_includes_training_fields(tmp_path: Path, monkeypatch):
    logs_dir = tmp_path / "prediction_log"
    logs_dir.mkdir(parents=True, exist_ok=True)
    date_str = "2026-02-14"
    log_file = logs_dir / f"{date_str}.json"
    log_file.write_text(
        json.dumps(
            {
                "ABC.NS": {
                    "predicted_return": 2.0,
                    "predicted_price": 102.0,
                    "current_price": 100.0,
                    "signal": "BUY",
                    "confidence": 67.0,
                    "strategy_price_at_open": 101.5,
                    "ai_last_prediction": 102.0,
                }
            }
        )
    )

    monkeypatch.setattr(server, "PREDICTION_LOG_DIR", logs_dir)
    monkeypatch.setattr(server, "ticker_names", {"ABC.NS": "ABC"})
    monkeypatch.setattr(
        server,
        "_get_close_prices_for_date",
        lambda tickers, _date: {"ABC.NS": 103.0},
    )
    monkeypatch.setattr(
        server,
        "_get_live_prices_batch",
        lambda tickers: {"^NSEI": {"price": 20000.0, "prev_close": 19900.0}},
    )
    monkeypatch.setattr(
        server.PredictionTracker,
        "check_outcomes",
        staticmethod(
            lambda _date: {
                "ABC.NS": {
                    "strategy_price_at_open": 101.5,
                    "ai_last_prediction": 102.0,
                    "strategy_direction_at_open": "UP",
                    "ai_direction_last": "UP",
                    "direction_comparison": True,
                    "checked_at": "2026-02-14T15:31:00+05:30",
                }
            }
        ),
    )

    with server.app.test_client() as client:
        resp = client.get(f"/api/expected-vs-actual?date={date_str}")
        assert resp.status_code == 200
        payload = resp.get_json()

    assert payload["results"]
    row = payload["results"][0]
    assert row["strategy_price_at_open"] == 101.5
    assert row["ai_last_prediction"] == 102.0
    assert row["actual_close"] == 103.0
    assert row["direction_comparison"] is True
    assert "strategy_return_pct" in row
    assert "ai_return_pct" in row
    assert "alpha_pct" in row
    assert row["open_price"] == 100.0
    assert row["close_price"] == 103.0
    pred_open_ts = datetime.fromisoformat(row["strategy_predicted_at_open"])
    assert pred_open_ts.hour == 9 and 15 <= pred_open_ts.minute <= 30


def test_api_expected_vs_actual_direction_uses_strategy_vs_open(
    tmp_path: Path, monkeypatch
):
    logs_dir = tmp_path / "prediction_log"
    logs_dir.mkdir(parents=True, exist_ok=True)
    date_str = "2026-02-12"
    (logs_dir / f"{date_str}.json").write_text(
        json.dumps(
            {
                "XYZ.NS": {
                    "predicted_return": 10.0,
                    "predicted_price": 110.0,
                    "current_price": 100.0,
                    "open_price": 100.0,
                    "strategy_price_at_open": 110.0,
                    "signal": "BUY",
                    "confidence": 75.0,
                    "timestamp": "2026-02-12T09:00:00+05:30",
                }
            }
        )
    )

    monkeypatch.setattr(server, "PREDICTION_LOG_DIR", logs_dir)
    monkeypatch.setattr(server, "ticker_names", {"XYZ.NS": "XYZ"})
    monkeypatch.setattr(
        server,
        "_get_close_prices_for_date",
        lambda tickers, _date: {"XYZ.NS": 90.0},
    )
    monkeypatch.setattr(
        server,
        "_get_live_prices_batch",
        lambda tickers: {"^NSEI": {"price": 20000.0, "prev_close": 19900.0}},
    )
    monkeypatch.setattr(
        server.PredictionTracker, "check_outcomes", staticmethod(lambda _date: {})
    )

    with server.app.test_client() as client:
        resp = client.get(f"/api/expected-vs-actual?date={date_str}")
        assert resp.status_code == 200
        payload = resp.get_json()

    row = payload["results"][0]
    assert row["strategy_direction_at_open"] == "UP"
    assert row["direction_actual"] == "DOWN"
    assert row["direction_comparison"] is False
