import pytest

from src.inference.predictor import LivePredictor
from webapp import server


def test_live_predictor_top_picks_ranges():
    predictor = LivePredictor()

    try:
        predictor.load_models()
        picks = predictor.predict_top_picks(n=20)
    except Exception as exc:
        pytest.skip(f"Live predictor integration unavailable: {exc}")

    if not picks:
        pytest.skip("No top picks returned from live integration run")

    assert all(abs(p["predicted_return"]) <= 50 for p in picks)
    assert all(p["current_price"] > 0 for p in picks)

    for pick in picks:
        predicted_return_decimal = pick["predicted_return"] / 100.0
        expected_price = round(pick["current_price"] * (1 + predicted_return_decimal), 2)
        assert abs(pick["predicted_price"] - expected_price) <= 0.02


def test_api_top_picks_shape_and_ranges(monkeypatch):
    class StubPredictor:
        def predict_top_picks(self, sectors=None, top_n=5, n=None):
            return [
                {
                    "ticker": "GOOD.NS",
                    "predicted_return": 12.5,
                    "predicted_price": 112.5,
                    "target_price": 112.5,
                    "current_price": 100.0,
                    "signal": "BUY",
                    "confidence": 72.0,
                    "model_agreement": 81.0,
                    "risk_reward": 1.8,
                },
                {
                    "ticker": "ZERO.NS",
                    "predicted_return": 4.0,
                    "predicted_price": 0.0,
                    "target_price": 0.0,
                    "current_price": 0.0,
                    "signal": "BUY",
                    "confidence": 60.0,
                    "model_agreement": 55.0,
                    "risk_reward": 1.0,
                },
            ]

    monkeypatch.setattr(server, "models_loaded", True)
    monkeypatch.setattr(server, "predictor", StubPredictor())
    monkeypatch.setattr(server, "ticker_names", {"GOOD.NS": "Good Co"})

    with server.app.test_client() as client:
        response = client.get("/api/top-picks?n=20")
        assert response.status_code == 200

        payload = response.get_json()
        assert isinstance(payload, list)
        assert len(payload) == 1

        pick = payload[0]
        for key in ("ticker", "predicted_return", "predicted_price", "current_price", "signal"):
            assert key in pick

        assert abs(pick["predicted_return"]) <= 50
        assert pick["current_price"] > 0
