from src.inference.predictor import LivePredictor


def test_top_picks_grouped_orders_by_confidence_then_score(monkeypatch):
    predictor = LivePredictor()
    predictor._loaded = True

    rows = {
        "AAA.NS": {
            "ticker": "AAA.NS",
            "signal": "BUY",
            "current_price": 100.0,
            "predicted_return": 2.0,
            "confidence": 80.0,
            "model_agreement": 55.0,
            "avg_volume_30d": 800000,
            "risk_reward": 1.4,
        },
        "BBB.NS": {
            "ticker": "BBB.NS",
            "signal": "BUY",
            "current_price": 100.0,
            "predicted_return": 3.2,
            "confidence": 74.0,
            "model_agreement": 90.0,
            "avg_volume_30d": 1900000,
            "risk_reward": 1.8,
        },
        "CCC.NS": {
            "ticker": "CCC.NS",
            "signal": "BUY",
            "current_price": 100.0,
            "predicted_return": 1.8,
            "confidence": 68.0,
            "model_agreement": 60.0,
            "avg_volume_30d": 500000,
            "risk_reward": 1.3,
        },
    }

    monkeypatch.setattr(predictor, "_resolve_top_pick_tickers", lambda **kwargs: ["AAA.NS", "BBB.NS", "CCC.NS"])
    monkeypatch.setattr(predictor, "predict_single", lambda ticker, use_cache=True: dict(rows[ticker]))

    grouped = predictor.predict_top_picks_grouped(top_n=3)
    buy = grouped["top_buy"]
    assert len(buy) == 3

    # Confidence-first ordering must place AAA before BBB even if BBB has larger return.
    assert buy[0]["ticker"] == "AAA.NS"
    assert buy[1]["ticker"] == "BBB.NS"

    # Liquidity factor should be present and positive for ranking explainability.
    assert all(float(r.get("liquidity_factor", 0) or 0) > 0 for r in buy)
    assert all(float(r.get("_score", 0) or 0) > 0 for r in buy)
