from src.inference.predictor import LivePredictor


def test_grouped_top_picks_use_signal_labels(monkeypatch):
    predictor = LivePredictor()
    predictor._loaded = True

    samples = {
        "A.NS": {"ticker": "A.NS", "signal": "BUY", "predicted_return": -1.0, "confidence": 70, "model_agreement": 70, "risk_reward": 1.2, "current_price": 100},
        "B.NS": {"ticker": "B.NS", "signal": "SELL", "predicted_return": 1.0, "confidence": 68, "model_agreement": 65, "risk_reward": 1.1, "current_price": 100},
        "C.NS": {"ticker": "C.NS", "signal": "HOLD", "predicted_return": 0.1, "confidence": 52, "model_agreement": 55, "risk_reward": 1.0, "current_price": 100},
    }

    monkeypatch.setattr(predictor, "_resolve_top_pick_tickers", lambda tickers=None, sectors=None: list(samples.keys()))
    monkeypatch.setattr(predictor, "predict_single", lambda ticker, use_cache=True: dict(samples[ticker]))

    grouped = predictor.predict_top_picks_grouped(top_n=5)
    assert [x["ticker"] for x in grouped["top_buy"]] == ["A.NS"]
    assert [x["ticker"] for x in grouped["top_sell"]] == ["B.NS"]
    assert [x["ticker"] for x in grouped["top_hold"]] == ["C.NS"]


def test_generate_signal_can_emit_buy_or_sell():
    predictor = LivePredictor()

    buy_signal, buy_conf = predictor._generate_signal(
        predicted_return=0.0045,  # 0.45%
        atr_pct=0.01,
        volume_ratio=1.2,
        rvol=1.2,
        model_agreement=0.75,
    )
    sell_signal, sell_conf = predictor._generate_signal(
        predicted_return=-0.0045,  # -0.45%
        atr_pct=0.01,
        volume_ratio=1.1,
        rvol=1.0,
        model_agreement=0.72,
    )

    assert buy_signal in {"BUY", "STRONG_BUY"}
    assert sell_signal in {"SELL", "STRONG_SELL"}
    assert buy_conf >= 54
    assert sell_conf >= 54
