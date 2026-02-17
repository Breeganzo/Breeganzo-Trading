from webapp import server


class _StubPredictor:
    def predict_top_picks_grouped(self, sectors=None, top_n=50):
        return {
            "top_buy": [
                {
                    "ticker": "AAA.NS",
                    "current_price": 100.0,
                    "target_price": 103.0,
                    "predicted_price": 103.0,
                    "predicted_return": 3.0,
                    "confidence": 80.0,
                    "model_agreement": 75.0,
                    "risk_reward": 1.8,
                    "liquidity_factor": 1.2,
                    "avg_volume_30d": 1_500_000,
                    "atr_pct": 2.5,
                    "timestamp": "2026-02-16T09:30:00+05:30",
                },
                {
                    "ticker": "BBB.NS",
                    "current_price": 250.0,
                    "target_price": 258.0,
                    "predicted_price": 258.0,
                    "predicted_return": 3.2,
                    "confidence": 76.0,
                    "model_agreement": 72.0,
                    "risk_reward": 1.5,
                    "liquidity_factor": 1.0,
                    "avg_volume_30d": 900_000,
                    "atr_pct": 3.1,
                    "timestamp": "2026-02-16T09:30:00+05:30",
                },
            ],
            "top_sell": [],
            "top_hold": [
                {
                    "ticker": "HOLD1.NS",
                    "current_price": 120.0,
                    "target_price": 121.0,
                    "predicted_price": 121.0,
                    "predicted_return": 0.8,
                    "signal": "HOLD",
                    "confidence": 68.0,
                    "model_agreement": 66.0,
                    "risk_reward": 1.3,
                    "liquidity_factor": 1.1,
                    "avg_volume_30d": 1_100_000,
                    "atr_pct": 2.0,
                    "timestamp": "2026-02-16T09:30:00+05:30",
                }
            ],
        }


def test_advisor_open_buy_list_respects_budget(monkeypatch):
    monkeypatch.setattr(server, "models_loaded", True)
    monkeypatch.setattr(server, "predictor", _StubPredictor())
    monkeypatch.setattr(server, "ticker_names", {"AAA.NS": "AAA", "BBB.NS": "BBB"})
    monkeypatch.setattr(server, "_read_portfolio_sim_state", lambda: {"cash": 40000.0})
    monkeypatch.setattr(
        server,
        "_estimate_entry_fee",
        lambda notional, trade_type="equity_delivery": round(notional * 0.001, 2),
    )

    with server.app.test_client() as client:
        resp = client.get("/api/advisor/open-buy-list?n=10&budget=40000")
        assert resp.status_code == 200
        payload = resp.get_json()

    assert payload["source"] == "strategy"
    assert payload["count"] <= 10
    assert payload["estimated_total_cost"] <= payload["budget"] + 1e-9
    for row in payload["picks"]:
        assert row["source"] == "strategy"
        assert row["est_trade_cost"] <= payload["budget"]
        assert row["suggested_qty"] > 0


def test_advisor_open_buy_list_hold_view(monkeypatch):
    monkeypatch.setattr(server, "models_loaded", True)
    monkeypatch.setattr(server, "predictor", _StubPredictor())
    monkeypatch.setattr(
        server,
        "ticker_names",
        {"AAA.NS": "AAA", "BBB.NS": "BBB", "HOLD1.NS": "HOLD1"},
    )
    monkeypatch.setattr(server, "_read_portfolio_sim_state", lambda: {"cash": 40000.0})
    monkeypatch.setattr(
        server,
        "_estimate_entry_fee",
        lambda notional, trade_type="equity_delivery": round(notional * 0.001, 2),
    )

    with server.app.test_client() as client:
        resp = client.get("/api/advisor/open-buy-list?n=10&budget=40000&view=hold")
        assert resp.status_code == 200
        payload = resp.get_json()

    assert payload["source"] == "strategy"
    assert payload["view"] == "hold"
    assert payload["count"] <= 10
    for row in payload["picks"]:
        assert row["advisor_view"] == "hold"
        assert row["advisor_action"] == "WATCH"
