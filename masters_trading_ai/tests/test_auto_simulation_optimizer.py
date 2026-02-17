from datetime import datetime

from webapp import server


def test_build_strategy_buy_candidates_strategy_first_with_sentiment_override(monkeypatch):
    monkeypatch.setattr(server, "_is_tradeable_ticker", lambda ticker: True)
    now_iso = datetime.now(server.IST).isoformat()
    rows = [
        {
            "ticker": "AAA.NS",
            "signal": "SELL",
            "predicted_return": 2.5,
            "current_price": 100.0,
            "strategy_price_at_open": 102.0,
            "confidence": 80,
            "model_agreement": 70,
            "atr_pct": 2.0,
            "risk_reward": 1.5,
            "timestamp": now_iso,
        },
        {
            "ticker": "BBB.NS",
            "signal": "HOLD",
            "predicted_return": 1.8,
            "current_price": 200.0,
            "strategy_price_at_open": 203.0,
            "confidence": 75,
            "model_agreement": 72,
            "atr_pct": 2.1,
            "risk_reward": 1.4,
            "weighted_sentiment_raw": 0.5,
            "sentiment_articles": 3,
            "sentiment_decay_weights": {"day0": 0.35},
            "timestamp": now_iso,
        },
        {
            "ticker": "CCC.NS",
            "signal": "BUY",
            "predicted_return": 3.2,
            "current_price": 300.0,
            "strategy_price_at_open": 303.0,
            "confidence": 72,
            "model_agreement": 68,
            "atr_pct": 1.1,
            "risk_reward": 1.6,
            "timestamp": now_iso,
        },
    ]
    live = {
        "AAA.NS": {"price": 100.0},
        "BBB.NS": {"price": 200.0},
        "CCC.NS": {"price": 300.0},
    }

    candidates, _warnings = server._build_strategy_buy_candidates(
        rows,
        live_prices=live,
        exclude_tickers=set(),
    )
    tickers = {r["ticker"] for r in candidates}
    assert "AAA.NS" not in tickers
    assert "BBB.NS" in tickers
    assert "CCC.NS" in tickers
    bbb = next(r for r in candidates if r["ticker"] == "BBB.NS")
    assert bbb["effective_action"] == "BUY"
    assert bbb["action_source"] in {"sentiment_upgrade_to_buy", "strategy_plus_sentiment"}


def test_run_sim_auto_check_sells_then_reinvests(monkeypatch):
    monkeypatch.setattr(server, "_is_tradeable_ticker", lambda ticker: True)
    monkeypatch.setattr(
        server,
        "_estimate_entry_fee",
        lambda notional, trade_type="equity_delivery": 0.0,
    )
    monkeypatch.setattr(
        server,
        "_estimate_exit_fee",
        lambda notional, trade_type="equity_delivery": 0.0,
    )
    monkeypatch.setattr(server, "models_loaded", True)
    monkeypatch.setattr(server, "predictor", object())
    monkeypatch.setattr(server, "_log_simulated_trade", lambda event: None)

    def _fake_live_prices(tickers):
        out = {}
        for t in tickers:
            if t == "AAA.NS":
                out[t] = {"price": 99.0}
            elif t == "BBB.NS":
                out[t] = {"price": 50.0}
            else:
                out[t] = {"price": 0.0}
        return out

    monkeypatch.setattr(server, "_get_live_prices_batch", _fake_live_prices)
    monkeypatch.setattr(
        server,
        "_get_prediction_snapshot",
        lambda **kwargs: {
            "items": [
                {
                    "ticker": "AAA.NS",
                    "signal": "SELL",
                    "predicted_return_pct": -0.8,
                    "current_price": 99.0,
                    "strategy_price_at_open": 98.2,
                    "confidence": 70,
                    "model_agreement": 68,
                    "atr_pct": 2.0,
                    "risk_reward": 1.4,
                },
                {
                    "ticker": "BBB.NS",
                    "signal": "BUY",
                    "predicted_return_pct": 1.8,
                    "current_price": 50.0,
                    "strategy_price_at_open": 50.0,
                    "confidence": 82,
                    "model_agreement": 74,
                    "atr_pct": 2.2,
                    "risk_reward": 1.5,
                },
            ]
        },
    )

    state = {
        "initial_cash": 10000.0,
        "cash": 5000.0,
        "open_positions": {
            "AAA.NS": {
                "ticker": "AAA.NS",
                "quantity": 10.0,
                "avg_entry_price": 100.0,
                "stop_loss_price": 85.0,
                "target_price": 130.0,
                "total_entry_fees": 0.0,
                "opened_at": datetime.now(server.IST).isoformat(),
            }
        },
        "closed_trades": [],
        "trade_history": [],
    }

    result = server._run_sim_auto_check(
        state,
        auto_buy_enabled=True,
        now_iso=datetime.now(server.IST).isoformat(),
        source="test",
    )
    sell_reasons = [e.get("reason") for e in result["events"]]
    assert "auto_strategy_sell_signal" in sell_reasons
    assert any(e.get("ticker") == "BBB.NS" for e in result["auto_buy_events"])
    assert "BBB.NS" in result["state"].get("open_positions", {})
