from datetime import datetime

from webapp import server


def test_build_strategy_buy_candidates_strategy_first_no_positive_upgrade(monkeypatch):
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
            "predicted_return": 0.1,
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
    # HOLD should not be upgraded to BUY purely by positive sentiment.
    assert "BBB.NS" not in tickers
    assert "CCC.NS" in tickers


def test_sentiment_influence_is_small_vs_strategy_edge(monkeypatch):
    monkeypatch.setattr(server, "_is_tradeable_ticker", lambda ticker: True)
    monkeypatch.setattr(server, "ADVISOR_SENTIMENT_INFLUENCE", 0.10)
    now_iso = datetime.now(server.IST).isoformat()
    base_row = {
        "ticker": "DDD.NS",
        "signal": "BUY",
        "predicted_return": 2.0,
        "current_price": 100.0,
        "strategy_price_at_open": 101.5,
        "confidence": 85,
        "model_agreement": 80,
        "atr_pct": 1.8,
        "risk_reward": 1.6,
        "timestamp": now_iso,
    }
    live = {"DDD.NS": {"price": 100.0}}

    neutral_rows = [dict(base_row, weighted_sentiment_raw=0.0, sentiment_articles=0)]
    mild_negative_rows = [
        dict(
            base_row,
            weighted_sentiment_raw=-0.2,
            sentiment_articles=2,
            sentiment_decay_weights={"day0": 0.35},
        )
    ]
    neutral = server._build_strategy_buy_candidates(
        neutral_rows, live_prices=live, exclude_tickers=set()
    )[0][0]
    with_sent = server._build_strategy_buy_candidates(
        mild_negative_rows, live_prices=live, exclude_tickers=set()
    )[0][0]

    delta = neutral["expected_edge"] - with_sent["expected_edge"]
    assert delta > 0
    assert with_sent["expected_edge"] > 0
    assert abs(delta) <= abs(neutral["strategy_edge_component"]) * 0.25


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


def test_optimize_allocations_fee_aware_and_sector_diversified(monkeypatch):
    monkeypatch.setattr(
        server,
        "_estimate_entry_fee",
        lambda notional, trade_type="equity_delivery": round(notional * 0.001, 2),
    )
    monkeypatch.setattr(server, "ADVISOR_MAX_PER_SECTOR", 1)
    monkeypatch.setattr(server, "MAX_POSITION_SIZE", 0.5)

    candidates = [
        {
            "ticker": "AAA.NS",
            "name": "AAA",
            "sector": "banking",
            "strategy_price_at_open": 100.0,
            "current_price": 100.0,
            "stop_loss_price": 95.0,
            "target_price": 110.0,
            "risk_reward": 1.5,
            "objective_utility": 0.08,
            "risk_metric": 0.02,
            "confidence": 80.0,
            "predicted_return_pct": 2.0,
            "signal": "BUY",
        },
        {
            "ticker": "BBB.NS",
            "name": "BBB",
            "sector": "banking",
            "strategy_price_at_open": 120.0,
            "current_price": 120.0,
            "stop_loss_price": 114.0,
            "target_price": 132.0,
            "risk_reward": 1.5,
            "objective_utility": 0.07,
            "risk_metric": 0.02,
            "confidence": 78.0,
            "predicted_return_pct": 1.8,
            "signal": "BUY",
        },
        {
            "ticker": "CCC.NS",
            "name": "CCC",
            "sector": "it",
            "strategy_price_at_open": 140.0,
            "current_price": 140.0,
            "stop_loss_price": 132.0,
            "target_price": 154.0,
            "risk_reward": 1.6,
            "objective_utility": 0.09,
            "risk_metric": 0.025,
            "confidence": 82.0,
            "predicted_return_pct": 2.1,
            "signal": "BUY",
        },
    ]

    picks = server._optimize_candidate_allocations(
        candidates,
        budget=10000.0,
        trade_type="equity_delivery",
        max_positions=3,
    )
    assert picks
    assert sum(float(p.get("est_trade_cost") or 0) for p in picks) <= 10000.0 + 1e-9
    sectors = {}
    for row in picks:
        sectors[row["sector"]] = sectors.get(row["sector"], 0) + 1
        assert float(row.get("est_trade_cost") or 0) > 0
        assert float(row.get("estimated_fee") or 0) >= 0
        assert int(row.get("suggested_qty") or 0) > 0
    assert max(sectors.values()) <= 1
