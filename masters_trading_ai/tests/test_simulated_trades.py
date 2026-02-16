from webapp import server


def test_simulated_trade_flow_stop_loss_autosell(monkeypatch, tmp_path):
    sim_file = tmp_path / "portfolio_sim.json"
    sim_log = tmp_path / "simulated_trades.jsonl"

    monkeypatch.setattr(server, "PORTFOLIO_SIM_FILE", sim_file)
    monkeypatch.setattr(server, "SIMULATED_TRADE_LOG_FILE", sim_log)
    monkeypatch.setattr(server, "_is_tradeable_ticker", lambda ticker: True)
    monkeypatch.setattr(server, "_estimate_entry_fee", lambda notional, trade_type="equity_delivery": 0.0)
    monkeypatch.setattr(server, "_estimate_exit_fee", lambda notional, trade_type="equity_delivery": 0.0)

    prices = {"AAA.NS": {"price": 94.0}}
    monkeypatch.setattr(server, "_get_live_prices_batch", lambda tickers: {t: prices.get(t, {"price": 0}) for t in tickers})

    with server.app.test_client() as client:
        reset = client.post("/api/simulate/trade", json={"action": "RESET", "budget": 40000})
        assert reset.status_code == 200

        bad_sell = client.post(
            "/api/simulate/trade",
            json={"action": "SELL", "ticker": "AAA.NS", "quantity": 1, "price": 100},
        )
        assert bad_sell.status_code == 400

        buy = client.post(
            "/api/simulate/trade",
            json={
                "action": "BUY",
                "ticker": "AAA.NS",
                "quantity": 10,
                "price": 100,
                "stop_loss_price": 95,
                "target_price": 110,
            },
        )
        assert buy.status_code == 200

        auto = client.post("/api/simulate/trade", json={"action": "AUTO_CHECK"})
        assert auto.status_code == 200
        auto_payload = auto.get_json()
        assert auto_payload["triggered_count"] == 1
        assert auto_payload["events"][0]["reason"] == "auto_stop_loss"

        summary = client.get("/api/simulate/portfolio")
        assert summary.status_code == 200
        payload = summary.get_json()
        assert payload["open_positions_count"] == 0
