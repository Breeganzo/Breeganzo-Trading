from webapp import server


def test_simulated_trade_flow_stop_loss_autosell(monkeypatch, tmp_path):
    sim_file = tmp_path / "portfolio_sim.json"
    sim_log = tmp_path / "simulated_trades.jsonl"
    sim_csv = tmp_path / "simulated_trades.csv"
    sim_xlsx = tmp_path / "simulated_trades.xlsx"
    report_dir = tmp_path / "daily_reports"

    monkeypatch.setattr(server, "PORTFOLIO_SIM_FILE", sim_file)
    monkeypatch.setattr(server, "SIMULATED_TRADE_LOG_FILE", sim_log)
    monkeypatch.setattr(server, "SIMULATED_TRADE_CSV_FILE", sim_csv)
    monkeypatch.setattr(server, "SIMULATED_TRADE_EXCEL_FILE", sim_xlsx)
    monkeypatch.setattr(server, "DAILY_TRADE_REPORT_DIR", report_dir)
    monkeypatch.setattr(server, "_is_tradeable_ticker", lambda ticker: True)
    monkeypatch.setattr(
        server,
        "_estimate_entry_fee",
        lambda notional, trade_type="equity_delivery": 0.0,
    )
    monkeypatch.setattr(
        server, "_estimate_exit_fee", lambda notional, trade_type="equity_delivery": 0.0
    )
    monkeypatch.setattr(server, "_is_market_trade_window", lambda *_: True)

    prices = {"AAA.NS": {"price": 94.0}}
    monkeypatch.setattr(
        server,
        "_get_live_prices_batch",
        lambda tickers: {t: prices.get(t, {"price": 0}) for t in tickers},
    )

    with server.app.test_client() as client:
        reset = client.post(
            "/api/simulate/trade", json={"action": "RESET", "budget": 40000}
        )
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
        assert sim_log.exists()
        assert sim_csv.exists()
        if server.OPENPYXL_AVAILABLE:
            assert sim_xlsx.exists()

        auto = client.post("/api/simulate/trade", json={"action": "AUTO_CHECK"})
        assert auto.status_code == 200
        auto_payload = auto.get_json()
        assert auto_payload["triggered_count"] == 1
        assert auto_payload["events"][0]["reason"] == "auto_stop_loss"

        summary = client.get("/api/simulate/portfolio")
        assert summary.status_code == 200
        payload = summary.get_json()
        assert payload["open_positions_count"] == 0
        tx = client.get("/api/simulate/transactions-summary")
        assert tx.status_code == 200
        tx_data = tx.get_json()
        assert tx_data["total_transactions"] >= 2
        assert tx_data["buy_transactions"] >= 1
        assert tx_data["sell_transactions"] >= 1


def test_simulated_position_delete_closes_open_lot(monkeypatch, tmp_path):
    sim_file = tmp_path / "portfolio_sim.json"
    sim_log = tmp_path / "simulated_trades.jsonl"
    sim_csv = tmp_path / "simulated_trades.csv"
    sim_xlsx = tmp_path / "simulated_trades.xlsx"
    report_dir = tmp_path / "daily_reports"

    monkeypatch.setattr(server, "PORTFOLIO_SIM_FILE", sim_file)
    monkeypatch.setattr(server, "SIMULATED_TRADE_LOG_FILE", sim_log)
    monkeypatch.setattr(server, "SIMULATED_TRADE_CSV_FILE", sim_csv)
    monkeypatch.setattr(server, "SIMULATED_TRADE_EXCEL_FILE", sim_xlsx)
    monkeypatch.setattr(server, "DAILY_TRADE_REPORT_DIR", report_dir)
    monkeypatch.setattr(server, "_is_tradeable_ticker", lambda ticker: True)
    monkeypatch.setattr(server, "_is_market_trade_window", lambda *_: True)
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
    monkeypatch.setattr(
        server,
        "_get_live_prices_batch",
        lambda tickers: {t: {"price": 102.0} for t in tickers},
    )

    with server.app.test_client() as client:
        reset = client.post(
            "/api/simulate/trade", json={"action": "RESET", "budget": 40000}
        )
        assert reset.status_code == 200

        buy = client.post(
            "/api/simulate/trade",
            json={
                "action": "BUY",
                "ticker": "INFY.NS",
                "quantity": 5,
                "price": 100.0,
                "stop_loss_price": 95.0,
                "target_price": 110.0,
            },
        )
        assert buy.status_code == 200

        close_resp = client.delete("/api/simulate/position/INFY.NS")
        assert close_resp.status_code == 200
        payload = close_resp.get_json()
        assert payload.get("ok") is True
        assert payload.get("event", {}).get("action") == "SELL"
        assert payload.get("event", {}).get("reason") == "manual_portfolio_remove"

        portfolio = client.get("/api/simulate/portfolio")
        assert portfolio.status_code == 200
        p = portfolio.get_json()
        assert p.get("open_positions_count") == 0


def test_simulated_trade_rejects_outside_market_hours(monkeypatch, tmp_path):
    sim_file = tmp_path / "portfolio_sim.json"
    sim_log = tmp_path / "simulated_trades.jsonl"
    sim_csv = tmp_path / "simulated_trades.csv"
    sim_xlsx = tmp_path / "simulated_trades.xlsx"
    report_dir = tmp_path / "daily_reports"

    monkeypatch.setattr(server, "PORTFOLIO_SIM_FILE", sim_file)
    monkeypatch.setattr(server, "SIMULATED_TRADE_LOG_FILE", sim_log)
    monkeypatch.setattr(server, "SIMULATED_TRADE_CSV_FILE", sim_csv)
    monkeypatch.setattr(server, "SIMULATED_TRADE_EXCEL_FILE", sim_xlsx)
    monkeypatch.setattr(server, "DAILY_TRADE_REPORT_DIR", report_dir)
    monkeypatch.setattr(server, "_is_tradeable_ticker", lambda ticker: True)
    monkeypatch.setattr(server, "_is_market_trade_window", lambda *_: False)

    with server.app.test_client() as client:
        reset = client.post(
            "/api/simulate/trade", json={"action": "RESET", "budget": 40000}
        )
        assert reset.status_code == 200

        buy = client.post(
            "/api/simulate/trade",
            json={
                "action": "BUY",
                "ticker": "AAA.NS",
                "quantity": 1,
                "price": 100,
            },
        )
        assert buy.status_code == 403
        payload = buy.get_json()
        assert "market hours" in payload.get("error", "").lower()

        auto = client.post("/api/simulate/trade", json={"action": "AUTO_CHECK"})
        assert auto.status_code == 200
        auto_payload = auto.get_json()
        assert auto_payload.get("ok") is True
        assert auto_payload.get("triggered_count") == 0


def test_simulated_trade_ledger_reports_sold_and_hold(monkeypatch, tmp_path):
    sim_file = tmp_path / "portfolio_sim.json"
    sim_log = tmp_path / "simulated_trades.jsonl"
    sim_csv = tmp_path / "simulated_trades.csv"
    sim_xlsx = tmp_path / "simulated_trades.xlsx"
    report_dir = tmp_path / "daily_reports"

    monkeypatch.setattr(server, "PORTFOLIO_SIM_FILE", sim_file)
    monkeypatch.setattr(server, "SIMULATED_TRADE_LOG_FILE", sim_log)
    monkeypatch.setattr(server, "SIMULATED_TRADE_CSV_FILE", sim_csv)
    monkeypatch.setattr(server, "SIMULATED_TRADE_EXCEL_FILE", sim_xlsx)
    monkeypatch.setattr(server, "DAILY_TRADE_REPORT_DIR", report_dir)
    monkeypatch.setattr(server, "_is_tradeable_ticker", lambda ticker: True)
    monkeypatch.setattr(server, "_is_market_trade_window", lambda *_: True)
    monkeypatch.setattr(
        server,
        "_estimate_entry_fee",
        lambda notional, trade_type="equity_delivery": 0.0,
    )
    monkeypatch.setattr(
        server, "_estimate_exit_fee", lambda notional, trade_type="equity_delivery": 0.0
    )
    monkeypatch.setattr(
        server,
        "_get_live_prices_batch",
        lambda tickers: {t: {"price": 104.0} for t in tickers},
    )

    with server.app.test_client() as client:
        reset = client.post(
            "/api/simulate/trade", json={"action": "RESET", "budget": 40000}
        )
        assert reset.status_code == 200

        buy = client.post(
            "/api/simulate/trade",
            json={
                "action": "BUY",
                "ticker": "INFY.NS",
                "quantity": 10,
                "price": 100,
                "stop_loss_price": 95,
                "target_price": 110,
            },
        )
        assert buy.status_code == 200

        sell = client.post(
            "/api/simulate/trade",
            json={"action": "SELL", "ticker": "INFY.NS", "quantity": 4, "price": 105},
        )
        assert sell.status_code == 200

        ledger = client.get("/api/simulate/trade-ledger")
        assert ledger.status_code == 200
        payload = ledger.get_json()
        assert payload["count"] >= 2
        statuses = {row.get("status") for row in payload.get("rows", [])}
        assert "SOLD" in statuses
        assert "HOLD" in statuses

        sold_rows = [r for r in payload["rows"] if r.get("status") == "SOLD"]
        hold_rows = [r for r in payload["rows"] if r.get("status") == "HOLD"]
        assert any(abs(float(r.get("quantity", 0)) - 4.0) < 1e-6 for r in sold_rows)
        assert any(abs(float(r.get("quantity", 0)) - 6.0) < 1e-6 for r in hold_rows)
        assert all(r.get("buy_timestamp") for r in sold_rows + hold_rows)


def test_sim_buy_respects_budget_plus_fee(monkeypatch, tmp_path):
    sim_file = tmp_path / "portfolio_sim.json"
    monkeypatch.setattr(server, "PORTFOLIO_SIM_FILE", sim_file)
    monkeypatch.setattr(server, "_is_tradeable_ticker", lambda ticker: True)
    monkeypatch.setattr(server, "_is_market_trade_window", lambda *_: True)
    monkeypatch.setattr(
        server,
        "_estimate_entry_fee",
        lambda notional, trade_type="equity_delivery": round(notional * 0.01, 2),
    )

    with server.app.test_client() as client:
        reset = client.post(
            "/api/simulate/trade",
            json={"action": "RESET", "budget": 1000, "clear_history": True},
        )
        assert reset.status_code == 200

        # 10 * 100 + 1% fee = 1010 -> should be rejected by budget.
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
        assert buy.status_code == 400
        assert "insufficient" in buy.get_json().get("error", "").lower()


def test_autocheck_same_cycle_sell_then_buy_updates_cache_and_no_dummy_values(
    monkeypatch, tmp_path
):
    sim_file = tmp_path / "portfolio_sim.json"
    sim_log = tmp_path / "simulated_trades.jsonl"
    sim_csv = tmp_path / "simulated_trades.csv"
    sim_xlsx = tmp_path / "simulated_trades.xlsx"
    report_dir = tmp_path / "daily_reports"

    monkeypatch.setattr(server, "PORTFOLIO_SIM_FILE", sim_file)
    monkeypatch.setattr(server, "SIMULATED_TRADE_LOG_FILE", sim_log)
    monkeypatch.setattr(server, "SIMULATED_TRADE_CSV_FILE", sim_csv)
    monkeypatch.setattr(server, "SIMULATED_TRADE_EXCEL_FILE", sim_xlsx)
    monkeypatch.setattr(server, "DAILY_TRADE_REPORT_DIR", report_dir)
    monkeypatch.setattr(server, "_is_tradeable_ticker", lambda ticker: True)
    monkeypatch.setattr(server, "_is_market_trade_window", lambda *_: True)
    monkeypatch.setattr(server, "models_loaded", True)
    monkeypatch.setattr(server, "predictor", object())
    monkeypatch.setattr(server, "_log_simulated_trade", lambda event: None)
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

    def _fake_prices(tickers):
        out = {}
        for t in tickers:
            if t == "AAA.NS":
                out[t] = {"price": 90.0}
            elif t == "BBB.NS":
                out[t] = {"price": 50.0}
            else:
                out[t] = {"price": 0.0}
        return out

    monkeypatch.setattr(server, "_get_live_prices_batch", _fake_prices)
    monkeypatch.setattr(
        server,
        "_get_prediction_snapshot",
        lambda **kwargs: {
            "items": [
                {
                    "ticker": "AAA.NS",
                    "signal": "SELL",
                    "predicted_return_pct": -1.2,
                    "current_price": 90.0,
                    "strategy_price_at_open": 99.0,
                    "confidence": 70,
                    "model_agreement": 68,
                    "atr_pct": 2.0,
                    "risk_reward": 1.4,
                },
                {
                    "ticker": "BBB.NS",
                    "signal": "BUY",
                    "predicted_return_pct": 2.0,
                    "current_price": 50.0,
                    "strategy_price_at_open": 50.0,
                    "entry_range_low": 49.5,
                    "entry_range_high": 50.5,
                    "confidence": 81,
                    "model_agreement": 73,
                    "atr_pct": 2.1,
                    "risk_reward": 1.5,
                },
            ]
        },
    )

    with server.app.test_client() as client:
        reset = client.post(
            "/api/simulate/trade",
            json={"action": "RESET", "budget": 40000, "clear_history": True},
        )
        assert reset.status_code == 200

        buy = client.post(
            "/api/simulate/trade",
            json={
                "action": "BUY",
                "ticker": "AAA.NS",
                "quantity": 10,
                "price": 100,
                "stop_loss_price": 95,
                "target_price": 110,
                "entry_range_low": 99,
                "entry_range_high": 101,
            },
        )
        assert buy.status_code == 200

        # Query-string invocation should work too.
        auto = client.post("/api/simulate/trade?action=AUTO_CHECK&auto_buy=true")
        assert auto.status_code == 200
        payload = auto.get_json()
        assert payload.get("triggered_count", 0) >= 2
        assert any(e.get("action") == "SELL" for e in payload.get("events", []))
        assert any(e.get("action") == "BUY" for e in payload.get("auto_buy_events", []))

    assert sim_file.exists()
    state = server.json.loads(sim_file.read_text())
    assert "open_positions" in state
    # No dummy placeholders/extreme junk values.
    for _, pos in dict(state.get("open_positions", {})).items():
        for key in ("avg_entry_price", "stop_loss_price", "target_price"):
            val = float(pos.get(key, 0) or 0)
            assert 0 < val < 1e9
