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
