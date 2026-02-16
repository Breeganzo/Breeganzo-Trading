# Daily Operations Runbook

## 1) Start services
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
python webapp/server.py
```

## 2) Verify market-window snapshot behavior (IST)
- `00:00-09:15`: `after_hours_live` should refresh live AI/strategy values.
- `09:15-09:30`: `premarket_open` snapshot window.
- `09:30-15:30`: `market_open_locked` strategy snapshot should stay fixed.
- `15:30-next day 00:00`: `after_hours_live` should refresh live AI/strategy values.
- Scheduled precompute: `09:28` and `15:31` IST (snapshot warming).

Checks:
```bash
curl -s 'http://localhost:5001/api/premarket-outlook' | jq '.snapshot_type,.captured_at,.captured_at_actual'
curl -s 'http://localhost:5001/api/debug/prediction-status/INFY.NS' | jq '.prediction_window,.snapshot_type,.snapshot_captured_at'
```

## 3) Trading Desk Advisor simulation checks
- Default simulation cash should be `₹40,000`.
- Open-buy list must respect budget + estimated fees.
- Auto-check should trigger simulated sells when stop-loss/target is hit.
- Auto-check can optionally trigger simulated auto-buys from snapshot candidates.
- BUY/SELL/AUTO_CHECK triggers run only during `09:30-15:30 IST` on trading days.

Checks:
```bash
curl -s 'http://localhost:5001/api/advisor/open-buy-list?n=10&budget=40000' | jq '.budget,.estimated_total_cost,.count'
curl -s 'http://localhost:5001/api/simulate/portfolio' | jq '.cash,.equity_value,.open_positions_count'
curl -s -X POST 'http://localhost:5001/api/simulate/trade' \
  -H 'Content-Type: application/json' \
  -d '{"action":"AUTO_CHECK","auto_buy":true}' | jq '.triggered_count,.auto_buy_events,.events'
curl -s 'http://localhost:5001/api/groq-status' | jq '.degraded_mode,.degraded_reason,.degraded_until_iso,.key_pool_size,.active_key_slot'
```

## 4) Expected vs Actual freeze checks
- At EOD, row values for a date should stabilize and not drift.
- Validate fields: `open_price`, `strategy_price_at_open`, `ai_price_at_open`, `close_price`, `actual_return_pct`, `alpha_pct`.

```bash
TODAY=$(date +%F)
curl -s "http://localhost:5001/api/expected-vs-actual?date=$TODAY" | jq '.date,.hit_rate_pct,.results[0]'
```

## 5) Retrain triggers
Run retraining early if any condition persists:
- Direction hit-rate drops > 7% week-over-week.
- Average alpha degrades for 10+ sessions.
- Feature drift alerts (KS or similar) are consistently failing.
- Drawdown regime shifts materially vs baseline.

## 6) Artifacts to preserve
- `cache/prediction_log/*.json`
- `cache/prediction_tracking/daily/*.json`
- `cache/portfolio_sim.json`
- `cache/prediction_log/simulated_trades.jsonl`

These are re-usable for post-trade analysis and model retraining.
