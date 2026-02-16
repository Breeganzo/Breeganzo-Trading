# Daily Operations Runbook

## 1) Start of day (before 09:15 IST)
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
python webapp/server.py
```

Checks:
- `GET /api/status` should show model load progress or `models_loaded: true`.
- `GET /api/premarket-outlook` may return `snapshot_type: "pending_market_open"` before open.

## 2) Market open capture window (09:15-09:30 IST)
Goal: ensure strategy snapshot is captured from market-open flow.

```bash
curl -s 'http://localhost:5001/api/premarket-outlook' | jq
```

Validate:
- `snapshot_type` should be `market_open` (or `near_open_fallback` if app started late).
- Each row should include:
  - `strategy_price_at_open`
  - `ai_predicted_price`
  - `strategy_predicted_at_open`
  - `ai_predicted_at_open`

## 3) Intraday monitoring
- Dashboard Stocks page updates live values every few seconds.
- Top Picks:
  - use dropdown for `Top 10 BUY / SELL / HOLD`.
  - premarket table + current-second panel should not show placeholder zero prices.

## 4) After-hours mode (>= 16:00 IST)
- UI switches to after-hours mode and locks premarket widgets.
- `expected-vs-actual` uses end-of-day close for same-day comparison.

```bash
curl -s "http://localhost:5001/api/expected-vs-actual?date=$(date +%F)" | jq
```

Validate fields:
- `strategy_price_at_open`
- `actual_close`
- `direction_comparison`
- `after_hours_mode`

## 5) Portfolio risk analytics
```bash
curl -s 'http://localhost:5001/api/risk-analytics' | jq
```

Validate:
- `portfolio_tickers` contains only held tickers.
- `ignored_tickers` contains non-portfolio requests.
- `monte_carlo.simulation_source` is `equity_curve`.

## 6) Daily quality gates
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
pytest -q
black --check src webapp tests
flake8 src webapp tests --max-line-length=120 --extend-ignore=E203,W503
mypy src/inference/predictor.py webapp/prediction_tracker.py webapp/server.py --ignore-missing-imports --follow-imports=silent
```

## 7) Weekly / monthly cadence
- Weekly: refresh data + inspect delisted ticker registry (`cache/delisted_tickers.csv`).
- Bi-weekly: retrain core models.
- Monthly: walk-forward/backtest review with cost/slippage checks.

## 8) Push workflow
```bash
cd /Users/anto/Trading_Project
git checkout feature/ui-risk-premarket-fixes
git add .
git commit -m "ui+backend: market-open snapshots, scrollable Groq popovers, risk/montecarlo fixes, cleanup, tests, docs"
git push -u origin feature/ui-risk-premarket-fixes
```
