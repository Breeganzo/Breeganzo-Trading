# Daily Operations

## 1) Start of day checklist
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
python webapp/server.py
```

Verify:
```bash
curl -s 'http://localhost:5001/api/status' | jq
curl -s 'http://localhost:5001/api/premarket-outlook' | jq
```

Confirm:
- models are loading/loaded
- premarket snapshot exists
- snapshot was captured within configured buffer if server was running before cutoff
- snapshot type matches session window:
  - `premarket_open` (09:15-09:30)
  - `market_open_locked` (09:30-15:30, frozen strategy)
  - `after_hours_live` (15:30 onwards)

## 2) Intraday health checks
```bash
curl -s 'http://localhost:5001/api/top-picks?n=20&grouped=true' | jq
curl -s 'http://localhost:5001/api/risk-analytics' | jq
```

Expect:
- no `current_price <= 0` in top picks
- `abs(predicted_return) <= 50`
- risk analytics only uses user portfolio tickers

## 3) End-of-day validation
```bash
TODAY=$(date +%F)
curl -s "http://localhost:5001/api/expected-vs-actual?date=$TODAY" | jq
curl -s 'http://localhost:5001/api/training-feedback' | jq
```

Ensure expected-vs-actual rows include:
- `open_price`, `close_price`
- `strategy_price_at_open`
- `ai_last_prediction`
- `actual_close`
- `strategy_return_pct`, `ai_return_pct`, `actual_return_pct`
- `alpha_pct` (actual vs strategy)
- `direction_comparison`

## 4) Retraining cadence
- Daily: monitor predictions and tracking.
- Weekly: refresh data/features.
- Bi-weekly: retrain ARIMA/GARCH and tree/sequence models as needed.
- Monthly: walk-forward + ensemble + backtest review.
- Quarterly: full retrain + risk validation.

## 5) Delisted ticker maintenance
```bash
curl -s 'http://localhost:5001/api/delisted-tickers?status=delisted_candidate' | jq
curl -L 'http://localhost:5001/api/delisted-tickers/export.csv' -o delisted_tickers.csv
```

Use this list to replace failing symbols in `config/tickers.yaml` before retraining.

## 6) CI and local gate before push
```bash
pytest -q
black --check src/inference/predictor.py webapp/server.py webapp/prediction_tracker.py tests/test_expected_vs_actual.py tests/test_prediction_endpoints.py
flake8 src/inference/predictor.py webapp/server.py webapp/prediction_tracker.py tests/test_expected_vs_actual.py tests/test_prediction_endpoints.py --max-line-length=120 --extend-ignore=E203,W503 || true
mypy --explicit-package-bases src/inference/predictor.py webapp/prediction_tracker.py webapp/server.py --ignore-missing-imports --follow-imports=silent || true
```

GitHub Actions workflow is located at repo root:
- `../.github/workflows/ci.yml`

## 7) Branch and PR flow
From `/Users/anto/Trading_Project`:
```bash
git checkout -b feature/predictor-productionize
git add .
git commit -m "productionize: sanitize predictions, premarket/outlook, expected-vs-actual persistence, tests, cleanup, docs"
git push -u origin feature/predictor-productionize
```

Open PR to `main`, wait for CI, then merge.
