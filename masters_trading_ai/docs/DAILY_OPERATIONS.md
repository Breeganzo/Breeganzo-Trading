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
- `strategy_price_at_open`
- `ai_last_prediction`
- `actual_close`
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
black --check src webapp tests
flake8 src webapp tests --max-line-length=120 --extend-ignore=E203,W503
mypy src/inference/predictor.py webapp/prediction_tracker.py webapp/server.py --ignore-missing-imports --follow-imports=silent
```

## 7) Branch and PR flow
From `/Users/anto/Trading_Project`:
```bash
git checkout -b feature/predictor-productionize
git add .
git commit -m "productionize: sanitize predictions, premarket/outlook, expected-vs-actual persistence, tests, cleanup, docs"
git push -u origin feature/predictor-productionize
```

Open PR to `main`, wait for CI, then merge.
