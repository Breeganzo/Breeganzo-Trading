# Masters Trading AI

AI-powered NSE dashboard with live prices, ensemble predictions, stock-level analysis, risk analytics, and Groq-assisted explanations.

## 1) Quick setup
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set minimum `.env` values:
```bash
GROQ_API_KEY=your_groq_key
FLASK_SECRET_KEY=change-this
```

## 2) Run locally
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
python webapp/server.py
```

Do not run `webapp/server.py` directly (zsh will show permission denied).

Open:
- Dashboard: `http://localhost:5001`
- Portfolio: `http://localhost:5001/portfolio`
- Risk Analytics: `http://localhost:5001/risk`

## 3) Verify app + predictions
```bash
curl -s 'http://localhost:5001/api/status' | jq
curl -s 'http://localhost:5001/api/top-picks?n=20&grouped=true' | jq
curl -s 'http://localhost:5001/api/price-tracker/RELIANCE.NS' | jq
curl -s 'http://localhost:5001/api/groq-price-forecast/RELIANCE.NS' | jq
```

## 4) Delisted / unavailable ticker tracking (new)
When yfinance repeatedly fails for a ticker, the app now writes it to:
- `cache/delisted_tickers.csv`

CSV columns:
- `ticker`
- `first_seen`
- `last_seen`
- `hit_count`
- `last_reason`
- `last_source`
- `status` (`watchlist`, `delisted_candidate`, `recovered`)

Rules:
- first/second failure: `watchlist`
- 3+ failures: `delisted_candidate`
- if data later appears again: `recovered`

APIs:
```bash
curl -s 'http://localhost:5001/api/delisted-tickers?min_hits=1' | jq
curl -s 'http://localhost:5001/api/delisted-tickers?status=delisted_candidate' | jq
curl -L 'http://localhost:5001/api/delisted-tickers/export.csv' -o delisted_tickers.csv
```

Before retraining, review `delisted_candidate` symbols and replace them in `config/tickers.yaml`.

## 5) Model rerun/retraining cadence (avoid running everything at once)

### Daily (market days)
- Run app + generate fresh predictions.
- Use `run_daily.sh`.

### Weekly (1x, weekend)
- Data + feature refresh:
  - `notebooks/01_data_download.ipynb`
  - `notebooks/03_feature_engineering.ipynb`
- Quick diagnostics:
  - `notebooks/11_model_comparison.ipynb`

### Every 2 weeks
- Refresh tree/sequence models:
  - `notebooks/07_xgboost.ipynb`
  - `notebooks/08_lightgbm.ipynb`
  - `notebooks/09_lstm.ipynb`

### Monthly
- Validate robustness and rebalance ensemble:
  - `notebooks/04_walk_forward_cv.ipynb`
  - `notebooks/12_ensemble.ipynb`
  - `notebooks/13_backtest.ipynb`

### Quarterly
- Heavy refresh and full report:
  - `notebooks/10_transformer.ipynb`
  - `notebooks/14_risk_analytics.ipynb`
  - `notebooks/17_final_report.ipynb`

### Trigger immediate retraining if any of these happen
- 20-day hit-rate drops below your threshold (example: <52%).
- Market regime shift (volatility spike/event risk).
- More than 10% universe changes in `config/tickers.yaml`.

## 6) Why AI price and Strategy price can look close
- `Strategy predicted price` = model output based on open/current context.
- `AI predicted price` = Groq narrative forecast with sentiment context.
- If Groq JSON parse fails, app now shows AI as unavailable instead of silently copying strategy price.

## 7) Tests
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
pytest -q
```

## 8) Troubleshooting

### Port 5001 already in use
```bash
lsof -iTCP:5001 -sTCP:LISTEN -n -P
kill $(lsof -tiTCP:5001 -sTCP:LISTEN)
```

### Start on another port
```bash
FLASK_PORT=5002 python webapp/server.py
```

### Models still loading
- Wait for `/api/status` => `models_loaded=true`.
- UI navbar also shows load progress.

## 9) Important reality check
This system can improve decision quality, but it cannot guarantee profit, and it cannot be made “perfect” for 5-10 years without ongoing maintenance. For production-grade use:
- keep strict position sizing and max drawdown limits,
- monitor live slippage/transaction costs,
- run monthly walk-forward validation,
- treat Groq outputs as explanatory context, not guaranteed price truth.
