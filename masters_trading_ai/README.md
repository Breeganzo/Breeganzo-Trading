# Masters Trading AI

AI-powered NSE dashboard with live prices, multi-model inference, ensemble predictions, top picks (BUY/SELL/HOLD), stock-level analysis, and risk analytics.

## What is included
- 6-model stack: `ARIMA`, `GARCH`, `XGBoost`, `LightGBM`, `LSTM`, `Transformer`
- Ensemble prediction with learned weights
- Prediction sanitizer and guardrails for runaway returns
- Top Picks split by actual signal (`BUY`, `SELL`, `HOLD`)
- Stock page with:
  - Open price
  - Strategy predicted price
  - Current price
  - Groq AI predicted price
- Portfolio tracking (add ticker + quantity + entry price)
- Risk analytics with Groq hover explainers

## 1) Setup (first time)
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from example:
```bash
cp .env.example .env
```

Minimum recommended in `.env`:
```bash
GROQ_API_KEY=your_groq_key
FLASK_SECRET_KEY=dev-secret-change-me
```

## 2) Run the app

### Option A: Start Flask server directly (recommended for development)
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
python webapp/server.py
```

If you are already inside `masters_trading_ai`:
```bash
source .venv/bin/activate
python webapp/server.py
```

Do NOT run:
```bash
webapp/server.py
```
That runs it as a shell command and causes `zsh: permission denied`.

Open:
- Dashboard: `http://localhost:5001`
- Risk page: `http://localhost:5001/risk`

### Option B: Use daily helper scripts
```bash
./run_daily.sh
```
or
```bash
./start_daily.sh
```

## 3) Verify it is working
In another terminal:
```bash
curl -s 'http://localhost:5001/api/status' | jq
curl -s 'http://localhost:5001/api/top-picks?n=20&grouped=true' | jq
curl -s 'http://localhost:5001/api/daily-analysis' | jq
```

You should see:
- `models_loaded: true`
- model load steps populated
- grouped top picks with `top_buy`, `top_sell`, `top_hold`

## 4) Run tests
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
pytest -q
```

## 5) Push changes to `main` (GitHub)

Use this flow from repo root (`/Users/anto/Trading_Project`):

```bash
# 1) Check branch and changes
git branch --show-current
git status

# 2) Create/switch to your feature branch (example)
git checkout -b fix/predictor-sanity

# 3) Stage and commit
git add .
git commit -m "your commit message"

# 4) Push branch to GitHub
git push -u origin fix/predictor-sanity
```

Open PR to merge into `main`:

```text
https://github.com/Breeganzo/Breeganzo-Trading/compare/main...fix/predictor-sanity?expand=1
```

After PR approval/merge, sync local `main`:

```bash
git checkout main
git pull origin main
```

## 6) Troubleshooting

### Port already in use
```bash
lsof -iTCP:5001 -sTCP:LISTEN -n -P
pkill -f "webapp/server.py"
```

### `zsh: permission denied: webapp/server.py`
Use Python to run it:
```bash
python webapp/server.py
```
or:
```bash
python3 webapp/server.py
```

### Models still loading
- Wait for `/api/status` to show `models_loaded=true`
- The navbar status in UI shows per-model load progress

### yfinance rate-limit issues
- Retry after a short delay; some endpoints depend on live market data
- You may temporarily see reduced ticker coverage until retry succeeds

### Groq unavailable
- Check `GROQ_API_KEY` in `.env`
- App still runs without Groq, but AI explanation/forecast endpoints degrade gracefully

## Key paths
- Server: `webapp/server.py`
- Inference: `src/inference/predictor.py`
- Dashboard JS: `webapp/static/js/app.js`
- Stock page JS: `webapp/static/js/stock.js`
- Risk page JS: `webapp/static/js/risk.js`
- Tests: `tests/`
