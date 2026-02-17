# Masters Trading AI

Production-focused NSE trading dashboard with:
- live yfinance prices,
- ensemble model predictions,
- premarket strategy snapshots,
- trading-desk advisor simulation (budget + fee constrained),
- expected-vs-actual tracking,
- portfolio-only risk analytics,
- Groq-assisted explainability.

## 1) Setup
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Minimum `.env` values:
```bash
GROQ_API_KEY=your_groq_key
FLASK_SECRET_KEY=change-this
```

Optional Groq failover keys:
```bash
GROQ_API_KEY_2=...
GROQ_API_KEY_3=...
GROQ_KEY_ROTATION_ENABLED=1
GROQ_MODEL=llama-3.3-70b-versatile
# Optional fallback model chain:
GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant
```

## 2) Run locally
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
python webapp/server.py
```

Open:
- `http://localhost:5001` (Dashboard)
- `http://localhost:5001/advisor` (Strategy Advisor simulation page)
- `http://localhost:5001/portfolio`
- `http://localhost:5001/risk`

If port 5001 is busy:
```bash
lsof -iTCP:5001 -sTCP:LISTEN -n -P
kill $(lsof -tiTCP:5001 -sTCP:LISTEN)
```

If you see `OSError: [Errno 24] Too many open files`:
```bash
ulimit -n 4096
```
and set conservative runtime knobs in `.env`:
```bash
FLASK_THREADED=1
YF_MAX_PARALLEL=2
YF_CALL_TIMEOUT_SEC=15
FD_SOFT_GUARD_RATIO=0.85
DAILY_ANALYSIS_TTL_SEC=90
PRICE_CACHE_TTL_SEC=30
```

## 3) Model-loading transparency
Use:
```bash
curl -s 'http://localhost:5001/api/status' | jq
```

It returns:
- `models_loaded`, `models_loading`, `model_load_elapsed_sec`, `model_load_progress`
- `premarket_config.max_buffer_minutes`
- `premarket_snapshot` metadata

## 4) Prediction unit contract (critical)
- Internal model return unit: **decimal** (`0.02` = +2%).
- API `predicted_return` unit: **percent** (`2.0` = +2%).
- Sanitizer guarantees:
  - `None/NaN` -> `0.0` (decimal)
  - if `abs(raw) > 2.0`, converts mistaken percent-to-decimal by `/100`
  - caps to `[-0.5, 0.5]` decimal (±50%)

## 5) Premarket behavior
Config (`config/settings.yaml`):
```yaml
webapp:
  premarket_max_buffer_minutes: 30
  premarket_default_tickers: 10
```

Snapshot windows (IST, server-side):
- `00:00-09:15`: `after_hours_live` (AI/strategy can refresh and fluctuate)
- `09:15-09:30`: `premarket_open` snapshot
- `09:30-15:30`: `market_open_locked` strategy snapshot (frozen)
- `15:30-next day 00:00`: `after_hours_live` (AI/strategy can refresh and fluctuate)

Scheduled precompute slots (IST):
- `09:28`: warm premarket/open snapshots for instant UI load
- `15:31`: warm after-hours snapshot for close-to-next-session flow

Timestamp behavior:
- In live windows (`after_hours_live`), displayed prediction timestamps use real generation time.
- During open-window fields (`*_at_open`) inside lock windows, timestamps are normalized to the 09:15-09:30 IST band.

Use latest stored snapshot:
```bash
curl -s 'http://localhost:5001/api/premarket-outlook?use_latest_stored=true' | jq
```

Endpoint:
```bash
curl -s 'http://localhost:5001/api/premarket-outlook' | jq
```

Response includes per ticker:
- `ticker`
- `current_price`
- `strategy_price_at_open`
- `ai_predicted_price`
- `strategy_direction`
- `ai_direction`
- `captured_at`

## 6) API contracts

### `/api/top-picks`
```bash
curl -s 'http://localhost:5001/api/top-picks?n=50' | jq
curl -s 'http://localhost:5001/api/top-picks?sectors=large_cap&grouped=true&n=10' | jq
```

Guarantees:
- rows with `current_price <= 0` are filtered out
- `abs(predicted_return) <= 50`
- `predicted_price` aligns with return contract
- max `n` is capped at `50`
- ranking is confidence-first, then score:
  - `score = |predicted_return_decimal| × (confidence/100) × (model_agreement/100) × liquidity_factor`

Top Picks vs DailyAdvisor:
- `Top Picks`: actionable ranked market list (BUY/SELL/HOLD filters).
- `DailyAdvisor` (`/advisor`): strategy execution simulation board with BUY candidates and post-buy HOLD/SOLD tracking.

### `/api/advisor/open-buy-list`
```bash
curl -s 'http://localhost:5001/api/advisor/open-buy-list?n=10&budget=40000' | jq
```

Returns strategy-only open-buy suggestions constrained by `budget` + estimated entry fees:
- `ticker`, `strategy_price_at_open`, `suggested_qty`
- `est_trade_cost`, `stop_loss_price`, `risk_reward`
- `liquidity_factor`, `avg_volume_30d`, `volatility_atr_pct`
- `source = strategy`
- Dynamic risk fields:
  - `stop_loss_pct` (ATR + confidence policy)
  - `target_price`
  - `sentiment_weighted_score`

### `/api/simulate/trade` and `/api/simulate/portfolio`
```bash
# Buy simulation
curl -s -X POST 'http://localhost:5001/api/simulate/trade' \
  -H 'Content-Type: application/json' \
  -d '{"action":"BUY","ticker":"INFY.NS","quantity":5,"price":1400,"stop_loss_price":1360,"target_price":1450}' | jq

# Auto stop-loss/target checks
curl -s -X POST 'http://localhost:5001/api/simulate/trade' \
  -H 'Content-Type: application/json' \
  -d '{"action":"AUTO_CHECK","auto_buy":true}' | jq

# Read simulated portfolio
curl -s 'http://localhost:5001/api/simulate/portfolio' | jq
```

Simulation state is persisted in:
- `cache/portfolio_sim.json`
- `cache/prediction_log/simulated_trades.jsonl`
- `cache/prediction_log/simulated_trades.csv`
- `cache/prediction_log/simulated_trades.xlsx` (when `openpyxl` is installed, one worksheet per IST date)

Simulation risk controls:
- ATR + confidence dynamic stop-loss
- position sizing by risk-per-trade and max-position cap
- trailing-stop updates after favorable move
- daily loss circuit breaker (`MAX_DAILY_LOSS`)
- BUY/SELL triggers are allowed only during `09:30-15:30 IST` on trading days
- AUTO_CHECK outside market hours returns a safe "skipped" response (no trades fired)
- SELL before BUY is blocked by quantity-sequence validation.
- auto-sell triggers on:
  - stop-loss hit,
  - target hit,
  - strategy signal turns `SELL` (strategy-first logic)
- same-cycle reinvestment:
  - after simulated sell frees cash, auto-buy optimizer can allocate into fresh strategy BUY candidates during market hours.
- strategy-first with sentiment adjustment:
  - baseline action comes from strategy signal;
  - same-day relevant sentiment can downgrade/upgrade action conservatively.
- portfolio optimizer:
  - risk-constrained utility optimization (budget cap, concentration cap, risk budget, fee-aware quantity rounding).
- simulation-to-portfolio sync:
  - each simulated BUY/SELL is mirrored into `cache/portfolio_trades.json` so Portfolio updates automatically.

Important: this is simulation only; no live brokerage order is sent.

### `/api/simulate/transactions-summary` and `/api/simulate/daily-report`
```bash
curl -s 'http://localhost:5001/api/simulate/transactions-summary' | jq
curl -s 'http://localhost:5001/api/simulate/daily-report' | jq
```

Returns:
- total transaction counts (BUY/SELL)
- buy/sell/total transaction costs
- total paid for buys, total received from sells, net cash flow
- realized PnL, recent transaction rows
- report file path for daily 15:45 IST summary

### `/api/expected-vs-actual`
```bash
curl -s 'http://localhost:5001/api/expected-vs-actual?date=2026-02-14' | jq
```

Includes:
- `open_price`, `close_price`, `actual_close`
- `strategy_price_at_open`
- `ai_last_prediction` (`ai_price_at_open`)
- `strategy_return_pct`, `ai_return_pct`, `actual_return_pct`
- `alpha_pct` (actual vs strategy, open-relative)
- `direction_comparison`
- `schema_version`

### `/api/strategy-price/<ticker>`
```bash
curl -s 'http://localhost:5001/api/strategy-price/INFY.NS' | jq
```

Returns strategy-only fields:
- `strategy_price`
- `rr_ratio`
- `strategy_generated_at`
- `predicted_return_decimal`

### `/api/debug/prediction-status/<ticker>`
```bash
curl -s 'http://localhost:5001/api/debug/prediction-status/INFY.NS' | jq
```

Provides cache/snapshot diagnostics and formula consistency checks.

### `/api/training-feedback`
```bash
curl -s 'http://localhost:5001/api/training-feedback' | jq
```

Export schema is retraining-friendly and includes direction comparison fields.

## 7) Dashboard behavior
- Top Picks supports dropdown: `BUY`, `SELL`, `HOLD` (SELL/HOLD restricted to portfolio symbols).
- Top Picks page is focused on actionable ranking + premarket/current snapshot.
- DailyAdvisor page (`/advisor`) is focused on strategy simulation execution and ledger review.
- New table: **Premarket vs Current**.
- DailyAdvisor includes:
  - default simulation cash ₹40,000,
  - BUY/SOLD/HOLD view selector,
  - open-buy list constrained by budget + estimated fees,
  - simulated BUY/SELL and auto stop-loss/target checks,
  - transaction totals (count + buy/sell/both costs) from server summary endpoint,
  - 1-second status refresh while page is visible.
- `0/null` prices render as `—` (not fake `₹0`).

Email notifications/report (optional):
- `TRADE_EMAIL_ENABLED=true`
- `TRADE_EMAIL_TO=anthonybreeganzo07@gmail.com`
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`
- Daily report scheduler defaults to `15:45 IST`:
  - `TRADE_DAILY_REPORT_HOUR_IST=15`
  - `TRADE_DAILY_REPORT_MINUTE_IST=45`

## 8) Risk analytics behavior
- `/api/risk-analytics` now uses **only user portfolio tickers**.
- Non-portfolio requested tickers are ignored and reported in `ignored_tickers`.
- No fallback to demo/model-pick portfolios.

## 9) Delisted ticker registry
Unavailable symbols are tracked in:
- `cache/delisted_tickers.csv`

Use:
```bash
curl -s 'http://localhost:5001/api/delisted-tickers?min_hits=1' | jq
curl -L 'http://localhost:5001/api/delisted-tickers/export.csv' -o delisted_tickers.csv
```

Before retraining, replace repeated failures in `config/tickers.yaml`.

## 10) Tests, lint, type-check
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
pytest -q
black --check src/inference/predictor.py webapp/server.py webapp/prediction_tracker.py tests/test_expected_vs_actual.py tests/test_prediction_endpoints.py
flake8 src/inference/predictor.py webapp/server.py webapp/prediction_tracker.py tests/test_expected_vs_actual.py tests/test_prediction_endpoints.py --max-line-length=120 --extend-ignore=E203,W503 || true
mypy --explicit-package-bases src/inference/predictor.py webapp/prediction_tracker.py webapp/server.py --ignore-missing-imports --follow-imports=silent || true
```

## 11) CI
GitHub Actions workflow:
- `../.github/workflows/ci.yml` (repo root)

Runs on `masters_trading_ai/**` changes:
- install deps
- `pytest -q`
- `black --check`
- `flake8`
- `mypy`

## 11.1) Groq API Usage Controls
This codebase uses caching and hard limits to avoid Groq overuse:
- UI explanation cache (`webapp/groq_explainer.py`): prompt cache + endpoint queue
- strict limits: `GROQ_GLOBAL_MAX_PER_MIN` and `GROQ_ENDPOINT_MAX_PER_MIN`
- multi-key failover: `GROQ_API_KEY` -> `GROQ_API_KEY_2` -> `GROQ_API_KEY_3` (or `GROQ_API_KEYS`)
- sticky-key behavior: keep using current key until it hits 429 exhaustion, then switch to next
- controls: `GROQ_KEY_ROTATION_ENABLED`
- model fallback chain: `GROQ_MODELS` (tries next model on same key before failing over key)
- AI forecast cache (`webapp/server.py`): `GROQ_FORECAST_TTL=900s`
- News-sentiment scoring cache (`src/inference/predictor.py`): `SENTIMENT_CACHE_TTL_SECONDS` (default 1800s)
- News-sentiment call budget: `GROQ_SENTIMENT_MAX_CALLS_PER_MINUTE` (default 8)
- Toggle sentiment Groq usage: `ENABLE_GROQ_NEWS_SENTIMENT` (`1`/`0`)
- degraded mode: if Groq returns `429`, app enters cached-only mode and shows a UI banner

Groq health endpoint:
```bash
curl -s 'http://localhost:5001/api/groq-status' | jq
```

Recommended `.env` knobs:
```bash
GROQ_API_KEY=...
GROQ_API_KEY_2=
GROQ_API_KEY_3=
GROQ_KEY_ROTATION_ENABLED=1
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant
ENABLE_GROQ_NEWS_SENTIMENT=1
GROQ_SENTIMENT_MAX_CALLS_PER_MINUTE=8
SENTIMENT_CACHE_TTL_SECONDS=1800
YF_NEWS_TTL_SECONDS=300
ADVISOR_MAX_PER_SECTOR=3
```

## 12) Branch workflow (feature -> main)
From project root (`/Users/anto/Trading_Project`):

```bash
git checkout -b feature/predictor-productionize
# make changes
git add .
git commit -m "productionize: sanitize predictions, premarket/outlook, expected-vs-actual persistence, tests, cleanup, docs"
git push -u origin feature/predictor-productionize
```

Create PR to `main`.
After approval + green CI, merge PR.

## 13) Architecture (data flow)
```mermaid
flowchart LR
    YF[yfinance Live + Daily] --> FP[FeaturePipeline]
    FP --> M1[XGBoost]
    FP --> M2[LightGBM]
    FP --> M3[LSTM]
    FP --> M4[Transformer]
    FP --> M5[ARIMA/GARCH]
    M1 --> ENS[Ensemble]
    M2 --> ENS
    M3 --> ENS
    M4 --> ENS
    M5 --> ENS
    ENS --> PRED[Predictor + Sanitizer]
    PRED --> API[Flask API]
    API --> UI[Dashboard UI]
    PRED --> TRACK[PredictionTracker]
    TRACK --> FB[/api/training-feedback]
```

## 14) Retraining cadence
- Daily: run app + monitor tracking.
- Weekly: data/feature refresh.
- Bi-weekly: retrain sequence/tree models.
- Monthly: walk-forward + ensemble + backtest review.
- Quarterly: full pipeline refresh.

Early retrain triggers:
- KS-test drift alert on core features.
- Direction hit-rate drop > 7% week-over-week.
- Sustained alpha degradation for 10+ trading sessions.
- Portfolio risk metrics degrade (Sharpe collapse / drawdown regime shift).

See `DAILY_OPERATIONS.md` for detailed runbook.

## 15) Trading disclaimer
This system supports decision quality, not guaranteed profit. Use risk limits, monitor regime shifts, and keep ongoing retraining/backtesting in loop.
All advisor automation paths in this app are simulation-only.
