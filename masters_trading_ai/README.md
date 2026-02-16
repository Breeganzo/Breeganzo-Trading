# Masters Trading AI

Production-focused NSE trading dashboard with:
- live yfinance prices,
- ensemble model predictions,
- premarket strategy snapshots,
- expected-vs-actual tracking,
- portfolio-only risk analytics,
- Groq-assisted explainability.

Project explainer for presentations and branch/CI flow:
- `docs/PROJECT_WORKFLOW_GUIDE.md`

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

## 2) Run locally
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
python webapp/server.py
```

Open:
- `http://localhost:5001` (Dashboard)
- `http://localhost:5001/portfolio`
- `http://localhost:5001/risk`

If port 5001 is busy:
```bash
lsof -iTCP:5001 -sTCP:LISTEN -n -P
kill $(lsof -tiTCP:5001 -sTCP:LISTEN)
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

## 5) Market-open snapshot behavior
Config (`config/settings.yaml`):
```yaml
webapp:
  premarket_max_buffer_minutes: 30
  premarket_default_tickers: 10
  after_hours_ui_hour: 16
  after_hours_ui_minute: 0
```

Snapshot contract:
- snapshot source is pinned to market-open flow (`09:15-09:30 IST`) with `snapshot_type: "market_open"`.
- if app starts late, fallback uses `snapshot_type: "near_open_fallback"` (same trading day only).
- before open, API may return `snapshot_type: "pending_market_open"` and empty items.

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
- `snapshot_type`
- `schema_version`

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

### `/api/expected-vs-actual`
```bash
curl -s 'http://localhost:5001/api/expected-vs-actual?date=2026-02-14' | jq
```

Includes:
- `strategy_price_at_open`
- `ai_last_prediction`
- `actual_close`
- `direction_comparison`
- `after_hours_mode`
- `snapshot_type`
- `schema_version`

### `/api/training-feedback`
```bash
curl -s 'http://localhost:5001/api/training-feedback' | jq
```

Export schema is retraining-friendly and includes direction comparison fields.

## 7) Dashboard behavior
- Top Picks supports dropdown: `Top 10 BUY`, `Top 10 SELL`, `Top 10 HOLD`.
- New table: **Premarket vs Current**.
- New table: **Current Second Snapshot** for highlighted ticker.
- `0/null` prices render as `—` (not fake `₹0`).
- After-hours mode (>= 16:00 IST or market closed) auto-locks premarket widgets and switches to EoD comparison context.

## 8) Groq explanation UX
- Metric explainers use persistent scrollable popovers.
- Popovers are keyboard accessible (`Enter`/`Space` to open, `Esc` to close).
- Long responses are clipped and normalized backend-side for UI readability.

## 9) Risk analytics behavior
- `/api/risk-analytics` now uses **only user portfolio tickers**.
- Non-portfolio requested tickers are ignored and reported in `ignored_tickers`.
- No fallback to demo/model-pick portfolios.
- Monte Carlo simulation is driven by the portfolio equity curve (`simulation_source: "equity_curve"`).
- Monte Carlo enforces minimum history (`>= 30` daily points in API flow).

## 10) Delisted ticker registry
Unavailable symbols are tracked in:
- `cache/delisted_tickers.csv`

Use:
```bash
curl -s 'http://localhost:5001/api/delisted-tickers?min_hits=1' | jq
curl -L 'http://localhost:5001/api/delisted-tickers/export.csv' -o delisted_tickers.csv
```

Before retraining, replace repeated failures in `config/tickers.yaml`.

## 11) Tests, lint, type-check
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
pytest -q
black --check \
  src/backtest/metrics.py \
  webapp/server.py \
  webapp/prediction_tracker.py \
  webapp/groq_explainer.py \
  tests/test_premarket_outlook.py \
  tests/test_expected_vs_actual.py \
  tests/test_premarket_snapshot.py \
  tests/test_risk_monte_carlo.py \
  tests/test_tooltip_popover.py \
  tests/test_artifact_cleanup.py
flake8 \
  tests/test_premarket_outlook.py \
  tests/test_expected_vs_actual.py \
  tests/test_premarket_snapshot.py \
  tests/test_risk_monte_carlo.py \
  tests/test_tooltip_popover.py \
  tests/test_artifact_cleanup.py \
  --max-line-length=120 --extend-ignore=E203,W503
mypy src/backtest/metrics.py tests/test_premarket_snapshot.py tests/test_risk_monte_carlo.py --ignore-missing-imports --follow-imports=silent
```

## 12) CI
GitHub Actions workflow:
- `.github/workflows/ci.yml`

Runs on `masters_trading_ai/**` changes:
- install deps
- `pytest -q`
- `black --check`
- `flake8`
- `mypy`

## 13) Branch workflow (feature -> main)
From project root (`/Users/anto/Trading_Project`):

```bash
git checkout -b feature/ui-risk-premarket-fixes
# make changes
git add .
git commit -m "ui+backend: market-open snapshots, scrollable Groq popovers, risk/montecarlo fixes, cleanup, tests, docs"
git push -u origin feature/ui-risk-premarket-fixes
```

Create PR to `main`.
After approval + green CI, merge PR.

## 14) Architecture (data flow)
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

## 15) Verification Runbook
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
pytest -q
python webapp/server.py
curl -s 'http://localhost:5001/api/premarket-outlook' | jq
curl -s \"http://localhost:5001/api/expected-vs-actual?date=$(date +%F)\" | jq
curl -s 'http://localhost:5001/api/risk-analytics' | jq
```

## 16) Retraining cadence
- Daily: run app + monitor tracking.
- Weekly: data/feature refresh.
- Bi-weekly: retrain sequence/tree models.
- Monthly: walk-forward + ensemble + backtest review.
- Quarterly: full pipeline refresh.

See `DAILY_OPERATIONS.md` for detailed runbook.

## 17) Trading disclaimer
This system supports decision quality, not guaranteed profit. Use risk limits, monitor regime shifts, and keep ongoing retraining/backtesting in loop.
