# Project + Workflow Guide

## 1) What this project does
`masters_trading_ai` is a production-style trading analytics app for NSE symbols. It combines:
- market data ingestion (`yfinance` + cached snapshots),
- ML strategy inference (ensemble + model fallback),
- UI/API serving through Flask,
- prediction tracking (expected vs actual),
- portfolio-only risk analytics,
- Groq-powered explanations for indicators/risk/strategy outputs.

Core idea:
- Internal prediction math uses **decimal returns** (`0.02` = 2%).
- API/UI return field `predicted_return` uses **percent** (`2.0` = 2%).
- Sanitizers/guards prevent runaway values and placeholder data from appearing as valid signals.

## 2) Important files and what each one is for

### Inference and model flow
- `src/inference/predictor.py`
  - Main prediction engine.
  - Loads models, computes ensemble return, sanitizes return units, computes predicted price.
  - Produces top-picks with checks (no zero/invalid prices).

### Server/API layer
- `webapp/server.py`
  - Flask app and API endpoints.
  - Provides `top-picks`, `premarket-outlook`, `expected-vs-actual`, risk endpoints, status endpoints.
  - Handles market-time mode logic and response shaping.

### Tracking/persistence
- `webapp/prediction_tracker.py`
  - Stores prediction snapshots and expected-vs-actual outcomes.
  - Persists fields needed for retraining feedback.

### Frontend
- `webapp/static/js/app.js`
  - Dashboard rendering, top picks, premarket/current comparisons.
- `webapp/static/js/stock.js`
  - Per-stock screen rendering and explainers.
- `webapp/static/js/risk.js`
  - Risk dashboard rendering and Monte Carlo related views.
- `webapp/templates/*.html`
  - Page layouts.
- `webapp/static/css/style.css`
  - Shared UI styling (including popover behavior).

### Backtest/risk internals
- `src/backtest/metrics.py`
  - Monte Carlo/equity metrics and related calculations.
- `src/portfolio/risk_manager.py`
  - Portfolio-level risk/stat functions.

### Config and data
- `config/settings.yaml`
  - Runtime knobs such as premarket buffer and UI timing.
- `config/tickers.yaml`
  - Universe and ticker groups (including large-cap).
- `cache/predictions.json`
  - Cached predictions.
- `cache/delisted_tickers.csv`
  - Delisted/unavailable symbol tracking for future retraining cleanup.

### Tests
- `tests/`
  - Unit + API + regression tests for predictor sanity, premarket logic, expected-vs-actual, risk, cleanup, etc.

## 3) GitHub workflow files and what they do
Workflow files are in repository root: `.github/workflows/`

- `.github/workflows/ci.yml`
  - Main quality gate for this project.
  - Installs dependencies and runs:
    - `pytest -q`
    - `black --check` (selected files)
    - `flake8` (selected files)
    - `mypy` (selected files)

- `.github/workflows/keep-alive.yml`
  - Utility workflow (keeps scheduled jobs active).

- `.github/workflows/morning-rankings.yml`
  - Scheduled/report style workflow for ranking jobs.

## 4) When CI runs (important)
`ci.yml` trigger config:
- `on: push`
- `on: pull_request`
- with path filters:
  - `masters_trading_ai/**`
  - `.github/workflows/ci.yml`

Meaning:
- CI runs on **any branch** (not just `main`) **if** pushed/PR changes touch `masters_trading_ai` or the CI file.
- If a push only changes files outside these paths, CI job will not trigger.

## 5) Local pre-push checks (same as CI intent)
From `masters_trading_ai`:

```bash
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
mypy \
  src/backtest/metrics.py \
  tests/test_premarket_snapshot.py \
  tests/test_risk_monte_carlo.py \
  --ignore-missing-imports --follow-imports=silent
```

## 6) Branching model to follow (dev -> test -> prod -> main)
Use these long-lived branches:
- `dev-1`
- `test-2`
- `prod-3`
- `main`

Promotion order:
1. Work and validate in `dev-1`.
2. Promote to `test-2` for test/UAT validation.
3. Promote to `prod-3` for release readiness.
4. Promote to `main` only after all checks pass.

### Step-by-step commands (standard flow)
From repo root: `/Users/anto/Trading_Project`

1. Start in `dev-1`:
```bash
git checkout dev-1
git pull
# make changes
git add .
git commit -m "your change"
git push origin dev-1
```

2. Promote `dev-1` -> `test-2`:
```bash
git checkout test-2
git pull
git merge --no-ff dev-1 -m "promote: dev-1 -> test-2"
git push origin test-2
```

3. Promote `test-2` -> `prod-3`:
```bash
git checkout prod-3
git pull
git merge --no-ff test-2 -m "promote: test-2 -> prod-3"
git push origin prod-3
```

4. Promote `prod-3` -> `main`:
```bash
git checkout main
git pull
git merge --no-ff prod-3 -m "promote: prod-3 -> main"
git push origin main
```

## 7) Verify CI status after push
- Open GitHub -> Actions tab.
- Confirm `CI` workflow ran for the pushed branch.
- If no run appears, confirm your commit touched `masters_trading_ai/**` or `.github/workflows/ci.yml`.

## 8) Quick explanation script (for demos/interviews)
Use this line:

"This project takes live market data, runs strategy + ML inference with strict unit checks, serves ranked opportunities via Flask APIs/UI, tracks expected-vs-actual outcomes for feedback retraining, and enforces quality with automated CI on every relevant branch push and PR."
