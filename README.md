# Trading_Project

This repository contains a full trading platform split into reusable modules:

- `frontend/`: Next.js UI (dashboard, portfolio, risk, orders, trades)
- `backend/`: FastAPI APIs (auth, portfolio, trades, orders, rankings, risk, live ticker)
- `masters_trading_ai/`: strategy + ML inference/simulation app
- `docs/`: setup and architecture docs
- `deployment/`: deployment configs

## What The Code Does

- Fetches live market prices and computes analytics.
- Tracks portfolio, trades, transaction costs, and risk metrics.
- Supports AI/risk explanations and ranking flows.
- Supports DB-based auto signal triggers (`BUY`/`SELL`/`HOLD`) with server-side background processing.
- Includes a separate ML/strategy simulation app (`masters_trading_ai`).
- Stores daily live market snapshots in database and supports CSV/XLSX export.
- Supports optional SMTP trade-notification emails for executed BUY trades.

## Trading_Project vs masters_trading_ai Webapp

- `Trading_Project` (`frontend` + `backend`) is the modular production-style app:
  - Next.js dashboard on `3000`
  - FastAPI APIs on `8000`
  - Supabase + Redis + optional Groq AI explanations
- `masters_trading_ai/webapp` is the strategy/ML sandbox app:
  - Flask app on `5001`
  - tighter coupling to local model inference/simulation files
  - useful for rapid strategy experiments and ML-focused workflows

In short: use `Trading_Project` for reusable frontend/backend/database architecture; use `masters_trading_ai` for strategy-model experimentation.

## Live Local URLs (running now)

- Frontend: `http://localhost:3000/dashboard`
- Backend API docs: `http://localhost:8000/docs`
- Backend health: `http://localhost:8000/health`

## Local Run (using venv)

### 1) Backend (FastAPI)

```bash
cd /Users/anto/Trading_Project/backend
source venv/bin/activate

# Production-like mode (Google OAuth enabled)
export AUTH_BYPASS_LOCAL=false

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Notes:
- Background auto-signal worker runs in backend startup.
- To enforce market-hours-only processing, keep `AUTO_SIGNAL_WORKER_MARKET_HOURS_ONLY=true`.
- To enable Groq explanations, set `GROQ_API_KEY` (and optional `GROQ_MODEL`) in `backend/.env`.
- For local bypass only (do not use in production): set `AUTH_BYPASS_LOCAL=true`.

### 2) Frontend (Next.js)

```bash
cd /Users/anto/Trading_Project/frontend

# first time only
npm install

# run against local backend
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1 \
NEXT_PUBLIC_WS_URL=ws://localhost:8000/api/v1/ticker/ws \
NEXT_PUBLIC_AUTH_BYPASS_LOCAL=false \
npx next dev -H 0.0.0.0 -p 3000
```

Open: `http://localhost:3000/dashboard`

### 3) Strategy/ML app (optional, separate runtime)

```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
python webapp/server.py
```

Open: `http://localhost:5001`

## If You Want Google Auth Again

Set local bypass off (recommended for production):

Backend:
```bash
export AUTH_BYPASS_LOCAL=false
```

Frontend:
```bash
export NEXT_PUBLIC_AUTH_BYPASS_LOCAL=false
```

Then configure Google OAuth env values in `backend/.env` and use the `/login` flow.

## New Live Data APIs

- `GET /api/v1/ticker/stocks/overview`
- `POST /api/v1/ticker/snapshot/today`
- `GET /api/v1/ticker/snapshot/today`
- `GET /api/v1/ticker/snapshot/today/export.csv`
- `GET /api/v1/ticker/snapshot/today/export.xlsx`

## Reusable Separation

For full component separation and workflow, see:
- `docs/PROJECT_COMPONENTS.md`
- `docs/SETUP_GUIDE.md`
