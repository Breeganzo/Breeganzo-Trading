# Project Components and Reusable Structure

This repository is split into reusable layers so each part can evolve independently.

## 1) Frontend (UI only)
- Path: `frontend/`
- Stack: Next.js + React
- Responsibilities:
  - Login/dashboard UI
  - Portfolio, risk, rankings, orders, trades views
  - Calls backend APIs only (no direct DB access)

## 2) Backend (API + business logic)
- Path: `backend/`
- Stack: FastAPI + SQLAlchemy + Redis cache
- Responsibilities:
  - Authentication and authorization
  - Portfolio/trade/order APIs
  - Risk/ranking/ticker APIs
  - Auto-signal processing (DB trigger queue)
  - Transaction cost calculations

## 3) Database (persistent state)
- Primary schema reference: `backend/sql/schema.sql`
- Documentation copy: `docs/schema.sql`
- Main tables:
  - `users`, `portfolio`, `trades`, `order_book`, `daily_returns`, `rankings`, `system_status`
  - `signal_triggers` (consume-once BUY/SELL/HOLD trigger queue)

## 4) ML strategy app (separate runtime)
- Path: `masters_trading_ai/`
- Responsibilities:
  - Strategy + AI prediction pipelines
  - Advisor simulation
  - Risk/expected-vs-actual tracker
  - Local caching and simulation logs

## 5) Model training artifacts/notebooks (separate from serving)
- Path: `masters_trading_ai/notebooks/` and training scripts under `masters_trading_ai/src/`
- Recommended:
  - Keep training notebooks and experiment outputs here
  - Keep deploy/runtime API code in `backend/`

## 6) Deployment configuration
- Path: `deployment/` and root `render.yaml`
- Responsibilities:
  - Hosting configuration (frontend/backend)
  - Environment setup for production

---

## Localhost Runbook

### Backend
```bash
cd backend
cp .env.example .env
# For local dev without Google auth:
# AUTH_BYPASS_LOCAL=true
# LOCAL_BYPASS_EMAIL=anthonybreeganzo02@gmail.com
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd frontend
cat > .env.local <<'EOF'
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000/api/v1/ticker/ws
NEXT_PUBLIC_AUTH_BYPASS_LOCAL=true
EOF
npm run dev -- --port 3000
```

### ML Strategy App (optional, separate)
```bash
cd masters_trading_ai
source .venv/bin/activate
python webapp/server.py
```

---

## Auto Signal Trigger Processing (server-side)

The backend now processes DB triggers in the background, even if UI is closed:
- Config:
  - `AUTO_SIGNAL_WORKER_ENABLED=true`
  - `AUTO_SIGNAL_INTERVAL_SEC=10`
  - `AUTO_SIGNAL_BATCH_SIZE=50`
  - `AUTO_SIGNAL_WORKER_MARKET_HOURS_ONLY=true`
- Trigger queue endpoints:
  - `GET /api/v1/orders/auto-signals`
  - `POST /api/v1/orders/auto-signals`
  - `DELETE /api/v1/orders/auto-signals/{signal_id}`
  - `POST /api/v1/orders/auto-signals/process`

Triggered signals are consumed once and deleted after execution to avoid repeated duplicate orders.
