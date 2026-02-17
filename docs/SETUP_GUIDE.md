# QuantDesk Pro — Production Setup Guide

## Architecture Overview

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Vercel (Free)  │     │  Render (Free)   │     │ Supabase (Free)  │
│   Next.js 14     │────▶│  FastAPI + WS    │────▶│  PostgreSQL      │
│   Tailwind CSS   │     │  Gunicorn/Uvicorn│     │                  │
└──────────────────┘     └──────┬───────────┘     └──────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Upstash (Free)     │
                    │   Redis Cache        │
                    └──────────────────────┘

          GitHub Actions: Morning cron (8:45 AM IST) + Keep-alive
```

---

## Step 1: Supabase PostgreSQL Setup

1. Go to [supabase.com](https://supabase.com) → Sign up → New Project
2. **Project name**: `quantdesk-pro`
3. **Database password**: Generate a strong password (save it!)
4. **Region**: Mumbai (`ap-south-1`) for lowest latency to NSE
5. Once created, go to **Settings → Database**
6. Copy the **Connection string (URI)** — it looks like:
   ```
   postgresql://postgres.[ref]:[password]@aws-0-ap-south-1.pooler.supabase.com:6543/postgres
   ```
7. **Important**: For async Python (asyncpg), use the **Session Mode** (port 5432) or **Transaction Mode** (port 6543)
8. The app auto-converts `postgresql://` to `postgresql+asyncpg://`

**Tables are auto-created** on first startup via SQLAlchemy `create_all()`.
Reference schema is now stored in:
- `backend/sql/schema.sql` (runtime schema reference)
- `docs/schema.sql` (documentation copy)

The schema includes `signal_triggers` for DB-driven BUY/SELL/HOLD trigger queues.

---

## Step 2: Upstash Redis Setup

1. Go to [upstash.com](https://upstash.com) → Sign up → Create Database
2. **Name**: `quantdesk-cache`
3. **Region**: `ap-south-1` (Mumbai)
4. **Type**: Regional
5. Once created, copy the **Redis URL** from the dashboard:
   ```
   rediss://default:[password]@[region].upstash.io:6379
   ```
6. **Free tier**: 10,000 commands/day — our keep-alive + ticker caching stays well within this

---

## Step 3: Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project: `QuantDesk Pro`
3. Go to **APIs & Services → OAuth consent screen**
   - User type: **External**
   - App name: `QuantDesk Pro`
   - Support email: `anthonybreeganzo02@gmail.com`
   - Authorized domains: `onrender.com`, `vercel.app`
4. Go to **Credentials → Create Credentials → OAuth 2.0 Client ID**
   - Application type: **Web application**
   - Authorized redirect URIs:
     - `http://localhost:8000/api/v1/auth/callback` (local dev)
     - `https://quantdesk-pro-api.onrender.com/api/v1/auth/callback` (production)
5. Copy **Client ID** and **Client Secret**

---

## Step 4: Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up → API Keys → Create API Key
3. Copy the key (starts with `gsk_`)

---

## Step 5: Environment Variables

### Backend `.env`:

```env
APP_ENV=production
DEBUG=false
SECRET_KEY=<generate-with: python3 -c "import secrets; print(secrets.token_hex(32))">

DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-0-ap-south-1.pooler.supabase.com:6543/postgres
REDIS_URL=rediss://default:[password]@[region].upstash.io:6379

GOOGLE_CLIENT_ID=<your-client-id>.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-<your-secret>
GOOGLE_REDIRECT_URI=https://quantdesk-pro-api.onrender.com/api/v1/auth/callback
ALLOWED_EMAIL=anthonybreeganzo02@gmail.com

GROQ_API_KEY=gsk_<your-key>

ALLOWED_ORIGINS=http://localhost:3000,https://quantdesk-pro.vercel.app
AUTH_BYPASS_LOCAL=false
LOCAL_BYPASS_EMAIL=anthonybreeganzo02@gmail.com
LOCAL_BYPASS_NAME=Local User

AUTO_SIGNAL_WORKER_ENABLED=true
AUTO_SIGNAL_INTERVAL_SEC=10
AUTO_SIGNAL_BATCH_SIZE=50
AUTO_SIGNAL_WORKER_MARKET_HOURS_ONLY=true
```

### Frontend `.env.local`:

```env
NEXT_PUBLIC_API_URL=https://quantdesk-pro-api.onrender.com/api/v1
NEXT_PUBLIC_WS_URL=wss://quantdesk-pro-api.onrender.com/api/v1/ticker/ws
NEXT_PUBLIC_AUTH_BYPASS_LOCAL=false
```

---

## Step 6: Deploy Backend to Render

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. **Root Directory**: `backend`
5. **Environment**: Docker
6. **Dockerfile Path**: `./Dockerfile`
7. **Plan**: Free
8. Add all environment variables from Step 5 in the Render dashboard
9. Deploy!

**Or** use the `render.yaml` blueprint:
- Render auto-detects `render.yaml` at the repo root
- Fill in the `sync: false` env vars in the dashboard

---

## Step 7: Deploy Frontend to Vercel

1. Go to [vercel.com](https://vercel.com) → Import Git Repository
2. Select your repo
3. **Framework Preset**: Next.js
4. **Root Directory**: `frontend`
5. Add environment variables:
   - `NEXT_PUBLIC_API_URL` = `https://quantdesk-pro-api.onrender.com/api/v1`
   - `NEXT_PUBLIC_WS_URL` = `wss://quantdesk-pro-api.onrender.com/api/v1/ticker/ws`
6. Deploy!

---

## Step 8: TOTP 2FA Setup (Google Authenticator)

After first login:

1. The app will prompt you to set up 2FA
2. Hit **POST /api/v1/auth/totp/setup** (happens automatically in the UI)
3. You'll receive a QR code and an `otpauth://` URI
4. **Scan the QR code** with Google Authenticator app
5. Enter the 6-digit code to verify
6. 2FA is now enabled — every login requires Google Authenticator code

**To set up manually:**
```bash
# After first Google OAuth login, call:
curl -X POST https://quantdesk-pro-api.onrender.com/api/v1/auth/totp/setup \
  -H "Authorization: Bearer <your-jwt-token>"

# Response includes:
# - secret: base32 TOTP secret
# - uri: otpauth://totp/QuantDesk%20Pro:anthonybreeganzo02@gmail.com?secret=XXX&issuer=QuantDesk%20Pro
# - qr_code: base64 PNG of QR code
```

---

## Step 9: Seed Portfolio

```bash
cd backend
python -m scripts.seed_portfolio
```

This seeds: **KTK Bank — 196 shares @ ₹211.48**

Or use the API:
```bash
curl -X POST https://quantdesk-pro-api.onrender.com/api/v1/portfolio/seed \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '[{"ticker": "KTKBANK.NS", "quantity": 196, "avg_buy_price": 211.48, "sector": "Banking & Finance"}]'
```

---

## Step 10: GitHub Actions Setup

Add these secrets to your GitHub repo (**Settings → Secrets → Actions**):

| Secret | Value |
|--------|-------|
| `API_BASE_URL` | `https://quantdesk-pro-api.onrender.com` |
| `CRON_JWT_TOKEN` | Generate a long-lived JWT (see below) |

**Generate CRON JWT Token:**
```python
from jose import jwt
from datetime import datetime, timedelta, timezone

token = jwt.encode(
    {"sub": "<your-user-uuid>", "email": "anthonybreeganzo02@gmail.com", "exp": datetime(2027, 1, 1, tzinfo=timezone.utc)},
    "<your-secret-key>",
    algorithm="HS256"
)
print(token)
```

---

## Step 11: Verify Deployment

```bash
# Health check
curl https://quantdesk-pro-api.onrender.com/health

# System validation
curl https://quantdesk-pro-api.onrender.com/api/v1/system/health

# Market regime
curl https://quantdesk-pro-api.onrender.com/api/v1/risk/regime
```

### Auto Signal Trigger Queue (consume-once)

```bash
# Add a pending BUY trigger
curl -X POST https://quantdesk-pro-api.onrender.com/api/v1/orders/auto-signals \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"INFY.NS","action":"BUY","quantity":2,"trigger_price_low":1400,"trigger_price_high":1410,"sentiment_min":-0.2}'

# List pending triggers
curl https://quantdesk-pro-api.onrender.com/api/v1/orders/auto-signals?status=PENDING \
  -H "Authorization: Bearer <your-jwt-token>"

# Process triggers once (uses live price + sentiment gate)
curl -X POST 'https://quantdesk-pro-api.onrender.com/api/v1/orders/auto-signals/process?limit=25' \
  -H "Authorization: Bearer <your-jwt-token>"
```

When a trigger executes, the trade is written to `trades` and the trigger row is deleted
to avoid redundant repeat execution.

---

## Free-Tier Limits & Optimization

| Service | Limit | Our Usage | Strategy |
|---------|-------|-----------|----------|
| Render | 750 hrs/mo, spins down after 15 min idle | ~720 hrs | GitHub Actions keep-alive during market hours |
| Supabase | 500 MB, 2 GB transfer | ~50 MB | Indexed queries, pagination |
| Upstash | 10K commands/day | ~5K | 5s price cache, 30m risk cache, 1h ranking cache |
| Vercel | 100 GB bandwidth | ~5 GB | Static assets, ISR |
| GitHub Actions | 2,000 min/mo | ~100 min | 2 workflows (cron + keep-alive) |

---

## Security Checklist

- [x] Single-user restriction (ALLOWED_EMAIL)
- [x] Google OAuth 2.0
- [x] TOTP 2FA (Google Authenticator)
- [x] JWT with expiry (12h)
- [x] CORS locked to specific origins
- [x] No secrets in code (all in env vars)
- [x] Non-root Docker user
- [x] Security headers (Vercel)
- [x] Input validation (Pydantic)
- [x] SQL injection prevention (SQLAlchemy ORM)
- [x] Rate limiting on AI endpoints
