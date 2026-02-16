# Masters Trading AI

AI-powered NSE dashboard with live prices, ensemble predictions, top picks (`BUY/SELL/HOLD`), stock analysis, and risk analytics.

## Quick start
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `.env` values (minimum):
```bash
GROQ_API_KEY=your_groq_key
FLASK_SECRET_KEY=dev-secret-change-me
```

## Run server
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
python webapp/server.py
```

Do **not** run:
```bash
webapp/server.py
```
It causes: `zsh: permission denied`.

Open:
- Dashboard: `http://localhost:5001`
- Risk page: `http://localhost:5001/risk`

## Verify
```bash
curl -s 'http://localhost:5001/api/status' | jq
curl -s 'http://localhost:5001/api/top-picks?n=20&grouped=true' | jq
curl -s 'http://localhost:5001/api/daily-analysis' | jq
```

Expected:
- `models_loaded: true`
- load steps populated
- grouped top picks: `top_buy`, `top_sell`, `top_hold`

## Run tests
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
pytest -q
```

## Troubleshooting

### Port already in use
```bash
lsof -iTCP:5001 -sTCP:LISTEN -n -P
# Kill process using 5001
kill $(lsof -tiTCP:5001 -sTCP:LISTEN)
```

Then run:
```bash
python webapp/server.py
```

If you prefer a different port:
```bash
FLASK_PORT=5002 python webapp/server.py
```

### `zsh: permission denied: webapp/server.py`
```bash
python webapp/server.py
```

### Models still loading
- Wait for `/api/status` to show `models_loaded=true`
- watch load progress in navbar or `/api/status`
