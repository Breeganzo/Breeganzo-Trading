# Masters AI Trading Bot — Daily Usage Guide

## Quick Start (Daily)

```bash
cd masters_trading_ai
./run_daily.sh
```

Open **http://localhost:5001** in your browser. That's it.

---

## What the App Does

1. **Loads 6 trained ML models**: ARIMA, GARCH, XGBoost, LightGBM, LSTM, Transformer
2. **Runs live predictions** for 124 NSE stocks across 5 sectors
3. **Ensemble combines** models using direction-accuracy weighted average (auto-selected)
4. **Shows signals**: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
5. **AI explanations** via Groq LLM (if `GROQ_API_KEY` is set in `.env`)
6. **Tracks prediction accuracy** — compares predictions vs actual closing prices

---

## Daily Workflow

### Before Market Open (8:30–9:00 AM IST)
1. Start the app: `./run_daily.sh`
2. Click **Top Picks** to see highest-conviction trades
3. Review individual stocks for entry/stop-loss/target levels
4. Check **Model Agreement** — higher % = more conviction

### During Market Hours (9:15 AM – 3:30 PM IST)
- Prices auto-refresh every 5 seconds
- Predictions are cached for 6 hours (click a stock to regenerate)
- IST clock runs in real-time on the dashboard

### After Market Close (3:30 PM IST)
- Click **Expected vs Actual** to review today's prediction accuracy
- The app logs all predictions with timestamps for tracking

### Stop the App
Press `Ctrl+C` in the terminal.

---

## When to Retrain Models

Retrain the ensemble **monthly** or when:
- Direction accuracy drops below 52% consistently
- Market regime changes significantly
- New stocks are added to the universe

```bash
./run_daily.sh --retrain
```

Or manually run notebook `12_ensemble.ipynb` in VS Code for full control.

---

## Configuration

| File | Purpose |
|------|---------|
| `.env` | API keys (GROQ_API_KEY, FLASK_SECRET_KEY) |
| `config/tickers.yaml` | Stock universe — add/remove tickers here |
| `config/model_params.yaml` | Model hyperparameters & ensemble strategy |
| `config/settings.yaml` | General settings (retrain schedule, thresholds) |

### Adding/Removing Stocks
Edit `config/tickers.yaml` and restart the app. No retraining needed for predictions (ARIMA/GARCH retrain per-ticker on the fly).

### Changing Ensemble Strategy
In `config/model_params.yaml`:
```yaml
ensemble:
  method: "auto"  # auto-selects best from 5 strategies
  # Or force one: "median", "simple_average", "weighted_average", "ridge", "confidence_filtered"
```

---

## Ensemble Model Details

The current ensemble uses **weighted_average** (auto-selected):
- **Transformer**: 54% weight — best individual direction accuracy
- **LSTM**: 36% weight — strong sequential pattern detection
- **GARCH**: 8% weight — volatility-informed predictions
- **ARIMA**: 3% weight — statistical baseline
- **XGBoost/LightGBM**: 0% weight — excluded due to data alignment gap

**Test direction accuracy: 52.0%** on 13,156 held-out samples.

With **confidence filtering** (|prediction| > 1%): **54.2% accuracy** on 38% of trades.

---

## File Structure

```
models/              Trained model files (joblib/pt)
cache/               Prediction cache + Groq explanation cache
data/predictions/    Historical prediction logs
config/              All configuration files
webapp/              Flask app + static files
notebooks/           Jupyter notebooks for training & analysis
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Models still loading" | Wait 3–5 seconds, models load on startup |
| No AI explanations | Add `GROQ_API_KEY` to `.env` |
| Chart times wrong | Clear browser cache (Ctrl+Shift+R) |
| Stale predictions | Delete `cache/predictions.json` |
| Port 5001 in use | `lsof -i :5001` then `kill <PID>` |
| Import errors | `pip install -r requirements.txt` |
