# Daily Operations Guide

## 📅 Running the App Daily

### Option 1: Quick Daily Run (Recommended)
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
./run_daily.sh
```

The `run_daily.sh` script automatically:
- Activates the virtual environment
- Generates predictions for all 124 tickers
- Tracks prediction accuracy
- Saves results to cache and logs

### Option 2: Manual Daily Run
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
python src/inference/predictor.py
```

### Option 3: Run Web App for Interactive Analysis
```bash
cd /Users/anto/Trading_Project/masters_trading_ai
.venv/bin/python -m webapp.server
```

Then open browser to: **http://localhost:5001**

---

## 🔄 Model Retraining Schedule

### Summary Table

| Model | Retrain Every | Why |
|-------|--------------|-----|
| **GARCH** | 7 days | Volatility changes quickly |
| **ARIMA** | 14 days | Captures recent trends |
| **LSTM** | 21 days | Deep learning needs fresh data |
| **Transformer** | 21 days | Deep learning needs fresh data |
| **XGBoost** | 30 days | Tree models are stable |
| **LightGBM** | 30 days | Tree models are stable |
| **Ensemble** | Daily (auto) | Combines all models automatically |

---

## 📋 Detailed Retraining Instructions

### Weekly (Every 7 Days) - High Priority
**GARCH Model** - Volatility changes rapidly in markets

```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate
jupyter notebook notebooks/06_garch.ipynb
# Run all cells to retrain and save model
```

### Bi-Weekly (Every 14 Days) - Medium-High Priority
**ARIMA Model** - Statistical time series model

```bash
jupyter notebook notebooks/05_arima.ipynb
# Run all cells to retrain and save model
```

### Tri-Weekly (Every 21 Days) - Medium Priority
**Deep Learning Models** - LSTM & Transformer

```bash
# LSTM
jupyter notebook notebooks/09_lstm.ipynb
# Run all cells to retrain and save model

# Transformer
jupyter notebook notebooks/10_transformer.ipynb
# Run all cells to retrain and save model
```

### Monthly (Every 30 Days) - Standard Priority
**Tree-Based Models** - XGBoost & LightGBM

```bash
# XGBoost
jupyter notebook notebooks/07_xgboost.ipynb
# Run all cells to retrain and save model

# LightGBM
jupyter notebook notebooks/08_lightgbm.ipynb
# Run all cells to retrain and save model
```

### After Any Retraining
**Ensemble Model** - Combines all models with optimal weights

```bash
jupyter notebook notebooks/12_ensemble.ipynb
# Run all cells to:
# 1. Calculate new optimal weights
# 2. Test ensemble performance
# 3. Save updated meta-model
```

---

## 📊 Complete Retraining Workflow (Quarterly)

Every **90 days** (or when market conditions change significantly), do a complete retrain:

```bash
cd /Users/anto/Trading_Project/masters_trading_ai
source .venv/bin/activate

# 1. Download fresh data
jupyter notebook notebooks/01_data_download.ipynb

# 2. Clean and validate
jupyter notebook notebooks/02_eda_cleaning.ipynb

# 3. Feature engineering
jupyter notebook notebooks/03_feature_engineering.ipynb

# 4. Walk-forward cross-validation
jupyter notebook notebooks/04_walk_forward_cv.ipynb

# 5. Retrain ALL models (in order)
jupyter notebook notebooks/05_arima.ipynb
jupyter notebook notebooks/06_garch.ipynb
jupyter notebook notebooks/07_xgboost.ipynb
jupyter notebook notebooks/08_lightgbm.ipynb
jupyter notebook notebooks/09_lstm.ipynb
jupyter notebook notebooks/10_transformer.ipynb

# 6. Rebuild ensemble
jupyter notebook notebooks/12_ensemble.ipynb

# 7. Run backtest
jupyter notebook notebooks/13_backtest.ipynb

# 8. Risk analytics
jupyter notebook notebooks/14_risk_analytics.ipynb
```

---

## 🎯 Recommended Schedule

### Daily
- ✅ Run `./run_daily.sh` to generate predictions
- ✅ Review prediction accuracy in webapp
- ✅ Check top picks in webapp dashboard

### Weekly (Sundays)
- 📈 Retrain GARCH model
- 📊 Review weekly performance metrics
- 🧹 Clean old cache files (optional)

### Bi-Weekly (1st & 15th)
- 📈 Retrain ARIMA model
- 📊 Review prediction accuracy trends

### Tri-Weekly (Every 21 days)
- 🤖 Retrain LSTM model
- 🤖 Retrain Transformer model
- 📈 Update ensemble weights

### Monthly (1st of month)
- 🌲 Retrain XGBoost model
- 🌲 Retrain LightGBM model
- 📈 Update ensemble weights
- 📊 Generate monthly performance report

### Quarterly (Jan 1, Apr 1, Jul 1, Oct 1)
- 🔄 Complete data refresh
- 🔄 Full retraining of all models
- 📊 Comprehensive backtest
- 📈 Risk analytics review

---

## 🧹 Cleanup Unwanted Files

All unwanted files have been removed. The following are **legitimate cache files** that improve performance:

```
cache/predictions.json          # Latest predictions (112K)
cache/groq_explanations/        # AI explanation cache (76K)
cache/fundamentals/             # Fundamental data cache (64K)
cache/prediction_tracking/      # Accuracy tracking (12K)
cache/prediction_log/           # Historical logs (4K)
```

### Manual Cleanup (if needed)
```bash
cd /Users/anto/Trading_Project/masters_trading_ai

# Clear Python cache
find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null

# Clear macOS files
find . -name ".DS_Store" -not -path "./.venv/*" -delete 2>/dev/null

# Clear Jupyter checkpoints
find . -type d -name ".ipynb_checkpoints" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null

# Clear old logs (older than 30 days)
find cache/prediction_log -name "*.json" -mtime +30 -delete 2>/dev/null
```

---

## 📈 Monitoring Model Performance

### Check Prediction Accuracy
```bash
# View accuracy history
cat cache/prediction_tracking/accuracy_history.json | python -m json.tool

# Or use the web app
.venv/bin/python -m webapp.server
# Navigate to: http://localhost:5001
```

### Signs You Need to Retrain Sooner

1. **Prediction accuracy drops below 55%** → Retrain all models
2. **Major market event** (earnings, policy change) → Retrain GARCH & ARIMA
3. **New trading regime** (bull to bear market) → Full retrain
4. **Volatility spike** → Retrain GARCH immediately

---

## 🔧 Troubleshooting

### Issue: Models not loading
```bash
# Check if model files exist
ls -lh models/

# Retrain missing models using notebooks
```

### Issue: Predictions taking too long
```bash
# Reduce ticker count in config/tickers.yaml
# Or run predictions in batches
```

### Issue: Server crashes
```bash
# Kill existing processes
pkill -f "python webapp/server.py"

# Restart server
.venv/bin/python -m webapp.server
```

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Daily predictions | `./run_daily.sh` |
| Start webapp | `.venv/bin/python -m webapp.server` |
| Check accuracy | Open http://localhost:5001 |
| Retrain GARCH | Open `notebooks/06_garch.ipynb` |
| Retrain ensemble | Open `notebooks/12_ensemble.ipynb` |
| Full retrain | Run notebooks 01-14 in order |

---

**Last Updated:** February 12, 2026
