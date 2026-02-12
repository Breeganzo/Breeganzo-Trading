# 🚀 AI Trading System for NSE India

AI-powered trading predictions for **124 Indian stocks** across 5 sectors using 6 machine learning models.

**What it does:** Predicts tomorrow's stock movements and shows you the top buy/sell opportunities.

---

## ⚡ Quick Start (First Time Setup)

### Step 1: Install
```bash
cd /Users/anto/Trading_Project/masters_trading_ai

# Create virtual environment and install packages
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Get Your FREE API Key
1. Go to https://console.groq.com/keys
2. Sign up (free)
3. Create API key
4. Create `.env` file in project folder:
```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

### Step 3: Make Script Executable (One Time Only)
```bash
chmod +x run_daily.sh
```

✅ **Setup Complete!**

---

## 📅 How to Run Daily

### **Every Morning** (Run This)
```bash
./run_daily.sh
```

**OR use the simple starter:**
```bash
./start_daily.sh
```

Then open your browser to: **http://localhost:5001**

**What this does:**
- ✅ Clears old predictions
- ✅ Generates fresh predictions for today
- ✅ Starts the web dashboard
- ✅ Shows you top buy/sell picks

**Press `Ctrl+C` to stop the server**

---

## 🔄 When to Retrain Models

### **Weekly** (Every Sunday)
```bash
./run_daily.sh --retrain
```

**What this does:**
- Updates all 6 AI models with latest data
- Takes 10-30 minutes
- Improves prediction accuracy

### **Never Run:**
❌ `python webapp/server.py` ← This is OLD (doesn't generate fresh predictions)

### **Always Run:**
✅ `./run_daily.sh` ← This is THE RIGHT WAY (generates fresh predictions)

---

## 📊 What You Get

| Feature | What It Shows |
|---------|---------------|
| **Dashboard** | All 124 stocks organized by sector |
| **Top Picks** | Best buy/sell opportunities ranked by AI |
| **Stock Detail** | Click any stock to see chart + prediction |
| **About Stock** | AI explains company + news sentiment |
| **Indicators** | RSI, MACD, ADX with buy/sell thresholds |
| **Accuracy** | How well predictions performed |
| **Live Prices** | Auto-updates every 5 seconds |

---

## 🎯 Understanding the Models

**6 AI Models Working Together:**
1. **ARIMA** - Statistical trend analyzer
2. **GARCH** - Volatility predictor  
3. **XGBoost** - Pattern recognition
4. **LightGBM** - Fast pattern detection
5. **LSTM** - Deep learning neural network
6. **Transformer** - Advanced attention mechanism

**Final Prediction = Weighted average of all 6 models**

---

## 📈 Trading Rules Built-In

| Rule | Value |
|------|-------|
| Capital | ₹50,000 |
| Stop Loss | -3% (auto-exit if stock drops 3%) |
| Take Profit | +10% (auto-sell if stock rises 10%) |
| Max Positions | 8 stocks at once |
| Max Per Stock | 12% of capital (₹6,000) |
| Max Sector | 35% of capital |

---

## 🗂️ Key Files

```
masters_trading_ai/
├── run_daily.sh          ← RUN THIS EVERY DAY
├── start_daily.sh        ← SIMPLE DAILY STARTER
├── DAILY_ROUTINE.txt     ← STEP-BY-STEP INSTRUCTIONS
├── .env                  ← Your API key goes here
├── webapp/
│   └── server.py         ← Flask web server
├── models/               ← Trained AI models (6 files)
├── config/
│   └── tickers.yaml      ← List of 124 stocks
└── cache/
    └── predictions.json  ← Today's predictions
```

---

## 🆘 Troubleshooting

### Server Won't Start
```bash
# Kill existing server
pkill -f "python webapp/server.py"

# Try again
./run_daily.sh
```

### Missing API Key Error
```bash
# Make sure .env file exists
cat .env

# Should show: GROQ_API_KEY=gsk_...
# If empty, get key from https://console.groq.com/keys
```

### Models Not Found
```bash
# Check if models exist
ls -lh models/

# Should see 7 files (.joblib and .pt files)
# If missing, run notebooks 05-12 to train models
```

---

## 📚 Advanced: Model Retraining Schedule

For best results, retrain models on this schedule:

| Model | How Often | Command |
|-------|-----------|---------|
| GARCH | Every 7 days | Run notebook `06_garch.ipynb` |
| ARIMA | Every 14 days | Run notebook `05_arima.ipynb` |
| LSTM | Every 21 days | Run notebook `09_lstm.ipynb` |
| Transformer | Every 21 days | Run notebook `10_transformer.ipynb` |
| XGBoost | Every 30 days | Run notebook `07_xgboost.ipynb` |
| LightGBM | Every 30 days | Run notebook `08_lightgbm.ipynb` |
| Ensemble | After any retrain | Run notebook `12_ensemble.ipynb` |

**OR** just run `./run_daily.sh --retrain` once a week for automated retraining.

---

## 🎓 Want to Understand the Code?

**Read these in order:**
1. [DAILY_OPERATIONS.md](docs/DAILY_OPERATIONS.md) - Daily usage guide
2. [SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md) - How everything works
3. [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md) - Project deep-dive for interviews

**Run Jupyter Notebooks (in order):**
```bash
jupyter notebook
# Open notebooks 00-17 in sequence
```

---

## ⚙️ Tech Stack

**Languages:** Python 3.14  
**ML:** XGBoost, LightGBM, PyTorch, scikit-learn, statsmodels  
**Web:** Flask, JavaScript, HTML/CSS  
**Data:** yfinance, pandas, numpy  
**AI:** Groq API (Llama 3.3 70B)

---

*For detailed technical documentation, see [SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md)*
