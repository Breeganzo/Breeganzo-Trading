# Masters AI Trading Bot - System Architecture & Workflow

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [ML Pipeline](#ml-pipeline)
6. [Ensemble Strategy](#ensemble-strategy)
7. [API Endpoints](#api-endpoints)

---

## System Overview

The Masters AI Trading Bot is an ML-powered stock prediction system for Indian markets (NSE/BSE). It combines multiple machine learning models with real-time market data to generate actionable trading signals.

### Key Features
- **6-Model Ensemble**: Combines ARIMA, GARCH, XGBoost, LightGBM, LSTM, and Transformer models
- **Real-time Predictions**: Live price tracking with IST timezone support
- **AI-Powered Explanations**: Groq LLM integration for contextual explanations
- **Risk Analytics**: ATR-based position sizing, stop-loss, and target calculation
- **Options Greeks**: Delta, Gamma, Theta, Vega calculations

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MASTERS AI TRADING BOT                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │   DATA LAYER     │     │   MODEL LAYER    │     │   WEB LAYER      │    │
│  │                  │     │                  │     │                  │    │
│  │  ┌────────────┐  │     │  ┌────────────┐  │     │  ┌────────────┐  │    │
│  │  │  yfinance  │  │────▶│  │   ARIMA    │  │     │  │   Flask    │  │    │
│  │  │  (prices)  │  │     │  └────────────┘  │     │  │   Server   │  │    │
│  │  └────────────┘  │     │  ┌────────────┐  │     │  └────────────┘  │    │
│  │                  │     │  │   GARCH    │  │            │           │    │
│  │  ┌────────────┐  │     │  └────────────┘  │            ▼           │    │
│  │  │  Features  │  │────▶│  ┌────────────┐  │     │  ┌────────────┐  │    │
│  │  │  Pipeline  │  │     │  │  XGBoost   │  │────▶│  │  REST API  │  │    │
│  │  └────────────┘  │     │  └────────────┘  │     │  │  Endpoints │  │    │
│  │                  │     │  ┌────────────┐  │     │  └────────────┘  │    │
│  │  ┌────────────┐  │     │  │  LightGBM  │  │            │           │    │
│  │  │Fundamentals│  │     │  └────────────┘  │            ▼           │    │
│  │  │   Cache    │  │     │  ┌────────────┐  │     │  ┌────────────┐  │    │
│  │  └────────────┘  │     │  │    LSTM    │  │     │  │    Web     │  │    │
│  │                  │     │  └────────────┘  │     │  │    UI      │  │    │
│  └──────────────────┘     │  ┌────────────┐  │     │  └────────────┘  │    │
│                           │  │Transformer │  │            │           │    │
│                           │  └────────────┘  │            ▼           │    │
│                           │        │         │     │  ┌────────────┐  │    │
│                           │        ▼         │     │  │   Groq     │  │    │
│                           │  ┌────────────┐  │     │  │  AI (LLM)  │  │    │
│                           │  │  ENSEMBLE  │  │     │  └────────────┘  │    │
│                           │  │ (weighted  │  │     │                  │    │
│                           │  │  average)  │  │     │                  │    │
│                           │  └────────────┘  │     │                  │    │
│                           │                  │     │                  │    │
│                           └──────────────────┘     └──────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Layer (`src/data/`)

| Component | File | Description |
|-----------|------|-------------|
| Downloader | `downloader.py` | Fetches OHLCV data from yfinance |
| Cleaner | `cleaner.py` | Handles missing data, outliers |
| Universe | `universe.py` | Manages stock ticker list |

### 2. Feature Engineering (`src/features/`)

| Component | File | Description |
|-----------|------|-------------|
| Technical | `technical.py` | RSI, MACD, Bollinger, ADX, ATR |
| Calendar | `calendar_features.py` | Day-of-week, month, expiry dates |
| Cross-Asset | `cross_asset.py` | NIFTY, VIX correlations |
| Fundamentals | `fundamentals.py` | P/E, P/B, ROE from yfinance |
| Pipeline | `pipeline.py` | Orchestrates all feature generation |

### 3. Models (`src/models/`)

| Model | File | Type | Description |
|-------|------|------|-------------|
| ARIMA | `arima_model.py` | Statistical | Autoregressive time series |
| GARCH | `garch_model.py` | Statistical | Volatility modeling |
| XGBoost | `xgboost_model.py` | ML | Gradient boosting |
| LightGBM | `lightgbm_model.py` | ML | Fast gradient boosting |
| LSTM | `lstm_model.py` | Deep Learning | Sequence modeling |
| Transformer | `transformer_model.py` | Deep Learning | Attention-based |
| **Ensemble** | `ensemble.py` | Meta-learner | Combines all models |

### 4. Inference (`src/inference/`)

| Component | File | Description |
|-----------|------|-------------|
| LivePredictor | `predictor.py` | Real-time predictions, caching |

### 5. Web Application (`webapp/`)

| Component | File | Description |
|-----------|------|-------------|
| Server | `server.py` | Flask app with REST API |
| Groq Explainer | `groq_explainer.py` | AI explanations via Groq |
| Tracker | `prediction_tracker.py` | Hit/miss tracking |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW DIAGRAM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   User Request                                                           │
│        │                                                                 │
│        ▼                                                                 │
│   ┌─────────┐      ┌─────────────┐      ┌───────────────┐               │
│   │  Flask  │      │   Check     │  No  │    Fetch      │               │
│   │ Server  │─────▶│   Cache     │─────▶│  Live Data    │               │
│   └─────────┘      └─────────────┘      │  (yfinance)   │               │
│                           │ Yes          └───────────────┘               │
│                           ▼                     │                        │
│                    ┌─────────────┐              │                        │
│                    │   Return    │              ▼                        │
│                    │   Cached    │      ┌───────────────┐               │
│                    │  Prediction │      │   Feature     │               │
│                    └─────────────┘      │  Engineering  │               │
│                                         └───────────────┘               │
│                                                │                        │
│                                                ▼                        │
│                                         ┌───────────────┐               │
│                                         │  Run 6 Base   │               │
│                                         │    Models     │               │
│                                         └───────────────┘               │
│                                                │                        │
│                                                ▼                        │
│                                         ┌───────────────┐               │
│                                         │   Ensemble    │               │
│                                         │   Weighted    │               │
│                                         │   Average     │               │
│                                         └───────────────┘               │
│                                                │                        │
│                                                ▼                        │
│                                         ┌───────────────┐               │
│                                         │   Generate    │               │
│                                         │   Signal &    │               │
│                                         │   Confidence  │               │
│                                         └───────────────┘               │
│                                                │                        │
│                                                ▼                        │
│                                         ┌───────────────┐               │
│                                         │  Cache &      │               │
│                                         │  Return       │               │
│                                         └───────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ML Pipeline

### Training Workflow (Notebook-based)

```
01_data_download.ipynb     → Download historical OHLCV data
        │
        ▼
02_eda_cleaning.ipynb      → Exploratory analysis, clean data
        │
        ▼
03_feature_engineering.ipynb → Generate technical indicators
        │
        ▼
04_walk_forward_cv.ipynb   → Set up cross-validation
        │
        ▼
05-10: Model Training      → Train individual models
   │
   ├── 05_arima.ipynb      (7 folds)
   ├── 06_garch.ipynb      (7 folds)
   ├── 07_xgboost.ipynb    (26 folds)
   ├── 08_lightgbm.ipynb   (26 folds)
   ├── 09_lstm.ipynb       (26 folds)
   └── 10_transformer.ipynb (26 folds)
        │
        ▼
11_model_comparison.ipynb  → Compare model metrics
        │
        ▼
12_ensemble.ipynb          → Train meta-learner ensemble
        │
        ▼
13_backtest.ipynb          → Full historical backtest
        │
        ▼
14_risk_analytics.ipynb    → Risk metrics, drawdown analysis
```

### Walk-Forward Cross-Validation

```
Time →
├────────────────────────────────────────────────────────────┤
│ Fold 1: [Train: Jan-Jun] [Validate: Jul-Aug] [Test: Sep]  │
│ Fold 2: [Train: Feb-Jul] [Validate: Aug-Sep] [Test: Oct]  │
│ Fold 3: [Train: Mar-Aug] [Validate: Sep-Oct] [Test: Nov]  │
│ ...                                                        │
├────────────────────────────────────────────────────────────┤
```

---

## Ensemble Strategy

### Direction-Accuracy Weighted Average

The ensemble uses **direction accuracy** to weight each model:

```
Weight(model) = max(0, DirectionAccuracy(model) - 0.50)

Normalized Weight = Weight(model) / Sum(All Weights)
```

### Current Model Weights

| Model | Direction Accuracy | Edge over 50% | Weight |
|-------|-------------------|---------------|--------|
| Transformer | 53.9% | +3.9% | ~54% |
| LSTM | 50.9% | +0.9% | ~36% |
| GARCH | 50.2% | +0.2% | ~8% |
| ARIMA | 50.1% | +0.1% | ~2% |
| XGBoost | 49.x% | 0% | 0% |
| LightGBM | 49.x% | 0% | 0% |

### Signal Generation

```python
# Confidence Score Calculation
base_confidence = 50 + (predicted_return / atr_pct) * 10
volume_bonus = +5 if volume_ratio > 1.2 else 0
rvol_bonus = +3 if rvol > 1.5 else 0
agreement_bonus = +8 if model_agreement >= 80% else
                  +3 if model_agreement >= 60% else
                  -5 if model_agreement < 40%

final_confidence = clamp(base_confidence + bonuses, 20, 95)
```

### Signal Thresholds

| Signal | Condition |
|--------|-----------|
| STRONG_BUY | predicted_return > 1.5 × ATR AND confidence > 65 |
| BUY | predicted_return > 0.5 × ATR AND confidence > 55 |
| HOLD | -0.5 × ATR < predicted_return < 0.5 × ATR |
| SELL | predicted_return < -0.5 × ATR AND confidence > 55 |
| STRONG_SELL | predicted_return < -1.5 × ATR AND confidence > 65 |

---

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard (all stocks) |
| GET | `/stock/<ticker>` | Stock detail page |
| GET | `/api/status` | System status, market hours |
| GET | `/api/prices` | Batch prices for all tickers |
| GET | `/api/predict/<ticker>` | Get ML prediction |
| GET | `/api/top-picks` | Top buy/sell recommendations |

### Data Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/intraday/<ticker>` | Intraday OHLCV |
| GET | `/api/history/<ticker>` | Historical OHLCV |
| GET | `/api/sectors` | Sector classification |

### AI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/explain` | Explain metric/indicator |
| GET | `/api/strategy/<ticker>` | Groq AI strategy |
| GET | `/api/overview/<ticker>` | Stock overview + sentiment |

### Tracking Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/tracking/daily` | Daily hit/miss summary |
| GET | `/api/tracking/monthly` | Monthly accuracy |
| GET | `/api/expected-vs-actual` | Prediction outcomes |

---

## Directory Structure

```
masters_trading_ai/
├── config/                   # Configuration files
│   ├── settings.yaml         # App settings
│   ├── tickers.yaml          # Stock universe
│   ├── model_params.yaml     # Model hyperparameters
│   └── groww_fees.yaml       # Fee structure
│
├── src/                      # Source code
│   ├── data/                 # Data fetching & cleaning
│   ├── features/             # Feature engineering
│   ├── models/               # ML models
│   ├── inference/            # Live prediction
│   ├── backtest/             # Backtesting engine
│   ├── options/              # Options pricing
│   ├── tracking/             # Performance tracking
│   └── utils/                # Utilities
│
├── webapp/                   # Web application
│   ├── server.py             # Flask server
│   ├── groq_explainer.py     # AI explanations
│   ├── prediction_tracker.py # Tracking logic
│   ├── templates/            # HTML templates
│   └── static/               # CSS, JS, images
│
├── models/                   # Saved model files (.joblib, .pt)
├── notebooks/                # Jupyter notebooks (01-18)
├── data/                     # Raw and processed data
├── cache/                    # Prediction cache
├── docs/                     # Documentation
│
├── requirements.txt          # Python dependencies
├── Makefile                  # Build automation
├── run_daily.sh              # Daily operations script
└── README.md                 # Project readme
```

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.14 |
| Web Framework | Flask |
| ML Libraries | scikit-learn, XGBoost, LightGBM |
| Deep Learning | PyTorch |
| Time Series | statsmodels, arch |
| Data | pandas, numpy, yfinance |
| AI Explanations | Groq (Llama 3.3 70B) |
| Frontend | HTML, CSS, JavaScript |
| Charts | Lightweight Charts |

---

*Document generated: February 2026*
*Version: 1.0*
