# Interview Guide — Masters Trading AI Project

A comprehensive guide for explaining this project in technical interviews. Covers what to say, how to explain design decisions, and how to handle follow-up questions.

---

## 1. The 30-Second Elevator Pitch

> "I built an end-to-end machine learning trading system for the Indian stock market. It covers 124 NSE equities, uses 6 different models — from classical time-series (ARIMA, GARCH) to deep learning (LSTM, Transformer) — combined through a Ridge meta-learner ensemble. The system has a complete pipeline: data ingestion, 200+ engineered features with anti-lookahead safeguards, walk-forward cross-validation, risk management with half-Kelly position sizing, and a live Flask web dashboard with REST API that generates daily trading signals."

---

## 2. The 2-Minute Deep Dive

### Problem Statement
"I wanted to build a system that goes beyond basic stock prediction. Most ML trading tutorials ignore critical issues like look-ahead bias, transaction costs, and position sizing. My goal was to build something production-quality that handles all these real-world constraints."

### Data Pipeline
"I pull 2 years of daily OHLCV data for 124 NSE equities using yfinance. The tickers span 5 sectors — Large-Cap, Banking, Mid-Cap Growth, High-Volatility, and Commodities-Linked — giving the models diverse market dynamics to learn from. I also pull cross-asset data: India VIX, Nifty 50, Bank Nifty, USD/INR, crude oil, and gold for regime detection."

### Feature Engineering
"I engineer over 200 features with strict no-lookahead design:
- 50+ technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, etc.)
- Cross-asset features (how VIX, gold, USD/INR correlate with each stock)
- Calendar features (day of week, month, expiry proximity)
- All normalised using rolling 252-day z-scores using only past data
- Features are lagged at t-1, t-2, t-5 before any normalisation
- Highly correlated features (>0.95) are dropped automatically"

### Model Architecture
"I use 6 diverse models to capture different market patterns:

1. **ARIMA** — Per-ticker time-series model. Each of the 121 stocks gets its own ARIMA with auto-selected order. Good for linear mean-reversion patterns.

2. **GARCH(1,1)** — Per-ticker volatility model with Student's t distribution. Captures volatility clustering — the tendency for high-volatility days to cluster together.

3. **XGBoost** — Gradient boosted trees with 1500 estimators. Strong at capturing non-linear feature interactions. L1/L2 regularised to prevent overfitting.

4. **LightGBM** — Similar to XGBoost but with leaf-wise growth and path smoothing. Often faster and sometimes more accurate.

5. **LSTM** — 2-layer recurrent neural network (48 hidden units). Processes 30-day sequences. Captures temporal dependencies that tree models can't — the order of features matters.

6. **Transformer** — Simplified Temporal Fusion Transformer with Variable Selection GRN and 4-head attention. Uses 45-day sequences. The attention mechanism can focus on the most relevant time steps."

### Ensemble
"I combine all 6 models through a Ridge regression meta-learner. During training, each model generates out-of-fold predictions, and Ridge learns optimal weights to combine them. This is called stacking — it's more principled than simple averaging because Ridge can learn which models are more reliable."

### Validation
"I use 8-fold walk-forward cross-validation — this is crucial for time-series data. You can't randomly shuffle because that creates data leakage. Each fold has:
- 504-day training window (~2 years)
- 63-day test window (~1 quarter)
- 5-day embargo between train and test to prevent label leakage
- 5-day purge to remove samples with overlapping prediction horizons"

### Risk Management
"The system has multiple layers of risk control:
- 3% stop-loss, 10% take-profit per position (R:R = 1:3.3)
- Maximum 8 concurrent positions
- No single position exceeds 12% of capital
- Sector concentration capped at 35%
- Half-Kelly position sizing — full Kelly is mathematically optimal but too aggressive, so we use half to reduce variance
- Daily drawdown limit of 3%, total drawdown limit of 15%
- Minimum 60% model confidence required to trade
- Transaction costs modelled using actual Groww brokerage fees including STT, stamp duty, SEBI charges"

### Application
"The Flask web application provides a real-time dashboard with sector-based stock browsing, individual stock analysis with interactive charts, ML prediction breakdowns, top picks ranking, and prediction accuracy tracking — all served through a REST API with 5-second auto-refresh."

---

## 3. Technical Deep-Dive Questions & Answers

### Q: "Why 6 models? Isn't that overkill?"
> "Each model captures different types of patterns. ARIMA handles linear autocorrelation, GARCH models volatility clustering, tree models capture non-linear feature interactions, and neural networks learn temporal sequences. Financial markets have all these dynamics simultaneously. In practice, the ensemble consistently outperforms any single model because errors from different model families tend to be uncorrelated."

### Q: "Explain walk-forward cross-validation vs. regular k-fold."
> "Regular k-fold randomly shuffles data, which for time series means your model could train on 2024 data and test on 2023 data — that's look-ahead bias. Walk-forward CV respects chronological order: fold 1 trains on months 1-24 and tests on months 25-27, fold 2 trains on months 4-27 and tests on months 28-30, and so on. The 5-day embargo between train and test prevents subtle label leakage from overlapping prediction horizons."

### Q: "What is Half-Kelly and why use it?"
> "Kelly criterion gives the mathematically optimal bet size: f* = (p × b - q) / b, where p is win probability, b is the win/loss ratio, and q = 1-p. But Kelly assumes you know exact probabilities — in practice we estimate them, so full Kelly leads to huge variance and potential ruin. Half-Kelly cuts the bet size in half, reducing variance by 75% while only sacrificing about 25% of growth rate. It's the standard practice in quantitative finance."

### Q: "How do you handle look-ahead bias?"
> "Multiple layers of protection:
> 1. Features are lagged (t-1, t-2, t-5) — we never use today's data to predict today
> 2. Rolling z-score normalisation uses a backward-looking 252-day window
> 3. Walk-forward CV with embargo and purge
> 4. Features are computed independently for each cross-validation fold
> 5. The target variable is the forward return — only calculated during training, never available to the model at prediction time"

### Q: "Why both ARIMA and GARCH?"
> "They forecast different things. ARIMA predicts the expected return (the mean of the distribution), while GARCH predicts volatility (the variance). Together they give a more complete picture — you want to know both where the price is heading AND how uncertain that prediction is. GARCH is particularly useful for position sizing: higher predicted volatility → smaller position size."

### Q: "How does the Transformer differ from a standard one?"
> "It's a Simplified Temporal Fusion Transformer (TFT). The key differences:
> 1. Variable Selection Network (GRN) — automatically learns which features are most important at each time step, rather than treating all features equally
> 2. The backbone is LSTM + Multi-Head Attention, not pure self-attention — this handles both local patterns (LSTM) and long-range dependencies (attention)
> 3. 4-head attention with head size 4 — much smaller than typical NLP transformers because financial time series have fewer meaningful patterns than language"

### Q: "What's your Sharpe ratio / returns?"
> "The goal of this project is demonstrating the ML engineering pipeline, not achieving production-level alpha. The backtest with realistic Groww transaction costs shows [refer to your actual results]. The more important metric is that walk-forward accuracy consistently exceeds random (50%), showing the models learn genuine patterns rather than noise."

### Q: "How do you prevent overfitting?"
> "Multiple mechanisms:
> 1. Walk-forward CV — models never see future data during training
> 2. Embargo/purge — prevents subtle temporal leakage
> 3. L1/L2 regularisation in XGBoost and LightGBM
> 4. Early stopping — tree models stop adding trees when validation loss stops improving
> 5. Dropout (0.5 for LSTM, 0.4 for Transformer) — randomly drops neurons during training
> 6. The Ridge meta-learner has its own regularisation (alpha=1.0)
> 7. Highly correlated features (>0.95) are automatically dropped
> 8. I have an overfitting detector that compares train vs test performance"

### Q: "Why Indian market (NSE) specifically?"
> "India's NSE is the world's largest derivatives market by volume and has high retail participation, making it an interesting test case. The Groww brokerage fee structure (with STT, stamp duty, exchange charges) creates realistic transaction cost modelling that many academic papers overlook. Also, building for a non-US market demonstrates that the pipeline is generalisable."

### Q: "How does the options pricing module work?"
> "I implemented Black-Scholes from scratch — delta, gamma, theta, vega, rho — using the analytical formulas. For implied volatility, I use a Newton-Raphson solver since there's no closed-form solution for IV. The options module also generates strategy signals: bull/bear spreads, straddles, and iron condors based on the predicted price move and IV percentile."

### Q: "What would you improve with more time?"
> "Three main areas:
> 1. **Alternative data** — Add sentiment from news/social media, order flow data, and earnings surprises
> 2. **Online learning** — Currently models are retrained periodically; I'd add incremental learning so models update daily
> 3. **Reinforcement learning** — Replace the rule-based position sizing with a PPO/SAC agent that learns optimal allocation
> 4. **Execution** — Add real-time order routing through a broker API (Zerodha Kite) for paper trading"

---

## 4. System Design Questions

### Q: "How does the system load models so fast?"
> "This was an interesting engineering challenge. ARIMA and GARCH are per-ticker models — 121 and 120 tickers respectively. Originally, loading meant re-fitting from raw data (10+ minutes). I fixed this by using statsmodels' `smooth()` for ARIMA and arch's `fix()` for GARCH — both restore the exact model from saved parameters without any optimisation. Total load time went from 10+ minutes to ~8 seconds."

### Q: "How would you deploy this to production?"
> "The architecture is already modular:
> 1. Data download runs as a scheduled cron job (pre-market)
> 2. Feature pipeline + prediction runs once after market data is fresh
> 3. Flask REST API serves the front-end — can scale to FastAPI + React
> 4. Models are serialised to joblib/pt files — could be stored in S3
> 5. Redis or PostgreSQL for prediction caching instead of JSON files
> 6. Docker container for reproducible deployment
> 7. Alerting via Telegram/email when high-confidence signals appear"

### Q: "What's the hardest bug you fixed?"
> "PyTorch segfaulting when loading models after statsmodels/ARIMA had been loaded. The C extensions from statsmodels were corrupting PyTorch's memory. The fix was to ensure PyTorch initialises BEFORE any statsmodels imports, and to load PyTorch models (.pt files) before statistical models (.joblib). This took significant debugging with faulthandler to identify."

---

## 5. Key Numbers to Remember

| Metric | Value |
|---|---|
| Tickers | 124 NSE equities |
| Sectors | 5 (Large-Cap, Banking, Mid-Cap, High-Vol, Commodities) |
| Features | 200+ |
| Models | 6 base + Ridge ensemble |
| Walk-Forward Folds | 8 |
| Training Window | 504 days (~2 years) |
| Test Window | 63 days (~1 quarter) |
| Capital | ₹50,000 |
| Max Positions | 8 concurrent |
| Stop-Loss / Take-Profit | 3% / 10% |
| Position Sizing | Half-Kelly |
| Prediction Horizons | 1, 5, 10, 20 days |
| Web App | Flask REST API + JavaScript dashboard |
| Load Time | ~8 seconds for all models |

---

## 6. Technologies to Highlight

**Python ecosystem:** pandas, numpy, scikit-learn, PyTorch, XGBoost, LightGBM, statsmodels, arch, scipy, plotly, Flask

**ML concepts:** Walk-forward CV, stacking ensemble, Ridge meta-learner, early stopping, regularisation, feature engineering, z-score normalisation

**Finance concepts:** ARIMA, GARCH, Black-Scholes, Greeks, Kelly criterion, Sharpe ratio, VaR/CVaR, drawdown analysis, anti-lookahead design

**Software engineering:** Modular OOP design (BaseModel abstract class), YAML config management, joblib serialisation, Flask REST API with decoupled frontend, git version control

---

## 7. Conversation Starters

If the interviewer hasn't asked yet, you can steer the conversation:

- *"One of the most interesting challenges was handling look-ahead bias in feature engineering..."*
- *"What I'm most proud of is the walk-forward validation — it prevents the overfitting that plagues most ML trading projects..."*
- *"The ARIMA/GARCH loading optimisation was a great systems engineering problem — reducing load time from 10 minutes to 8 seconds..."*
- *"Building Black-Scholes from scratch gave me a deep understanding of options pricing and numerical methods..."*

---

## 8. What NOT to Claim

- Don't claim the system generates guaranteed profits — no ML system does
- Don't overstate accuracy — be honest about the gap between backtest and live trading
- Emphasise the engineering pipeline and methodology, not the returns
- Acknowledge limitations: no alternative data, no real-time execution yet, models need periodic retraining
- Be clear this is a research/learning project, not production HFT

---

*Preparation time: Review this doc 30 minutes before the interview. Focus on sections 2-3.*
