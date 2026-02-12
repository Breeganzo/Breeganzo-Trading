"""
Constants used across the entire trading bot pipeline.
All magic numbers live here so they can be changed in one place.
"""

import os
from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# =============================================================================
# Market Constants
# =============================================================================
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.065           # India 10-year government bond yield (~6.5%)
BENCHMARK_TICKER = "^NSEI"       # Nifty 50 index
INDIA_VIX_TICKER = "^INDIAVIX"

# =============================================================================
# Sector Mapping (for concentration limits)
# =============================================================================
SECTOR_MAP = {
    "large_cap": "Blue Chip",
    "banking": "Banking & Finance",
    "mid_cap": "Mid Cap Growth",
    "high_volatility": "High Volatility",
    "commodities": "Commodities-Linked",
}

# =============================================================================
# Prediction Horizons
# =============================================================================
HORIZONS = [1, 5, 10, 20]  # Days ahead

# =============================================================================
# Walk-Forward Defaults
# =============================================================================
DEFAULT_TRAIN_WINDOW = 504    # ~2 years
DEFAULT_TEST_WINDOW = 63      # ~1 quarter
DEFAULT_EMBARGO_DAYS = 5
DEFAULT_PURGE_DAYS = 5
DEFAULT_N_FOLDS = 8

# =============================================================================
# Feature Engineering
# =============================================================================
SMA_PERIODS = [5, 10, 20, 50, 200]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14
ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
STOCHASTIC_PERIOD = 14
CCI_PERIOD = 20
ADX_PERIOD = 14
WILLIAMS_R_PERIOD = 14
RETURN_PERIODS = [1, 5, 10, 21]
VOLATILITY_WINDOWS = [5, 10, 21, 63]

# =============================================================================
# Model Training
# =============================================================================
RANDOM_STATE = 42

# =============================================================================
# Display
# =============================================================================
DISCLAIMER = """
⚠️  DISCLAIMER: This is an academic research project for educational purposes only.
It is NOT financial advice. Past performance does not guarantee future results.
The authors are not responsible for any trading losses. Always consult a qualified
financial advisor before making investment decisions.
"""
