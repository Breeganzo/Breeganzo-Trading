"""
ARIMA Model (Statistical Baseline)
====================================
Auto-Regressive Integrated Moving Average for per-ticker return forecasting.

ARIMA captures linear autoregressive patterns in return series that
ML models may overlook. Unlike tree/neural models, ARIMA is:
  - Univariate: models each ticker's return series independently
  - Interpretable: AR, I, MA orders have clear statistical meaning
  - Fast: fits in seconds, not hours

Key difference from XGBoost/LightGBM/LSTM/Transformer:
  - Those models train on a pooled panel of all tickers (features × samples)
  - ARIMA trains per-ticker on the target return series itself
  - Integration is at the ensemble level: ARIMA predictions become a column
    in the stacking DataFrame alongside other model predictions
"""

import numpy as np
import pandas as pd
import yaml
import warnings
import joblib
from pathlib import Path
from typing import Optional

from .base import BaseModel
from ..utils.constants import CONFIG_DIR, RANDOM_STATE


class ARIMAModel(BaseModel):
    """
    ARIMA wrapper implementing the BaseModel interface.

    Fits one ARIMA model per ticker. The `fit()` method expects
    X_train to contain a '_ticker' column (or be indexed with ticker info)
    and uses only y_train (returns) for model fitting.

    For walk-forward integration, the notebook handles the per-ticker
    loop and passes predictions to the ensemble in the standard format.
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(name="arima", task="regression")

        if params is None:
            config_path = CONFIG_DIR / "model_params.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                params = config.get("arima", {})
            else:
                params = {}

        self.max_p = params.get("max_p", 5)
        self.max_d = params.get("max_d", 2)
        self.max_q = params.get("max_q", 5)
        self.default_order = tuple(params.get("default_order", [2, 0, 1]))
        self.forecast_horizon = params.get("forecast_horizon", 5)
        self.models = {}  # {ticker: fitted ARIMA result}
        self.orders = {}  # {ticker: (p, d, q) selected}

    def fit_ticker(
        self,
        returns: pd.Series,
        ticker: str,
        auto_order: bool = True,
    ) -> dict:
        """
        Fit ARIMA on a single ticker's return series.

        Parameters
        ----------
        returns : pd.Series
            Historical returns for one ticker (chronologically sorted)
        ticker : str
            Ticker symbol (used as key in self.models)
        auto_order : bool
            If True, select (p,d,q) via AIC. Otherwise use default_order.

        Returns
        -------
        dict with fit info (order, AIC, n_obs)
        """
        from statsmodels.tsa.arima.model import ARIMA

        # Clean data
        clean = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 60:
            return {"ticker": ticker, "status": "skipped", "reason": "too few observations"}

        best_order = self.default_order
        best_aic = np.inf

        if auto_order:
            # Grid search over (p, d, q) — keep it practical (max 2)
            for p in range(0, min(self.max_p + 1, 3)):
                for d in range(0, min(self.max_d + 1, 2)):
                    for q in range(0, min(self.max_q + 1, 3)):
                        if p == 0 and q == 0:
                            continue
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                m = ARIMA(clean.values, order=(p, d, q))
                                res = m.fit()
                                if res.aic < best_aic:
                                    best_aic = res.aic
                                    best_order = (p, d, q)
                        except Exception:
                            continue

        # Fit with best order
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(clean.values, order=best_order)
                result = model.fit()
                self.models[ticker] = result
                self.orders[ticker] = best_order
                return {
                    "ticker": ticker,
                    "status": "fitted",
                    "order": best_order,
                    "aic": result.aic,
                    "n_obs": len(clean),
                }
        except Exception as e:
            return {"ticker": ticker, "status": "failed", "reason": str(e)}

    def forecast_ticker(self, ticker: str, steps: int = None) -> float:
        """
        Forecast the next `steps`-day return for a specific ticker.

        Returns the cumulative forecast (sum of step-ahead forecasts).
        """
        if steps is None:
            steps = self.forecast_horizon

        if ticker not in self.models:
            return 0.0

        try:
            result = self.models[ticker]
            fc = result.forecast(steps=steps)
            # Return the cumulative forecast over the horizon
            return float(fc.sum())
        except Exception:
            return 0.0

    def _fit_one_ticker(self, args):
        """Helper for parallel fitting (must be top-level-picklable)."""
        ticker_returns, ticker = args
        import signal as _sig

        # Timeout per ticker (60s max)
        class _Timeout(Exception):
            pass

        def _handler(signum, frame):
            raise _Timeout()

        old = _sig.signal(_sig.SIGALRM, _handler)
        _sig.alarm(30)
        try:
            info = self.fit_ticker(ticker_returns, ticker, auto_order=True)
        except _Timeout:
            info = {"ticker": ticker, "status": "timeout"}
        finally:
            _sig.alarm(0)
            _sig.signal(_sig.SIGALRM, old)
        return info

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict:
        """
        Fit ARIMA per-ticker from panel data.

        X_train must have a '_ticker' column and a '_date' column.
        y_train is the return series aligned with X_train rows.
        """
        if "_ticker" not in X_train.columns:
            raise ValueError("X_train must contain '_ticker' column for ARIMA per-ticker fitting")

        tickers = X_train["_ticker"].unique()
        fitted_count = 0

        for ticker in tickers:
            mask = X_train["_ticker"] == ticker
            ticker_returns = y_train[mask.values].copy()

            # Sort by date if available
            if "_date" in X_train.columns:
                ticker_dates = X_train.loc[mask, "_date"]
                sort_idx = ticker_dates.argsort()
                ticker_returns = ticker_returns.iloc[sort_idx]

            info = self._fit_one_ticker((ticker_returns, ticker))
            if info["status"] == "fitted":
                fitted_count += 1

        self.is_fitted = True
        self.train_metrics = {
            "fitted_tickers": fitted_count,
            "total_tickers": len(tickers),
            "fit_rate": fitted_count / max(len(tickers), 1),
        }
        return self.train_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for panel data.

        X must have '_ticker' column. Returns forecasted cumulative
        return for each row based on the ticker's fitted ARIMA.
        """
        if not self.is_fitted:
            raise ValueError("ARIMA model not fitted.")

        if "_ticker" not in X.columns:
            return np.zeros(len(X))

        predictions = np.zeros(len(X))
        for ticker in X["_ticker"].unique():
            mask = (X["_ticker"] == ticker).values
            predictions[mask] = self.forecast_ticker(ticker, self.forecast_horizon)

        return predictions

    def feature_importance(self) -> Optional[pd.Series]:
        """Return AR/MA coefficient summary across tickers."""
        if not self.models:
            return None

        ar_coeffs = []
        ma_coeffs = []
        for ticker, result in self.models.items():
            params = result.params
            order = self.orders[ticker]
            p, d, q = order
            # AR coefficients
            for i in range(p):
                key = f"ar.L{i+1}"
                if key in result.params.index if hasattr(result.params, 'index') else False:
                    ar_coeffs.append(abs(result.params[key]))
            # MA coefficients
            for i in range(q):
                key = f"ma.L{i+1}"
                if key in result.params.index if hasattr(result.params, 'index') else False:
                    ma_coeffs.append(abs(result.params[key]))

        importance = {}
        if ar_coeffs:
            importance["AR (avg |coeff|)"] = np.mean(ar_coeffs)
        if ma_coeffs:
            importance["MA (avg |coeff|)"] = np.mean(ma_coeffs)
        importance["Fitted tickers"] = len(self.models)

        return pd.Series(importance).sort_values(ascending=False) if importance else None

    def save(self, path: Path):
        """Save all per-ticker models to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "models": {
                ticker: {
                    "params": result.params,
                    "order": self.orders[ticker],
                    "aic": result.aic,
                    "nobs": result.nobs,
                    "resid": result.resid,
                    "fittedvalues": result.fittedvalues,
                    # Store enough for re-forecasting
                    "model_data": result.data.endog,
                }
                for ticker, result in self.models.items()
            },
            "name": self.name,
            "task": self.task,
            "is_fitted": self.is_fitted,
            "train_metrics": self.train_metrics,
            "forecast_horizon": self.forecast_horizon,
            "orders": self.orders,
        }
        joblib.dump(data, path)
        print(f"  Saved {self.name} ({len(self.models)} tickers) to {path}")

    def load(self, path: Path):
        """Load per-ticker models from disk using smooth() for instant restore."""
        from statsmodels.tsa.arima.model import ARIMA

        path = Path(path)
        data = joblib.load(path)
        self.name = data["name"]
        self.task = data["task"]
        self.is_fitted = data["is_fitted"]
        self.train_metrics = data["train_metrics"]
        self.forecast_horizon = data["forecast_horizon"]
        self.orders = data["orders"]

        # Restore models using smooth() — instant, no re-optimization
        self.models = {}
        for ticker, mdata in data["models"].items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    order = mdata["order"]
                    endog = mdata["model_data"]
                    params = mdata["params"]
                    # Convert pandas Series to numpy array if needed
                    if hasattr(params, 'values'):
                        params = params.values
                    params = np.asarray(params, dtype=float)
                    m = ARIMA(endog, order=order)
                    # smooth() restores from params without re-optimizing
                    result = m.smooth(params)
                    self.models[ticker] = result
                    self.orders[ticker] = order
            except Exception:
                pass  # Skip ticker — never re-fit (too slow)

        print(f"  Loaded {self.name} ({len(self.models)} tickers) from {path}")
