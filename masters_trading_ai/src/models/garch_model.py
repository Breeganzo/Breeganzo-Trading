"""
GARCH Model (Volatility Forecasting)
=======================================
Generalized Autoregressive Conditional Heteroskedasticity for
per-ticker volatility and return forecasting.

GARCH models capture **volatility clustering** — the stylized fact
that large returns (positive or negative) tend to cluster together.
This is critical for:
  - Position sizing (reduce exposure during high-volatility regimes)
  - Risk management (VaR, CVaR estimation)
  - Ensemble diversification (volatility signal is uncorrelated with
    mean-return predictions from LSTM/XGBoost)

Architecture:
  - Mean model: AR(1) or constant mean
  - Variance model: GARCH(1,1) with Student-t innovations
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
  - Forecasts both conditional mean AND conditional volatility
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


class GARCHModel(BaseModel):
    """
    GARCH wrapper implementing the BaseModel interface.

    Fits one GARCH model per ticker. Outputs:
      - Predicted return (conditional mean forecast)
      - Predicted volatility (conditional std forecast) — bonus signal

    For ensemble integration, only the mean forecast is used as the
    prediction column. Volatility can be accessed separately for
    position sizing and risk management.
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(name="garch", task="regression")

        if params is None:
            config_path = CONFIG_DIR / "model_params.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                params = config.get("garch", {})
            else:
                params = {}

        self.p = params.get("p", 1)  # GARCH order
        self.q = params.get("q", 1)  # ARCH order
        self.mean_model = params.get("mean_model", "AR")
        self.ar_order = params.get("ar_order", 1)
        self.distribution = params.get("distribution", "studentst")
        self.forecast_horizon = params.get("forecast_horizon", 5)
        self.models = {}  # {ticker: fitted GARCH result}
        self.volatility_forecasts = {}  # {ticker: vol forecast}
        self.rescale_factors = {}  # {ticker: rescale factor}

    def fit_ticker(self, returns: pd.Series, ticker: str) -> dict:
        """
        Fit GARCH(p,q) on a single ticker's return series.

        Parameters
        ----------
        returns : pd.Series
            Daily returns for one ticker (chronologically sorted)
        ticker : str
            Ticker symbol

        Returns
        -------
        dict with fit info (params, loglikelihood, n_obs)
        """
        from arch import arch_model

        # Clean and scale returns (arch library works better with percentage returns)
        clean = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 100:
            return {"ticker": ticker, "status": "skipped", "reason": "too few observations"}

        # Scale to percentage returns for numerical stability
        clean_pct = clean * 100

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(
                    clean_pct,
                    mean=self.mean_model,
                    lags=self.ar_order if self.mean_model == "AR" else 0,
                    vol="Garch",
                    p=self.p,
                    q=self.q,
                    dist=self.distribution,
                    rescale=False,
                )
                result = model.fit(disp="off", show_warning=False)
                self.models[ticker] = result
                self.rescale_factors[ticker] = 100.0  # We scaled by 100

                return {
                    "ticker": ticker,
                    "status": "fitted",
                    "loglikelihood": result.loglikelihood,
                    "aic": result.aic,
                    "bic": result.bic,
                    "n_obs": result.nobs,
                    "params": dict(result.params),
                }
        except Exception as e:
            return {"ticker": ticker, "status": "failed", "reason": str(e)}

    def forecast_ticker(
        self, ticker: str, steps: int = None
    ) -> tuple[float, float]:
        """
        Forecast return and volatility for a specific ticker.

        Returns
        -------
        (mean_forecast, vol_forecast) : tuple[float, float]
            mean_forecast: cumulative mean return over horizon
            vol_forecast: average conditional std over horizon
        """
        if steps is None:
            steps = self.forecast_horizon

        if ticker not in self.models:
            return 0.0, 0.0

        try:
            result = self.models[ticker]
            fc = result.forecast(horizon=steps)
            scale = self.rescale_factors.get(ticker, 100.0)

            # Mean forecast (rescale back from percentage)
            mean_fc = fc.mean.iloc[-1].values
            cum_mean = float(mean_fc.sum()) / scale

            # Volatility forecast (rescale back)
            var_fc = fc.variance.iloc[-1].values
            avg_vol = float(np.sqrt(var_fc.mean())) / scale

            self.volatility_forecasts[ticker] = avg_vol
            return cum_mean, avg_vol
        except Exception:
            return 0.0, 0.0

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict:
        """
        Fit GARCH per-ticker from panel data.

        X_train must have '_ticker' and '_date' columns.
        y_train is the return series aligned with X_train rows.
        """
        if "_ticker" not in X_train.columns:
            raise ValueError("X_train must contain '_ticker' column for GARCH per-ticker fitting")

        tickers = X_train["_ticker"].unique()
        fit_info = []
        fitted_count = 0

        for ticker in tickers:
            mask = X_train["_ticker"] == ticker
            ticker_returns = y_train[mask.values].copy()

            # Sort by date
            if "_date" in X_train.columns:
                ticker_dates = X_train.loc[mask, "_date"]
                sort_idx = ticker_dates.argsort()
                ticker_returns = ticker_returns.iloc[sort_idx]

            info = self.fit_ticker(ticker_returns, ticker)
            fit_info.append(info)
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
        Generate return predictions for panel data.

        X must have '_ticker' column. Returns forecasted conditional
        mean return for each row.
        """
        if not self.is_fitted:
            raise ValueError("GARCH model not fitted.")

        if "_ticker" not in X.columns:
            return np.zeros(len(X))

        predictions = np.zeros(len(X))
        for ticker in X["_ticker"].unique():
            mask = (X["_ticker"] == ticker).values
            mean_fc, _ = self.forecast_ticker(ticker, self.forecast_horizon)
            predictions[mask] = mean_fc

        return predictions

    def predict_volatility(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate volatility predictions for panel data.

        Returns forecasted conditional std for each row.
        Useful for position sizing and risk management.
        """
        if not self.is_fitted:
            raise ValueError("GARCH model not fitted.")

        if "_ticker" not in X.columns:
            return np.ones(len(X)) * 0.02  # Default 2% daily vol

        vol_preds = np.zeros(len(X))
        for ticker in X["_ticker"].unique():
            mask = (X["_ticker"] == ticker).values
            _, vol_fc = self.forecast_ticker(ticker, self.forecast_horizon)
            vol_preds[mask] = vol_fc if vol_fc > 0 else 0.02

        return vol_preds

    def feature_importance(self) -> Optional[pd.Series]:
        """Return average GARCH parameters across fitted tickers."""
        if not self.models:
            return None

        all_params = {}
        for ticker, result in self.models.items():
            for param_name, param_val in result.params.items():
                if param_name not in all_params:
                    all_params[param_name] = []
                all_params[param_name].append(abs(param_val))

        importance = {
            name: np.mean(vals) for name, vals in all_params.items()
        }
        importance["Fitted tickers"] = len(self.models)

        return pd.Series(importance).sort_values(ascending=False)

    def save(self, path: Path):
        """Save all per-ticker GARCH models to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model summaries and parameters (not the full arch result,
        # which can have serialization issues). Store enough to re-fit.
        data = {
            "models_data": {
                ticker: {
                    "params": dict(result.params),
                    "aic": result.aic,
                    "bic": result.bic,
                    "nobs": result.nobs,
                    "resid": result.resid.values,
                    "endog": result.model.y,  # Original data
                }
                for ticker, result in self.models.items()
            },
            "name": self.name,
            "task": self.task,
            "is_fitted": self.is_fitted,
            "train_metrics": self.train_metrics,
            "forecast_horizon": self.forecast_horizon,
            "p": self.p,
            "q": self.q,
            "mean_model": self.mean_model,
            "ar_order": self.ar_order,
            "distribution": self.distribution,
            "rescale_factors": self.rescale_factors,
            "orders": {"p": self.p, "q": self.q},
        }
        joblib.dump(data, path)
        print(f"  Saved {self.name} ({len(self.models)} tickers) to {path}")

    def load(self, path: Path):
        """Load per-ticker GARCH models from disk with fast parameter restore."""
        from arch import arch_model

        path = Path(path)
        data = joblib.load(path)
        self.name = data["name"]
        self.task = data["task"]
        self.is_fitted = data["is_fitted"]
        self.train_metrics = data["train_metrics"]
        self.forecast_horizon = data["forecast_horizon"]
        self.p = data.get("p", 1)
        self.q = data.get("q", 1)
        self.mean_model = data.get("mean_model", "AR")
        self.ar_order = data.get("ar_order", 1)
        self.distribution = data.get("distribution", "studentst")
        self.rescale_factors = data.get("rescale_factors", {})

        # Fast restore: use fix() to set params without re-optimization
        self.models = {}
        for ticker, mdata in data["models_data"].items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    endog = mdata["endog"]
                    model = arch_model(
                        endog,
                        mean=self.mean_model,
                        lags=self.ar_order if self.mean_model == "AR" else 0,
                        vol="Garch",
                        p=self.p,
                        q=self.q,
                        dist=self.distribution,
                        rescale=False,
                    )
                    # fix() sets params directly — no optimization loop
                    saved_params = mdata["params"]
                    param_values = list(saved_params.values()) if isinstance(saved_params, dict) else list(saved_params)
                    result = model.fix(param_values)
                    self.models[ticker] = result
                    self.rescale_factors[ticker] = 100.0
            except Exception:
                pass  # Skip ticker — never re-fit (too slow)

        print(f"  Loaded {self.name} ({len(self.models)} tickers) from {path}")
