"""
XGBoost Model Wrapper
======================
Gradient boosted trees for tabular financial data.

XGBoost is typically the strongest performer on engineered tabular features.
It handles missing values natively and provides SHAP-based interpretability.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Optional

from .base import BaseModel
from ..utils.constants import CONFIG_DIR, RANDOM_STATE


class XGBoostModel(BaseModel):
    """
    XGBoost wrapper implementing the BaseModel interface.

    Supports both regression (range prediction) and classification (direction).
    Uses early stopping on validation set to prevent overfitting.
    """

    def __init__(
        self,
        task: str = "regression",
        params: Optional[dict] = None,
    ):
        super().__init__(name="xgboost", task=task)

        # Load default params from config
        if params is None:
            config_path = CONFIG_DIR / "model_params.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                key = "regressor" if task == "regression" else "classifier"
                params = config.get("xgboost", {}).get(key, {})
            else:
                params = {}

        self.params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        self.early_stopping_rounds = params.get("early_stopping_rounds", 50)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict:
        """
        Train XGBoost model with early stopping.

        Parameters
        ----------
        X_train, y_train : Training data
        X_val, y_val : Validation data for early stopping

        Returns
        -------
        dict : Training and validation metrics
        """
        # Select only numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train = X_train[numeric_cols].copy()
        if X_val is not None:
            X_val = X_val[numeric_cols].copy()

        # Handle any remaining NaN
        X_train = X_train.fillna(0)
        if X_val is not None:
            X_val = X_val.fillna(0)

        self.feature_names = numeric_cols

        # In XGBoost 2+, early_stopping_rounds is a constructor param
        model_params = dict(self.params)
        if X_val is not None and self.early_stopping_rounds:
            model_params["early_stopping_rounds"] = self.early_stopping_rounds

        if self.task == "regression":
            self.model = xgb.XGBRegressor(**model_params)
        else:
            self.model = xgb.XGBClassifier(**model_params)

        # Fit with eval set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        self.is_fitted = True

        # Compute metrics
        train_pred = self.model.predict(X_train)
        self.train_metrics = self.evaluate(y_train, train_pred)

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            self.val_metrics = self.evaluate(y_val, val_pred)

        return {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "best_iteration": getattr(self.model, "best_iteration", None),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = X[self.feature_names].copy().fillna(0)

        if self.task == "classification":
            # Return probability of class 1
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def feature_importance(self) -> Optional[pd.Series]:
        """Return feature importances sorted by value."""
        if not self.is_fitted:
            return None
        importance = self.model.feature_importances_
        return pd.Series(
            importance, index=self.feature_names
        ).sort_values(ascending=False)

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for model interpretability.

        SHAP (SHapley Additive exPlanations) values show how each feature
        contributes to each individual prediction.
        """
        try:
            import shap
            X = X[self.feature_names].copy().fillna(0)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            return shap_values
        except ImportError:
            print("Install shap package for SHAP analysis: pip install shap")
            return None
