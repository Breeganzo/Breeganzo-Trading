"""
LightGBM Model Wrapper
========================
Histogram-based gradient boosting — faster than XGBoost on large datasets.

LightGBM uses leaf-wise tree growth (vs level-wise in XGBoost),
which often leads to better accuracy but risks overfitting on small data.
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Optional

from .base import BaseModel
from ..utils.constants import CONFIG_DIR, RANDOM_STATE


class LightGBMModel(BaseModel):
    """
    LightGBM wrapper implementing the BaseModel interface.

    Advantages over XGBoost:
    - Faster training (histogram-based, leaf-wise)
    - Lower memory usage
    - Native categorical feature support
    - Better handling of high-cardinality features
    """

    def __init__(
        self,
        task: str = "regression",
        params: Optional[dict] = None,
    ):
        super().__init__(name="lightgbm", task=task)

        if params is None:
            config_path = CONFIG_DIR / "model_params.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                key = "regressor" if task == "regression" else "classifier"
                params = config.get("lightgbm", {}).get(key, {})
            else:
                params = {}

        self.params = params
        self.early_stopping_rounds = params.pop("early_stopping_rounds", 50)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict:
        """Train LightGBM with early stopping."""
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train = X_train[numeric_cols].copy()
        if X_val is not None:
            X_val = X_val[numeric_cols].copy()

        X_train = X_train.fillna(0)
        if X_val is not None:
            X_val = X_val.fillna(0)

        self.feature_names = numeric_cols

        if self.task == "regression":
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            self.model = lgb.LGBMClassifier(**self.params)

        callbacks = [lgb.log_evaluation(period=0)]  # Suppress output
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        self.is_fitted = True

        train_pred = self.model.predict(X_train)
        self.train_metrics = self.evaluate(y_train, train_pred)

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            self.val_metrics = self.evaluate(y_val, val_pred)

        return {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "best_iteration": getattr(self.model, "best_iteration_", None),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = X[self.feature_names].copy().fillna(0)

        if self.task == "classification":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def feature_importance(self) -> Optional[pd.Series]:
        """Return feature importances."""
        if not self.is_fitted:
            return None
        importance = self.model.feature_importances_
        return pd.Series(
            importance, index=self.feature_names
        ).sort_values(ascending=False)

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for interpretability."""
        try:
            import shap
            X = X[self.feature_names].copy().fillna(0)
            explainer = shap.TreeExplainer(self.model)
            return explainer.shap_values(X)
        except ImportError:
            print("Install shap: pip install shap")
            return None
