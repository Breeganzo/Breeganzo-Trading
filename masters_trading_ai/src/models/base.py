"""
Abstract Base Model
====================
All models inherit from this interface to ensure consistent API
across XGBoost, LightGBM, LSTM, and Transformer implementations.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.

    Every model must implement:
    - fit(X_train, y_train, X_val, y_val)
    - predict(X_test)
    - feature_importance()
    - save(path)
    - load(path)
    """

    def __init__(self, name: str, task: str = "regression"):
        """
        Parameters
        ----------
        name : str
            Model name (e.g., "xgboost", "lightgbm", "lstm")
        task : str
            "regression" (predict range) or "classification" (predict direction)
        """
        self.name = name
        self.task = task
        self.model = None
        self.is_fitted = False
        self.train_metrics = {}
        self.val_metrics = {}

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict:
        """
        Train the model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training targets
        X_val : pd.DataFrame, optional
            Validation features (for early stopping)
        y_val : pd.Series, optional
            Validation targets

        Returns
        -------
        dict
            Training metrics {metric_name: value}
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on

        Returns
        -------
        np.ndarray
            Predictions (regression values or class probabilities)
        """
        pass

    @abstractmethod
    def feature_importance(self) -> Optional[pd.Series]:
        """
        Return feature importances.

        Returns
        -------
        pd.Series or None
            Feature names → importance values, sorted descending.
            None if model doesn't support feature importance.
        """
        pass

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "name": self.name,
            "task": self.task,
            "is_fitted": self.is_fitted,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "feature_names": getattr(self, "feature_names", None),
        }, path)
        print(f"  Saved {self.name} to {path}")

    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)
        data = joblib.load(path)
        self.model = data["model"]
        self.name = data["name"]
        self.task = data["task"]
        self.is_fitted = data["is_fitted"]
        self.train_metrics = data["train_metrics"]
        self.val_metrics = data["val_metrics"]
        self.feature_names = data.get("feature_names", None)
        print(f"  Loaded {self.name} from {path}")

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> dict:
        """
        Evaluate predictions against ground truth.

        Returns
        -------
        dict
            Metrics appropriate for the task type.
        """
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            accuracy_score, precision_score, recall_score, f1_score,
        )

        if self.task == "regression":
            return {
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MAE": mean_absolute_error(y_true, y_pred),
                "R²": r2_score(y_true, y_pred),
                "MAPE": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100,
            }
        else:
            y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred
            return {
                "Accuracy": accuracy_score(y_true, y_pred_binary),
                "Precision": precision_score(y_true, y_pred_binary, zero_division=0),
                "Recall": recall_score(y_true, y_pred_binary, zero_division=0),
                "F1": f1_score(y_true, y_pred_binary, zero_division=0),
            }

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', task='{self.task}', {status})"
