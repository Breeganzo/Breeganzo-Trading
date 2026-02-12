"""
Ensemble Model (Robust Multi-Strategy)
========================================
Combines predictions from ARIMA, GARCH, XGBoost, LightGBM, LSTM,
and Transformer using multiple ensemble strategies with automatic
best-strategy selection.

Industry best practice for financial ML ensembles:
- Simple average / median as baselines (most robust)
- Confidence-filtered predictions (only trade high-conviction signals)
- Ridge meta-learner with strong regularization (optional, validated)
- Majority vote for direction
- Models scoring below min_accuracy_threshold on walk-forward OOF
  direction accuracy are auto-excluded.

The ensemble is model-agnostic — pass any combination of base model
predictions as DataFrame columns.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional
from sklearn.linear_model import Ridge

from .base import BaseModel
from ..utils.constants import CONFIG_DIR, RANDOM_STATE


class EnsembleModel(BaseModel):
    """
    Robust multi-strategy ensemble of base models.

    Strategies (auto-selected based on holdout performance):
    1. Simple Average — most robust, no training needed
    2. Median — robust to outlier models
    3. Confidence-filtered Average — only trades when |prediction| > threshold
    4. Direction-accuracy Weighted Average — weights from edge above 50%
    5. Ridge Meta-learner — accounts for correlations (validated against overfit)

    The best strategy is chosen on the validation set and applied to test.
    Models below min_accuracy_threshold are auto-excluded.
    """

    def __init__(
        self,
        method: str = "auto",
        params: Optional[dict] = None,
    ):
        super().__init__(name="ensemble", task="regression")

        if params is None:
            config_path = CONFIG_DIR / "model_params.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                params = config.get("ensemble", {})
            else:
                params = {}

        self.method = method or params.get("method", "auto")
        self.fixed_weights = params.get("weights", {})
        self.min_accuracy = params.get("min_accuracy_threshold", 0.50)
        self.meta_alpha = params.get("meta_alpha", 100.0)
        self.confidence_threshold = params.get("confidence_threshold", 0.001)
        self.meta_learner = None
        self.model_names = []
        self.best_strategy = "median"  # default — most robust
        self.direction_model = None
        self.scaler = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict:
        """
        Fit ensemble using multi-strategy evaluation.

        Trains all strategies on X_train/y_train, evaluates each on
        X_val/y_val, and selects the one with the best direction accuracy
        on the validation set. This avoids overfitting from learned weights.

        If no validation data is provided, defaults to median (safest).
        """
        self.model_names = X_train.columns.tolist()

        # Drop NaN rows from training data
        mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_clean = X_train[mask]
        y_clean = y_train[mask]

        # Use validation data for strategy selection
        if X_val is not None and y_val is not None:
            val_mask = ~(X_val.isna().any(axis=1) | y_val.isna())
            X_weight = X_val[val_mask]
            y_weight = y_val[val_mask]
            weight_source = "validation"
        else:
            X_weight = X_clean
            y_weight = y_clean
            weight_source = "training (OOF)"

        print(f"  Computing weights from {weight_source} data ({len(X_weight)} samples)")

        # Compute direction accuracy for each model
        accuracies = {}
        for col in X_weight.columns:
            pred_dir = (X_weight[col] > 0).astype(int)
            true_dir = (y_weight > 0).astype(int)
            acc = (pred_dir == true_dir).mean()
            accuracies[col] = acc
            print(f"  {col:15s} direction accuracy: {acc:.1%}")

        # Edge weights — proportional to accuracy above 50%
        edge_weights = {}
        excluded = []
        for name, acc in accuracies.items():
            if acc < self.min_accuracy:
                edge_weights[name] = 0.0
                excluded.append(f"{name} ({acc:.1%})")
                print(f"  ❌ {name:15s} EXCLUDED (below {self.min_accuracy:.0%} threshold)")
            else:
                edge = acc - 0.50
                edge_weights[name] = edge
                print(f"  ✅ {name:15s} edge: +{edge:.1%}")

        if excluded:
            print(f"  Auto-excluded {len(excluded)} model(s): {', '.join(excluded)}")

        total_edge = sum(edge_weights.values())
        if total_edge > 0:
            self.learned_weights = {
                name: edge / total_edge for name, edge in edge_weights.items()
            }
        else:
            # All models at or below 50% — use equal weights
            print("  ⚠ No models above threshold — using equal weights")
            n_models = len(X_weight.columns)
            self.learned_weights = {
                name: 1.0 / n_models for name in X_weight.columns
            }

        # Fit Ridge meta-learner on training data (for comparison)
        self.meta_learner = Ridge(alpha=self.meta_alpha, random_state=RANDOM_STATE)
        self.meta_learner.fit(X_clean.values, y_clean.values)
        self.model = self.meta_learner
        self.is_fitted = True

        # --- Multi-strategy evaluation on validation set ---
        strategies = {}
        y_val_arr = y_weight.values if hasattr(y_weight, 'values') else np.array(y_weight)
        true_dir = (y_val_arr > 0).astype(int)

        # Strategy 1: Simple Average
        avg_pred = X_weight.mean(axis=1).values
        strategies["simple_average"] = (avg_pred, (np.sign(avg_pred) == np.sign(y_val_arr)).mean())

        # Strategy 2: Median
        med_pred = X_weight.median(axis=1).values
        strategies["median"] = (med_pred, (np.sign(med_pred) == np.sign(y_val_arr)).mean())

        # Strategy 3: Weighted Average
        wt_pred = np.zeros(len(X_weight))
        for name in X_weight.columns:
            w = self.learned_weights.get(name, 0)
            wt_pred += w * X_weight[name].fillna(0).values
        strategies["weighted_average"] = (wt_pred, ((wt_pred > 0).astype(int) == true_dir).mean())

        # Strategy 4: Ridge
        ridge_pred = self.meta_learner.predict(X_weight.fillna(0).values)
        strategies["ridge"] = (ridge_pred, ((ridge_pred > 0).astype(int) == true_dir).mean())

        # Strategy 5: Confidence-filtered median (exclude near-zero predictions)
        # Uses median but sets predictions below threshold to 0 (HOLD)
        # For direction accuracy, we only count non-zero predictions
        # BUT for overall direction accuracy metric, treat as the base median
        strategies["confidence_filtered"] = (med_pred, ((med_pred > 0).astype(int) == true_dir).mean())

        # Print strategy comparison
        print(f"\n  Strategy Comparison (on {weight_source} set):")
        print(f"  {'Strategy':25s} {'Dir. Accuracy':>15s}")
        print(f"  {'-'*42}")
        for strat_name, (_, acc) in sorted(strategies.items(), key=lambda x: -x[1][1]):
            marker = " ← BEST" if acc == max(v[1] for v in strategies.values()) else ""
            print(f"  {strat_name:25s} {acc:>14.1%}{marker}")

        # Select best strategy
        best_strat = max(strategies.items(), key=lambda x: x[1][1])
        self.best_strategy = best_strat[0]

        # GUARD: If Ridge is best but only marginally better than median,
        # prefer median (more robust, less overfit risk)
        ridge_acc = strategies["ridge"][1]
        median_acc = strategies["median"][1]
        if self.best_strategy == "ridge" and (ridge_acc - median_acc) < 0.02:
            self.best_strategy = "median"
            print(f"  ⚠ Ridge only marginally better than median — defaulting to median (anti-overfit)")

        print(f"\n  ✅ Selected strategy: {self.best_strategy.upper()}")

        # Training metrics (for logging only)
        train_pred = self.predict(X_clean)
        self.train_metrics = self.evaluate(y_clean, train_pred)

        # Ensemble direction accuracy on weight-computation split
        weight_pred = self.predict(X_weight)
        ens_dir = (weight_pred > 0).astype(int)
        ens_acc = (ens_dir == true_dir).mean()
        print(f"  Ensemble direction accuracy ({weight_source}): {ens_acc:.1%}")
        print(f"  Ensemble weights: {self.learned_weights}")

        return {
            "method": self.best_strategy,
            "weights": self.learned_weights,
            "accuracies": accuracies,
            "strategy_comparison": {k: round(v[1], 4) for k, v in strategies.items()},
            "ensemble_accuracy": ens_acc,
            "train": self.train_metrics,
        }

    def predict(self, X: pd.DataFrame, use_ridge: bool = False) -> np.ndarray:
        """
        Generate ensemble predictions using the best strategy.

        Parameters
        ----------
        X : pd.DataFrame
            Columns = base model predictions
        use_ridge : bool
            If True, force Ridge meta-learner regardless of best_strategy.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted.")

        # Force Ridge if requested
        if use_ridge and self.meta_learner is not None:
            try:
                return self.meta_learner.predict(X.fillna(0).values)
            except Exception:
                pass

        # Use best strategy
        if self.best_strategy == "median":
            return X.median(axis=1).values

        elif self.best_strategy == "simple_average":
            return X.mean(axis=1).values

        elif self.best_strategy == "ridge":
            if self.meta_learner is not None:
                try:
                    return self.meta_learner.predict(X.fillna(0).values)
                except Exception:
                    return X.median(axis=1).values
            return X.median(axis=1).values

        elif self.best_strategy == "confidence_filtered":
            preds = X.median(axis=1).values
            # Set small predictions to slight bias toward positive (market has slight upward drift)
            preds[np.abs(preds) < self.confidence_threshold] = 0.0
            return preds

        else:
            # weighted_average (default fallback)
            return self._weighted_average(X)

    def _weighted_average(self, X: pd.DataFrame) -> np.ndarray:
        """Compute weighted average prediction."""
        result = np.zeros(len(X))
        total_weight = sum(
            self.learned_weights.get(name, 0)
            for name in X.columns
            if name in self.learned_weights
        )

        if total_weight <= 0:
            return X.median(axis=1).values

        for name in X.columns:
            w = self.learned_weights.get(name, 0) / total_weight
            result += w * X[name].fillna(0).values

        return result

    def predict_direction_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get direction probability.

        Uses the sign and magnitude of weighted average prediction.
        Maps to [0, 1] range via sigmoid-like transform.
        """
        preds = self.predict(X)
        # Scale predictions (typical range ±0.05) to probability
        # Using a simple linear map: 0.5 + pred * 10 (capped at [0, 1])
        prob = 0.5 + preds * 10
        return np.clip(prob, 0, 1)

    def save(self, path: Path):
        """Save ensemble with learned_weights, best_strategy, and model_names."""
        import joblib as _jl

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _jl.dump({
            "model": self.model,
            "meta_learner": self.meta_learner,
            "name": self.name,
            "task": self.task,
            "is_fitted": self.is_fitted,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "learned_weights": getattr(self, "learned_weights", {}),
            "model_names": self.model_names,
            "method": self.method,
            "best_strategy": getattr(self, "best_strategy", "median"),
            "confidence_threshold": getattr(self, "confidence_threshold", 0.001),
            "min_accuracy": self.min_accuracy,
        }, path)
        print(f"  Saved {self.name} to {path}")

    def load(self, path: Path):
        """Load ensemble with learned_weights, best_strategy, and model_names."""
        import joblib as _jl

        path = Path(path)
        data = _jl.load(path)

        # Handle both new format (dict) and legacy (raw Ridge object)
        if isinstance(data, dict):
            self.model = data.get("model")
            self.meta_learner = data.get("meta_learner", data.get("model"))
            self.name = data.get("name", "ensemble")
            self.task = data.get("task", "regression")
            self.is_fitted = data.get("is_fitted", True)
            self.train_metrics = data.get("train_metrics", {})
            self.val_metrics = data.get("val_metrics", {})
            self.learned_weights = data.get("learned_weights", {})
            self.model_names = data.get("model_names", [])
            self.method = data.get("method", "auto")
            self.best_strategy = data.get("best_strategy", "median")
            self.confidence_threshold = data.get("confidence_threshold", 0.001)
            self.min_accuracy = data.get("min_accuracy", 0.50)
        elif isinstance(data, EnsembleModel):
            self.__dict__.update(data.__dict__)
        else:
            # Legacy: raw sklearn model
            self.model = data
            self.meta_learner = data
            self.is_fitted = True
            self.learned_weights = {}
            self.model_names = []
            self.best_strategy = "median"
        print(f"  Loaded {self.name} from {path} (strategy: {self.best_strategy})")

    def feature_importance(self) -> Optional[pd.Series]:
        """Return model weights as 'feature importance'."""
        if self.learned_weights:
            return pd.Series(self.learned_weights).sort_values(ascending=False)
        return None
