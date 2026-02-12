"""
LSTM Model (PyTorch)
=====================
Long Short-Term Memory recurrent neural network for sequence-based
stock price range prediction.

LSTM captures temporal dependencies in the feature sequence that
tree-based models cannot — the order of features matters.

Architecture:
    Input (seq_len × n_features)
    → LSTM (2 layers, 128 hidden)
    → Dropout (0.3)
    → Fully Connected → Output (1 or n_horizons)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset, DataLoader

from .base import BaseModel
from ..utils.constants import CONFIG_DIR, RANDOM_STATE


class StockSequenceDataset(Dataset):
    """
    PyTorch Dataset that creates sequences from tabular stock features.

    For each sample at time t, the input is the sequence of features
    from [t - seq_len + 1, ..., t] and the target is the label at t.

    This ensures NO look-ahead: the sequence at time t only uses
    data available at or before time t.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 30):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_val = self.y[idx + self.seq_len - 1]
        return x_seq, y_val


class LSTMNetwork(nn.Module):
    """
    LSTM neural network with dropout and residual connection.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use only the last time step's output
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        x = self.dropout(last_output)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class LSTMModel(BaseModel):
    """
    LSTM wrapper implementing the BaseModel interface.

    Handles sequence construction, training with early stopping,
    and prediction. Uses GPU if available.
    """

    def __init__(
        self,
        task: str = "regression",
        params: Optional[dict] = None,
    ):
        super().__init__(name="lstm", task=task)

        if params is None:
            config_path = CONFIG_DIR / "model_params.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                params = config.get("lstm", {})
            else:
                params = {}

        self.seq_len = params.get("sequence_length", 30)
        self.hidden_size = params.get("hidden_size", 128)
        self.num_layers = params.get("num_layers", 2)
        self.dropout = params.get("dropout", 0.3)
        self.batch_size = params.get("batch_size", 64)
        self.lr = params.get("learning_rate", 0.001)
        self.weight_decay = params.get("weight_decay", 0.0001)
        self.max_epochs = params.get("max_epochs", 100)
        self.patience = params.get("patience", 15)
        self.scheduler_factor = params.get("scheduler_factor", 0.5)
        self.scheduler_patience = params.get("scheduler_patience", 5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_names = None
        self.scaler_mean = None
        self.scaler_std = None

    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Prepare data: select numeric, normalize, create sequences."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        y_values = y.values

        return X_numeric, y_values, numeric_cols

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict:
        """
        Train LSTM with early stopping.

        Steps:
        1. Normalize features (z-score using training stats)
        2. Create sequences of length seq_len
        3. Train with Adam optimizer + ReduceLROnPlateau scheduler
        4. Early stop when validation loss stops improving
        """
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        X_tr, y_tr, self.feature_names = self._prepare_data(X_train, y_train)

        # Normalize using training statistics (prevent look-ahead)
        self.scaler_mean = X_tr.mean(axis=0)
        self.scaler_std = X_tr.std(axis=0) + 1e-10
        X_tr = (X_tr - self.scaler_mean) / self.scaler_std

        # Create datasets
        train_dataset = StockSequenceDataset(X_tr, y_tr, self.seq_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=False)  # Keep temporal order

        val_loader = None
        if X_val is not None and y_val is not None:
            X_v, y_v, _ = self._prepare_data(X_val, y_val)
            X_v = (X_v - self.scaler_mean) / self.scaler_std
            val_dataset = StockSequenceDataset(X_v, y_v, self.seq_len)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                    shuffle=False)

        # Build model
        input_size = X_tr.shape[1]
        output_size = 1
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=output_size,
        ).to(self.device)

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        if self.task == "regression":
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        train_losses = []
        val_losses = []

        for epoch in range(self.max_epochs):
            # Train
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(train_loss)

            # Validate
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        pred = self.model(X_batch)
                        val_loss += criterion(pred, y_batch).item()
                        n_val += 1

                val_loss = val_loss / max(n_val, 1)
                val_losses.append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_fitted = True
        self.train_losses = train_losses
        self.val_losses = val_losses

        # Compute final metrics
        train_pred = self._predict_array(X_tr)
        # Trim predictions to match sequence output length
        y_train_trimmed = y_tr[self.seq_len - 1:]
        if len(train_pred) > len(y_train_trimmed):
            train_pred = train_pred[:len(y_train_trimmed)]
        self.train_metrics = self.evaluate(
            pd.Series(y_train_trimmed[:len(train_pred)]),
            train_pred
        )

        return {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "epochs_trained": len(train_losses),
            "best_val_loss": best_val_loss,
        }

    def _predict_array(self, X: np.ndarray) -> np.ndarray:
        """Predict from numpy array."""
        self.model.eval()
        dataset = StockSequenceDataset(X, np.zeros(len(X)), self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch)
                if self.task == "classification":
                    pred = torch.sigmoid(pred)
                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions) if predictions else np.array([])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions from DataFrame."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        X_numeric = X[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).values
        X_normalized = (X_numeric - self.scaler_mean) / self.scaler_std
        # Replace any remaining NaN/inf from normalization
        X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return self._predict_array(X_normalized)

    def feature_importance(self) -> Optional[pd.Series]:
        """
        LSTM doesn't have built-in feature importance.
        Returns None — use SHAP or gradient-based methods instead.
        """
        return None

    def save(self, path: Path):
        """Save LSTM model in both .pt (PyTorch) and .joblib (portable) formats.

        The .joblib version converts state_dict tensors to numpy arrays,
        making it safe to load without torch.load() pickle issues.
        This is critical for web app deployment (Flask, etc.).
        """
        import joblib as _jl

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state": self.model.state_dict() if self.model else None,
            "config": {
                "input_size": len(self.feature_names) if self.feature_names else 0,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "seq_len": self.seq_len,
            },
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "task": self.task,
            "is_fitted": self.is_fitted,
        }

        # 1. Save .pt (standard PyTorch format)
        torch.save(payload, path)
        print(f"  Saved {self.name} to {path}")

        # 2. Save .joblib (portable — converts tensors to numpy)
        joblib_path = path.with_suffix(".joblib")
        portable = dict(payload)
        if portable["model_state"] is not None:
            portable["model_state"] = {
                k: v.cpu().numpy() for k, v in portable["model_state"].items()
            }
        if isinstance(portable.get("scaler_mean"), np.ndarray):
            pass  # already numpy
        _jl.dump(portable, joblib_path)
        print(f"  Saved {self.name} (portable) to {joblib_path}")

    def load(self, path: Path):
        """Load LSTM model. Tries .pt first (native), falls back to .joblib."""
        import joblib as _jl

        path = Path(path)
        data = None
        loaded_from = None

        # Try .pt first (native PyTorch format — avoids numpy→tensor conversion issues)
        pt_path = path.with_suffix(".pt")
        if pt_path.exists():
            try:
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                loaded_from = pt_path
            except Exception:
                data = None

        # Fall back to .joblib
        if data is None:
            joblib_path = path.with_suffix(".joblib")
            if joblib_path.exists():
                try:
                    data = _jl.load(joblib_path)
                    if data.get("model_state") is not None:
                        data["model_state"] = {
                            k: torch.as_tensor(v).clone()
                            for k, v in data["model_state"].items()
                        }
                    loaded_from = joblib_path
                except Exception:
                    data = None

        if data is None:
            raise FileNotFoundError(f"No model found at {path}")

        self.feature_names = data["feature_names"]
        self.scaler_mean = data["scaler_mean"]
        self.scaler_std = data["scaler_std"]
        self.task = data["task"]
        self.is_fitted = data["is_fitted"]
        self.seq_len = data["config"]["seq_len"]

        if data["model_state"] is not None:
            self.model = LSTMNetwork(
                input_size=data["config"]["input_size"],
                hidden_size=data["config"]["hidden_size"],
                num_layers=data["config"]["num_layers"],
                dropout=data["config"]["dropout"],
            ).to(self.device)
            self.model.load_state_dict(data["model_state"], strict=False)

        print(f"  Loaded {self.name} from {loaded_from}")
