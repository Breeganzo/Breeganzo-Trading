"""
Temporal Fusion Transformer Model
===================================
Multi-horizon forecasting with interpretable attention.

The TFT architecture provides:
1. Variable selection networks — learns which features matter
2. Gated residual networks — controls information flow
3. Multi-head attention — captures long-range temporal dependencies
4. Multi-horizon output — predicts 1, 5, 10, 20 days simultaneously

Simplified implementation using PyTorch (no pytorch-forecasting dependency
for portability). For the full TFT, install pytorch-forecasting.
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
from ..utils.constants import CONFIG_DIR, RANDOM_STATE, HORIZONS


class MultiHorizonDataset(Dataset):
    """Dataset that returns sequences and multi-horizon targets."""

    def __init__(self, X: np.ndarray, y_dict: dict[int, np.ndarray], seq_len: int = 60):
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y_dict : dict, {horizon: np.ndarray of targets}
        seq_len : int
        """
        self.X = torch.FloatTensor(X)
        self.y_dict = {h: torch.FloatTensor(v) for h, v in y_dict.items()}
        self.seq_len = seq_len
        self.horizons = sorted(y_dict.keys())

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_vals = torch.stack([
            self.y_dict[h][idx + self.seq_len - 1]
            for h in self.horizons
        ])
        return x_seq, y_vals


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network — controls information flow."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        self.gate = nn.Linear(output_size, output_size)
        self.sigmoid = nn.Sigmoid()

        # Skip connection
        if input_size != output_size:
            self.skip = nn.Linear(input_size, output_size)
        else:
            self.skip = None

    def forward(self, x):
        skip = self.skip(x) if self.skip is not None else x
        h = self.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        gate = self.sigmoid(self.gate(h))
        return self.layer_norm(gate * h + (1 - gate) * skip)


class SimplifiedTFT(nn.Module):
    """
    Simplified Temporal Fusion Transformer.

    Architecture:
    1. Input projection layer
    2. GRN-based feature selection
    3. LSTM encoder (captures local temporal patterns)
    4. Multi-head self-attention (captures long-range dependencies)
    5. Multi-horizon output heads
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_attention_heads: int = 4,
        dropout: float = 0.2,
        n_horizons: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Variable selection
        self.variable_selection = GatedResidualNetwork(
            input_size, hidden_size, hidden_size, dropout
        )

        # Temporal processing (LSTM)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_layer_norm = nn.LayerNorm(hidden_size)

        # Output GRN
        self.output_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )

        # Multi-horizon output heads
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_horizons)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Parameters
        ----------
        x : tensor, shape (batch, seq_len, input_size)

        Returns
        -------
        tensor, shape (batch, n_horizons)
        """
        batch_size, seq_len, _ = x.shape

        # Variable selection (applied per time step)
        selected = self.variable_selection(x)  # (batch, seq_len, hidden)

        # LSTM
        lstm_out, _ = self.lstm(selected)  # (batch, seq_len, hidden)

        # Self-attention
        attn_out, self.attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        attn_out = self.attn_layer_norm(attn_out + lstm_out)  # Residual

        # Use last time step
        last = attn_out[:, -1, :]  # (batch, hidden)

        # Output GRN
        output = self.output_grn(last)
        output = self.dropout(output)

        # Multi-horizon predictions
        predictions = torch.cat([
            head(output) for head in self.output_heads
        ], dim=-1)  # (batch, n_horizons)

        return predictions


class TransformerModel(BaseModel):
    """
    Simplified TFT wrapper implementing the BaseModel interface.

    Predicts multiple horizons simultaneously (1, 5, 10, 20 days).
    """

    def __init__(
        self,
        task: str = "regression",
        params: Optional[dict] = None,
        horizons: list[int] = None,
    ):
        super().__init__(name="transformer", task=task)

        if params is None:
            config_path = CONFIG_DIR / "model_params.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                params = config.get("transformer", {})
            else:
                params = {}

        self.seq_len = params.get("sequence_length", 60)
        self.hidden_size = params.get("hidden_size", 64)
        self.num_attention_heads = params.get("num_attention_heads", 4)
        self.dropout_rate = params.get("dropout", 0.2)
        self.batch_size = params.get("batch_size", 64)
        self.lr = params.get("learning_rate", 0.001)
        self.max_epochs = params.get("max_epochs", 80)
        self.patience = params.get("patience", 10)
        self.grad_clip = params.get("gradient_clip_val", 0.5)

        self.horizons = horizons or HORIZONS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_names = None
        self.scaler_mean = None
        self.scaler_std = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | dict,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series | dict] = None,
    ) -> dict:
        """
        Train Simplified TFT.

        y_train can be:
        - pd.Series: single horizon (will be wrapped in dict)
        - dict: {horizon: pd.Series} for multi-horizon
        """
        torch.manual_seed(RANDOM_STATE)

        # Prepare features
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = numeric_cols
        X_tr = X_train[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

        self.scaler_mean = X_tr.mean(axis=0)
        self.scaler_std = X_tr.std(axis=0) + 1e-10
        X_tr = (X_tr - self.scaler_mean) / self.scaler_std

        # Prepare targets
        if isinstance(y_train, pd.Series):
            y_dict_train = {1: y_train.values}
        else:
            y_dict_train = {h: v.values for h, v in y_train.items()}

        train_dataset = MultiHorizonDataset(X_tr, y_dict_train, self.seq_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_v = X_val[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
            X_v = (X_v - self.scaler_mean) / self.scaler_std
            if isinstance(y_val, pd.Series):
                y_dict_val = {1: y_val.values}
            else:
                y_dict_val = {h: v.values for h, v in y_val.items()}
            val_dataset = MultiHorizonDataset(X_v, y_dict_val, self.seq_len)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Build model
        n_horizons = len(y_dict_train)
        self.model = SimplifiedTFT(
            input_size=len(numeric_cols),
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout_rate,
            n_horizons=n_horizons,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                n_val = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        pred = self.model(X_batch)
                        val_loss += criterion(pred, y_batch).item()
                        n_val += 1

                val_loss /= max(n_val, 1)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True
        return {"epochs_trained": epoch + 1, "best_val_loss": best_val_loss}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict all horizons. Returns (n_samples, n_horizons)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        X_numeric = X[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).values
        X_normalized = (X_numeric - self.scaler_mean) / self.scaler_std
        X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=0.0, neginf=0.0)

        self.model.eval()
        dummy_y = {h: np.zeros(len(X_normalized)) for h in [1]}
        dataset = MultiHorizonDataset(X_normalized, dummy_y, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch)
                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions) if predictions else np.array([])

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Return attention weights from last forward pass."""
        if hasattr(self.model, "attention_weights") and self.model.attention_weights is not None:
            return self.model.attention_weights.detach().cpu().numpy()
        return None

    def feature_importance(self) -> Optional[pd.Series]:
        """TFT uses attention — see get_attention_weights() instead."""
        return None

    def save(self, path: Path):
        """Save TFT model in both .pt (PyTorch) and .joblib (portable) formats.

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
                "num_attention_heads": self.num_attention_heads,
                "dropout": self.dropout_rate,
                "seq_len": self.seq_len,
                "n_horizons": len(self.horizons),
            },
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "horizons": self.horizons,
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
        _jl.dump(portable, joblib_path)
        print(f"  Saved {self.name} (portable) to {joblib_path}")

    def load(self, path: Path):
        """Load TFT model. Tries .pt first (native), falls back to .joblib."""
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
        self.horizons = data["horizons"]
        self.task = data["task"]
        self.is_fitted = data["is_fitted"]
        self.seq_len = data["config"]["seq_len"]

        if data["model_state"] is not None:
            cfg = data["config"]
            self.model = SimplifiedTFT(
                input_size=cfg["input_size"],
                hidden_size=cfg["hidden_size"],
                num_attention_heads=cfg["num_attention_heads"],
                dropout=cfg["dropout"],
                n_horizons=cfg["n_horizons"],
            ).to(self.device)
            self.model.load_state_dict(data["model_state"], strict=False)

        print(f"  Loaded {self.name} from {loaded_from}")
