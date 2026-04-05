"""Neural Network models (MLP + LSTM) for sports predictions.

MLP: feedforward network on current-game features.
LSTM: sequence model on last N games per team, captures momentum/fatigue
patterns that fixed-window averages miss.

Package: torch (pip install torch)
"""

import logging
import os
import math
from collections import defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    # Auto-detect GPU
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        GPU_NAME = torch.cuda.get_device_name(0)
        logging.info("GPU detected: %s", GPU_NAME)
    else:
        DEVICE = torch.device("cpu")
        GPU_NAME = None
except (ImportError, OSError):
    HAS_TORCH = False
    DEVICE = None
    GPU_NAME = None
    logging.debug("PyTorch not available. Neural network models disabled.")
    # Stub so class definitions don't crash
    class _ModuleStub:
        pass
    class _nn:
        Module = _ModuleStub
    nn = _nn()


# ── MLP Classifier ─────────────────────────────────────────────────

class MLPNet(nn.Module):
    """Multi-layer perceptron for game outcome classification."""

    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPPredictor:
    """MLP binary classifier for game outcome prediction."""

    def __init__(self, sport="nfl", hidden_dims=(64, 32), lr=0.001,
                 epochs=100, batch_size=32, **kwargs):
        self.sport = sport
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.feature_names = []
        self._fitted = False
        self._mean = None
        self._std = None

    def _normalize(self, X):
        """Z-score normalize features."""
        if self._mean is None:
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0) + 1e-8
        return (X - self._mean) / self._std

    def train(self, X, y, feature_names=None):
        """Train MLP on feature matrix X and binary labels y."""
        if not HAS_TORCH:
            logging.warning("PyTorch not available")
            return

        if isinstance(X, np.ndarray):
            pass
        else:
            X = np.array(X)
        y = np.array(y, dtype=np.float32)

        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        X_norm = self._normalize(X)

        dev = DEVICE or torch.device("cpu")
        X_tensor = torch.FloatTensor(X_norm).to(dev)
        y_tensor = torch.FloatTensor(y).to(dev)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = MLPNet(X.shape[1], self.hidden_dims).to(dev)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCELoss()

        self.model.train()
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(dev), y_batch.to(dev)
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            # Early stopping
            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    logging.debug("MLP early stop at epoch %d (loss=%.4f)", epoch, avg_loss)
                    break

        self._fitted = True
        logging.debug("MLP trained: %d features, final loss=%.4f", X.shape[1], best_loss)

    def predict_proba(self, X):
        """Predict win probability."""
        if not self._fitted or self.model is None:
            return np.full(len(X), 0.5)

        X = np.array(X) if not isinstance(X, np.ndarray) else X
        X_norm = self._normalize(X)
        X_tensor = torch.FloatTensor(X_norm)

        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_tensor).numpy()

        return np.clip(probs, 0.01, 0.99)

    def predict_single(self, features_dict):
        """Predict from a feature dictionary."""
        if not self._fitted:
            return 0.5
        X = np.array([[features_dict.get(f, 0) for f in self.feature_names]])
        return float(self.predict_proba(X)[0])

    def save(self, filepath=None):
        if filepath is None:
            filepath = f"{self.sport}_mlp_model.pt"
        if self.model:
            torch.save({
                "model_state": self.model.state_dict(),
                "feature_names": self.feature_names,
                "hidden_dims": self.hidden_dims,
                "mean": self._mean,
                "std": self._std,
            }, filepath)

    def load(self, filepath=None):
        if filepath is None:
            filepath = f"{self.sport}_mlp_model.pt"
        if os.path.exists(filepath):
            data = torch.load(filepath, weights_only=False)
            self.feature_names = data["feature_names"]
            self.hidden_dims = data["hidden_dims"]
            self._mean = data["mean"]
            self._std = data["std"]
            self.model = MLPNet(len(self.feature_names), self.hidden_dims)
            self.model.load_state_dict(data["model_state"])
            self._fitted = True
            return True
        return False


# ── LSTM Sequence Model ────────────────────────────────────────────

class LSTMNet(nn.Module):
    """LSTM for team game sequences."""

    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),  # *2 for concat of both teams
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, home_seq, away_seq):
        """
        home_seq: (batch, seq_len, features)
        away_seq: (batch, seq_len, features)
        """
        _, (h_home, _) = self.lstm(home_seq)  # h: (n_layers, batch, hidden)
        _, (h_away, _) = self.lstm(away_seq)

        # Use last layer's hidden state
        home_repr = h_home[-1]  # (batch, hidden)
        away_repr = h_away[-1]

        combined = torch.cat([home_repr, away_repr], dim=1)
        return self.fc(combined).squeeze(-1)


class LSTMPredictor:
    """LSTM-based sequence predictor for sports outcomes.

    Processes the last N games for each team as a sequence,
    capturing momentum, fatigue, and form patterns.
    """

    def __init__(self, sport="nfl", seq_len=10, hidden_dim=64,
                 lr=0.001, epochs=80, batch_size=32, **kwargs):
        self.sport = sport
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.feature_names = []
        self._fitted = False
        self._mean = None
        self._std = None
        self.team_sequences = defaultdict(list)

    def add_game(self, team, features_dict):
        """Record a game's features for a team's sequence."""
        self.team_sequences[team].append(features_dict)

    def _get_sequence(self, team):
        """Get padded sequence of last seq_len games for a team."""
        seq = self.team_sequences.get(team, [])
        if not seq:
            return None

        # Get the last seq_len games
        recent = seq[-self.seq_len:]

        # Convert to numpy array
        if not self.feature_names:
            self.feature_names = list(recent[0].keys())

        rows = []
        for game in recent:
            rows.append([game.get(f, 0) for f in self.feature_names])

        arr = np.array(rows, dtype=np.float32)

        # Pad if shorter than seq_len
        if len(arr) < self.seq_len:
            pad = np.zeros((self.seq_len - len(arr), arr.shape[1]), dtype=np.float32)
            arr = np.vstack([pad, arr])

        return arr

    def train(self, home_teams, away_teams, labels, feature_names=None):
        """Train LSTM from accumulated team sequences.

        Args:
            home_teams: list of home team names (one per game)
            away_teams: list of away team names
            labels: binary labels (1=home win)
            feature_names: feature names per game
        """
        if not HAS_TORCH:
            logging.warning("PyTorch not available")
            return

        if feature_names:
            self.feature_names = feature_names

        home_seqs = []
        away_seqs = []
        valid_labels = []

        for home, away, label in zip(home_teams, away_teams, labels):
            h_seq = self._get_sequence(home)
            a_seq = self._get_sequence(away)
            if h_seq is not None and a_seq is not None:
                home_seqs.append(h_seq)
                away_seqs.append(a_seq)
                valid_labels.append(label)

        if len(home_seqs) < 20:
            logging.warning("Not enough sequences for LSTM training (%d)", len(home_seqs))
            return

        home_tensor = torch.FloatTensor(np.array(home_seqs))
        away_tensor = torch.FloatTensor(np.array(away_seqs))
        label_tensor = torch.FloatTensor(np.array(valid_labels, dtype=np.float32))

        input_dim = home_tensor.shape[2]

        dataset = TensorDataset(home_tensor, away_tensor, label_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = LSTMNet(input_dim, self.hidden_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCELoss()

        self.model.train()
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for h_batch, a_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model(h_batch, a_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break

        self._fitted = True
        logging.debug("LSTM trained: seq_len=%d, %d features, loss=%.4f",
                     self.seq_len, input_dim, best_loss)

    def predict_single(self, home_team, away_team):
        """Predict probability from team sequences."""
        if not self._fitted or self.model is None:
            return 0.5

        h_seq = self._get_sequence(home_team)
        a_seq = self._get_sequence(away_team)

        if h_seq is None or a_seq is None:
            return 0.5

        self.model.eval()
        with torch.no_grad():
            h_tensor = torch.FloatTensor(h_seq).unsqueeze(0)
            a_tensor = torch.FloatTensor(a_seq).unsqueeze(0)
            prob = self.model(h_tensor, a_tensor).item()

        return max(0.01, min(0.99, prob))

    def save(self, filepath=None):
        if filepath is None:
            filepath = f"{self.sport}_lstm_model.pt"
        if self.model:
            torch.save({
                "model_state": self.model.state_dict(),
                "feature_names": self.feature_names,
                "seq_len": self.seq_len,
                "hidden_dim": self.hidden_dim,
            }, filepath)

    def load(self, filepath=None):
        if filepath is None:
            filepath = f"{self.sport}_lstm_model.pt"
        if os.path.exists(filepath) and HAS_TORCH:
            data = torch.load(filepath, weights_only=False)
            self.feature_names = data["feature_names"]
            self.seq_len = data["seq_len"]
            self.hidden_dim = data["hidden_dim"]
            input_dim = len(self.feature_names)
            self.model = LSTMNet(input_dim, self.hidden_dim)
            self.model.load_state_dict(data["model_state"])
            self._fitted = True
            return True
        return False
