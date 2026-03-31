"""
strategy_nn.py

Three small feedforward neural networks trained on demo data:

  1. WinPredictor      — P(T wins round) given both teams' economy/equipment
  2. AttackPredictor   — P(attack A | attack B | no plant) given T-side features
  3. FormationClassifier — CT defensive formation given economy/context

All models use CUDA when available and are intentionally tiny (< 500 params
each) to avoid overfitting on ~300 training rounds.

Usage
-----
    from strategy_nn import train_all_models, load_models

    models = train_all_models("src/demo")   # train from scratch
    models = load_models("models/")         # or load saved weights

    wp = models["win_predictor"]
    print(wp.predict_single(t_tier=2, ct_tier=1, t_streak=0, ct_streak=2,
                            round_in_half=5, side_is_t=1))
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


# ---------------------------------------------------------------------------
# 1. Win Probability Predictor
# ---------------------------------------------------------------------------

class WinPredictorNet(nn.Module):
    """Predict P(T wins round) from team-level features."""

    def __init__(self, n_features: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class WinPredictor:
    """Wrapper with training, evaluation, and single-sample prediction."""

    FEATURE_COLS = [
        "t_equip_tier", "ct_equip_tier",
        "t_loss_streak", "ct_loss_streak",
        "t_avg_money", "ct_avg_money",
        "round_in_half", "is_second_half",
    ]
    MONEY_SCALE = 16_000.0

    def __init__(self):
        self.model = WinPredictorNet(len(self.FEATURE_COLS)).to(DEVICE)
        self.trained = False

    def _prepare(self, df: pd.DataFrame):
        X = df[self.FEATURE_COLS].copy().astype(float)
        X["t_avg_money"] /= self.MONEY_SCALE
        X["ct_avg_money"] /= self.MONEY_SCALE
        X["t_loss_streak"] /= 5.0
        X["ct_loss_streak"] /= 5.0
        X["t_equip_tier"] /= 2.0
        X["ct_equip_tier"] /= 2.0
        X["round_in_half"] /= 12.0
        y = df["t_won"].values.astype(float)
        return (torch.tensor(X.values, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.float32).to(DEVICE))

    def train(self, df: pd.DataFrame, epochs: int = 400, lr: float = 3e-3,
              verbose: bool = True) -> dict:
        X, y = self._prepare(df)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.model.train()
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            avg_loss = total_loss / len(X)
            history.append(avg_loss)
            if verbose and (epoch + 1) % 100 == 0:
                acc = self._accuracy(X, y)
                print(f"  WinPredictor epoch {epoch+1}/{epochs}: "
                      f"loss={avg_loss:.4f}, acc={acc:.1%}")

        self.trained = True
        acc = self._accuracy(X, y)
        return {"final_loss": history[-1], "accuracy": acc, "epochs": epochs}

    def _accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            pred = (self.model(X) > 0.5).float()
            return (pred == y).float().mean().item()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self._prepare(df)
        self.model.eval()
        with torch.no_grad():
            return self.model(X).cpu().numpy()

    def predict_single(self, t_tier: int, ct_tier: int,
                       t_streak: int, ct_streak: int,
                       round_in_half: int = 5,
                       side_is_t: int = 1) -> float:
        tier_to_money = {0: 1800, 1: 3200, 2: 5500}
        row = pd.DataFrame([{
            "t_equip_tier": t_tier, "ct_equip_tier": ct_tier,
            "t_loss_streak": t_streak, "ct_loss_streak": ct_streak,
            "t_avg_money": tier_to_money.get(t_tier, 3000),
            "ct_avg_money": tier_to_money.get(ct_tier, 3000),
            "round_in_half": round_in_half,
            "is_second_half": 0,
            "t_won": 0,
        }])
        return float(self.predict(row)[0])


# ---------------------------------------------------------------------------
# 2. Attack Site Predictor
# ---------------------------------------------------------------------------

class AttackPredictorNet(nn.Module):
    """Predict P(attack A | attack B | no_plant)."""

    def __init__(self, n_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


ATTACK_LABELS = {"A": 0, "B": 1, "no_plant": 2}
ATTACK_NAMES = {0: "A", 1: "B", 2: "no_plant"}


class AttackPredictor:
    """Predict which bomb site T will target."""

    FEATURE_COLS = [
        "t_avg_money", "t_avg_equip", "t_loss_streak",
        "t_util_count", "t_rifles", "t_smgs",
        "round_in_half", "is_second_half",
        "smoke_A", "smoke_MID",
    ]
    MONEY_SCALE = 16_000.0

    def __init__(self):
        self.model = AttackPredictorNet(len(self.FEATURE_COLS)).to(DEVICE)
        self.trained = False

    def _prepare(self, df: pd.DataFrame):
        X = df[self.FEATURE_COLS].copy().astype(float)
        X["t_avg_money"] /= self.MONEY_SCALE
        X["t_avg_equip"] /= self.MONEY_SCALE
        X["t_loss_streak"] /= 5.0
        X["t_util_count"] /= 25.0
        X["t_rifles"] /= 5.0
        X["t_smgs"] /= 5.0
        X["round_in_half"] /= 12.0

        y = df["attack_site"].map(ATTACK_LABELS).fillna(2).astype(int).values
        return (torch.tensor(X.values, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.long).to(DEVICE))

    def train(self, df: pd.DataFrame, epochs: int = 500, lr: float = 3e-3,
              verbose: bool = True) -> dict:
        X, y = self._prepare(df)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            avg_loss = total_loss / len(X)
            history.append(avg_loss)
            if verbose and (epoch + 1) % 100 == 0:
                acc = self._accuracy(X, y)
                print(f"  AttackPredictor epoch {epoch+1}/{epochs}: "
                      f"loss={avg_loss:.4f}, acc={acc:.1%}")

        self.trained = True
        acc = self._accuracy(X, y)
        return {"final_loss": history[-1], "accuracy": acc, "epochs": epochs}

    def _accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X).argmax(dim=1)
            return (pred == y).float().mean().item()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self._prepare(df)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict_single(self, t_avg_money: float, t_loss_streak: int,
                       t_util_count: int = 10, t_rifles: int = 3,
                       round_in_half: int = 5) -> dict:
        row = pd.DataFrame([{
            "t_avg_money": t_avg_money, "t_avg_equip": t_avg_money * 0.8,
            "t_loss_streak": t_loss_streak,
            "t_util_count": t_util_count, "t_rifles": t_rifles, "t_smgs": 0,
            "round_in_half": round_in_half, "is_second_half": 0,
            "smoke_A": 0, "smoke_MID": 0,
            "attack_site": "no_plant",
        }])
        probs = self.predict_proba(row)[0]
        return {ATTACK_NAMES[i]: float(probs[i]) for i in range(3)}


# ---------------------------------------------------------------------------
# 3. CT Formation Classifier
# ---------------------------------------------------------------------------

FORMATION_CLASSES = ["2-1-2", "1-2-2", "1-1-3", "2-2-1", "3-1-1",
                     "1-1-2", "0-2-3", "2-0-3", "other"]
FORMATION_TO_IDX = {f: i for i, f in enumerate(FORMATION_CLASSES)}


def _map_formation(fmt: str) -> int:
    if fmt in FORMATION_TO_IDX:
        return FORMATION_TO_IDX[fmt]
    return FORMATION_TO_IDX["other"]


class FormationClassifierNet(nn.Module):
    """Predict CT defensive formation class."""

    def __init__(self, n_features: int = 7, n_classes: int = len(FORMATION_CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.ReLU(),
            nn.Linear(12, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FormationClassifier:
    """Predict CT defensive setup."""

    FEATURE_COLS = [
        "ct_avg_money", "ct_loss_streak", "ct_rifles",
        "ct_util_count", "round_in_half", "is_second_half",
        "t_loss_streak",
    ]
    MONEY_SCALE = 16_000.0

    def __init__(self):
        self.model = FormationClassifierNet(len(self.FEATURE_COLS)).to(DEVICE)
        self.trained = False

    def _prepare(self, df: pd.DataFrame):
        X = df[self.FEATURE_COLS].copy().astype(float)
        X["ct_avg_money"] /= self.MONEY_SCALE
        X["ct_loss_streak"] /= 5.0
        X["ct_rifles"] /= 5.0
        X["ct_util_count"] /= 25.0
        X["round_in_half"] /= 12.0
        X["t_loss_streak"] /= 5.0

        y = df["ct_formation"].map(_map_formation).fillna(
            FORMATION_TO_IDX["other"]).astype(int).values
        return (torch.tensor(X.values, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.long).to(DEVICE))

    def train(self, df: pd.DataFrame, epochs: int = 500, lr: float = 3e-3,
              verbose: bool = True) -> dict:
        X, y = self._prepare(df)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            avg_loss = total_loss / len(X)
            history.append(avg_loss)
            if verbose and (epoch + 1) % 100 == 0:
                acc = self._accuracy(X, y)
                print(f"  FormationClassifier epoch {epoch+1}/{epochs}: "
                      f"loss={avg_loss:.4f}, acc={acc:.1%}")

        self.trained = True
        acc = self._accuracy(X, y)
        return {"final_loss": history[-1], "accuracy": acc, "epochs": epochs}

    def _accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X).argmax(dim=1)
            return (pred == y).float().mean().item()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self._prepare(df)
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(X), dim=1).cpu().numpy()

    def predict_single(self, ct_avg_money: float, ct_loss_streak: int,
                       ct_rifles: int = 3, round_in_half: int = 5) -> dict:
        row = pd.DataFrame([{
            "ct_avg_money": ct_avg_money, "ct_loss_streak": ct_loss_streak,
            "ct_rifles": ct_rifles, "ct_util_count": 10,
            "round_in_half": round_in_half, "is_second_half": 0,
            "t_loss_streak": 0,
            "ct_formation": "other",
        }])
        probs = self.predict_proba(row)[0]
        return {FORMATION_CLASSES[i]: float(probs[i])
                for i in range(len(FORMATION_CLASSES))}


# ===========================================================================
# Event Sequence Predictor (LSTM)
# ===========================================================================

N_EVENT_TYPES = 5   # kill, smoke, flash, he, plant
N_SEQ_ZONES = 5     # A, B, MID, CT_BASE, T_BASE
EVENT_DIM = N_EVENT_TYPES + 1 + N_SEQ_ZONES + 1 + 1  # 13 per event
MAX_SEQ_LEN = 30
ATTACK_CLASSES = ["A", "B", "no_plant"]


class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int = EVENT_DIM, hidden: int = 64,
                 n_classes: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(32, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, lengths=None):
        """x: (batch, seq_len, input_dim)"""
        encoded = self.encoder(x)
        out, (h_n, _) = self.lstm(encoded)
        last_hidden = h_n[-1]
        return self.head(last_hidden)


class EventSequencePredictor:
    """LSTM that predicts attack site from a sequence of in-round events."""

    def __init__(self, hidden: int = 64):
        self.model = _LSTMNet(EVENT_DIM, hidden, len(ATTACK_CLASSES)).to(DEVICE)
        self.trained = False
        self.label_map = {c: i for i, c in enumerate(ATTACK_CLASSES)}

    @staticmethod
    def _encode_event(ev: dict) -> list[float]:
        """Encode a single event dict into a fixed-size vector."""
        vec = [0.0] * EVENT_DIM
        # One-hot event type (5 dims, indices 0-4)
        t_idx = ev.get("type_idx", 0)
        if 0 <= t_idx < N_EVENT_TYPES:
            vec[t_idx] = 1.0
        # Actor side (1 dim, index 5)
        vec[N_EVENT_TYPES] = float(ev.get("actor_side_is_t", 0))
        # One-hot zone (5 dims, indices 6-10)
        z_idx = ev.get("zone_idx", 2)
        if 0 <= z_idx < N_SEQ_ZONES:
            vec[N_EVENT_TYPES + 1 + z_idx] = 1.0
        # Time normalized (1 dim, index 11)
        vec[N_EVENT_TYPES + 1 + N_SEQ_ZONES] = float(ev.get("time_norm", 0))
        # Headshot (1 dim, index 12)
        vec[N_EVENT_TYPES + 1 + N_SEQ_ZONES + 1] = float(ev.get("is_headshot", 0))
        return vec

    def _events_to_tensor(self, event_list: list[dict]) -> torch.Tensor:
        """Convert a list of event dicts to a padded tensor (1, seq_len, dim)."""
        seq = [self._encode_event(e) for e in event_list[:MAX_SEQ_LEN]]
        while len(seq) < MAX_SEQ_LEN:
            seq.append([0.0] * EVENT_DIM)
        return torch.tensor([seq], dtype=torch.float32, device=DEVICE)

    def predict(self, events: list[dict]) -> dict[str, float]:
        """Predict attack site from event sequence. Returns class probabilities."""
        self.model.eval()
        with torch.no_grad():
            x = self._events_to_tensor(events)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return {ATTACK_CLASSES[i]: float(probs[i]) for i in range(len(ATTACK_CLASSES))}

    def predict_at_checkpoints(self, events: list[dict],
                               checkpoints: list[int] | None = None,
                               ) -> list[tuple[int, dict[str, float]]]:
        """Predict at multiple event counts to show how prediction evolves.

        Returns list of (n_events, probabilities) tuples.
        """
        if not events:
            return [(0, {"A": 0.33, "B": 0.33, "no_plant": 0.34})]

        if checkpoints is None:
            n = len(events)
            checkpoints = sorted(set(
                [1, max(1, n // 4), max(1, n // 2), max(1, 3 * n // 4), n]))

        results = []
        self.model.eval()
        with torch.no_grad():
            for cp in checkpoints:
                if cp > len(events):
                    cp = len(events)
                x = self._events_to_tensor(events[:cp])
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred = {ATTACK_CLASSES[i]: float(probs[i])
                        for i in range(len(ATTACK_CLASSES))}
                results.append((cp, pred))
        return results

    def train(self, sequences: list[dict], n_epochs: int = 300,
              lr: float = 1e-3, verbose: bool = True) -> dict:
        """Train from extracted event sequences.

        Each sequence dict has: events (list[dict]), attack_site (str)
        """
        X_all = []
        y_all = []

        for seq_data in sequences:
            events = seq_data["events"]
            label = self.label_map.get(seq_data["attack_site"], 2)

            if not events:
                continue

            # Generate training samples at different event counts
            n = len(events)
            sample_points = sorted(set([1, max(1, n // 3),
                                        max(1, 2 * n // 3), n]))

            for sp in sample_points:
                encoded = [self._encode_event(e) for e in events[:sp]]
                while len(encoded) < MAX_SEQ_LEN:
                    encoded.append([0.0] * EVENT_DIM)
                encoded = encoded[:MAX_SEQ_LEN]
                X_all.append(encoded)
                y_all.append(label)

        if not X_all:
            return {"error": "no training data"}

        X = torch.tensor(X_all, dtype=torch.float32, device=DEVICE)
        y = torch.tensor(y_all, dtype=torch.long, device=DEVICE)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(dim=-1) == yb).sum().item()
                total += len(yb)

            if verbose and (epoch + 1) % 50 == 0:
                acc = correct / total if total else 0
                print(f"  LSTM epoch {epoch+1}/{n_epochs}: "
                      f"loss={total_loss/total:.4f}, acc={acc:.1%}")

        self.trained = True
        acc = correct / total if total else 0
        return {"final_loss": total_loss / total, "accuracy": acc,
                "n_samples": total, "n_sequences": len(sequences)}


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_models(models: dict, save_dir: str = "models") -> None:
    os.makedirs(save_dir, exist_ok=True)
    for name, wrapper in models.items():
        torch.save(wrapper.model.state_dict(),
                   os.path.join(save_dir, f"{name}.pt"))
    print(f"Models saved to {save_dir}/")


def load_models(save_dir: str = "models") -> dict:
    models = {
        "win_predictor": WinPredictor(),
        "attack_predictor": AttackPredictor(),
        "formation_classifier": FormationClassifier(),
    }
    for name, wrapper in models.items():
        path = os.path.join(save_dir, f"{name}.pt")
        if os.path.exists(path):
            wrapper.model.load_state_dict(
                torch.load(path, map_location=DEVICE, weights_only=True))
            wrapper.trained = True

    # LSTM event sequence predictor
    lstm = EventSequencePredictor()
    lstm_path = os.path.join(save_dir, "event_sequence_predictor.pt")
    if os.path.exists(lstm_path):
        lstm.model.load_state_dict(
            torch.load(lstm_path, map_location=DEVICE, weights_only=True))
        lstm.trained = True
    models["event_sequence_predictor"] = lstm

    return models


# ---------------------------------------------------------------------------
# Train all from demo data
# ---------------------------------------------------------------------------

def train_all_models(demo_dir: str = "src/demo",
                     save_dir: str = "models",
                     verbose: bool = True) -> dict:
    """Extract data from all demos and train all four NN models
    (3 MLPs + 1 LSTM)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from training_data import extract_all

    if verbose:
        print(f"Using device: {DEVICE}")
        if DEVICE.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print()

    data = extract_all(demo_dir, include_rl=False, include_sequences=True,
                       verbose=verbose)
    df = data["rounds"]
    sequences = data["event_sequences"]

    if df.empty:
        print("No training data extracted!")
        return {}

    if verbose:
        print(f"\nTraining on {len(df)} rounds from {df['demo'].nunique()} demos")
        print(f"Event sequences: {len(sequences)} rounds")
        print(f"{'='*60}\n")

    models = {
        "win_predictor": WinPredictor(),
        "attack_predictor": AttackPredictor(),
        "formation_classifier": FormationClassifier(),
    }

    # Train win predictor
    if verbose:
        print("--- Win Predictor ---")
    stats_wp = models["win_predictor"].train(df, verbose=verbose)
    if verbose:
        print(f"  Final: loss={stats_wp['final_loss']:.4f}, "
              f"acc={stats_wp['accuracy']:.1%}\n")

    # Train attack predictor
    if verbose:
        print("--- Attack Predictor ---")
    stats_ap = models["attack_predictor"].train(df, verbose=verbose)
    if verbose:
        print(f"  Final: loss={stats_ap['final_loss']:.4f}, "
              f"acc={stats_ap['accuracy']:.1%}\n")

    # Train formation classifier
    if verbose:
        print("--- Formation Classifier ---")
    stats_fc = models["formation_classifier"].train(df, verbose=verbose)
    if verbose:
        print(f"  Final: loss={stats_fc['final_loss']:.4f}, "
              f"acc={stats_fc['accuracy']:.1%}\n")

    # Train LSTM event sequence predictor
    if verbose:
        print("--- Event Sequence Predictor (LSTM) ---")
    lstm = EventSequencePredictor()
    if sequences:
        stats_lstm = lstm.train(sequences, n_epochs=300, verbose=verbose)
        if verbose:
            print(f"  Final: loss={stats_lstm['final_loss']:.4f}, "
                  f"acc={stats_lstm['accuracy']:.1%}, "
                  f"samples={stats_lstm['n_samples']}\n")
    else:
        if verbose:
            print("  No event sequences available for training.\n")
    models["event_sequence_predictor"] = lstm

    save_models(models, save_dir)

    if verbose:
        print(f"\n{'='*60}")
        print("  Model parameter counts:")
        for name, wrapper in models.items():
            n_params = sum(p.numel() for p in wrapper.model.parameters())
            print(f"    {name:25s}: {n_params:,} parameters")
        print(f"{'='*60}")

        print("\n--- Sample predictions ---")
        wp = models["win_predictor"]
        print(f"  T full buy vs CT eco:   P(T win) = "
              f"{wp.predict_single(t_tier=2, ct_tier=0, t_streak=0, ct_streak=1):.1%}")
        print(f"  T eco vs CT full buy:   P(T win) = "
              f"{wp.predict_single(t_tier=0, ct_tier=2, t_streak=1, ct_streak=0):.1%}")
        print(f"  T full vs CT full:      P(T win) = "
              f"{wp.predict_single(t_tier=2, ct_tier=2, t_streak=0, ct_streak=0):.1%}")

        ap = models["attack_predictor"]
        pred = ap.predict_single(t_avg_money=5000, t_loss_streak=0)
        print(f"\n  T attack (rich, streak 0): {pred}")
        pred = ap.predict_single(t_avg_money=2000, t_loss_streak=2)
        print(f"  T attack (poor, streak 2): {pred}")

        if lstm.trained:
            sample_events = [
                {"type_idx": 0, "actor_side_is_t": 1, "zone_idx": 1,
                 "time_norm": 0.2, "is_headshot": 0},
                {"type_idx": 1, "actor_side_is_t": 1, "zone_idx": 1,
                 "time_norm": 0.25, "is_headshot": 0},
            ]
            pred = lstm.predict(sample_events)
            print(f"\n  LSTM (T kill at B + T smoke at B): {pred}")

    return models


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_all_models()
