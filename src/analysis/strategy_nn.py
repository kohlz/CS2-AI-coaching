"""
strategy_nn.py

Neural network models for CS2 coaching: pre-round formation and attack
predictors plus LSTM classifiers that track in-round formation/attack state.

Classes: PreRoundFormation, PreRoundAttack, FormationClassifier_T, FormationClassifier_CT.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


FORMATION_CLASSES = ["2-1-2", "1-2-2", "1-1-3", "2-2-1", "3-1-1",
                     "1-1-2", "0-2-3", "2-0-3", "other"]
FORMATION_TO_IDX = {f: i for i, f in enumerate(FORMATION_CLASSES)}


def _map_formation(fmt: str) -> int:
    if fmt in FORMATION_TO_IDX:
        return FORMATION_TO_IDX[fmt]
    return FORMATION_TO_IDX["other"]


class PreRoundFormationNet(nn.Module):
    """Predict enemy formation from HMM tier probabilities + context."""

    def __init__(self, n_features: int = 8, n_classes: int = len(FORMATION_CLASSES),
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreRoundFormation:
    """Pre-round formation predictor using HMM economy tier probabilities
    plus prior-round tendency features."""

    ECON_COLS = [
        "p_broke", "p_low", "p_medium", "p_high", "p_rich",
        "predicted_avg_money", "round_in_half", "is_second_half",
    ]
    PRIOR_COLS = [
        "prev_plant_A", "prev_plant_B", "prev_plant_none", "prev_no_history",
        "prev_t_won", "prev_t_tier", "prev_ct_tier",
        "rounds_since_plant_A", "rounds_since_plant_B", "streak_same_site",
    ]
    FEATURE_COLS = ECON_COLS + PRIOR_COLS
    MONEY_SCALE = 16_000.0
    TARGET_COL = "ct_formation"

    def __init__(self):
        self.model = PreRoundFormationNet(len(self.FEATURE_COLS)).to(DEVICE)
        self.trained = False

    def _prepare(self, df: pd.DataFrame):
        df = df.copy()
        for col in self.PRIOR_COLS:
            if col not in df.columns:
                if col == "prev_no_history":
                    df[col] = 1
                elif col == "prev_t_won":
                    df[col] = 0.5
                else:
                    df[col] = 0.0

        X = df[self.FEATURE_COLS].copy().astype(float)
        X["predicted_avg_money"] /= self.MONEY_SCALE
        X["round_in_half"] /= 12.0

        y = df[self.TARGET_COL].map(_map_formation).fillna(
            FORMATION_TO_IDX["other"]).astype(int).values
        return (torch.tensor(X.values, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.long).to(DEVICE))

    def train(self, df: pd.DataFrame, epochs: int = 800, lr: float = 3e-3,
              weight_decay: float = 1e-3, patience: int = 80,
              verbose: bool = True) -> dict:
        X, y = self._prepare(df)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr,
                               weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=40)
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        best_state = None
        wait = 0

        loss_history: list[float] = []
        acc_history: list[float] = []
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            avg_loss = total_loss / len(X)
            loss_history.append(avg_loss)
            acc_history.append(self._accuracy(X, y))
            scheduler.step(avg_loss)

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in
                              self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if verbose and (epoch + 1) % 100 == 0:
                print(f"  PreRoundFormation epoch {epoch+1}/{epochs}: "
                      f"loss={avg_loss:.4f}, acc={acc_history[-1]:.1%}")

            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.trained = True
        acc = self._accuracy(X, y)
        return {"final_loss": best_loss, "accuracy": acc,
                "epochs": epoch + 1,
                "loss_history": loss_history,
                "acc_history": acc_history}

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

    def predict_single(self, tier_probs: dict, predicted_avg_money: float,
                       round_in_half: int = 5,
                       is_second_half: int = 0,
                       prior: dict | None = None) -> dict:
        """Predict formation distribution. ``prior`` carries prior-round
        tendency features (see PRIOR_COLS); when omitted falls back to
        "no history" defaults."""
        prior = prior or {}
        row = pd.DataFrame([{
            "p_broke": tier_probs.get("BROKE", 0.2),
            "p_low": tier_probs.get("LOW", 0.2),
            "p_medium": tier_probs.get("MEDIUM", 0.2),
            "p_high": tier_probs.get("HIGH", 0.2),
            "p_rich": tier_probs.get("RICH", 0.2),
            "predicted_avg_money": predicted_avg_money,
            "round_in_half": round_in_half,
            "is_second_half": is_second_half,
            "prev_plant_A": prior.get("prev_plant_A", 0),
            "prev_plant_B": prior.get("prev_plant_B", 0),
            "prev_plant_none": prior.get("prev_plant_none", 0),
            "prev_no_history": prior.get("prev_no_history", 1),
            "prev_t_won": prior.get("prev_t_won", 0.5),
            "prev_t_tier": prior.get("prev_t_tier", 0.0),
            "prev_ct_tier": prior.get("prev_ct_tier", 0.0),
            "rounds_since_plant_A": prior.get("rounds_since_plant_A", 1.0),
            "rounds_since_plant_B": prior.get("rounds_since_plant_B", 1.0),
            "streak_same_site": prior.get("streak_same_site", 0.0),
            self.TARGET_COL: "other",
        }])
        probs = self.predict_proba(row)[0]
        return {FORMATION_CLASSES[i]: float(probs[i])
                for i in range(len(FORMATION_CLASSES))}


ATTACK_SITE_CLASSES = ["A", "B", "no_plant"]
ATTACK_SITE_TO_IDX = {c: i for i, c in enumerate(ATTACK_SITE_CLASSES)}


class PreRoundAttack(PreRoundFormation):
    """Same feature vector as PreRoundFormation; predicts T attack site
    (A / B / no_plant) instead of CT formation."""

    TARGET_COL = "attack_site"

    def __init__(self):
        self.model = PreRoundFormationNet(
            n_features=len(self.FEATURE_COLS),
            n_classes=len(ATTACK_SITE_CLASSES),
        ).to(DEVICE)
        self.trained = False

    def _prepare(self, df: pd.DataFrame):
        df = df.copy()
        for col in self.PRIOR_COLS:
            if col not in df.columns:
                if col == "prev_no_history":
                    df[col] = 1
                elif col == "prev_t_won":
                    df[col] = 0.5
                else:
                    df[col] = 0.0

        X = df[self.FEATURE_COLS].copy().astype(float)
        X["predicted_avg_money"] /= self.MONEY_SCALE
        X["round_in_half"] /= 12.0

        def _map_attack(s: str) -> int:
            return ATTACK_SITE_TO_IDX.get(str(s), ATTACK_SITE_TO_IDX["no_plant"])

        y = df[self.TARGET_COL].map(_map_attack).fillna(
            ATTACK_SITE_TO_IDX["no_plant"]).astype(int).values
        return (torch.tensor(X.values, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.long).to(DEVICE))

    def predict_single(self, tier_probs: dict, predicted_avg_money: float,
                       round_in_half: int = 5,
                       is_second_half: int = 0,
                       prior: dict | None = None) -> dict:
        prior = prior or {}
        row = pd.DataFrame([{
            "p_broke": tier_probs.get("BROKE", 0.2),
            "p_low": tier_probs.get("LOW", 0.2),
            "p_medium": tier_probs.get("MEDIUM", 0.2),
            "p_high": tier_probs.get("HIGH", 0.2),
            "p_rich": tier_probs.get("RICH", 0.2),
            "predicted_avg_money": predicted_avg_money,
            "round_in_half": round_in_half,
            "is_second_half": is_second_half,
            "prev_plant_A": prior.get("prev_plant_A", 0),
            "prev_plant_B": prior.get("prev_plant_B", 0),
            "prev_plant_none": prior.get("prev_plant_none", 0),
            "prev_no_history": prior.get("prev_no_history", 1),
            "prev_t_won": prior.get("prev_t_won", 0.5),
            "prev_t_tier": prior.get("prev_t_tier", 0.0),
            "prev_ct_tier": prior.get("prev_ct_tier", 0.0),
            "rounds_since_plant_A": prior.get("rounds_since_plant_A", 1.0),
            "rounds_since_plant_B": prior.get("rounds_since_plant_B", 1.0),
            "streak_same_site": prior.get("streak_same_site", 0.0),
            self.TARGET_COL: "no_plant",
        }])
        probs = self.predict_proba(row)[0]
        return {ATTACK_SITE_CLASSES[i]: float(probs[i])
                for i in range(len(ATTACK_SITE_CLASSES))}


N_EVENT_TYPES = 6   # kill, smoke, flash, he, plant, molotov
N_SEQ_ZONES = 5     # A, B, MID, CT_BASE, T_BASE
EVENT_DIM = N_EVENT_TYPES + 1 + N_SEQ_ZONES + 1 + 1
MAX_SEQ_LEN = 30
ATTACK_CLASSES = ["A", "B", "no_plant"]


class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int = EVENT_DIM, hidden: int = 64,
                 n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(32, hidden, num_layers=2, batch_first=True,
                            dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, lengths=None):
        """x: (batch, seq_len, input_dim)"""
        encoded = self.encoder(x)
        out, (h_n, _) = self.lstm(encoded)
        last_hidden = self.drop(h_n[-1])
        return self.head(last_hidden)


class FormationClassifier_T:
    """LSTM that predicts T attack site (A/B/no_plant) from event sequences.
    Used when playing CT to predict where T is attacking."""

    def __init__(self, hidden: int = 64):
        self.model = _LSTMNet(EVENT_DIM, hidden, len(ATTACK_CLASSES)).to(DEVICE)
        self.trained = False
        self.label_map = {c: i for i, c in enumerate(ATTACK_CLASSES)}

    @staticmethod
    def _encode_event(ev: dict) -> list[float]:
        vec = [0.0] * EVENT_DIM
        t_idx = ev.get("type_idx", 0)
        if 0 <= t_idx < N_EVENT_TYPES:
            vec[t_idx] = 1.0
        vec[N_EVENT_TYPES] = float(ev.get("actor_side_is_t", 0))
        z_idx = ev.get("zone_idx", 2)
        if 0 <= z_idx < N_SEQ_ZONES:
            vec[N_EVENT_TYPES + 1 + z_idx] = 1.0
        vec[N_EVENT_TYPES + 1 + N_SEQ_ZONES] = float(ev.get("time_norm", 0))
        vec[N_EVENT_TYPES + 1 + N_SEQ_ZONES + 1] = float(ev.get("is_headshot", 0))
        return vec

    def _events_to_tensor(self, event_list: list[dict]) -> torch.Tensor:
        seq = [self._encode_event(e) for e in event_list[:MAX_SEQ_LEN]]
        while len(seq) < MAX_SEQ_LEN:
            seq.append([0.0] * EVENT_DIM)
        return torch.tensor([seq], dtype=torch.float32, device=DEVICE)

    def predict(self, events: list[dict]) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            x = self._events_to_tensor(events)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return {ATTACK_CLASSES[i]: float(probs[i]) for i in range(len(ATTACK_CLASSES))}

    def predict_at_checkpoints(self, events: list[dict],
                               checkpoints: list[int] | None = None,
                               ) -> list[tuple[int, dict[str, float]]]:
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

    def train(self, sequences: list[dict], n_epochs: int = 500,
              lr: float = 1e-3, weight_decay: float = 1e-4,
              patience: int = 60, verbose: bool = True) -> dict:
        X_all = []
        y_all = []

        for seq_data in sequences:
            events = seq_data["events"]
            label = self.label_map.get(seq_data["attack_site"], 2)

            if not events:
                continue

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

        optimizer = optim.Adam(self.model.parameters(), lr=lr,
                               weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=30)
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        best_state = None
        wait = 0
        loss_history: list[float] = []
        acc_history: list[float] = []

        for epoch in range(n_epochs):
            self.model.train()
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

            avg_loss = total_loss / total if total else 0
            epoch_acc = correct / total if total else 0
            loss_history.append(avg_loss)
            acc_history.append(epoch_acc)
            scheduler.step(avg_loss)

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in
                              self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  FormationClassifier_T epoch {epoch+1}/{n_epochs}: "
                      f"loss={avg_loss:.4f}, acc={epoch_acc:.1%}")

            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.trained = True
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X).argmax(dim=-1)
            acc = (pred == y).float().mean().item()
        return {"final_loss": best_loss, "accuracy": acc,
                "n_samples": len(X), "n_sequences": len(sequences),
                "loss_history": loss_history,
                "acc_history": acc_history}


# Import formation constants from training_data
try:
    from training_data import (
        ALL_CT_FORMATIONS, CT_FORMATION_TO_IDX, N_CT_FORMATIONS,
        CT_ALIVE_MASK, CT_FORMATIONS_BY_ALIVE,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from training_data import (
        ALL_CT_FORMATIONS, CT_FORMATION_TO_IDX, N_CT_FORMATIONS,
        CT_ALIVE_MASK, CT_FORMATIONS_BY_ALIVE,
    )

# Per-event input dimension = base event + ct_alive + pre-round prior
CT_PRIOR_DIM = len(FORMATION_CLASSES)
CT_EVENT_DIM = EVENT_DIM + 1 + CT_PRIOR_DIM


class _CTFormationLSTMNet(nn.Module):
    """LSTM for CT formation prediction with alive-aware output masking."""

    def __init__(self, input_dim: int = CT_EVENT_DIM, hidden: int = 48,
                 n_classes: int = N_CT_FORMATIONS, dropout: float = 0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(32, hidden, num_layers=2, batch_first=True,
                            dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)
        self.n_classes = n_classes

    def forward(self, x, alive_mask=None):
        """x: (batch, seq_len, input_dim), alive_mask: (batch, n_classes) bool"""
        encoded = self.encoder(x)
        out, (h_n, _) = self.lstm(encoded)
        last_hidden = self.drop(h_n[-1])
        logits = self.head(last_hidden)

        if alive_mask is not None:
            logits = logits.masked_fill(~alive_mask, float("-inf"))

        return logits


class FormationClassifier_CT:
    """LSTM that predicts CT player distribution per zone with alive-aware
    output masking. Used when playing T."""

    def __init__(self, hidden: int = 48, dropout: float = 0.4):
        self.model = _CTFormationLSTMNet(
            CT_EVENT_DIM, hidden, N_CT_FORMATIONS, dropout).to(DEVICE)
        self.trained = False

    @staticmethod
    def _encode_event(ev: dict, ct_alive: int,
                      prior: list[float] | None = None) -> list[float]:
        """Encode event + ct_alive + per-round pre-round formation prior."""
        vec = [0.0] * CT_EVENT_DIM
        t_idx = ev.get("type_idx", 0)
        if 0 <= t_idx < N_EVENT_TYPES:
            vec[t_idx] = 1.0
        vec[N_EVENT_TYPES] = float(ev.get("actor_side_is_t", 0))
        z_idx = ev.get("zone_idx", 2)
        if 0 <= z_idx < N_SEQ_ZONES:
            vec[N_EVENT_TYPES + 1 + z_idx] = 1.0
        vec[N_EVENT_TYPES + 1 + N_SEQ_ZONES] = float(ev.get("time_norm", 0))
        vec[N_EVENT_TYPES + 1 + N_SEQ_ZONES + 1] = float(ev.get("is_headshot", 0))
        vec[EVENT_DIM] = ct_alive / 5.0

        if prior is None:
            prior = [1.0 / CT_PRIOR_DIM] * CT_PRIOR_DIM
        for i in range(min(CT_PRIOR_DIM, len(prior))):
            vec[EVENT_DIM + 1 + i] = float(prior[i])
        return vec

    def _build_alive_mask(self, ct_alive: int) -> torch.Tensor:
        """Build a boolean mask for valid formations given ct_alive."""
        alive_clamped = max(1, min(ct_alive, 5))
        mask_list = CT_ALIVE_MASK.get(alive_clamped,
                                      [True] * N_CT_FORMATIONS)
        return torch.tensor([mask_list], dtype=torch.bool, device=DEVICE)

    def predict(self, events: list[dict],
                ct_alive_per_event: list[int],
                prior: list[float] | None = None) -> dict[str, float]:
        """Predict CT formation from event sequence, ct_alive counts, and
        an optional 9-dim pre-round prior over FORMATION_CLASSES."""
        self.model.eval()
        if not events or not ct_alive_per_event:
            return {}

        ct_alive = ct_alive_per_event[-1] if ct_alive_per_event else 5

        seq = [self._encode_event(e, ca, prior)
               for e, ca in zip(events[:MAX_SEQ_LEN],
                                ct_alive_per_event[:MAX_SEQ_LEN])]
        while len(seq) < MAX_SEQ_LEN:
            seq.append([0.0] * CT_EVENT_DIM)

        x = torch.tensor([seq], dtype=torch.float32, device=DEVICE)
        mask = self._build_alive_mask(ct_alive)

        with torch.no_grad():
            logits = self.model(x, alive_mask=mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        result = {}
        for i, label in enumerate(ALL_CT_FORMATIONS):
            if probs[i] > 0.01:
                result[label] = float(probs[i])
        return result

    def predict_readable(self, events: list[dict],
                         ct_alive_per_event: list[int],
                         prior: list[float] | None = None) -> dict:
        """Predict and return human-readable formation (e.g. '2-1-2')."""
        raw = self.predict(events, ct_alive_per_event, prior=prior)
        if not raw:
            return {"formation": "unknown", "confidence": 0.0, "detail": {}}
        top_label = max(raw, key=raw.get)
        parts = top_label.split("_", 1)
        fmt = parts[1] if len(parts) == 2 else top_label
        return {
            "formation": fmt,
            "confidence": raw[top_label],
            "ct_alive": int(parts[0]) if len(parts) == 2 else 5,
            "detail": raw,
        }

    def train(self, ct_sequences: list[dict], n_epochs: int = 600,
              lr: float = 1e-3, weight_decay: float = 5e-4,
              patience: int = 80, verbose: bool = True) -> dict:
        """Train from CT formation sequences (events, formation_labels,
        ct_alive_at_event per sequence)."""
        X_all = []
        y_all = []
        masks_all = []

        for seq_data in ct_sequences:
            events = seq_data["events"]
            labels = seq_data["formation_labels"]
            alive_list = seq_data["ct_alive_at_event"]
            prior = seq_data.get("pre_round_prior")

            if not events or not labels:
                continue

            n = len(events)
            sample_points = sorted(set([1, max(1, n // 4),
                                        max(1, n // 2),
                                        max(1, 3 * n // 4), n]))

            for sp in sample_points:
                if sp > len(labels):
                    sp = len(labels)
                if sp == 0:
                    continue

                target_label = labels[sp - 1]
                target_idx = CT_FORMATION_TO_IDX.get(target_label, 0)
                ct_alive = alive_list[sp - 1] if sp - 1 < len(alive_list) else 5

                encoded = [self._encode_event(e,
                                              alive_list[j] if j < len(alive_list) else 5,
                                              prior)
                           for j, e in enumerate(events[:sp])]
                while len(encoded) < MAX_SEQ_LEN:
                    encoded.append([0.0] * CT_EVENT_DIM)
                encoded = encoded[:MAX_SEQ_LEN]

                alive_clamped = max(1, min(ct_alive, 5))
                mask = CT_ALIVE_MASK.get(alive_clamped,
                                         [True] * N_CT_FORMATIONS)

                X_all.append(encoded)
                y_all.append(target_idx)
                masks_all.append(mask)

        if not X_all:
            return {"error": "no training data"}

        X = torch.tensor(X_all, dtype=torch.float32, device=DEVICE)
        y = torch.tensor(y_all, dtype=torch.long, device=DEVICE)
        M = torch.tensor(masks_all, dtype=torch.bool, device=DEVICE)

        dataset = TensorDataset(X, y, M)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr,
                               weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=30)
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        best_state = None
        wait = 0
        loss_history: list[float] = []
        acc_history: list[float] = []

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for xb, yb, mb in loader:
                optimizer.zero_grad()
                logits = self.model(xb, alive_mask=None)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                masked_logits = logits.masked_fill(~mb, float("-inf"))
                correct += (masked_logits.argmax(dim=-1) == yb).sum().item()
                total += len(yb)

            avg_loss = total_loss / total if total else 0
            epoch_acc = correct / total if total else 0
            loss_history.append(avg_loss)
            acc_history.append(epoch_acc)
            scheduler.step(avg_loss)

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in
                              self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  FormationClassifier_CT epoch {epoch+1}/{n_epochs}: "
                      f"loss={avg_loss:.4f}, acc={epoch_acc:.1%}")

            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.trained = True
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X, alive_mask=M)
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean().item()
        return {"final_loss": best_loss, "accuracy": acc,
                "n_samples": len(X), "n_sequences": len(ct_sequences),
                "loss_history": loss_history,
                "acc_history": acc_history}


EventSequencePredictor = FormationClassifier_T

# Stub classes so old model files don't crash on load
class WinPredictor:
    def __init__(self):
        self.trained = False
        self.model = nn.Sequential(nn.Linear(8, 1))

class AttackPredictor:
    def __init__(self):
        self.trained = False
        self.model = nn.Sequential(nn.Linear(10, 3))

class FormationClassifier:
    """Legacy formation classifier — replaced by PreRoundFormation."""
    def __init__(self):
        self.trained = False
        self.model = nn.Sequential(nn.Linear(7, len(FORMATION_CLASSES)))


def save_models(models: dict, save_dir: str = "models") -> None:
    os.makedirs(save_dir, exist_ok=True)
    for name, wrapper in models.items():
        if hasattr(wrapper, "model") and hasattr(wrapper.model, "state_dict"):
            torch.save(wrapper.model.state_dict(),
                       os.path.join(save_dir, f"{name}.pt"))
    print(f"Models saved to {save_dir}/")


def load_models(save_dir: str = "models") -> dict:
    models = {}

    # Pre-round formation
    prf = PreRoundFormation()
    prf_path = os.path.join(save_dir, "preround_formation.pt")
    if os.path.exists(prf_path):
        try:
            prf.model.load_state_dict(
                torch.load(prf_path, map_location=DEVICE, weights_only=True))
            prf.trained = True
        except RuntimeError:
            print(f"  [warn] {prf_path} has old feature shape — retrain needed.")
    models["preround_formation"] = prf

    # Pre-round attack (T-side intent)
    pra = PreRoundAttack()
    pra_path = os.path.join(save_dir, "preround_attack.pt")
    if os.path.exists(pra_path):
        try:
            pra.model.load_state_dict(
                torch.load(pra_path, map_location=DEVICE, weights_only=True))
            pra.trained = True
        except RuntimeError:
            print(f"  [warn] {pra_path} has old feature shape — retrain needed.")
    models["preround_attack"] = pra

    # FormationClassifier_T (LSTM, attack site)
    fc_t = FormationClassifier_T()
    fc_t_path = os.path.join(save_dir, "formation_classifier_t.pt")
    if os.path.exists(fc_t_path):
        fc_t.model.load_state_dict(
            torch.load(fc_t_path, map_location=DEVICE, weights_only=True))
        fc_t.trained = True
    else:
        # Try legacy name
        legacy_path = os.path.join(save_dir, "event_sequence_predictor.pt")
        if os.path.exists(legacy_path):
            fc_t.model.load_state_dict(
                torch.load(legacy_path, map_location=DEVICE, weights_only=True))
            fc_t.trained = True
    models["formation_classifier_t"] = fc_t

    # FormationClassifier_CT (LSTM, alive-aware)
    fc_ct = FormationClassifier_CT()
    fc_ct_path = os.path.join(save_dir, "formation_classifier_ct.pt")
    if os.path.exists(fc_ct_path):
        fc_ct.model.load_state_dict(
            torch.load(fc_ct_path, map_location=DEVICE, weights_only=True))
        fc_ct.trained = True
    models["formation_classifier_ct"] = fc_ct

    # Legacy aliases for backward compat
    models["event_sequence_predictor"] = fc_t

    return models


_PRIOR_COLS_PASSTHROUGH = [
    "prev_plant_A", "prev_plant_B", "prev_plant_none", "prev_no_history",
    "prev_t_won", "prev_t_tier", "prev_ct_tier",
    "rounds_since_plant_A", "rounds_since_plant_B", "streak_same_site",
]


def _soft_tier_probs(tier: str) -> dict:
    """Smear a hard tier label into a soft distribution (70% assigned,
    30% spread over the others)."""
    from info_model import ECON_TIERS
    probs = {t: 0.0 for t in ECON_TIERS}
    probs[tier] = 0.7
    for t in ECON_TIERS:
        if t != tier:
            probs[t] = 0.3 / (len(ECON_TIERS) - 1)
    return probs


def _build_prf_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build PreRoundFormation DF (uses CT money to predict CT formation)."""
    from info_model import money_to_tier
    tier_rows = []
    for _, row in df.iterrows():
        ct_money = row["ct_avg_money"]
        probs = _soft_tier_probs(money_to_tier(ct_money))
        out = {
            "p_broke": probs["BROKE"],
            "p_low": probs["LOW"],
            "p_medium": probs["MEDIUM"],
            "p_high": probs["HIGH"],
            "p_rich": probs["RICH"],
            "predicted_avg_money": ct_money,
            "round_in_half": row["round_in_half"],
            "is_second_half": row["is_second_half"],
            "ct_formation": row["ct_formation"],
        }
        for col in _PRIOR_COLS_PASSTHROUGH:
            out[col] = row[col] if col in row.index else (
                1 if col == "prev_no_history" else 0.0)
        tier_rows.append(out)
    return pd.DataFrame(tier_rows)


def _attach_ct_priors(ct_sequences: list[dict],
                      prf: PreRoundFormation,
                      rounds_df: pd.DataFrame,
                      label_smooth: float = 0.15) -> None:
    """Attach a 9-dim pre_round_prior to each CT formation sequence by
    mixing PRF predictions with label-smoothed ground truth."""
    from info_model import money_to_tier
    rounds_by_key = {
        (str(r["demo"]), int(r["round_num"])): r
        for _, r in rounds_df.iterrows()
    }
    uniform = [1.0 / CT_PRIOR_DIM] * CT_PRIOR_DIM
    for seq in ct_sequences:
        key = (str(seq.get("demo", "")), int(seq.get("round_num", -1)))
        r = rounds_by_key.get(key)
        if r is None or not prf.trained:
            seq["pre_round_prior"] = uniform
            continue

        probs = _soft_tier_probs(money_to_tier(r["ct_avg_money"]))
        try:
            prior_dict = prf.predict_single(
                tier_probs=probs,
                predicted_avg_money=float(r["ct_avg_money"]),
                round_in_half=int(r["round_in_half"]),
                is_second_half=int(r["is_second_half"]),
                prior={
                    "prev_plant_A": int(r.get("prev_plant_A", 0)),
                    "prev_plant_B": int(r.get("prev_plant_B", 0)),
                    "prev_plant_none": int(r.get("prev_plant_none", 0)),
                    "prev_no_history": int(r.get("prev_no_history", 1)),
                    "prev_t_won": float(r.get("prev_t_won", 0.5)),
                    "prev_t_tier": float(r.get("prev_t_tier", 0.0)),
                    "prev_ct_tier": float(r.get("prev_ct_tier", 0.0)),
                    "rounds_since_plant_A": float(r.get("rounds_since_plant_A", 1.0)),
                    "rounds_since_plant_B": float(r.get("rounds_since_plant_B", 1.0)),
                    "streak_same_site": float(r.get("streak_same_site", 0.0)),
                },
            )
        except Exception:
            seq["pre_round_prior"] = uniform
            continue

        prior_vec = [prior_dict.get(c, 0.0) for c in FORMATION_CLASSES]
        gt_fmt = str(r["ct_formation"])
        gt_idx = FORMATION_TO_IDX.get(gt_fmt, FORMATION_TO_IDX["other"])
        gt_vec = [label_smooth / (CT_PRIOR_DIM - 1)] * CT_PRIOR_DIM
        gt_vec[gt_idx] = 1.0 - label_smooth
        mixed = [0.5 * p + 0.5 * g for p, g in zip(prior_vec, gt_vec)]
        s = sum(mixed)
        if s > 0:
            mixed = [v / s for v in mixed]
        seq["pre_round_prior"] = mixed


def _build_pra_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build PreRoundAttack DF (uses T money to predict T attack site)."""
    from info_model import money_to_tier
    tier_rows = []
    for _, row in df.iterrows():
        t_money = row["t_avg_money"]
        probs = _soft_tier_probs(money_to_tier(t_money))
        out = {
            "p_broke": probs["BROKE"],
            "p_low": probs["LOW"],
            "p_medium": probs["MEDIUM"],
            "p_high": probs["HIGH"],
            "p_rich": probs["RICH"],
            "predicted_avg_money": t_money,
            "round_in_half": row["round_in_half"],
            "is_second_half": row["is_second_half"],
            "attack_site": row["attack_site"],
        }
        for col in _PRIOR_COLS_PASSTHROUGH:
            out[col] = row[col] if col in row.index else (
                1 if col == "prev_no_history" else 0.0)
        tier_rows.append(out)
    return pd.DataFrame(tier_rows)


def train_all_models(
    demo_dir: str = "src/demo",
    save_dir: str = "models",
    verbose: bool = True,
    *,
    train_data: dict | None = None,
    val_data: dict | None = None,
) -> dict:
    """Train all NN models.

    If *train_data* is provided (dict with keys rounds, event_sequences,
    ct_formation_sequences), it is used directly — skipping demo extraction.
    If *val_data* is also provided, validation metrics are reported.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if verbose:
        print(f"Using device: {DEVICE}")
        if DEVICE.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print()

    if train_data is not None:
        df = train_data["rounds"]
        sequences = train_data.get("event_sequences", [])
        ct_formations = train_data.get("ct_formation_sequences", [])
    else:
        from training_data import extract_all
        data = extract_all(demo_dir, include_rl=False, include_sequences=True,
                           include_ct_formations=True, verbose=verbose)
        df = data["rounds"]
        sequences = data["event_sequences"]
        ct_formations = data["ct_formation_sequences"]

    if df.empty:
        print("No training data extracted!")
        return {}

    if verbose:
        n_demos = df["demo"].nunique() if "demo" in df.columns else "?"
        print(f"\nTraining on {len(df)} rounds from {n_demos} demos")
        print(f"Event sequences: {len(sequences)} rounds")
        print(f"CT formation sequences: {len(ct_formations)} rounds")
        print(f"{'='*60}\n")

    models = {}
    all_stats = {}

    prf_df = _build_prf_df(df)
    prf_val_df = _build_prf_df(val_data["rounds"]) if val_data and not val_data["rounds"].empty else None

    if verbose:
        print("--- Pre-Round Formation Predictor ---")
    prf = PreRoundFormation()
    stats_prf = prf.train(prf_df, verbose=verbose)
    models["preround_formation"] = prf
    all_stats["preround_formation"] = {"train": stats_prf}

    if prf_val_df is not None and not prf_val_df.empty:
        X_val, y_val = prf._prepare(prf_val_df)
        val_acc = prf._accuracy(X_val, y_val)
        all_stats["preround_formation"]["val"] = {"accuracy": val_acc}
        if verbose:
            print(f"  Train acc={stats_prf['accuracy']:.1%}, Val acc={val_acc:.1%}\n")
    elif verbose:
        print(f"  Final: loss={stats_prf['final_loss']:.4f}, "
              f"acc={stats_prf['accuracy']:.1%}\n")

    pra_df = _build_pra_df(df)
    pra_val_df = _build_pra_df(val_data["rounds"]) if val_data and not val_data["rounds"].empty else None

    if verbose:
        print("--- Pre-Round Attack Predictor ---")
    pra = PreRoundAttack()
    stats_pra = pra.train(pra_df, verbose=verbose)
    models["preround_attack"] = pra
    all_stats["preround_attack"] = {"train": stats_pra}

    if pra_val_df is not None and not pra_val_df.empty:
        X_val, y_val = pra._prepare(pra_val_df)
        val_acc_pra = pra._accuracy(X_val, y_val)
        all_stats["preround_attack"]["val"] = {"accuracy": val_acc_pra}
        if verbose:
            print(f"  Train acc={stats_pra['accuracy']:.1%}, Val acc={val_acc_pra:.1%}\n")
    elif verbose:
        print(f"  Final: loss={stats_pra['final_loss']:.4f}, "
              f"acc={stats_pra['accuracy']:.1%}\n")

    if verbose:
        print("--- FormationClassifier_T (LSTM) ---")
    fc_t = FormationClassifier_T()
    if sequences:
        stats_fct = fc_t.train(sequences, n_epochs=500, verbose=verbose)
        all_stats["formation_classifier_t"] = {"train": stats_fct}

        val_seqs = val_data.get("event_sequences", []) if val_data else []
        if val_seqs:
            val_acc_t = _eval_lstm_t(fc_t, val_seqs)
            all_stats["formation_classifier_t"]["val"] = {"accuracy": val_acc_t}
            if verbose:
                print(f"  Train acc={stats_fct['accuracy']:.1%}, Val acc={val_acc_t:.1%}\n")
        elif verbose:
            print(f"  Final: loss={stats_fct['final_loss']:.4f}, "
                  f"acc={stats_fct['accuracy']:.1%}, "
                  f"samples={stats_fct['n_samples']}\n")
    else:
        if verbose:
            print("  No event sequences available for training.\n")
    models["formation_classifier_t"] = fc_t

    if verbose:
        print("--- FormationClassifier_CT (LSTM, alive-aware) ---")
    fc_ct = FormationClassifier_CT()
    if ct_formations:
        # Attach pre-round formation prior from the just-trained PRF.
        _attach_ct_priors(ct_formations, prf, df)
        val_ct = val_data.get("ct_formation_sequences", []) if val_data else []
        if val_ct and val_data is not None and not val_data["rounds"].empty:
            _attach_ct_priors(val_ct, prf, val_data["rounds"])

        stats_fcct = fc_ct.train(ct_formations, n_epochs=600, verbose=verbose)
        all_stats["formation_classifier_ct"] = {"train": stats_fcct}

        if val_ct:
            val_acc_ct = _eval_lstm_ct(fc_ct, val_ct)
            all_stats["formation_classifier_ct"]["val"] = {"accuracy": val_acc_ct}
            if verbose:
                print(f"  Train acc={stats_fcct['accuracy']:.1%}, Val acc={val_acc_ct:.1%}\n")
        elif verbose:
            print(f"  Final: loss={stats_fcct['final_loss']:.4f}, "
                  f"acc={stats_fcct['accuracy']:.1%}, "
                  f"samples={stats_fcct['n_samples']}\n")
    else:
        if verbose:
            print("  No CT formation sequences available for training.\n")
    models["formation_classifier_ct"] = fc_ct

    models["event_sequence_predictor"] = fc_t

    save_models(models, save_dir)

    if verbose:
        print(f"\n{'='*60}")
        print("  Model parameter counts:")
        for name, wrapper in models.items():
            if name == "event_sequence_predictor":
                continue
            if hasattr(wrapper, "model"):
                n_params = sum(p.numel() for p in wrapper.model.parameters())
                print(f"    {name:30s}: {n_params:,} parameters")
        print(f"{'='*60}")

    return {"models": models, "stats": all_stats}


def _eval_lstm_t(fc_t: FormationClassifier_T,
                 sequences: list[dict]) -> float:
    """Evaluate FormationClassifier_T accuracy on a set of sequences."""
    correct, total = 0, 0
    for seq_data in sequences:
        events = seq_data.get("events", [])
        label = seq_data.get("attack_site", "")
        if not events or label not in ("A", "B", "no_plant"):
            continue
        pred = fc_t.predict(events)
        if pred:
            top = max(pred, key=pred.get)
            if top == label:
                correct += 1
            total += 1
    return correct / max(total, 1)


def _eval_lstm_ct(fc_ct: FormationClassifier_CT,
                  ct_sequences: list[dict]) -> float:
    """Evaluate FormationClassifier_CT accuracy on a set of sequences."""
    correct, total = 0, 0
    for seq_data in ct_sequences:
        events = seq_data.get("events", [])
        labels = seq_data.get("formation_labels", [])
        alive_counts = seq_data.get("ct_alive_at_event", [])
        if not events or not labels or not alive_counts:
            continue
        last_label = labels[-1]
        raw = fc_ct.predict(events, alive_counts)
        if raw:
            top = max(raw, key=raw.get)
            if top == last_label:
                correct += 1
            total += 1
    return correct / max(total, 1)


if __name__ == "__main__":
    result = train_all_models()
    if result:
        print("\nTraining complete.")
