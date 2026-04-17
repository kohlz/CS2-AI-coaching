# Formation Classifier_T — T Attack LSTM

**Source:** [`src/analysis/strategy_nn.py`](../../src/analysis/strategy_nn.py)
— class `FormationClassifier_T` / `_LSTMNet`

**Checkpoint:** `models/formation_classifier_t.pt`

## Role in the Pipeline

Reads a sequence of in-round events (kills, smokes, flashes, HE,
plants, molotovs) and predicts the **T attack distribution**:

```
{"A": 0.88, "B": 0.10, "no_plant": 0.02}
```

Used when the target player is **CT** — the LSTM tells the CT player
which site T is committing to as the round unfolds, then the
[CT Q-learner](qlearning.md) picks an action (HOLD / ROTATE /
RETAKE / ...).

Returning to [main pipeline](../README.md#1-pipeline).

## Python Packages

- `torch.nn.LSTM`, `torch.nn.Linear`, `torch.nn.Dropout`
- `torch.optim.Adam` + `ReduceLROnPlateau`
- `torch.utils.data.DataLoader` / `TensorDataset`

## Architecture

```
Input frame (14-dim per event) ──▶ Linear(14 → 32) ─▶ ReLU ─▶ Dropout(0.3)
                                                        │
                                                        ▼
                              LSTM(32 → 64, 2 layers, dropout=0.3)
                                                        │
                                                        ▼
                                             Dropout(0.3)
                                                        │
                                                        ▼
                                          Linear(64 → 3)   →  softmax
```

**Per-event encoding (14 dims).** Concatenation of:

- `N_EVENT_TYPES = 6` one-hot — `kill / smoke / flash / he / plant / molotov`
- `actor_side_is_t` (1 dim) — was the actor a T?
- `N_SEQ_ZONES = 5` one-hot — `A / B / MID / CT_BASE / T_BASE`
- `time_norm` (1 dim) — normalized round time
- `is_headshot` (1 dim)

Sequences are padded / truncated to `MAX_SEQ_LEN = 30`.

## Training Data

- Source: `data/event_sequences_train.json` (and val/test splits),
  produced by [`training_data.py`](../../src/analysis/training_data.py).
  Each entry has:
  ```json
  {"events": [...], "attack_site": "A" | "B" | "no_plant"}
  ```
- **Curriculum sampling.** For a sequence of length `n`, the training
  loop emits 4 prefixes at `[1, n/3, 2n/3, n]` — this teaches the LSTM
  to predict from partial information at every stage of the round, not
  just from a fully-played-out sequence.
- Optimizer: Adam, `lr = 1e-3`, `weight_decay = 1e-4`,
  `ReduceLROnPlateau(patience=30)`. Up to 500 epochs, early-stopping
  patience 60. Batch size 64.
- Loss: cross-entropy over 3 classes.

## Evaluation

On the held-out test split:

| Metric | Value |
|---|---:|
| Test accuracy | 83.2% |
| Macro-F1 | 0.842 |
| Weighted-F1 | 0.833 |
| Majority-class baseline | 38.6% |
| n test sequences | 101 |

This is the strongest model in the project. The fact that Macro-F1
(0.84) is essentially equal to Accuracy (0.83) is an important quality
signal: the model is not gaining its accuracy by collapsing to one
class — it predicts A, B, and `no_plant` with comparable F1.
Per-class F1 in
[`reports/f1_per_class.png`](../../reports/f1_per_class.png) shows
~0.85 / 0.85 / 0.83 across the three classes.

The lift over the majority baseline is **2.2×**.

## Output Use

- `generate_report.py` calls `predict_at_checkpoints(events,
  checkpoints)` to emit a formation update in the event timeline at
  each natural breakpoint of the round (kill, plant, etc.).
- Predictions are filtered with a delta threshold so the report only
  prints the formation when the dominant probability *changes* — the
  raw sequence is too verbose otherwise.
