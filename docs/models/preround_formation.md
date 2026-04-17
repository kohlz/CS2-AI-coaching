# Pre-Round Formation — Feedforward NN

**Source:** [`src/analysis/strategy_nn.py`](../../src/analysis/strategy_nn.py)
— class `PreRoundFormation` / `PreRoundFormationNet`

**Checkpoint:** `models/preround_formation.pt`

## Role in the Pipeline

Given the enemy team's economy state and prior-round tendencies,
predict the enemy's **CT formation** (e.g. `2-1-2`, `1-2-2`, `3-1-1`,
`other`) before the round starts. Used when the target player is on
**T side** — the T player wants to know the CT setup pre-round.

This model also supplies the **pre-round prior** that seeds
`FormationClassifier_CT` during inference (the LSTM revises the prior
based on in-round events). See
[FormationClassifier_CT](formation_classifier_ct.md#pre-round-prior).

Returning to [main pipeline](../README.md#1-pipeline).

## Python Packages

- `torch` (`torch.nn`, `torch.optim`) — FNN architecture, training
- `pandas`, `numpy` — feature DataFrame and arrays
- `torch.utils.data.DataLoader` / `TensorDataset` — mini-batching

## Input Features (18 total)

**Economy features (8)** — direct HMM outputs:

| Feature | Source |
|---|---|
| `p_broke`, `p_low`, `p_medium`, `p_high`, `p_rich` | `EconomyHMM.predict_tier_probs()` |
| `predicted_avg_money` | `EconomyHMM.predict_avg_money()`, normalized by `$16 000` |
| `round_in_half` | Round index within the half, normalized `/12` |
| `is_second_half` | 0 or 1 |

**Prior-round features (10)** — added to improve pre-round signal (see
[docs/README §3.6](../README.md#36-limitations--future-work)):

| Feature | Description |
|---|---|
| `prev_plant_A`, `prev_plant_B`, `prev_plant_none` | One-hot of what T did last round |
| `prev_no_history` | 1 on round 0 / round 12 (half boundaries) |
| `prev_t_won` | Did T win last round? |
| `prev_t_tier`, `prev_ct_tier` | Economy tier at start of last round |
| `rounds_since_plant_A`, `rounds_since_plant_B` | Rounds since last plant on each site |
| `streak_same_site` | Consecutive plants on the same site |

All prior features reset at the half boundary so the feature vector
doesn't leak information across the side swap at round 12.

## Output

Softmax distribution over 9 CT formation classes:

```
["2-1-2", "1-2-2", "1-1-3", "2-2-1", "3-1-1",
 "1-1-2", "0-2-3", "2-0-3", "other"]
```

## Architecture

`PreRoundFormationNet` — 3-layer MLP with BatchNorm and Dropout:

```
Linear(18 → 64) -> BN -> ReLU -> Dropout(0.3)
Linear(64 → 32) -> BN -> ReLU -> Dropout(0.3)
Linear(32 → 9)
```

## Training Data

- Produced by [`training_data.py`](../../src/analysis/training_data.py)
  as `data/rounds_train.csv` etc. (see
  [Training-Data Extraction & Splits](../pipeline/training_data.md)).
- Labels `ct_formation` are extracted per round by sampling CT
  positions at several ticks early in the round, counting players
  per zone (A / MID / B), mapping counts to a class and taking the
  most common.
- Loss: cross-entropy. Optimizer: Adam, `lr = 3e-3`,
  `weight_decay = 1e-3`, `ReduceLROnPlateau(patience=40)`. Up to 800
  epochs with early-stopping patience 80.
- Batch size 64. Devices: CUDA if available.

Training histories (per-epoch loss + accuracy) are saved into
`data/training_results.json` under `nn.preround_formation.train`.

## Evaluation

On the held-out test split:

| Metric | Value |
|---|---:|
| Test accuracy | 53.7% |
| Macro-F1 | 0.112 |
| Weighted-F1 | 0.492 |
| Majority-class baseline accuracy | 61.1% |
| n test samples | 108 |

### 5.3 Why below baseline?

The majority class (`other`) makes up 61% of the test set. A
trivial "always predict other" classifier would therefore score
better on raw accuracy. The FNN's Macro-F1 of 0.11 reveals that it
collapses into 1–2 classes, which is the worst-case failure for
an imbalanced 9-way task.

The most likely causes:

1. **Weak signal in features.** Economy tier alone is a coarse
   proxy for formation — many tiers are compatible with many setups.
2. **Thin class tail.** Only 22 test rounds have the second-most
   common label (`2-2-1`); classes `0-2-3` / `2-0-3` each have 0
   support.
3. **Sample size.** The training set has 720 rounds across 9
   classes; label noise (formation extraction uses coarse zone
   counting) compounds at these sizes.

See the per-class breakdown in
[`reports/f1_per_class.png`](../../reports/f1_per_class.png). This
model is still useful as the prior seed for
`FormationClassifier_CT`, because even a weak prior carries more
information than a uniform one.

## Output Use

Consumed by:

- `generate_report.py` — prints the top prediction into the
  "Pre-Round" section when the player is T.
- `FormationClassifier_CT._encode_event(prior=...)` — the softmax
  output is concatenated into every LSTM input frame during T-side
  rounds.
