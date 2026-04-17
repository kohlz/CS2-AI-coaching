# Formation Classifier_CT — CT Formation LSTM (Alive-Aware)

**Source:** [`src/analysis/strategy_nn.py`](../../src/analysis/strategy_nn.py)
— class `FormationClassifier_CT` / `_CTFormationLSTMNet`

**Checkpoint:** `models/formation_classifier_ct.pt`

## Role in the Pipeline

Predicts the **CT player distribution per zone** as a round unfolds —
e.g. `5_2-1-2` (5 alive: 2 at A, 1 mid, 2 at B), `4_2-1-1` after a
kill, `3_1-1-1` after another, etc. Used when the target player is
**T** — the T player wants to know how many CTs are at each zone so
the [T Q-learner](qlearning.md) can recommend a side.

The model is **alive-aware**: it consumes `ct_alive` as a feature and
applies a logit mask at inference time so it can never output a
formation whose zone counts disagree with the number of CTs alive.

Returning to [main pipeline](../README.md#1-pipeline).

## Python Packages

- `torch.nn.LSTM`, `torch.nn.Linear`, `torch.nn.Dropout`
- `torch.optim.Adam` + `ReduceLROnPlateau`
- `torch.nn.utils.clip_grad_norm_` (gradient clipping at norm 1.0)

Constants `ALL_CT_FORMATIONS`, `CT_ALIVE_MASK`,
`CT_FORMATIONS_BY_ALIVE` are imported from
[`training_data.py`](../../src/analysis/training_data.py).

## Architecture

```
Input frame (24-dim) ─▶ Linear(24 → 32) ─▶ ReLU ─▶ Dropout(0.4)
                                              │
                                              ▼
                  LSTM(32 → 48, 2 layers, dropout=0.4)
                                              │
                                              ▼
                                    Dropout(0.4)
                                              │
                                              ▼
                              Linear(48 → ~30)
                                              │
                          alive_mask  ─────▶ masked_fill(-inf)  ─▶ softmax
```

**Per-event encoding (24 dims).** Concatenation of:

- 14-dim event vector identical to `FormationClassifier_T`
- `ct_alive / 5.0` (1 dim)
- `pre_round_prior` (9 dims) — the
  [PreRoundFormation](preround_formation.md) softmax output for this
  round, **broadcast onto every event**.

The prior is the same vector at every timestep of a given round: the
LSTM gets "you were expecting 2-1-2 pre-round" baked into every input
frame and learns to revise it based on incoming events.

### Pre-Round Prior

The 9-dim `pre_round_prior` slot is filled at training time by the
`_attach_ct_priors(...)` helper in `strategy_nn.py`, which runs
`PreRoundFormation` over the training rounds and blends its output
with the ground-truth one-hot. At inference time it is filled by the
live `PreRoundFormation` softmax. This is what lets the LSTM "start"
each round with the FNN's belief and revise it from the events.

### Alive-Aware Masking

`CT_ALIVE_MASK` is a precomputed dict mapping `ct_alive ∈ {1..5}` to a
boolean vector of length `N_CT_FORMATIONS`. A formation is valid for
`ct_alive = k` iff its zone counts sum to `k`. At inference:

```python
logits = self.head(...)
logits = logits.masked_fill(~mask, float("-inf"))
return softmax(logits)
```

so e.g. `5_2-1-2` is impossible when `ct_alive = 4`. This is applied
**only at inference** — during training the model sees raw logits so
that it learns from the `ct_alive` input feature naturally rather
than depending on the mask.

## Output

```python
{"formation": "2-1-2", "ct_alive": 5, "confidence": 0.41,
 "detail": {"5_2-1-2": 0.41, "5_1-2-2": 0.18, ...}}
```

## Training Data

- Source: `data/event_sequences_train.json` filtered to T-side
  rounds. Each sequence dict has:
  ```json
  {"events":             [...],           // shared with FC_T
   "formation_labels":   ["5_2-1-2", ...],
   "ct_alive_at_event":  [5, 5, 4, ...],
   "pre_round_prior":    [...]            // 9-dim, optional
  }
  ```
- **Curriculum sampling.** Each sequence emits 5 prefixes at
  `[1, n/4, n/2, 3n/4, n]` so the LSTM is supervised at multiple
  in-round timestamps.
- Optimizer: Adam, `lr = 1e-3`, `weight_decay = 5e-4`,
  `ReduceLROnPlateau(patience=30)`, gradient clipping at 1.0. Up to
  600 epochs, early-stopping patience 80. Batch size 32.
- Loss: cross-entropy over the full ~30-class set, **without** the
  alive mask (so the model sees the full distribution).

## Evaluation

On the held-out test split:

| Metric | Value |
|---|---:|
| Test accuracy | 53.2% |
| Macro-F1 | 0.136 |
| Weighted-F1 | 0.469 |
| Majority-class baseline | 21.3% |
| n test sequences | 94 |

The model achieves **2.5× the majority baseline** in raw accuracy,
but per-class F1 is dominated by a few well-supported formations
(typically `1-1-0-0` and `1-0-0-1` post-trade configurations); the
long tail of rare formations (e.g. `0-2-3` with 5 alive) is
under-predicted because each has 0–3 test samples. Per-class detail
is in [`reports/f1_per_class.png`](../../reports/f1_per_class.png).

## Output Use

- `generate_report.py` calls `predict_readable(events, ct_alive_per_event,
  prior=...)` once per natural breakpoint in the round.
- The first prediction in a T-side round always agrees with the
  pre-round prior at full weight; later predictions can diverge as
  events arrive.
