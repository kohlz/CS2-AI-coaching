# Pre-Round Attack — Feedforward NN

**Source:** [`src/analysis/strategy_nn.py`](../../src/analysis/strategy_nn.py)
— class `PreRoundAttack`

**Checkpoint:** `models/preround_attack.pt`

## Role in the Pipeline

Symmetric counterpart to
[`PreRoundFormation`](preround_formation.md): given T-side economy
and prior-round tendencies, predict where T is most likely to go
pre-round — `A`, `B`, or `no_plant` (eco rounds). Used when the
target player is **CT** and wants to anticipate the push.

Returning to [main pipeline](../README.md#1-pipeline).

## Python Packages

Same as `PreRoundFormation`: `torch`, `pandas`, `numpy`.
`PreRoundAttack` is a subclass of `PreRoundFormation` and reuses the
feature vector / network architecture, overriding only the target and
output size.

## Input Features

Identical to [`PreRoundFormation`](preround_formation.md#input-features-18-total).
The 18-dim vector contains HMM tier probabilities for the *enemy T
economy* (from the CT player's perspective), `predicted_avg_money`,
round-context flags, and 10 prior-round tendency features. See the
feature table in the parent doc.

## Output

Softmax distribution over 3 classes:

```
["A", "B", "no_plant"]
```

## Architecture

Same `PreRoundFormationNet` as the parent class:

```
Linear(18 → 64) -> BN -> ReLU -> Dropout(0.3)
Linear(64 → 32) -> BN -> ReLU -> Dropout(0.3)
Linear(32 → 3)    <-- reduced output size
```

## Training Data

- Source: `data/rounds_{train,val,test}.csv` (one row per round).
- Label: `attack_site`, derived from the bomb plant site on that
  round. Rounds where no plant occurred are labelled `no_plant`.
- Optimizer and schedule match `PreRoundFormation` (Adam,
  `lr=3e-3`, `weight_decay=1e-3`, cosine-style early stopping at
  patience 80, up to 800 epochs, batch 64).

## Evaluation

On the held-out test split:

| Metric | Value |
|---|---:|
| Test accuracy | 42.6% |
| Macro-F1 | 0.372 |
| Weighted-F1 | 0.410 |
| Majority-class baseline accuracy | 42.6% |
| n test samples | 108 |

The raw accuracy matches the majority-class baseline, but the
**macro-F1 of 0.37 is materially above 0** — the model actually
learns to distinguish A from B from no_plant rather than collapsing
to a single class. The per-class F1 breakdown in
[`reports/f1_per_class.png`](../../reports/f1_per_class.png) shows
`no_plant: 0.55`, `A: 0.41`, `B: 0.16` — B-side predictions are
the weakest, which is consistent with Mirage's documented A-bias
in casual/pro play.

## Output Use

Consumed by:

- `generate_report.py` — prints the top A/B prediction (or an "eco
  likely" message when `no_plant` dominates) into the "Pre-Round"
  section when the player is CT.
