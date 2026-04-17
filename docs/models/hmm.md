# Economy HMM — Enemy Tier Prediction

**Source:** [`src/analysis/info_model.py`](../../src/analysis/info_model.py)

## Role in the Pipeline

The Economy HMM maintains a belief over the enemy team's current
**economy tier** — one of `BROKE`, `LOW`, `MEDIUM`, `HIGH`, `RICH`.
Its output is the upstream feature that drives:

- The MDP buy evaluation ([MDP docs](economy_mdp.md))
- `PreRoundFormation` FNN inputs ([doc](preround_formation.md))
- `PreRoundAttack` FNN inputs ([doc](preround_attack.md))

Returning to [main pipeline](../README.md#1-pipeline).

## Python Packages

- `math` — softmax-distance transitions
- `dataclasses` — `EconObservation`
- No third-party ML dependency. The HMM is a plain Python object
  with dict beliefs over 5 tiers.

## Model Specification

**Hidden state (5 tiers):**

| Tier | Money range ($) | Expected avg ($) |
|---|---|---:|
| `BROKE`  | 0 – 1 500       |   750 |
| `LOW`    | 1 500 – 3 000   | 2 250 |
| `MEDIUM` | 3 000 – 5 000   | 4 000 |
| `HIGH`   | 5 000 – 8 000   | 6 500 |
| `RICH`   | 8 000 – 16 000  | 10 000 |

**Observations per round (`EconObservation` in `info_model.py`):**

1. `enemy_won_prev` — whether the enemy team won the *previous* round
2. `round_end_type` — `elimination / bomb / time / close / dominant`
3. `best_weapon_seen` — `pistol / smg / rifle / awp / unknown`
4. `enemy_survivors` — number of enemies alive at round end (0–5)
5. `enemy_loss_streak` — consecutive losses (0–5)
6. `enemy_win_streak` — consecutive wins (0–5)

**Transition model** `P(tier_t | tier_{t-1}, enemy_won_prev)`.

The transition matrix is **derived from the CS2 economy rules** (not
hand-tuned). For each `(current_tier, won_prev)` pair:

- Compute expected post-round money using the real game formulas
  (`WIN_REWARD_ELIM + AVG_KILL_INCOME_WIN` on a win, and a marginal
  over streaks `{1: 0.40, 2: 0.30, 3: 0.15, 4: 0.10, 5: 0.05}` of
  `loss_bonus(streak) + AVG_KILL_INCOME_LOSS` on a loss).
- Convert that expected money to a distribution over next-round tiers
  via an exponential softmax on distance to each tier's midpoint
  (scale = 3000 on win, 2500 on loss).

This is done once at import time (`TRANSITION = _derive_transition_matrix()`).

**Emission model** `P(observation | tier)`. The joint emission is a
product of five independent signals, each defined as a categorical
conditional table:

- `_WEAPON_EMISSION` — `awp` is rare on `BROKE` (0.01) and common on
  `RICH` (0.37).
- `_END_TYPE_EMISSION` — dominant wins correlate with `RICH`.
- `_SURVIVORS_EMISSION` — 0 survivors correlates with `BROKE`,
  3+ survivors with `RICH`.
- `_LOSS_STREAK_EMISSION` — strong signal: a 4-loss streak is ~4× more
  likely under `BROKE` than `RICH`.
- `_WIN_STREAK_EMISSION` — symmetric signal in the other direction.

All five are combined multiplicatively with a lower clamp of `1e-12`.

**Inference** is the standard forward algorithm:

```
predict: belief' = T · belief
update:  belief'' ∝ belief' * emission(obs | tier)
```

exposed as `predict_step(...)` and `update(obs)` on the `EconomyHMM`
class. A convenience method `predict_tier_probs(enemy_won_prev,
prior_obs)` returns the posterior over tiers for use by downstream
consumers.

## Training Data

The HMM is **not trained** by backprop / EM — its transition matrix is
analytically derived from CS2 economy constants, and its emission
tables are encoded directly from domain knowledge about how buys look
at each tier. What we do need is a stream of `EconObservation`s for
inference; these are extracted per round in
[`training_data.py`](../../src/analysis/training_data.py) via the
`_derive_econ_observation` logic (see
[Training-Data Extraction & Splits](../pipeline/training_data.md)).

## Output Format

`predict_tier_probs(...)` returns a dict:

```python
{"BROKE": 0.08, "LOW": 0.14, "MEDIUM": 0.41, "HIGH": 0.27, "RICH": 0.10}
```

`predict_avg_money(...)` returns the expected dollar value:

```
E[money] = Σ_t TIER_AVG_MONEY[t] · P(t)
```

Both are consumed verbatim as features by the pre-round FNNs.

## Known Caveats

- Emission tables are domain-knowledge priors, not data-driven.
  Re-estimating them from demo data is a natural extension.
- The HMM tracks *team* economy; it does not maintain per-player
  beliefs. That's sufficient for coaching signal but not for a
  full per-enemy loadout prediction.
