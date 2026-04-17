# Tactical Q-Learning — T and CT

**Source:** [`src/analysis/tactical_rl.py`](../../src/analysis/tactical_rl.py)
— classes `_SideQLearner`, `TacticalQLearner_T`, `TacticalQLearner_CT`

**Checkpoints:** `models/tactical_ql_t.npz`, `models/tactical_ql_ct.npz`

## Role in the Pipeline

Two **side-specific tabular Q-learners** that, given the current
in-round state, recommend the next tactical action. They are paired
with the LSTM formation classifiers in the report:

- LSTM tells the player **where the enemies are**.
- Q-learner tells the player **what to do about it**.

Returning to [main pipeline](../README.md#1-pipeline).

## Python Packages

- `numpy` — Q-tables, visit counts, sampling
- `pandas` — RL transition DataFrame
- No deep-learning library — these are tabular, not neural.

## State Space

A 6-tuple, indexed identically for both sides:

```
(alive_adv, bomb_status, time_bucket, zone, recent_event, team_support)
   7 × 3 × 4 × 5 × 5 × 3  =  6 300 discrete states per side
```

| Component | Cardinality | Meaning |
|---|---:|---|
| `alive_adv`     | 7 | `t_alive − ct_alive`, clamped to `[−3, +3]` |
| `bomb_status`   | 3 | not_planted / planted / defused-or-detonated |
| `time_bucket`   | 4 | round time quartile |
| `zone`          | 5 | A / B / MID / CT_BASE / T_BASE |
| `recent_event`  | 5 | none / teammate_died / enemy_killed / grenade / bomb_planted |
| `team_support`  | 3 | ALONE (0) / SUPPORTED (1 nearby) / GROUPED (2+ nearby) |

## Action Space

Six actions per side:

- **T**: `EXECUTE, PEEK, TRADE, FALL_BACK, UTILITY, LURK`
- **CT**: `HOLD, ROTATE, RETAKE, PUSH, FALL_BACK, UTILITY`

## Algorithm

**Dual Q-table tabular Q-learning**:

```
Q_kill[s,a]  ←  trained on normalized kill / site reward (per-tick shaping)
Q_win[s,a]   ←  trained on terminal win/loss reward (sparse)

Q_recommend[s,a]  =  α_blend · Q_kill[s,a]  +  (1 − α_blend) · Q_win[s,a]
```

Both tables share the same SGD-style update:

```
Q[s,a] += lr · (r + γ · max_a' Q[s', a']  − Q[s,a])    if not terminal
Q[s,a] += lr · (r − Q[s,a])                            if terminal
```

**Hyperparameters** (applied to both T and CT learners):

| Parameter | Value | Notes |
|---|---|---|
| `alpha_lr` | 0.1 → 0.01 | decayed via `α / (1 + (9/(N−1)) · pass)` |
| `gamma` | 0.95 | weights sparse terminal win reward more |
| `blend_alpha` | 0.25 | `Q_kill` contributes 25% of recommendation |
| `n_passes` | 50 | full sweeps of the transition log per training run |
| reward normalization | ÷ 5.0 | applied in `training_data.py` so kill+site reward fits in `[−1, +1]` |

## Training Data

- Source: `data/rl_v2_train.csv` (and val/test splits) — one row per
  player-tick from `training_data.py` with columns:
  - state: `alive_adv, bomb_status, time_bucket, zone_idx, recent_event, team_support`
  - action: `action_ss` (the side-specific action label inferred from
    the player's behavior)
  - rewards: `kill_reward`, `win_reward`
  - housekeeping: `demo, round_num, player, tick, is_terminal, side`
- Side filter: T learner reads only `side == "T"` rows; CT learner
  only `side == "CT"`.
- Round termination: when `t_alive == 0` or `ct_alive == 0` the
  episode is treated as terminal — the last transition uses
  `Q(s') = 0`. This is enforced both in `training_data.py` (rows are
  truncated) and in `_SideQLearner._update` (the `done` flag).

## Evaluation

`metrics.evaluate_all()` computes **expert-action agreement**: the
fraction of test transitions where the Q-learner's argmax matches
the action the human player actually took. Random-policy agreement
in a 6-action space is 1/6 ≈ 16.7%; we report against 1/7 (14.3%)
to be conservative.

| Q-learner | Val agreement | Test agreement | State coverage | SA pairs visited |
|---|---:|---:|---:|---:|
| T-side  | 34.8% | 32.3% | 16.2% | 2 144 |
| CT-side | 42.0% | 41.9% | 17.6% | 2 483 |

See [`reports/qlearner_agreement.png`](../../reports/qlearner_agreement.png)
and [`reports/ql_coverage.png`](../../reports/ql_coverage.png).

The state-coverage figure is informative: only ~17% of the 6 300
states ever appear in the training data on either side. The model
recommends a uniform action for unvisited states. Expanding the
demo pool would directly raise coverage.

## Output Use

- `generate_report.py` calls `recommend(...)` at each event timestamp
  and emits a one-liner like:

  ```
  [RL_CT] ROTATE -- Rotate toward B
  ```

- The recommendation is suppressed once the target player dies
  (round-over-for-you logic).

## Known Caveats

- Tabular Q is an honest baseline at this dataset size — moving to
  function approximation (DQN) would be premature given 16% state
  coverage.
- The "expert action" labels in the training data are inferred
  heuristically from movements / kill positions. They are noisy
  ground truth, which caps the achievable agreement metric.
- The `team_support` state component is computed from teammates in
  the *same or adjacent* zone; that adjacency is hand-defined for
  Mirage and would need redoing for other maps.
