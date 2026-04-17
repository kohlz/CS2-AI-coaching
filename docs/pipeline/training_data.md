# Training-Data Extraction & Splits

**Source files:**

- [`src/analysis/training_data.py`](../../src/analysis/training_data.py)
- [`src/analysis/dataset_builder.py`](../../src/analysis/dataset_builder.py)
- [`src/analysis/train_pipeline.py`](../../src/analysis/train_pipeline.py)

## Role in the Pipeline

Turns the raw demo files in `src/demo/train_demos/` into the labelled
tables every model trains on. Splits the resulting rounds into
`train / val / test` and persists everything under `data/`.

Returning to [main pipeline](../README.md#1-pipeline).

## Python Packages

- `demoparser2` (via `demo_parser.py`)
- `pandas`, `numpy`
- `hashlib` — deterministic round-level split keys
- `json` — sequence + split-info persistence

## What Gets Extracted

`training_data.extract_all(demo_dir)` returns a dict with four
tables:

| Key | Format | One row per | Used by |
|---|---|---|---|
| `rounds` | DataFrame | round | [PreRoundFormation](../models/preround_formation.md), [PreRoundAttack](../models/preround_attack.md) |
| `rl_transitions` | DataFrame | player-tick | [Q-learners](../models/qlearning.md) |
| `event_sequences` | list[dict] | round | [FormationClassifier_T](../models/formation_classifier_t.md), [FormationClassifier_CT](../models/formation_classifier_ct.md) |
| `ct_formations` | list[dict] | round | sample weight / label diagnostics |

### `rounds` columns (selected)

Pre-round HMM features:

- `p_broke, p_low, p_medium, p_high, p_rich`
- `predicted_avg_money`, `round_in_half`, `is_second_half`

Prior-round tendency features (added in the model-quality pass):

- `prev_plant_A, prev_plant_B, prev_plant_none, prev_no_history`
- `prev_t_won, prev_t_tier, prev_ct_tier`
- `rounds_since_plant_A, rounds_since_plant_B, streak_same_site`

Labels:

- `ct_formation` — coarse zone counting from CT positions
- `attack_site` — `A` / `B` / `no_plant`

### `rl_transitions` columns

State: `alive_adv, bomb_status, time_bucket, zone_idx, recent_event,
team_support`. Action: `action_ss` (side-specific). Reward:
`kill_reward, win_reward` (both already normalized by 5.0).
Bookkeeping: `demo, round_num, player, tick, side, is_terminal`.

Reward normalization keeps the dual Q-table updates well-scaled —
without it `Q_kill` would dominate `Q_win` by ~10×.

### `event_sequences` items

Each item is `{"events": [...], "attack_site": "...", "ct_alive_at_event":
[...], "formation_labels": [...], "pre_round_prior": [...]}`.

`events` use the per-event encoding documented in
[FormationClassifier_T](../models/formation_classifier_t.md#architecture).

## Filters

- **Knife rounds** are dropped at extraction time (knife rounds do
  not pay real economy bonuses, so leaving them in would poison the
  HMM).
- **The held-out demo** `260319mirage.dem` is kept in `src/demo/`
  *outside* `train_demos/`; the dataset builder only walks
  `train_demos/` so this file can never enter the dataset.
- **Round termination** truncates RL transitions and LSTM event
  sequences at the tick when either side reaches 0 alive.

## Splits

`dataset_builder._split_rounds(...)` splits **by `(demo, round_num)`
pairs**, not by demo. Each round is treated as an independent example.

The split is deterministic: round keys are sorted by an MD5 hash and
the first 70% / next 15% / last 15% land in `train` / `val` / `test`.
This:

1. Reproduces exactly across machines / runs.
2. Mixes rounds from every demo into every split, which is
   appropriate because rounds within a demo share economy state but
   not formation / event labels — so weak leakage is acceptable for
   the task and we get much higher diversity per split.

The split mapping is persisted to `data/split_info.json` so that
later re-extracts can re-use it.

| Split | Rounds | Sequences (T LSTM) | Sequences (CT LSTM) |
|---:|---:|---:|---:|
| Train | 720 | 499 | 441 |
| Val   | 108 | 108 | 100 |
| Test  | 108 | 101 |  94 |

## Saved Artifacts

After `python src/analysis/train_pipeline.py extract`:

```
data/
  rounds_train.csv  rounds_val.csv  rounds_test.csv
  rl_v1_*.csv  rl_v2_*.csv
  event_sequences_*.json
  ct_formations_*.json
  split_info.json
```

After `python src/analysis/train_pipeline.py train`:

```
models/
  preround_formation.pt
  preround_attack.pt
  formation_classifier_t.pt
  formation_classifier_ct.pt
  tactical_ql_t.npz
  tactical_ql_ct.npz

data/
  training_results.json    # accuracy / F1 / loss histories per model
```
