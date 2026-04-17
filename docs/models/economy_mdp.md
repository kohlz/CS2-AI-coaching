# Economy MDP — Optimal Buy Policy

**Source:** [`src/analysis/economy_mdp.py`](../../src/analysis/economy_mdp.py)

## Role in the Pipeline

The MDP solves for the **optimal buy decision** (`SAVE` / `FORCE` /
`FULL_BUY`) in every reachable economy state, then compares the
player's actual buy to that optimum to produce coaching feedback.
Its output is one of the two pre-round signals shown in the final
report (the other is the [HMM](hmm.md)).

Returning to [main pipeline](../README.md#1-pipeline).

## Python Packages

- `numpy` — state / value / policy tensors
- `dataclasses` — `EconomyPolicy`
- No ML library; the MDP is solved by Value Iteration in pure NumPy.
- Economy constants are imported from
  [`info_model.py`](../../src/analysis/info_model.py) — the HMM is the
  single source of truth for CS2 money rules.

## Model Specification

**States.** `(money_bin, my_loss_streak, enemy_loss_streak)`:

- `money_bin` — 0 to 32, binning `$0–$16 000` in `$500` steps.
- `my_loss_streak` — 0 to 5 (capped at `MAX_LOSS_STREAK`).
- `enemy_loss_streak` — 0 to 5.

Total state count: `33 × 6 × 6 = 1 188` per side.

**Actions.** `{SAVE, FORCE, FULL_BUY}` with side-specific equipment
costs:

| Side | SAVE | FORCE | FULL_BUY |
|---|---:|---:|---:|
| T   | $200 | $2 600 | $4 700 |
| CT  | $200 | $2 800 | $5 500 |

**Transition model.** For each `(state, action)`:

- Win probability is looked up in a 3×3 `WIN_PROB` matrix indexed by
  `(my_equipment_tier, opponent_equipment_tier)`. The opponent tier
  distribution is derived from the enemy's loss streak via
  `_opponent_equip_dist(enemy_loss_streak)` (e.g. `streak=4` → 75%
  chance opponent is full-buying).
- On win: `my_streak → 0`, `enemy_streak → +1`, money updated by
  `next_money_win(side, money_after_buy)` from `info_model.py`.
- On loss: `my_streak → +1`, `enemy_streak → 0`, money updated by
  `next_money_loss(side, money_after_buy, new_streak)`.

**Rewards.** Sparse: `+1.0` on round win, `−0.3` on loss.

**Discount.** `γ = 0.85`, infinite-horizon.

**Solver.** Value Iteration with `ε = 1e-6`, capped at 500 sweeps.
The relevant code is `solve_economy_mdp(side)` which returns an
`EconomyPolicy` dataclass with `V` and `policy` tensors.

## Training Data

The MDP is **not learned from data** — it is solved analytically from
the CS2 economy rules encoded in
[`info_model.py`](../../src/analysis/info_model.py). The only
data-dependent step is evaluation: per-round buy quality is computed
by looking up the observed `(money, my_streak, enemy_streak)` in the
solved policy and checking whether the player's actual buy matched
the recommendation. That evaluation feeds the "Pre-Round (Economy)"
line in the generated report.

## Output Format

Two consumers use the solved policy:

- `EconomyPolicy.recommend(money, loss_streak, enemy_loss_streak)`
  returns one of `{0: SAVE, 1: FORCE, 2: FULL_BUY}`.
- Post-hoc evaluation inside `generate_report.py` also separately
  checks the player's **weapon tier, utility count, armor, and kit**
  against what a team in the predicted enemy tier would expect — this
  is why the MDP is decoupled from per-item evaluation (avoids state
  explosion on the solver side).

Example coaching output line:

```
[MDP] Good buy (FULL_BUY at $5 500). Waste: $0
```

## Known Caveats

- The opponent-equipment lookup `_opponent_equip_dist` is a legacy
  heuristic from before the HMM consolidation. The MDP still works
  correctly with it because the streak is a strong proxy, but the
  more principled path (using HMM tier probabilities directly
  inside `_expected_win_prob`) is a natural next integration step.
- The reward is a coarse win/lose signal. A richer reward that
  includes per-round expected damage or eco-quality could tighten
  the policy but at the cost of more state dimensions.
