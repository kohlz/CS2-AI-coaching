# Economy MDP — How It Works

This document explains the Markov Decision Process (MDP) used in
`src/analysis/economy_mdp.py` to model CS2 round-by-round buy decisions
and evaluate a player's economic choices against the computed optimal policy.

---

## 1. Motivation

In Counter-Strike 2, money management between rounds is one of the most
impactful strategic decisions a player makes. Buying too aggressively
(force-buying) wastes money and leaves the team under-equipped in later
rounds; buying too conservatively (eco-ing when a full buy is affordable)
throws away winnable rounds. A good economy strategy maximizes the total
number of rounds won over the course of a half.

This module frames that sequential decision problem as a **finite-state,
infinite-horizon, discounted MDP** and solves it with **Value Iteration**
to obtain the optimal buy policy for every reachable economy state.

---

## 2. CS2 Economy Rules (Simplified)

The MDP encodes the following real CS2 money mechanics:

| Rule | Value |
|------|-------|
| Starting money (each half) | $800 |
| Money cap | $16,000 |
| Round win reward (elimination / time) | $3,250 |
| Round win reward (bomb explode for T, defuse for CT) | $3,500 |
| T bomb-plant bonus (all T players, even on loss) | $800 |
| Loss bonus (consecutive losses 1–5+) | $1,400 → $1,900 → $2,400 → $2,900 → $3,400 |
| Loss bonus reset | Back to $1,400 after a win |
| Kill reward (rifles, pistols, LMGs) | $300 |
| Kill reward (SMGs, excl. P90) | $600 |
| Kill reward (shotguns, excl. XM1014) | $900 |
| Kill reward (knife) | $1,500 |
| Kill reward (AWP / Zeus) | $100 |

The loss bonus ladder is the most strategically important rule — it
creates a tension between spending now (lower win probability but you
still get money) and saving for a full buy later (higher win probability
once fully equipped).

---

## 3. MDP Formulation

### 3.1 State Space

A state is a tuple of two variables:

| Variable | Range | Description |
|----------|-------|-------------|
| `money` | $0, $500, $1000, …, $16,000 (33 bins) | Team average money at round start, discretised into $500 steps |
| `loss_streak` | 0, 1, 2, 3, 4, 5 | Consecutive losses so far (capped at 5) |

This gives **33 × 6 = 198 states** per side. Two separate MDPs are
solved — one for T side and one for CT side — because costs and win
probabilities differ, producing **396 total states**.

### 3.2 Action Space

Three discrete actions represent the three standard buy tiers in CS2:

| Action | Label | Approximate Cost (T / CT) | What It Means |
|--------|-------|---------------------------|---------------|
| 0 | **SAVE** | $200 / $200 | Pistol round or eco — spend almost nothing |
| 1 | **FORCE** | $2,600 / $2,800 | SMG or shotgun + partial armor/utility |
| 2 | **FULL_BUY** | $4,700 / $5,500 | Rifle + full armor + helmet + full utility |

An action is **only available** if the player can afford it (i.e.
`money ≥ EQUIP_COST[side][action]`). The `SAVE` action is always
available.

### 3.3 Transition Model

Taking action `a` in state `(money, loss_streak)` leads to:

1. **Spend**: `money_after_buy = money − cost(a)`
2. **Win probability**: `p_win = Σ_j  P(win | my_tier, opp_tier_j) × P(opp_tier_j)`

   The opponent's equipment distribution is estimated from the player's
   loss streak (if the player has been losing, the opponent has been
   winning and is likely fully equipped):

   | My loss streak | P(opp eco) | P(opp force) | P(opp full) |
   |----------------|-----------|-------------|------------|
   | 0 (just won) | 0.45 | 0.30 | 0.25 |
   | 1 | 0.10 | 0.20 | 0.70 |
   | 2+ | 0.05 | 0.10 | 0.85 |

   The base win probabilities `P(win | my_tier, opp_tier)` are stored in
   a 3×3 matrix per side. For example, a T-side full buy vs opponent full
   buy has a 48% win rate; a CT-side eco vs opponent full buy has only 15%.

3. **Win branch** (probability `p_win`):
   - Next money = `money_after_buy + win_reward + avg_kill_income_win + side_bonuses`
   - Loss streak resets to 0
   - Capped at $16,000

4. **Loss branch** (probability `1 − p_win`):
   - Loss streak increments: `new_streak = min(streak + 1, 5)`
   - Next money = `money_after_buy + loss_bonus(new_streak) + avg_kill_income_loss + side_bonuses`
   - Capped at $16,000

### 3.4 Reward Function

| Outcome | Reward |
|---------|--------|
| Win the round | +1.0 |
| Lose the round | −0.3 |

The asymmetric reward (loss penalty is not −1.0) reflects that losing a
round is bad but not catastrophic — you still accumulate loss bonus money
and the game continues. The negative penalty discourages gambling on low
win-probability buys.

### 3.5 Discount Factor

**γ = 0.85**

A discount factor below 1.0 ensures the value function converges and
encodes the idea that winning the current round matters more than winning
a hypothetical future round. At γ = 0.85, a round win five rounds from
now is worth about 0.85⁵ ≈ 0.44 of an immediate win.

---

## 4. Solving the MDP: Value Iteration

The optimal value function `V*(s)` and policy `π*(s)` are computed using
the **Bellman optimality equation**:

```
V*(money, streak) = max_a [ p_win(a) × (R_win + γ × V*(s_win))
                          + (1 − p_win(a)) × (R_loss + γ × V*(s_loss)) ]
```

where:
- `s_win = (next_money_win, 0)` — money after winning, streak resets
- `s_loss = (next_money_loss, streak+1)` — money after losing, streak increments

### Algorithm

```
Initialise V(s) = 0 for all states
Repeat until convergence (max |V_new − V| < 1e-6 or 500 iterations):
    For each state (money_bin, streak):
        For each affordable action a ∈ {SAVE, FORCE, FULL_BUY}:
            Compute Q(s, a) using the Bellman equation above
        V_new(s) = max_a Q(s, a)
        π(s) = argmax_a Q(s, a)
    V ← V_new
```

Because the state space is small (198 states per side), Value Iteration
converges in well under 100 iterations and runs in milliseconds.

### Output

The solver returns an `EconomyPolicy` object containing:
- `V` — a 33×6 numpy array of optimal state values
- `policy` — a 33×6 numpy array where each entry is 0 (SAVE), 1 (FORCE), or 2 (FULL_BUY)
- `recommend(money, loss_streak)` — look up the optimal action for any state
- `value(money, loss_streak)` — look up the expected discounted future reward

---

## 5. Player Buy-Decision Classifier

Before comparing against the optimal policy, the system must determine
what the player **actually did**. The function `classify_buy_decision`
maps a player's post-buy loadout to one of the three action tiers:

| Condition (checked in order) | Classification |
|------------------------------|----------------|
| Primary weapon is a rifle or sniper (AK-47, M4, AWP, etc.) | FULL_BUY |
| Primary weapon is an SMG, shotgun, or budget rifle (Galil, FAMAS) | FORCE |
| Equipment value ≥ $3,500 | FULL_BUY |
| Equipment value ≥ $1,500 | FORCE |
| Has armor + secondary or utility | FORCE |
| Otherwise | SAVE |

---

## 6. Round-by-Round Evaluation Pipeline

The `evaluate_player_economy` function walks through every round and
produces a `BuyEvaluation` for each:

1. **Skip special rounds**: knife round (round 0), pistol rounds (round 1
   and first round of second half), and any round where the player has
   under $1,000 (economy-reset scenarios).

2. **Detect halftime**: when the player's side flips (T→CT or CT→T), the
   loss streak resets to 0 and the next round is treated as a pistol
   round.

3. **Classify actual buy**: use the classifier above on the player's
   post-buy inventory.

4. **Handle equipment carry-over**: if the player survived the previous
   round, they keep their gear. A player carrying a rifle from a won
   round isn't making a "buy decision" — they already have the
   equipment. The evaluation marks these as `(carried equipment)` and
   counts them as optimal.

5. **Compare against MDP policy**: look up
   `policy.recommend(money, loss_streak)` and compare to the actual buy
   tier.

6. **Compute expected values**: for both the actual and optimal action,
   calculate the one-step Bellman Q-value (immediate reward plus
   discounted future value). This quantifies how much value the player
   gained or lost by deviating from optimal.

7. **Generate a coaching note** for sub-optimal decisions:
   - *"Over-buying: should save for better buy next round"* — when the
     player force-buys or full-buys but the MDP says to save
   - *"Under-buying: had enough for full buy"* — when the player saves
     or forces but could afford a full buy
   - *"Could not afford optimal buy"* — when the optimal action costs
     more than the player has

---

## 7. Summary and Grading

The `economy_summary` function aggregates all per-round evaluations into
a report card:

| Metric | Description |
|--------|-------------|
| `total_rounds` | Number of rounds evaluated (excluding knife/pistol) |
| `fresh_buy_rounds` | Rounds where the player made an active buy decision |
| `team_drop_rounds` | Rounds where equipment was carried from previous round |
| `optimal_decisions` | Count of rounds matching the MDP-optimal action |
| `fresh_buy_accuracy` | Accuracy on rounds where a real buy decision was made |
| `overall_accuracy` | Accuracy across all evaluated rounds |
| `over_buys` | Mistakes where the player bought more than optimal |
| `under_buys` | Mistakes where the player bought less than optimal |
| `avg_wp_loss_per_mistake` | Average win-probability lost per mistake |
| `grade` | Letter grade based on overall accuracy |

### Grading Scale

| Accuracy | Grade |
|----------|-------|
| ≥ 90% | A |
| ≥ 75% | B |
| ≥ 60% | C |
| ≥ 45% | D |
| < 45% | F |

---

## 8. Example: Interpreting the Policy

Running the solver and printing the T-side policy produces a table like:

```
Money     W       L1       L2       L3       L4       L5
--------------------------------------------------------------
 $0      SAVE     SAVE     SAVE     SAVE     SAVE     SAVE
 $500    SAVE     SAVE     SAVE     SAVE     SAVE     SAVE
 ...
 $2500   SAVE     SAVE     SAVE     SAVE     SAVE     SAVE
 $3000   FORCE    FORCE    SAVE     SAVE     SAVE     SAVE
 ...
 $4500   FULL_BUY FULL_BUY SAVE     SAVE     SAVE     SAVE
 $5000   FULL_BUY FULL_BUY FULL_BUY FORCE    FORCE    FORCE
 ...
```

Reading this table: at $4,500 after just winning (column W), the T-side
optimal action is FULL_BUY. But at $4,500 after two consecutive losses
(column L2), the optimal action is SAVE — because the loss bonus is
building and saving guarantees a full buy next round with teammates also
having enough money.

---

## 9. Data Flow

```
  .dem file
     │
     ▼
  demo_parser.py ──► RoundData (with PlayerRound per player)
                        │
                        │  start_money, equipment_value,
                        │  primary_weapon, armor, alive_at_end, ...
                        ▼
                  economy_mdp.py
                     ┌──────────────────────────┐
                     │ 1. solve_economy_mdp()    │ ← runs once per side
                     │    Value Iteration        │
                     │    → EconomyPolicy (T/CT) │
                     ├──────────────────────────┤
                     │ 2. classify_buy_decision()│ ← per round per player
                     │    loadout → SAVE/FORCE/  │
                     │              FULL_BUY     │
                     ├──────────────────────────┤
                     │ 3. evaluate_player_economy│ ← walks all rounds
                     │    actual vs optimal      │
                     │    → list[BuyEvaluation]  │
                     ├──────────────────────────┤
                     │ 4. economy_summary()      │
                     │    → accuracy, grade,     │
                     │      mistake breakdown    │
                     └──────────────────────────┘
                        │
                        ▼
                  Report Generator (templates.py / report_generator.py)
```

---

## 10. Key Design Decisions

1. **Discretised money in $500 bins** — keeps the state space at 33 bins
   (198 states per side) so Value Iteration runs in milliseconds. The
   $500 granularity is fine enough to distinguish eco/force/full buy
   thresholds.

2. **Loss streak capped at 5** — CS2 caps the loss bonus at 5
   consecutive losses ($3,400). Further losses don't change the bonus,
   so modelling streaks beyond 5 adds no information.

3. **Opponent equipment estimated from loss streak** — rather than
   tracking the opponent's exact money (which would double the state
   space), the model infers a distribution over opponent equipment tiers
   from the player's own loss streak. This is a reasonable heuristic:
   if the player has been losing, the opponent has been winning and is
   probably fully equipped.

4. **Side-specific models** — T and CT have different equipment costs
   (CT buys are more expensive), different win probabilities (CT has a
   structural defensive advantage), and different bonuses (T gets bomb
   plant money, CT gets kill-sharing bonuses). The MDP is solved
   independently for each side.

5. **Equipment carry-over handling** — the evaluation pipeline detects
   when a player survived the previous round and carries equipment. These
   rounds are not penalised even if the "policy" would recommend a
   different tier, because the player didn't make a fresh buy decision.
