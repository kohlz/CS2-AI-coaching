# Economy MDP v2 — How It Works

This document explains the Markov Decision Process (MDP) used in
`src/analysis/economy_mdp.py` to model CS2 round-by-round buy decisions
and evaluate a player's economic choices against the computed optimal policy.

**v2 addition**: the MDP now tracks the **enemy team's loss streak** as
an observable state variable, enabling direct opponent economy prediction
from the scoreboard instead of relying on a crude heuristic.

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
| Loss bonus (consecutive losses 1-5+) | $1,400 / $1,900 / $2,400 / $2,900 / $3,400 |
| Loss bonus reset | Back to $1,400 after a win |
| Kill reward (rifles, pistols, LMGs) | $300 |
| Kill reward (SMGs, excl. P90) | $600 |
| Kill reward (shotguns, excl. XM1014) | $900 |
| Kill reward (knife) | $1,500 |
| Kill reward (AWP / Zeus) | $100 |

The loss bonus ladder is the most strategically important rule -- it
creates a tension between spending now (lower win probability but you
still get money) and saving for a full buy later (higher win probability
once fully equipped).

---

## 3. MDP Formulation

### 3.1 State Space

A state is a tuple of **three** variables:

| Variable | Range | Description |
|----------|-------|-------------|
| `money` | $0, $500, $1000, ..., $16,000 (33 bins) | Team average money at round start, discretised into $500 steps |
| `my_loss_streak` | 0, 1, 2, 3, 4, 5 | Player's team consecutive losses (capped at 5) |
| `enemy_loss_streak` | 0, 1, 2, 3, 4, 5 | Opponent team consecutive losses (capped at 5) |

This gives **33 x 6 x 6 = 1,188 states** per side. Two separate MDPs
are solved -- one for T side and one for CT side -- producing
**2,376 total states**.

The enemy loss streak is observable in CS2: the scoreboard shows round
results, so a player always knows the opponent's win/loss history.

### 3.2 Action Space

Three discrete actions represent the three standard buy tiers in CS2:

| Action | Label | Approximate Cost (T / CT) | What It Means |
|--------|-------|---------------------------|---------------|
| 0 | **SAVE** | $200 / $200 | Pistol round or eco -- spend almost nothing |
| 1 | **FORCE** | $2,600 / $2,800 | SMG or shotgun + partial armor/utility |
| 2 | **FULL_BUY** | $4,700 / $5,500 | Rifle + full armor + helmet + full utility |

An action is **only available** if the player can afford it (i.e.
`money >= EQUIP_COST[side][action]`). The `SAVE` action is always
available.

### 3.3 Transition Model

Taking action `a` in state `(money, my_streak, enemy_streak)` leads to:

1. **Spend**: `money_after_buy = money - cost(a)`

2. **Win probability**: `p_win = SUM_j  P(win | my_tier, opp_tier_j) x P(opp_tier_j | enemy_streak)`

   The opponent's equipment distribution is estimated directly from
   their observable loss streak:

   | Enemy loss streak | P(opp eco) | P(opp force) | P(opp full) | Rationale |
   |-------------------|-----------|-------------|------------|-----------|
   | 0 (just won) | 0.05 | 0.10 | 0.85 | Flush with cash from win reward |
   | 1 | 0.65 | 0.20 | 0.15 | Classic eco after losing with full buy |
   | 2 | 0.30 | 0.45 | 0.25 | Building up, might force |
   | 3 | 0.10 | 0.25 | 0.65 | $2,400 loss bonus -- 3rd-round buy |
   | 4 | 0.05 | 0.20 | 0.75 | $2,900 bonus, can full buy |
   | 5+ | 0.05 | 0.15 | 0.80 | Max $3,400 bonus, definitely buying |

   The base win probabilities `P(win | my_tier, opp_tier)` are stored in
   a 3x3 matrix per side. For example, a T-side full buy vs opponent full
   buy has a 48% win rate; a CT-side eco vs opponent full buy has only 15%.

3. **Win branch** (probability `p_win`):
   - Next money = `money_after_buy + win_reward + avg_kill_income + side_bonuses`
   - My loss streak resets to 0
   - **Enemy loss streak increments**: `enemy_streak' = min(enemy_streak + 1, 5)`
   - Capped at $16,000

4. **Loss branch** (probability `1 - p_win`):
   - My loss streak increments: `my_streak' = min(my_streak + 1, 5)`
   - **Enemy loss streak resets to 0** (they won)
   - Next money = `money_after_buy + loss_bonus(my_streak') + avg_kill_income_loss + side_bonuses`
   - Capped at $16,000

The streak transitions are the key v2 insight: winning and losing affect
**both** streaks in opposite directions, which the MDP captures exactly.

### 3.4 Reward Function

| Outcome | Reward |
|---------|--------|
| Win the round | +1.0 |
| Lose the round | -0.3 |

The asymmetric reward (loss penalty is not -1.0) reflects that losing a
round is bad but not catastrophic -- you still accumulate loss bonus money
and the game continues. The negative penalty discourages gambling on low
win-probability buys.

### 3.5 Discount Factor

**gamma = 0.85**

A discount factor below 1.0 ensures the value function converges and
encodes the idea that winning the current round matters more than winning
a hypothetical future round. At gamma = 0.85, a round win five rounds from
now is worth about 0.85^5 = 0.44 of an immediate win.

---

## 4. Solving the MDP: Value Iteration

The optimal value function `V*(s)` and policy `pi*(s)` are computed using
the **Bellman optimality equation**:

```
V*(money, k, ek) = max_a [
    p_win(a, ek) x (R_win + gamma x V*(s_win)) +
    (1 - p_win(a, ek)) x (R_loss + gamma x V*(s_loss))
]
```

where:
- `s_win  = (next_money_win,  0,            min(ek+1, 5))` -- I won, my streak resets, enemy's grows
- `s_loss = (next_money_loss,  min(k+1, 5),  0)`           -- I lost, my streak grows, enemy's resets

### Algorithm

```
Initialise V(s) = 0 for all 1,188 states
Repeat until convergence (max |V_new - V| < 1e-6 or 500 iterations):
    For each state (money_bin, my_streak, enemy_streak):
        For each affordable action a in {SAVE, FORCE, FULL_BUY}:
            Compute Q(s, a) using the Bellman equation above
        V_new(s) = max_a Q(s, a)
        pi(s) = argmax_a Q(s, a)
    V <- V_new
```

With 1,188 states per side (6x more than v1), Value Iteration still
converges in under 100 iterations and completes in about 3 seconds.

### Output

The solver returns an `EconomyPolicy` object containing:
- `V` -- a 33x6x6 numpy array of optimal state values
- `policy` -- a 33x6x6 numpy array where each entry is 0 (SAVE), 1 (FORCE), or 2 (FULL_BUY)
- `recommend(money, loss_streak, enemy_loss_streak)` -- look up the optimal action
- `value(money, loss_streak, enemy_loss_streak)` -- look up the expected discounted future reward

---

## 5. Player Buy-Decision Classifier

Before comparing against the optimal policy, the system must determine
what the player **actually did**. The function `classify_buy_decision`
maps a player's post-buy loadout to one of the three action tiers:

| Condition (checked in order) | Classification |
|------------------------------|----------------|
| Primary weapon is a rifle or sniper (AK-47, M4, AWP, etc.) | FULL_BUY |
| Primary weapon is an SMG, shotgun, or budget rifle (Galil, FAMAS) | FORCE |
| Equipment value >= $3,500 | FULL_BUY |
| Equipment value >= $1,500 | FORCE |
| Has armor + secondary or utility | FORCE |
| Otherwise | SAVE |

---

## 6. Round-by-Round Evaluation Pipeline

The `evaluate_player_economy` function walks through every round and
produces a `BuyEvaluation` for each:

1. **Skip special rounds**: knife round (round 0), pistol rounds (round 1
   and first round of second half), and any round where the player has
   under $1,000 (economy-reset scenarios).

2. **Detect halftime**: when the player's side flips (T->CT or CT->T),
   **both** loss streaks reset to 0 and the next round is treated as a
   pistol round.

3. **Track enemy loss streak**: after each round, update both streaks:
   - Win: `my_streak = 0`, `enemy_streak += 1`
   - Loss: `my_streak += 1`, `enemy_streak = 0`

4. **Classify actual buy**: use the classifier above on the player's
   post-buy inventory.

5. **Handle equipment carry-over**: if the player survived the previous
   round, they keep their gear. A player carrying a rifle from a won
   round isn't making a "buy decision" -- they already have the
   equipment. The evaluation marks these as `(carried equipment)` and
   counts them as optimal.

6. **Compare against MDP policy**: look up
   `policy.recommend(money, loss_streak, enemy_loss_streak)` and compare
   to the actual buy tier.

7. **Compute expected values**: for both the actual and optimal action,
   calculate the one-step Bellman Q-value (immediate reward plus
   discounted future value). This quantifies how much value the player
   gained or lost by deviating from optimal.

8. **Generate coaching notes**:
   - *"Over-buying: should save for better buy next round"*
   - *"Under-buying: had enough for full buy"*
   - *"Could not afford optimal buy"*
   - **Enemy economy prediction**: *"Enemy lost 1 -- likely ECO round"*
     or *"Enemy lost 3 -- loss bonus allows FULL BUY"*

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
| `mistakes_vs_enemy_eco` | Mistakes in rounds where enemy was on eco (streak 1) |
| `mistakes_vs_enemy_buy` | Mistakes in rounds where enemy just won (streak 0) |
| `avg_wp_loss_per_mistake` | Average win-probability lost per mistake |
| `grade` | Letter grade based on overall accuracy |

### Grading Scale

| Accuracy | Grade |
|----------|-------|
| >= 90% | A |
| >= 75% | B |
| >= 60% | C |
| >= 45% | D |
| < 45% | F |

---

## 8. Example: How Enemy Streak Changes the Policy

The v2 model produces different policies depending on the enemy's
economic situation. Compare T-side at $3,000 with 0 losses (just won):

| Enemy streak | Optimal action | Why |
|-------------|---------------|-----|
| 0 (enemy just won) | **FORCE** | Enemy is full buying -- you need equipment to compete |
| 1 (enemy lost 1) | **FORCE** | Enemy is eco -- even a force buy wins easily |
| 3 (enemy lost 3) | **FORCE** or **SAVE** | Enemy recovered with loss bonus -- save to match their full buy |

At higher money ($5,500+), the policy almost always recommends FULL_BUY
regardless of enemy streak. The interesting decisions happen in the
$2,500-$5,000 range where the enemy's economic state determines whether
forcing or saving is optimal.

### CT-side with enemy on 3-loss streak

When the enemy (T-side) has lost 3 rounds, they have accumulated $2,400
in loss bonus and can likely full buy. The CT policy responds by becoming
more conservative -- preferring FORCE over FULL_BUY up to $8,000+ in
some states, to ensure economic sustainability against a well-equipped
opponent over multiple rounds.

---

## 9. Data Flow

```
  .dem file
     |
     v
  demo_parser.py --> RoundData (with PlayerRound per player)
                        |
                        |  start_money, equipment_value,
                        |  primary_weapon, armor, alive_at_end,
                        |  round winner (for enemy streak tracking)
                        v
                  economy_mdp.py
                     +------------------------------+
                     | 1. solve_economy_mdp()        | <-- runs once per side
                     |    Value Iteration on 3D      |
                     |    state (money, k, ek)       |
                     |    --> EconomyPolicy (T/CT)   |
                     +------------------------------+
                     | 2. classify_buy_decision()    | <-- per round per player
                     |    loadout --> SAVE/FORCE/    |
                     |                FULL_BUY       |
                     +------------------------------+
                     | 3. evaluate_player_economy    | <-- walks all rounds
                     |    tracks both loss streaks   |
                     |    actual vs optimal          |
                     |    + enemy buy prediction     |
                     |    --> list[BuyEvaluation]    |
                     +------------------------------+
                     | 4. economy_summary()          |
                     |    --> accuracy, grade,       |
                     |      mistake breakdown,       |
                     |      vs-eco / vs-buy splits   |
                     +------------------------------+
                        |
                        v
                  Report Generator (templates.py / report_generator.py)
```

---

## 10. Key Design Decisions

1. **Discretised money in $500 bins** -- keeps the state space at 33 bins.
   Combined with the two streak dimensions, this gives 1,188 states per
   side -- still trivial for Value Iteration (converges in ~3 seconds).

2. **Loss streak capped at 5** -- CS2 caps the loss bonus at 5
   consecutive losses ($3,400). Further losses don't change the bonus,
   so modelling streaks beyond 5 adds no information.

3. **Enemy loss streak as observable state (v2)** -- in v1, opponent
   equipment was guessed from the player's own loss streak (a crude
   heuristic). In v2, the enemy's loss streak is tracked directly from
   round outcomes. This is realistic: in CS2 the scoreboard reveals
   round history, so every player knows the opponent's streak. The
   improvement enables:
   - More accurate opponent buy-tier prediction
   - Context-aware coaching notes ("enemy is eco, you can force")
   - Different optimal policies depending on whether the opponent is
     rich or broke

4. **Side-specific models** -- T and CT have different equipment costs
   (CT buys are more expensive), different win probabilities (CT has a
   structural defensive advantage), and different bonuses (T gets bomb
   plant money, CT gets kill-sharing bonuses). The MDP is solved
   independently for each side.

5. **Equipment carry-over handling** -- the evaluation pipeline detects
   when a player survived the previous round and carries equipment. These
   rounds are not penalised even if the "policy" would recommend a
   different tier, because the player didn't make a fresh buy decision.

6. **Streak transition symmetry** -- when the player wins, their streak
   resets to 0 while the enemy's increments. When the player loses, the
   opposite happens. This means the MDP naturally models the economic
   "seesaw" between teams -- a key dynamic in competitive CS2 where
   economy leads shift back and forth.

---

## 11. v1 vs v2 Comparison

| Aspect | v1 | v2 |
|--------|----|----|
| State space | (money, my_streak) -- 198 per side | (money, my_streak, enemy_streak) -- 1,188 per side |
| Opponent modelling | Inferred from own streak (heuristic) | Directly from enemy's observable streak |
| Enemy buy prediction | None | Per-round coaching note |
| Summary stats | General accuracy | + breakdown by enemy eco vs enemy buy rounds |
| Solve time | ~0.5s | ~3s |
| Policy sensitivity | Same policy regardless of enemy state | Different policy per enemy streak |
