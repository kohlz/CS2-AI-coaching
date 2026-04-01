
# Simplified POMDP / Belief-State Model (v1 - Event/State Based)

## Why POMDP-style Modeling?
In CS2, we do not observe the full game state (enemy intentions, rotations, exact positions).
However, we can observe **events and HUD signals** (time, alive counts, economy, hp/armor, weapon).
We maintain a **belief** about "how risky the situation is" and output a recommendation label.

This v1 model focuses on macro coaching decisions (SAVE/BUY) because they are more reliably
inferred from demo/HUD signals than fine-grained spatial positioning.

## Hidden State (Simplified)
We use a coarse hidden state representing the situation difficulty/risk:

- DangerLevel ∈ {LOW, MEDIUM, HIGH}

Interpretation:
- LOW: favorable/low-risk situation
- MEDIUM: uncertain/mixed
- HIGH: high-risk (outnumbered, low economy, low time, etc.)

## Observations (Event/State Signals)
From demo/HUD we can observe:
- `time_left`
- `team_alive`, `enemy_alive`
- `money`
- `hp`, `armor`
- `has_rifle` / weapon indicators

Optional weak signals:
- `place_probs` / `place_id` (VPR), if available

## Belief State
A probability distribution over DangerLevel:
- belief = P(DangerLevel)

We update the belief using simple rules (v1 baseline). Later, weights can be learned.

## Belief Update (Rule-Based v1)
Examples:
- If `team_alive` is much lower than `enemy_alive`, increase HIGH risk
- If `money` is very low, increase HIGH risk
- If `time_left` is low and we are outnumbered, increase HIGH risk
- If `hp/armor` are low, increase risk

## Actions / Outputs
We output a label:
- SAVE, BUY, FORCE_BUY, PLAY_SAFE

Mapping idea:
- HIGH risk + low money + has_rifle → SAVE
- good money + stable situation → BUY
- medium money + needs impact → FORCE_BUY
- uncertain situation → PLAY_SAFE

## Extension (Later)
- Replace rule-based mapping with learned policy (MDP/RL)
- Add more labels and context (bomb planted, site, utility, etc.)
- Use demo parsing to build training dataset and evaluate decisions quantitatively
