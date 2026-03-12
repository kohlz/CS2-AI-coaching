
# Simplified POMDP / Belief-State Model (v0)

## Why POMDP-style Modeling?
In CS2, the true enemy positions are not fully observable. We only have partial observations (visual cues, teammate events, timing).  
We therefore maintain a **belief state** over likely enemy presence in coarse map regions, and update it as new observations arrive.

## Hidden State (Simplified)
We model enemy presence at a coarse level:

- EnemyRegion ∈ {"A", "B", "MID", "UNKNOWN"}

## Observations
We use partial observations such as:
- time remaining in round (`time_left`)
- teammate death site (`teammate_death_site`)
- VPR output as a distribution over map regions (`place_probs`)
- (optional) utility cues like smoke, and any "seen enemy" signals

## Belief State
A probability distribution over regions:
- belief = P(EnemyRegion)

We initialize or partially set belief using VPR output (`place_probs`) as a prior, then update it with simple rules.

## Belief Update (Rule-Based v0)
Example intuition:
- If teammate dies at B, increase belief(B)
- If time_left is low, rotating is more urgent
- If smoke blocks a site, reduce confidence of direct information and rely more on belief
- If "seen enemy at A", increase belief(A) strongly

This is a first prototype update rule (not a full optimal POMDP solver).

## Actions / Outputs
We output a label from:
- ROTATE_A, ROTATE_B, HOLD, REPOSITION, PLAY_SAFE

The label is chosen based on the updated belief and simple decision logic.

## Extension (Later)
If time allows, we may:
- use a finer region grid
- learn update weights from data
- evaluate policies under simulated scenarios
