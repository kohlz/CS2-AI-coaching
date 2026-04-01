# Observation Schema (v1 - Event/State Based)

This document defines the input format used by the decision / belief layer.
We prioritize **demo/HUD-derived** signals (time, alive counts, economy, hp/armor, weapon)
because VPR/CV may be unstable.

## Core Fields (MVP)
Minimum required fields to run the first prototype:

- `time_left` (int): seconds remaining in the round (e.g., 60)
- `team_alive` (int): number of alive teammates (including self)
- `enemy_alive` (int): number of alive enemies
- `hp` (int): player health (0-100)
- `armor` (int): player armor (0-100)
- `money` (int): current money (e.g., 200, 3500)
- `has_rifle` (bool): whether player has a primary rifle (coarse indicator)

## Optional Fields (Later)
These fields can be added when available:

- `team_kills` (int)
- `team_deaths` (int)
- `weapon_type` (str): e.g., "rifle", "awp", "pistol", "smg"
- `utility_smoke` (bool), `utility_flash` (bool), `utility_molotov` (bool)
- `bomb_planted` (bool)
- `round_number` (int)

## Optional VPR Fields (Weak Signal)
If VPR becomes usable, we can include it as an optional weak prior:

- `place_probs` (dict): {"A": 0.2, "B": 0.7, "MID": 0.1}
- `place_id` (str): top-1 region label

The belief layer can ignore these when unreliable.

## Example Observation (Economy/Situation)
```json
{
  "time_left": 60,
  "team_alive": 0,
  "enemy_alive": 5,
  "hp": 100,
  "armor": 100,
  "money": 200,
  "has_rifle": true
}
{
  "time_left": 23,
  "teammate_death_site": "B",
  "place_probs": {"A": 0.20, "B": 0.70, "MID": 0.10},
  "smoke_B": true
}
