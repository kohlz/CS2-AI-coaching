# Observation Schema (v0)

This document defines the input format used by the belief/POMDP layer.

## Core Fields (MVP)
These fields are the minimum required to run the first prototype.

- `time_left` (int): seconds remaining in the round
- `teammate_death_site` (str): one of {"A", "B", "NONE"}
- `place_probs` (dict): probability distribution over coarse map regions  
  Example: {"A": 0.15, "B": 0.75, "MID": 0.10}

## Optional Fields (Later)
These fields can be added when available.

- `smoke_A` (bool)
- `smoke_B` (bool)
- `saw_enemy_A` (bool)
- `saw_enemy_B` (bool)
- `player_health` (int)
- `teammates_alive` (int)

## Example Observation
```json
{
  "time_left": 23,
  "teammate_death_site": "B",
  "place_probs": {"A": 0.20, "B": 0.70, "MID": 0.10},
  "smoke_B": true
}
