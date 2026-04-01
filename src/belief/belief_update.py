from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional


# Optional VPR regions (weak signal)
REGIONS = ["A", "B", "MID"]

# Output labels (v1: macro decisions)
LABELS = ["SAVE", "BUY", "FORCE_BUY", "PLAY_SAFE", "ROTATE", "HOLD", "REPOSITION"]


@dataclass
class Observation:
    # Core event/state fields (MVP)
    time_left: int
    team_alive: int
    enemy_alive: int
    hp: int
    armor: int
    money: int
    has_rifle: bool

    # Optional VPR fields (weak signal)
    place_probs: Optional[Dict[str, float]] = None
    place_id: Optional[str] = None

    # Optional extra fields (future)
    bomb_planted: bool = False


def normalize(dist: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(v, 0.0) for v in dist.values())
    if s <= 1e-9:
        return {k: 1.0 / len(dist) for k in dist}
    return {k: max(v, 0.0) / s for k, v in dist.items()}


def make_vpr_prior(obs: Observation) -> Optional[Dict[str, float]]:
    """
    Optional weak prior from VPR. If VPR is missing or unreliable, return None.
    """
    if obs.place_probs is not None:
        prior = dict(obs.place_probs)
        # confidence check: if too flat, treat as unreliable
        top_prob = max(prior.values()) if prior else 0.0
        if top_prob < 0.45:
            return None
        return normalize(prior)

    if obs.place_id in REGIONS:
        return {r: (1.0 if r == obs.place_id else 0.0) for r in REGIONS}

    return None


def belief_update(obs: Observation) -> Dict[str, float]:
    """
    Belief over a simplified hidden state: DangerLevel ∈ {LOW, MEDIUM, HIGH}.
    Event/state driven baseline (v1).
    """
    # Start from neutral
    belief = {"LOW": 1.0, "MEDIUM": 1.0, "HIGH": 1.0}

    # 1) Alive count disadvantage
    diff = obs.enemy_alive - obs.team_alive
    if diff >= 3:
        belief["HIGH"] += 2.0
        belief["LOW"] -= 0.5
    elif diff == 2:
        belief["HIGH"] += 1.2
    elif diff == 1:
        belief["HIGH"] += 0.6
    elif diff <= -2:
        # we have advantage
        belief["LOW"] += 1.2
        belief["HIGH"] -= 0.3

    # 2) Money / economy pressure
    if obs.money < 1000:
        belief["HIGH"] += 1.5
    elif obs.money < 2000:
        belief["HIGH"] += 0.8
    elif obs.money >= 4000:
        belief["LOW"] += 0.6

    # 3) Survivability
    if obs.hp < 40:
        belief["HIGH"] += 0.7
    if obs.armor < 40:
        belief["HIGH"] += 0.4

    # 4) Time pressure (simple)
    if obs.time_left <= 20 and diff >= 1:
        belief["HIGH"] += 0.6

    # 5) Optional VPR (weak signal) - doesn't dominate
    vpr = make_vpr_prior(obs)
    if vpr is not None:
        # If VPR says we're likely MID, treat as slightly uncertain
        if vpr.get("MID", 0.0) >= 0.6:
            belief["MEDIUM"] += 0.2

    # Ensure non-negative and normalize
    belief = {k: max(0.0, v) for k, v in belief.items()}
    return normalize(belief)


def choose_label(belief: Dict[str, float], obs: Observation) -> str:
    """
    Map belief + observation to a macro coaching label (v1 baseline).
    """
    danger = max(belief, key=belief.get)
    danger_prob = belief[danger]

    # Hard rule: extremely low money => SAVE
    if obs.money < 1000:
        return "SAVE"

    # If we are outnumbered and have a rifle but low money => SAVE (keep the rifle)
    if obs.has_rifle and obs.money < 2000 and obs.team_alive <= 1 and obs.enemy_alive >= 3:
        return "SAVE"

    # High danger and not enough money => play safe / save
    if danger == "HIGH" and danger_prob >= 0.45:
        return "PLAY_SAFE" if obs.money >= 2000 else "SAVE"

    # Economy decisions (coarse)
    if obs.money >= 4000:
        return "BUY"
    if 2000 <= obs.money < 4000:
        return "FORCE_BUY"

    # Default
    return "PLAY_SAFE"


def run_once(obs_dict: Dict) -> Tuple[Dict[str, float], str]:
    obs = Observation(**obs_dict)
    b = belief_update(obs)
    label = choose_label(b, obs)
    return b, label


if __name__ == "__main__":
    # Case 1: Example mentioned by team discussion -> expect SAVE
    example_obs_1 = {
        "time_left": 60,
        "team_alive": 0,
        "enemy_alive": 5,
        "hp": 100,
        "armor": 100,
        "money": 200,
        "has_rifle": True,
    }
    belief, label = run_once(example_obs_1)
    print("[case1] belief =", belief)
    print("[case1] label  =", label)

    # Case 2: Good money -> expect BUY
    example_obs_2 = {
        "time_left": 70,
        "team_alive": 5,
        "enemy_alive": 5,
        "hp": 100,
        "armor": 100,
        "money": 5200,
        "has_rifle": False,
    }
    belief, label = run_once(example_obs_2)
    print("[case2] belief =", belief)
    print("[case2] label  =", label)

    # Case 3: Medium money -> expect FORCE_BUY
    example_obs_3 = {
        "time_left": 55,
        "team_alive": 3,
        "enemy_alive": 4,
        "hp": 80,
        "armor": 60,
        "money": 3000,
        "has_rifle": False,
    }
    belief, label = run_once(example_obs_3)
    print("[case3] belief =", belief)
    print("[case3] label  =", label)
