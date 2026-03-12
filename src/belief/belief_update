from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


REGIONS = ["A", "B", "MID"]
LABELS = ["ROTATE_A", "ROTATE_B", "HOLD", "REPOSITION", "PLAY_SAFE"]


@dataclass
class Observation:
    time_left: int
    teammate_death_site: str  # "A", "B", "NONE"
    place_probs: Dict[str, float]  # e.g. {"A":0.2,"B":0.7,"MID":0.1}
    smoke_A: bool = False
    smoke_B: bool = False
    saw_enemy_A: bool = False
    saw_enemy_B: bool = False


def normalize(dist: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(v, 0.0) for v in dist.values())
    if s <= 1e-9:
        # fallback to uniform
        return {k: 1.0 / len(dist) for k in dist}
    return {k: max(v, 0.0) / s for k, v in dist.items()}


def belief_update(obs: Observation) -> Dict[str, float]:
    """
    Rule-based belief update (v0).
    Uses VPR place_probs as a prior, then adjusts using teammate death + simple cues.
    """
    belief = dict(obs.place_probs)

    # Strong evidence: saw enemy
    if obs.saw_enemy_A:
        belief["A"] += 1.5
    if obs.saw_enemy_B:
        belief["B"] += 1.5

    # Evidence: teammate death site
    if obs.teammate_death_site == "A":
        belief["A"] += 1.0
    elif obs.teammate_death_site == "B":
        belief["B"] += 1.0

    # Utility cue (very simple): smoke at a site reduces certainty of direct info
    # Here we just slightly smooth belief when smoke exists.
    if obs.smoke_A or obs.smoke_B:
        belief["MID"] += 0.1

    return normalize(belief)


def choose_label(belief: Dict[str, float], obs: Observation) -> str:
    """
    Convert belief + context into a recommendation label (v0).
    """
    top_region = max(belief, key=belief.get)
    top_prob = belief[top_region]

    # If time is low and belief is confident, recommend rotate to the likely site
    if obs.time_left <= 25 and top_prob >= 0.55:
        if top_region == "A":
            return "ROTATE_A"
        if top_region == "B":
            return "ROTATE_B"

    # If belief not confident, suggest holding or playing safe depending on time
    if top_prob < 0.55:
        return "HOLD" if obs.time_left > 25 else "PLAY_SAFE"

    # Otherwise, default conservative
    return "REPOSITION"


def run_once(obs_dict: Dict) -> Tuple[Dict[str, float], str]:
    obs = Observation(**obs_dict)
    b = belief_update(obs)
    label = choose_label(b, obs)
    return b, label


if __name__ == "__main__":
    # Toy demo (you can run this locally later)
    example_obs = {
        "time_left": 22,
        "teammate_death_site": "B",
        "place_probs": {"A": 0.20, "B": 0.70, "MID": 0.10},
        "smoke_A": False,
        "smoke_B": True,
        "saw_enemy_A": False,
        "saw_enemy_B": False,
    }
    belief, label = run_once(example_obs)
    print("belief =", belief)
    print("label  =", label)
