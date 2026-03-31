"""
info_model.py

Economy HMM — Hidden Markov Model for predicting enemy team economy.

Hidden state:  enemy team economy tier
    BROKE ($0-1500), LOW ($1500-3000), MEDIUM ($3000-5000),
    HIGH ($5000-8000), RICH ($8000+)

Observations (per round, based on PREVIOUS round's outcome):
    1. Round result (win/loss)
    2. How the round ended (elimination, bomb, time, close, dominant)
    3. Weapons seen in killfeed (pistol-only, SMG, rifle, AWP)
    4. Number of enemy survivors
    5. Win/loss streak

References:
    - Zeng et al. (2020) "Learning to Reason in Round-based Games"
    - Xenopoulos et al. (2021) "Optimal Team Economic Decisions in CS"
    - StarCraft opponent modeling with Bayesian/HMM approaches
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Economy tiers
# ---------------------------------------------------------------------------

ECON_TIERS = ["BROKE", "LOW", "MEDIUM", "HIGH", "RICH"]
TIER_IDX = {t: i for i, t in enumerate(ECON_TIERS)}
N_TIERS = len(ECON_TIERS)

TIER_MONEY_RANGES = {
    "BROKE":  (0, 1500),
    "LOW":    (1500, 3000),
    "MEDIUM": (3000, 5000),
    "HIGH":   (5000, 8000),
    "RICH":   (8000, 16000),
}

TIER_AVG_MONEY = {
    "BROKE":  750,
    "LOW":    2250,
    "MEDIUM": 4000,
    "HIGH":   6500,
    "RICH":   10000,
}


# ---------------------------------------------------------------------------
# Round observation — extracted from previous round
# ---------------------------------------------------------------------------

@dataclass
class EconObservation:
    """Observation from the previous round used to predict current enemy economy."""
    enemy_won_prev: bool
    round_end_type: str       # "elimination", "bomb", "time", "close", "dominant"
    best_weapon_seen: str     # "pistol", "smg", "rifle", "awp", "unknown"
    enemy_survivors: int      # 0-5
    enemy_loss_streak: int    # 0-5+
    enemy_win_streak: int     # 0-5+


# ---------------------------------------------------------------------------
# Transition model: P(tier_t | tier_{t-1}, enemy_won_prev)
# ---------------------------------------------------------------------------

# CS2 loss bonuses: 1st loss $1400, 2nd $1900, 3rd $2400, 4th $2900, 5th+ $3400
# Win reward: $3250 base (T) or $3250 (CT), plus kill rewards ~$300-600/kill

def _build_transition_matrix() -> dict[str, dict[bool, dict[str, float]]]:
    """Build P(next_tier | current_tier, enemy_won_previous_round)."""
    T = {}

    # After enemy WIN — they get $3250+ and keep equipment
    T["BROKE"] = {
        True:  {"BROKE": 0.02, "LOW": 0.15, "MEDIUM": 0.50, "HIGH": 0.28, "RICH": 0.05},
        False: {"BROKE": 0.10, "LOW": 0.55, "MEDIUM": 0.25, "HIGH": 0.08, "RICH": 0.02},
    }
    T["LOW"] = {
        True:  {"BROKE": 0.01, "LOW": 0.05, "MEDIUM": 0.30, "HIGH": 0.45, "RICH": 0.19},
        False: {"BROKE": 0.15, "LOW": 0.40, "MEDIUM": 0.30, "HIGH": 0.12, "RICH": 0.03},
    }
    T["MEDIUM"] = {
        True:  {"BROKE": 0.01, "LOW": 0.03, "MEDIUM": 0.15, "HIGH": 0.45, "RICH": 0.36},
        False: {"BROKE": 0.20, "LOW": 0.35, "MEDIUM": 0.30, "HIGH": 0.12, "RICH": 0.03},
    }
    T["HIGH"] = {
        True:  {"BROKE": 0.01, "LOW": 0.02, "MEDIUM": 0.07, "HIGH": 0.35, "RICH": 0.55},
        False: {"BROKE": 0.25, "LOW": 0.30, "MEDIUM": 0.25, "HIGH": 0.15, "RICH": 0.05},
    }
    T["RICH"] = {
        True:  {"BROKE": 0.01, "LOW": 0.02, "MEDIUM": 0.05, "HIGH": 0.22, "RICH": 0.70},
        False: {"BROKE": 0.30, "LOW": 0.30, "MEDIUM": 0.20, "HIGH": 0.12, "RICH": 0.08},
    }

    return T


TRANSITION = _build_transition_matrix()


# ---------------------------------------------------------------------------
# Emission model: P(observation features | economy tier)
# ---------------------------------------------------------------------------

# P(best_weapon_seen | tier) — what weapons enemies use reveals their money
_WEAPON_EMISSION = {
    #                  pistol   smg    rifle   awp   unknown
    "BROKE":  {"pistol": 0.75, "smg": 0.10, "rifle": 0.05, "awp": 0.01, "unknown": 0.09},
    "LOW":    {"pistol": 0.30, "smg": 0.35, "rifle": 0.20, "awp": 0.02, "unknown": 0.13},
    "MEDIUM": {"pistol": 0.10, "smg": 0.20, "rifle": 0.50, "awp": 0.08, "unknown": 0.12},
    "HIGH":   {"pistol": 0.05, "smg": 0.08, "rifle": 0.55, "awp": 0.22, "unknown": 0.10},
    "RICH":   {"pistol": 0.03, "smg": 0.05, "rifle": 0.45, "awp": 0.37, "unknown": 0.10},
}

# P(round_end_type | tier) — how round ends is influenced by enemy economy
_END_TYPE_EMISSION = {
    #                  elimination  bomb    time   close  dominant
    "BROKE":  {"elimination": 0.25, "bomb": 0.15, "time": 0.25, "close": 0.20, "dominant": 0.15},
    "LOW":    {"elimination": 0.20, "bomb": 0.20, "time": 0.15, "close": 0.25, "dominant": 0.20},
    "MEDIUM": {"elimination": 0.18, "bomb": 0.25, "time": 0.10, "close": 0.22, "dominant": 0.25},
    "HIGH":   {"elimination": 0.15, "bomb": 0.30, "time": 0.08, "close": 0.17, "dominant": 0.30},
    "RICH":   {"elimination": 0.12, "bomb": 0.35, "time": 0.05, "close": 0.13, "dominant": 0.35},
}

# P(survivors_bucket | tier)
_SURVIVORS_EMISSION = {
    #                  0      1      2      3+
    "BROKE":  {0: 0.50, 1: 0.25, 2: 0.15, 3: 0.10},
    "LOW":    {0: 0.35, 1: 0.30, 2: 0.20, 3: 0.15},
    "MEDIUM": {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
    "HIGH":   {0: 0.15, 1: 0.20, 2: 0.30, 3: 0.35},
    "RICH":   {0: 0.10, 1: 0.15, 2: 0.30, 3: 0.45},
}


def _emission_prob(obs: EconObservation, tier: str) -> float:
    """P(observation | economy tier).

    Combines weapon, end-type, and survivors sub-models multiplicatively.
    """
    p = 1.0

    # Weapon signal
    weapon_probs = _WEAPON_EMISSION.get(tier, {})
    p *= weapon_probs.get(obs.best_weapon_seen, 0.10)

    # Round end type signal
    end_probs = _END_TYPE_EMISSION.get(tier, {})
    p *= end_probs.get(obs.round_end_type, 0.20)

    # Survivors bucket
    surv_bucket = min(obs.enemy_survivors, 3)
    surv_probs = _SURVIVORS_EMISSION.get(tier, {})
    p *= surv_probs.get(surv_bucket, 0.25)

    return max(p, 1e-12)


# ---------------------------------------------------------------------------
# Economy HMM inference
# ---------------------------------------------------------------------------

def _normalize(belief: dict[str, float]) -> dict[str, float]:
    s = sum(max(v, 0.0) for v in belief.values())
    if s < 1e-12:
        return {t: 1.0 / N_TIERS for t in ECON_TIERS}
    return {t: max(v, 0.0) / s for t, v in belief.items()}


class EconomyHMM:
    """HMM for predicting enemy team economy tier."""

    def __init__(self):
        self.belief: dict[str, float] = {t: 1.0 / N_TIERS for t in ECON_TIERS}
        self.history: list[dict] = []

    def reset(self):
        """Reset to uniform prior (start of match)."""
        self.belief = {t: 1.0 / N_TIERS for t in ECON_TIERS}
        self.history = []

    def predict_step(self, enemy_won_prev: bool) -> dict[str, float]:
        """Apply transition model: P(tier_t | tier_{t-1}, won_prev)."""
        new_belief = {t: 0.0 for t in ECON_TIERS}
        for t_from in ECON_TIERS:
            trans = TRANSITION[t_from][enemy_won_prev]
            for t_to in ECON_TIERS:
                new_belief[t_to] += self.belief[t_from] * trans[t_to]
        return _normalize(new_belief)

    def update_step(self, belief: dict[str, float],
                    obs: EconObservation) -> dict[str, float]:
        """Incorporate observation via emission model."""
        for t in ECON_TIERS:
            belief[t] *= _emission_prob(obs, t)
        return _normalize(belief)

    def observe(self, obs: EconObservation) -> dict:
        """Full predict-update cycle. Returns prediction result."""
        predicted = self.predict_step(obs.enemy_won_prev)
        self.belief = self.update_step(predicted, obs)

        best_tier = max(self.belief, key=self.belief.get)
        avg_money = sum(self.belief[t] * TIER_AVG_MONEY[t] for t in ECON_TIERS)

        result = {
            "predicted_tier": best_tier,
            "tier_probs": dict(self.belief),
            "predicted_avg_money": int(avg_money),
        }
        self.history.append(result)
        return result

    def predict_current(self) -> dict:
        """Return current belief without updating."""
        best_tier = max(self.belief, key=self.belief.get)
        avg_money = sum(self.belief[t] * TIER_AVG_MONEY[t] for t in ECON_TIERS)
        return {
            "predicted_tier": best_tier,
            "tier_probs": dict(self.belief),
            "predicted_avg_money": int(avg_money),
        }


# ---------------------------------------------------------------------------
# Helper: classify round end type from match data
# ---------------------------------------------------------------------------

def classify_round_end(rd, enemy_side: str) -> str:
    """Determine how a round ended from the perspective of the enemy team.

    Returns one of: "elimination", "bomb", "time", "close", "dominant"
    """
    has_bomb_explode = False
    has_bomb_defuse = False
    for ev in rd.events:
        if ev.event_type == "bomb_explode":
            has_bomb_explode = True
        elif ev.event_type == "bomb_defuse":
            has_bomb_defuse = True

    if has_bomb_explode or has_bomb_defuse:
        return "bomb"

    # Check for elimination
    if enemy_side == "T":
        enemy_players = rd.t_players
        our_players = rd.ct_players
    else:
        enemy_players = rd.ct_players
        our_players = rd.t_players

    enemy_alive = sum(1 for p in enemy_players if p.alive_at_end)
    our_alive = sum(1 for p in our_players if p.alive_at_end)

    if enemy_alive == 0:
        return "elimination"

    if our_alive == 0:
        return "elimination"

    total_alive = enemy_alive + our_alive
    if total_alive <= 3:
        return "close"
    if our_alive >= 4 or enemy_alive >= 4:
        return "dominant"

    return "time"


def classify_best_weapon(rd, enemy_side: str) -> str:
    """Determine the best weapon category seen from the enemy team in killfeed."""
    weapon_tiers = {
        "awp": "awp", "ssg08": "rifle",
        "ak47": "rifle", "m4a1": "rifle", "m4a1_silencer": "rifle",
        "sg556": "rifle", "aug": "rifle", "galil": "rifle", "famas": "rifle",
        "mp9": "smg", "mac10": "smg", "mp7": "smg", "ump45": "smg",
        "p90": "smg", "bizon": "smg", "mp5sd": "smg",
        "glock": "pistol", "usp_silencer": "pistol", "hkp2000": "pistol",
        "p250": "pistol", "tec9": "pistol", "fiveseven": "pistol",
        "deagle": "pistol", "elite": "pistol", "cz75_auto": "pistol",
        "revolver": "pistol",
        "nova": "smg", "xm1014": "smg", "sawedoff": "smg", "mag7": "smg",
        "negev": "rifle", "m249": "rifle",
    }

    best = "unknown"
    rank = {"unknown": 0, "pistol": 1, "smg": 2, "rifle": 3, "awp": 4}

    enemy_names = set()
    if enemy_side == "T":
        enemy_names = {p.name for p in rd.t_players}
    else:
        enemy_names = {p.name for p in rd.ct_players}

    for ev in rd.events:
        if ev.event_type == "kill":
            attacker = ev.data.get("attacker", "")
            if attacker in enemy_names:
                weapon = ev.data.get("weapon", "").lower()
                tier = weapon_tiers.get(weapon, "unknown")
                if rank.get(tier, 0) > rank.get(best, 0):
                    best = tier

    return best


def count_enemy_survivors(rd, enemy_side: str) -> int:
    """Count how many enemies survived the round."""
    if enemy_side == "T":
        return sum(1 for p in rd.t_players if p.alive_at_end)
    else:
        return sum(1 for p in rd.ct_players if p.alive_at_end)


# ---------------------------------------------------------------------------
# Build observations from match data
# ---------------------------------------------------------------------------

def build_economy_observations(
    rounds: list,
    target_player: str,
) -> list[EconObservation]:
    """Build economy observations from match rounds.

    For round i, the observation is based on what happened in round i-1.
    The first round has no previous data, so we skip it.
    """
    observations = []

    enemy_loss_streak = 0
    enemy_win_streak = 0

    for i in range(1, len(rounds)):
        prev_rd = rounds[i - 1]
        p = prev_rd.get_player(target_player)
        if p is None:
            observations.append(None)
            continue

        our_side = p.side
        enemy_side = "T" if our_side == "CT" else "CT"

        enemy_won = prev_rd.winner == enemy_side

        if enemy_won:
            enemy_win_streak += 1
            enemy_loss_streak = 0
        else:
            enemy_loss_streak += 1
            enemy_win_streak = 0

        end_type = classify_round_end(prev_rd, enemy_side)
        best_weapon = classify_best_weapon(prev_rd, enemy_side)
        survivors = count_enemy_survivors(prev_rd, enemy_side)

        observations.append(EconObservation(
            enemy_won_prev=enemy_won,
            round_end_type=end_type,
            best_weapon_seen=best_weapon,
            enemy_survivors=survivors,
            enemy_loss_streak=enemy_loss_streak,
            enemy_win_streak=enemy_win_streak,
        ))

    return observations


def predict_enemy_economy(
    rounds: list,
    target_player: str,
) -> list[dict]:
    """Run the Economy HMM across a match, returning per-round predictions.

    Returns a list with one entry per round (first round gets uniform prior).
    Each entry: {round_num, predicted_tier, tier_probs, predicted_avg_money}
    """
    hmm = EconomyHMM()
    results = []

    observations = build_economy_observations(rounds, target_player)

    # Round 0: no previous data — use uniform prior
    if rounds:
        results.append({
            "round_num": rounds[0].round_num,
            **hmm.predict_current(),
        })

    for i, obs in enumerate(observations):
        rd = rounds[i + 1] if (i + 1) < len(rounds) else None
        if obs is None:
            results.append({
                "round_num": rd.round_num if rd else i + 1,
                **hmm.predict_current(),
            })
        else:
            pred = hmm.observe(obs)
            results.append({
                "round_num": rd.round_num if rd else i + 1,
                **pred,
            })

    return results
