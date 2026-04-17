"""
info_model.py

Single source of truth for CS2 economy rules and the Economy HMM that
predicts the enemy team's economy tier from per-round observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ===========================================================================
# CS2 Economy Constants
# ===========================================================================

MONEY_CAP = 16_000
STARTING_MONEY = 800

LOSS_BONUS = [1_400, 1_900, 2_400, 2_900, 3_400]  # indexed by streak-1
MAX_LOSS_STREAK = 5

WIN_REWARD_ELIM = 3_250
WIN_REWARD_BOMB = 3_500     # T win by bomb explosion
WIN_REWARD_DEFUSE = 3_500   # CT win by defuse

T_BOMB_PLANT_BONUS = 800    # all T players, even on loss

KILL_REWARDS = {
    "knife": 1500, "bayonet": 1500,
    "nova": 900, "mag7": 900, "sawedoff": 900,
    "mp9": 600, "mp7": 600, "mp5sd": 600, "ump45": 600,
    "mac10": 600, "bizon": 600,
    "xm1014": 300, "p90": 300,
    "glock": 300, "hkp2000": 300, "usp_silencer": 300, "elite": 300,
    "p250": 300, "fiveseven": 300, "tec9": 300, "deagle": 300,
    "cz75a": 300, "revolver": 300,
    "ak47": 300, "m4a1": 300, "m4a1_silencer": 300, "sg556": 300,
    "aug": 300, "galilar": 300, "famas": 300,
    "m249": 300, "negev": 300,
    "hegrenade": 300, "inferno": 300, "molotov": 300,
    "awp": 100, "ssg08": 300, "g3sg1": 300, "scar20": 300,
    "taser": 100,
}

CT_KILL_SHARE_BONUS = 50    # per CT per T kill (max $250)

AVG_KILL_INCOME_WIN = 600    # ~2 kills * $300 avg
AVG_KILL_INCOME_LOSS = 200   # ~0.6 kills * $300
T_PLANT_RATE_ON_LOSS = 0.35  # bomb planted in ~35% of T round losses
CT_KILL_BONUS_WIN = 200      # ~4 T kills * $50
CT_KILL_BONUS_LOSS = 75      # ~1.5 T kills * $50


# ---------------------------------------------------------------------------
# Economy helper functions (shared by HMM + MDP)
# ---------------------------------------------------------------------------

def loss_bonus(streak: int) -> int:
    """Loss bonus received after accumulating ``streak`` consecutive losses."""
    if streak <= 0:
        return 0
    idx = min(streak, MAX_LOSS_STREAK) - 1
    return LOSS_BONUS[idx]


def next_money_win(side: str, money_after_buy: int) -> int:
    """Money at the start of next round after winning."""
    income = WIN_REWARD_ELIM + AVG_KILL_INCOME_WIN
    if side == "CT":
        income += CT_KILL_BONUS_WIN
        income += (WIN_REWARD_DEFUSE - WIN_REWARD_ELIM) * 0.5
    else:
        income += (WIN_REWARD_BOMB - WIN_REWARD_ELIM) * 0.5
    return min(money_after_buy + int(income), MONEY_CAP)


def next_money_loss(side: str, money_after_buy: int, new_streak: int) -> int:
    """Money at the start of next round after losing."""
    income = loss_bonus(new_streak) + AVG_KILL_INCOME_LOSS
    if side == "T":
        income += int(T_BOMB_PLANT_BONUS * T_PLANT_RATE_ON_LOSS)
    else:
        income += CT_KILL_BONUS_LOSS
    return min(money_after_buy + int(income), MONEY_CAP)


def expected_income_after_loss(streak: int, side: str) -> int:
    """Approximate total income a team receives after a loss with given streak."""
    return loss_bonus(streak) + AVG_KILL_INCOME_LOSS + (
        int(T_BOMB_PLANT_BONUS * T_PLANT_RATE_ON_LOSS) if side == "T"
        else CT_KILL_BONUS_LOSS)


def money_to_tier(money: float) -> str:
    """Map a dollar amount to an economy tier string."""
    if money < 1500:
        return "BROKE"
    if money < 3000:
        return "LOW"
    if money < 5000:
        return "MEDIUM"
    if money < 8000:
        return "HIGH"
    return "RICH"


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

def _derive_transition_matrix() -> dict[str, dict[bool, dict[str, float]]]:
    """Build P(next_tier | current_tier, enemy_won_previous_round)."""
    T: dict[str, dict[bool, dict[str, float]]] = {}

    for tier in ECON_TIERS:
        T[tier] = {True: {}, False: {}}
        lo, hi = TIER_MONEY_RANGES[tier]
        mid = (lo + hi) / 2.0

        win_income = WIN_REWARD_ELIM + AVG_KILL_INCOME_WIN
        expected_after_win = min(mid + win_income, MONEY_CAP)
        raw_win = {}
        for t2 in ECON_TIERS:
            t2_lo, t2_hi = TIER_MONEY_RANGES[t2]
            t2_mid = (t2_lo + t2_hi) / 2.0
            dist = abs(expected_after_win - t2_mid)
            raw_win[t2] = math.exp(-dist / 3000.0)
        s = sum(raw_win.values())
        T[tier][True] = {t2: v / s for t2, v in raw_win.items()}

        raw_loss = {t2: 0.0 for t2 in ECON_TIERS}
        streak_weights = {1: 0.40, 2: 0.30, 3: 0.15, 4: 0.10, 5: 0.05}
        for streak, w in streak_weights.items():
            bonus = loss_bonus(streak)
            expected_after_loss = min(mid * 0.3 + bonus + AVG_KILL_INCOME_LOSS,
                                     MONEY_CAP)
            for t2 in ECON_TIERS:
                t2_lo, t2_hi = TIER_MONEY_RANGES[t2]
                t2_mid = (t2_lo + t2_hi) / 2.0
                dist = abs(expected_after_loss - t2_mid)
                raw_loss[t2] += w * math.exp(-dist / 2500.0)
        s = sum(raw_loss.values())
        T[tier][False] = {t2: v / s for t2, v in raw_loss.items()}

    return T


TRANSITION = _derive_transition_matrix()


# ---------------------------------------------------------------------------
# Emission model: P(observation features | economy tier)
# ---------------------------------------------------------------------------

_WEAPON_EMISSION = {
    "BROKE":  {"pistol": 0.75, "smg": 0.10, "rifle": 0.05, "awp": 0.01, "unknown": 0.09},
    "LOW":    {"pistol": 0.30, "smg": 0.35, "rifle": 0.20, "awp": 0.02, "unknown": 0.13},
    "MEDIUM": {"pistol": 0.10, "smg": 0.20, "rifle": 0.50, "awp": 0.08, "unknown": 0.12},
    "HIGH":   {"pistol": 0.05, "smg": 0.08, "rifle": 0.55, "awp": 0.22, "unknown": 0.10},
    "RICH":   {"pistol": 0.03, "smg": 0.05, "rifle": 0.45, "awp": 0.37, "unknown": 0.10},
}

_END_TYPE_EMISSION = {
    "BROKE":  {"elimination": 0.25, "bomb": 0.15, "time": 0.25, "close": 0.20, "dominant": 0.15},
    "LOW":    {"elimination": 0.20, "bomb": 0.20, "time": 0.15, "close": 0.25, "dominant": 0.20},
    "MEDIUM": {"elimination": 0.18, "bomb": 0.25, "time": 0.10, "close": 0.22, "dominant": 0.25},
    "HIGH":   {"elimination": 0.15, "bomb": 0.30, "time": 0.08, "close": 0.17, "dominant": 0.30},
    "RICH":   {"elimination": 0.12, "bomb": 0.35, "time": 0.05, "close": 0.13, "dominant": 0.35},
}

_SURVIVORS_EMISSION = {
    "BROKE":  {0: 0.50, 1: 0.25, 2: 0.15, 3: 0.10},
    "LOW":    {0: 0.35, 1: 0.30, 2: 0.20, 3: 0.15},
    "MEDIUM": {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
    "HIGH":   {0: 0.15, 1: 0.20, 2: 0.30, 3: 0.35},
    "RICH":   {0: 0.10, 1: 0.15, 2: 0.30, 3: 0.45},
}

_LOSS_STREAK_EMISSION = {
    "BROKE":  {0: 0.10, 1: 0.20, 2: 0.25, 3: 0.25, 4: 0.20},
    "LOW":    {0: 0.20, 1: 0.30, 2: 0.25, 3: 0.15, 4: 0.10},
    "MEDIUM": {0: 0.35, 1: 0.25, 2: 0.20, 3: 0.12, 4: 0.08},
    "HIGH":   {0: 0.50, 1: 0.25, 2: 0.12, 3: 0.08, 4: 0.05},
    "RICH":   {0: 0.60, 1: 0.20, 2: 0.10, 3: 0.06, 4: 0.04},
}

_WIN_STREAK_EMISSION = {
    "BROKE":  {0: 0.60, 1: 0.20, 2: 0.10, 3: 0.06, 4: 0.04},
    "LOW":    {0: 0.50, 1: 0.25, 2: 0.12, 3: 0.08, 4: 0.05},
    "MEDIUM": {0: 0.35, 1: 0.25, 2: 0.20, 3: 0.12, 4: 0.08},
    "HIGH":   {0: 0.20, 1: 0.25, 2: 0.25, 3: 0.18, 4: 0.12},
    "RICH":   {0: 0.10, 1: 0.20, 2: 0.25, 3: 0.25, 4: 0.20},
}


def _emission_prob(obs: EconObservation, tier: str) -> float:
    """P(observation | economy tier)."""
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

    # Loss streak signal
    loss_bucket = min(obs.enemy_loss_streak, 4)
    loss_probs = _LOSS_STREAK_EMISSION.get(tier, {})
    p *= loss_probs.get(loss_bucket, 0.20)

    # Win streak signal
    win_bucket = min(obs.enemy_win_streak, 4)
    win_probs = _WIN_STREAK_EMISSION.get(tier, {})
    p *= win_probs.get(win_bucket, 0.20)

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
    """Build economy observations from match rounds (one per round after the first)."""
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
    """Run the Economy HMM across a match and return one prediction per round.

    Each entry: {round_num, predicted_tier, tier_probs, predicted_avg_money}.
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
