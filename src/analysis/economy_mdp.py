"""
economy_mdp.py

Model CS2 round economy as an MDP and solve the optimal buy policy via
Value Iteration, then evaluate a target player's buy decisions against
that policy. Economy constants come from info_model.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from info_model import (
    MONEY_CAP, STARTING_MONEY, LOSS_BONUS, MAX_LOSS_STREAK,
    WIN_REWARD_ELIM, WIN_REWARD_BOMB, WIN_REWARD_DEFUSE,
    T_BOMB_PLANT_BONUS, KILL_REWARDS, CT_KILL_SHARE_BONUS,
    AVG_KILL_INCOME_WIN, AVG_KILL_INCOME_LOSS,
    T_PLANT_RATE_ON_LOSS, CT_KILL_BONUS_WIN, CT_KILL_BONUS_LOSS,
    loss_bonus, next_money_win, next_money_loss,
)

MONEY_STEP = 500
N_MONEY_BINS = MONEY_CAP // MONEY_STEP + 1
N_STREAKS = MAX_LOSS_STREAK + 1

# Actions
SAVE = 0
FORCE = 1
FULL_BUY = 2
ACTIONS = [SAVE, FORCE, FULL_BUY]
ACTION_NAMES = {SAVE: "SAVE", FORCE: "FORCE", FULL_BUY: "FULL_BUY"}

EQUIP_COST = {
    "T":  {SAVE: 200,  FORCE: 2_600, FULL_BUY: 4_700},
    "CT": {SAVE: 200,  FORCE: 2_800, FULL_BUY: 5_500},
}

ACTION_TIER = {SAVE: 0, FORCE: 1, FULL_BUY: 2}

# Win probability matrix: P(win | my_tier, opp_tier)
WIN_PROB = {
    "T": np.array([
        [0.45, 0.28, 0.12],
        [0.65, 0.45, 0.30],
        [0.82, 0.65, 0.48],
    ]),
    "CT": np.array([
        [0.55, 0.30, 0.15],
        [0.70, 0.52, 0.35],
        [0.88, 0.68, 0.52],
    ]),
}

# Reward structure
WIN_REWARD_MDP = 1.0
LOSS_PENALTY_MDP = -0.3

# Discount factor for infinite-horizon MDP
GAMMA = 0.85

# Value Iteration convergence threshold
VI_EPSILON = 1e-6
VI_MAX_ITER = 500


def _money_to_bin(money: int) -> int:
    return min(max(0, money // MONEY_STEP), N_MONEY_BINS - 1)


def _bin_to_money(b: int) -> int:
    return b * MONEY_STEP


def _opponent_equip_dist(enemy_loss_streak: int) -> np.ndarray:
    """Estimate opponent equipment distribution from their observable loss streak."""
    distributions = {
        0: np.array([0.05, 0.10, 0.85]),
        1: np.array([0.65, 0.20, 0.15]),
        2: np.array([0.30, 0.45, 0.25]),
        3: np.array([0.10, 0.25, 0.65]),
        4: np.array([0.05, 0.20, 0.75]),
        5: np.array([0.05, 0.15, 0.80]),
    }
    k = min(enemy_loss_streak, MAX_LOSS_STREAK)
    return distributions[k]


def _expected_win_prob(side: str, my_tier: int, enemy_loss_streak: int) -> float:
    """Expected win probability given my equipment tier and enemy loss streak."""
    opp_dist = _opponent_equip_dist(enemy_loss_streak)
    return float(WIN_PROB[side][my_tier] @ opp_dist)


@dataclass
class EconomyPolicy:
    """Solved MDP policy for one side (T or CT)."""
    side: str
    V: np.ndarray
    policy: np.ndarray

    def recommend(self, money: int, loss_streak: int,
                  enemy_loss_streak: int = 0) -> int:
        b = _money_to_bin(money)
        k = min(loss_streak, MAX_LOSS_STREAK)
        ek = min(enemy_loss_streak, MAX_LOSS_STREAK)
        return int(self.policy[b, k, ek])

    def value(self, money: int, loss_streak: int,
              enemy_loss_streak: int = 0) -> float:
        b = _money_to_bin(money)
        k = min(loss_streak, MAX_LOSS_STREAK)
        ek = min(enemy_loss_streak, MAX_LOSS_STREAK)
        return float(self.V[b, k, ek])


def solve_economy_mdp(side: str, gamma: float = GAMMA) -> EconomyPolicy:
    """Solve the economy MDP for one side using Value Iteration.

    Returns an EconomyPolicy with the optimal value function and policy.
    """
    V = np.zeros((N_MONEY_BINS, N_STREAKS, N_STREAKS))
    policy = np.zeros((N_MONEY_BINS, N_STREAKS, N_STREAKS), dtype=int)

    costs = EQUIP_COST[side]

    for iteration in range(VI_MAX_ITER):
        V_new = np.zeros_like(V)

        for b in range(N_MONEY_BINS):
            money = _bin_to_money(b)
            for k in range(N_STREAKS):
                for ek in range(N_STREAKS):
                    best_val = -1e9
                    best_act = SAVE

                    for a in ACTIONS:
                        cost = costs[a]
                        actual_cost = min(cost, money)

                        if a != SAVE and cost > money:
                            continue

                        money_after = money - actual_cost
                        tier = ACTION_TIER[a]
                        p_win = _expected_win_prob(side, tier, ek)

                        # Win branch
                        next_m_win = next_money_win(side, money_after)
                        b_win = _money_to_bin(next_m_win)
                        k_win = 0
                        ek_win = min(ek + 1, MAX_LOSS_STREAK)

                        # Loss branch
                        k_loss = min(k + 1, MAX_LOSS_STREAK)
                        ek_loss = 0
                        next_m_loss = next_money_loss(side, money_after, k_loss)
                        b_loss = _money_to_bin(next_m_loss)

                        q = (p_win * (WIN_REWARD_MDP +
                                      gamma * V[b_win, k_win, ek_win]) +
                             (1 - p_win) * (LOSS_PENALTY_MDP +
                                            gamma * V[b_loss, k_loss, ek_loss]))

                        if q > best_val:
                            best_val = q
                            best_act = a

                    V_new[b, k, ek] = best_val
                    policy[b, k, ek] = best_act

        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < VI_EPSILON:
            break

    return EconomyPolicy(side=side, V=V, policy=policy)


# Rifles / snipers that indicate a full buy
_FULL_BUY_PRIMARIES = {
    "AK-47", "M4A4", "M4A1-S", "SG 553", "AUG", "AWP",
    "SSG 08", "SCAR-20", "G3SG1",
}
# SMGs / shotguns that indicate a force buy
_FORCE_PRIMARIES = {
    "MP9", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "MAC-10",
    "XM1014", "Nova", "MAG-7", "Sawed-Off", "M249", "Negev",
    "Galil AR", "FAMAS",
}


def classify_buy_decision(
    primary_weapon: Optional[str],
    secondary_weapon: Optional[str],
    has_helmet: bool,
    armor: int,
    equipment_value: int,
    utilities: list[str],
) -> int:
    """Classify a player's round buy into SAVE / FORCE / FULL_BUY."""
    if primary_weapon in _FULL_BUY_PRIMARIES:
        return FULL_BUY

    if primary_weapon in _FORCE_PRIMARIES:
        return FORCE

    if equipment_value >= 3500:
        return FULL_BUY
    if equipment_value >= 1500:
        return FORCE

    if armor >= 50 and (secondary_weapon or utilities):
        return FORCE

    return SAVE


@dataclass
class BuyEvaluation:
    """Evaluation of a single round's buy decision."""
    round_num: int
    side: str
    money: int
    loss_streak: int
    enemy_loss_streak: int
    actual_action: int
    optimal_action: int
    actual_name: str
    optimal_name: str
    is_optimal: bool
    actual_win_prob: float
    optimal_win_prob: float
    win_prob_diff: float
    actual_value: float
    optimal_value: float
    enemy_buy_prediction: str = ""
    note: str = ""
    # Enhanced fields (Economy HMM + matchup evaluation)
    enemy_predicted_tier: str = ""
    enemy_tier_probs: dict = field(default_factory=dict)
    enemy_predicted_money: int = 0
    weapon_matchup_note: str = ""
    team_avg_money: int = 0
    team_can_fullbuy_next: bool = True
    player_can_fullbuy_next: bool = True
    is_drop_or_pickup: bool = False
    upgrade_path_note: str = ""
    posthoc: Optional[PostHocDetail] = None


@dataclass
class PostHocDetail:
    """Fine-grained evaluation of a round's loadout beyond macro SAVE/FORCE/FULL."""
    weapon_tier: str = "pistol"          # pistol / smg / rifle / awp
    weapon_appropriate: bool = True      # correct tier for predicted enemy economy?
    utility_level: int = 0               # 0 / 1-2 / 3+
    utility_sufficient: bool = True
    has_armor: bool = False
    has_helmet: bool = False
    armor_appropriate: bool = True
    has_kit: bool = False                # CT only
    kit_note: str = ""
    waste: int = 0                       # money that could have been spent usefully
    waste_note: str = ""


_WEAPON_TIER_MAP = {
    "AK-47": "rifle", "M4A4": "rifle", "M4A1-S": "rifle",
    "SG 553": "rifle", "AUG": "rifle", "AWP": "awp",
    "SSG 08": "rifle", "SCAR-20": "rifle", "G3SG1": "rifle",
    "Galil AR": "rifle", "FAMAS": "rifle",
    "MP9": "smg", "MP7": "smg", "MP5-SD": "smg", "UMP-45": "smg",
    "P90": "smg", "PP-Bizon": "smg", "MAC-10": "smg",
    "XM1014": "shotgun", "Nova": "shotgun", "MAG-7": "shotgun", "Sawed-Off": "shotgun",
    "M249": "lmg", "Negev": "lmg",
}

_WEAPON_TIER_RANK = {"pistol": 0, "smg": 1, "shotgun": 1, "rifle": 2, "awp": 3, "lmg": 2}

# Approximate costs for waste calculation
_UTIL_COST = 300   # average grenade cost
_ARMOR_COST = 650  # kevlar
_HELMET_COST = 1000  # kevlar+helmet
_KIT_COST = 400


def evaluate_posthoc(
    primary_weapon: Optional[str],
    has_helmet: bool,
    armor: int,
    utilities: list[str],
    has_kit: bool,
    money: int,
    side: str,
    enemy_predicted_tier: str,
    actual_action: int,
) -> PostHocDetail:
    """Evaluate weapon/armor/utility/kit independently of the MDP macro decision."""
    detail = PostHocDetail()

    # Weapon tier
    w_tier = _WEAPON_TIER_MAP.get(primary_weapon or "", "pistol")
    detail.weapon_tier = w_tier

    enemy_strong = enemy_predicted_tier in ("HIGH", "RICH")
    enemy_weak = enemy_predicted_tier in ("BROKE", "LOW")
    if enemy_strong and w_tier in ("pistol", "smg"):
        detail.weapon_appropriate = False
    elif enemy_weak and w_tier == "awp":
        detail.weapon_appropriate = False

    # Utility
    n_util = len(utilities) if utilities else 0
    detail.utility_level = n_util
    if actual_action == FULL_BUY and n_util == 0:
        detail.utility_sufficient = False

    # Armor
    detail.has_armor = armor > 0
    detail.has_helmet = has_helmet
    if actual_action >= FORCE and armor == 0:
        detail.armor_appropriate = False

    # Kit (CT only)
    detail.has_kit = has_kit
    if side == "CT" and actual_action == FULL_BUY and not has_kit:
        detail.kit_note = "Missing defuse kit on full buy"

    # Waste: leftover money that could have been spent
    spent_estimate = 0
    if w_tier == "awp":
        spent_estimate += 4750
    elif w_tier == "rifle":
        spent_estimate += 2800
    elif w_tier in ("smg", "shotgun"):
        spent_estimate += 1500
    elif w_tier == "pistol":
        spent_estimate += 200

    if has_helmet:
        spent_estimate += _HELMET_COST
    elif armor > 0:
        spent_estimate += _ARMOR_COST

    spent_estimate += n_util * _UTIL_COST
    if has_kit:
        spent_estimate += _KIT_COST

    leftover = max(0, money - spent_estimate)
    if actual_action == FULL_BUY and leftover > 1000:
        detail.waste = leftover
        detail.waste_note = f"${leftover} unspent on full buy — buy more utility or upgrade"
    elif actual_action == SAVE and money > 5000:
        detail.waste = 0
        detail.waste_note = "Saving with high bank — consider force to stay competitive"

    return detail


def _is_knife_or_pistol(round_num: int) -> bool:
    """Rounds 0-1 are knife + pistol in competitive/Faceit."""
    return round_num <= 1


def _predict_enemy_buy(enemy_loss_streak: int,
                       hmm_pred: dict | None = None) -> str:
    """Human-readable prediction of enemy buy tier for coaching notes."""
    if hmm_pred:
        tier = hmm_pred.get("predicted_tier", "")
        money = hmm_pred.get("predicted_avg_money", 0)
        probs = hmm_pred.get("tier_probs", {})
        top_p = probs.get(tier, 0)
        tier_label = {
            "BROKE": "ECO round",
            "LOW": "FORCE buy or upgraded pistols",
            "MEDIUM": "FORCE buy or light rifles",
            "HIGH": "FULL BUY with rifles",
            "RICH": "FULL BUY with AWP possible",
        }.get(tier, "unknown")
        return f"Enemy ~${money} ({tier_label}, {top_p:.0%} confidence)"

    if enemy_loss_streak == 0:
        return "Enemy won last round — expect FULL BUY"
    elif enemy_loss_streak == 1:
        return "Enemy lost 1 — likely ECO round"
    elif enemy_loss_streak == 2:
        return "Enemy lost 2 — possible FORCE buy"
    elif enemy_loss_streak >= 3:
        return f"Enemy lost {enemy_loss_streak} — loss bonus allows FULL BUY"
    return ""


_WEAPON_TIER = {
    "AK-47": "rifle", "M4A4": "rifle", "M4A1-S": "rifle",
    "SG 553": "rifle", "AUG": "rifle", "AWP": "awp",
    "SSG 08": "rifle", "SCAR-20": "rifle", "G3SG1": "rifle",
    "Galil AR": "rifle", "FAMAS": "rifle",
    "MP9": "smg", "MP7": "smg", "MP5-SD": "smg", "UMP-45": "smg",
    "P90": "smg", "PP-Bizon": "smg", "MAC-10": "smg",
    "XM1014": "shotgun", "Nova": "shotgun", "MAG-7": "shotgun", "Sawed-Off": "shotgun",
    "M249": "lmg", "Negev": "lmg",
}

# Full buy cost threshold per side
_FULL_BUY_COST = {"T": 4700, "CT": 5500}


def _weapon_matchup_note(primary_weapon: str | None,
                         enemy_tier: str,
                         side: str) -> str:
    """Evaluate whether the player's weapon matches the predicted enemy economy."""
    if not primary_weapon:
        return ""
    w_tier = _WEAPON_TIER.get(primary_weapon, "pistol")

    if enemy_tier in ("BROKE", "LOW"):
        if w_tier == "awp":
            return f"AWP vs predicted eco — overkill, could save ${2750 if side == 'T' else 4750} for team"
        if w_tier == "rifle":
            return "Rifle vs eco — solid, but SMG would earn more kill reward"
        if w_tier == "smg":
            return "SMG vs eco — good economy choice, extra kill reward"
        return ""

    if enemy_tier in ("HIGH", "RICH"):
        if w_tier == "smg":
            return "SMG vs predicted full buy — outgunned at range"
        if w_tier == "pistol":
            return "Pistol vs predicted full buy — major disadvantage"
        if w_tier == "awp":
            return "AWP vs full buy — strong pick potential"
        return ""

    return ""


def _team_economy_analysis(
    rd, target_player: str, side: str,
) -> tuple[int, bool, bool]:
    """Analyze team economy.

    Returns (team_avg_money, team_can_fullbuy_next, player_can_fullbuy_next).
    """
    teammates = rd.t_players if side == "T" else rd.ct_players
    moneys = [p.start_money for p in teammates if p.start_money is not None]
    if not moneys:
        return (0, True, True)

    team_avg = int(sum(moneys) / len(moneys))
    fullbuy_cost = _FULL_BUY_COST[side]

    p = rd.get_player(target_player)
    player_money = p.start_money if p else 0

    expected_income = 1900
    team_can = (team_avg + expected_income) >= fullbuy_cost
    player_can = (player_money + expected_income) >= fullbuy_cost

    return (team_avg, team_can, player_can)


def _detect_drop_or_pickup(primary_weapon: str | None,
                           money: int, side: str) -> bool:
    """Detect if the player likely received a dropped weapon or picked one up."""
    if primary_weapon is None:
        return False

    weapon_costs = {
        "AK-47": 2700, "M4A4": 3100, "M4A1-S": 2900,
        "SG 553": 3000, "AUG": 3300, "AWP": 4750,
        "SSG 08": 1700, "SCAR-20": 5000, "G3SG1": 5000,
        "Galil AR": 1800, "FAMAS": 2050,
    }
    cost = weapon_costs.get(primary_weapon, 0)
    if cost == 0:
        return False

    min_total_buy = cost + 650
    return money < min_total_buy


def evaluate_player_economy(
    rounds: list,  # list[RoundData]
    target_player: str,
    hmm_predictions: list[dict] | None = None,
) -> list[BuyEvaluation]:
    """Evaluate the target player's buy decisions across all rounds.

    Parameters
    ----------
    rounds : list[RoundData]
        Parsed round data from demo_parser.
    target_player : str
        In-game name to analyze.
    hmm_predictions : list[dict] or None
        Per-round Economy HMM predictions from info_model.predict_enemy_economy.

    Returns
    -------
    list[BuyEvaluation]
        Per-round evaluation against optimal policy.
    """
    t_policy = solve_economy_mdp("T")
    ct_policy = solve_economy_mdp("CT")

    hmm_by_round: dict[int, dict] = {}
    if hmm_predictions:
        for pred in hmm_predictions:
            hmm_by_round[pred["round_num"]] = pred

    evaluations: list[BuyEvaluation] = []
    loss_streak = 0
    enemy_loss_streak = 0
    prev_alive = False
    prev_side: Optional[str] = None
    is_pistol = False

    for rd in rounds:
        p = rd.get_player(target_player)
        if p is None:
            prev_alive = False
            continue

        side = p.side
        if side not in ("T", "CT"):
            prev_alive = False
            continue

        # Detect half-time side switch → reset both streaks
        if prev_side is not None and prev_side != side:
            loss_streak = 0
            enemy_loss_streak = 0
            prev_alive = False
            is_pistol = True

        prev_side = side
        money = p.start_money

        skip = _is_knife_or_pistol(rd.round_num) or is_pistol or money < 1000
        if skip:
            is_pistol = False
            won = (rd.winner == side)
            if won:
                loss_streak = 0
                enemy_loss_streak = min(enemy_loss_streak + 1, MAX_LOSS_STREAK)
            else:
                loss_streak = min(loss_streak + 1, MAX_LOSS_STREAK)
                enemy_loss_streak = 0
            prev_alive = p.alive_at_end
            continue

        policy = t_policy if side == "T" else ct_policy

        actual_equip = classify_buy_decision(
            p.primary_weapon, p.secondary_weapon,
            p.has_helmet, p.armor,
            p.equipment_value, p.utilities,
        )

        best_affordable = SAVE
        for a in [FULL_BUY, FORCE]:
            if EQUIP_COST[side][a] <= money:
                best_affordable = a
                break

        optimal = policy.recommend(money, loss_streak, enemy_loss_streak)

        note = ""

        if prev_alive and actual_equip > best_affordable:
            actual = actual_equip
            is_optimal = True
            note = "(carried equipment)"
        elif EQUIP_COST[side][actual_equip] > money and actual_equip > best_affordable:
            actual = actual_equip
            is_optimal = (actual_equip >= optimal)
            if is_optimal:
                note = "(carried/picked up)"
        else:
            actual = actual_equip
            is_optimal = (actual == optimal)

        actual_tier = ACTION_TIER[actual]
        optimal_tier = ACTION_TIER[optimal]
        actual_wp = _expected_win_prob(side, actual_tier, enemy_loss_streak)
        optimal_wp = _expected_win_prob(side, optimal_tier, enemy_loss_streak)

        cost_actual = min(EQUIP_COST[side][actual], money)
        money_after_actual = money - cost_actual
        ek_win = min(enemy_loss_streak + 1, MAX_LOSS_STREAK)
        actual_val = (actual_wp * (WIN_REWARD_MDP + GAMMA * policy.value(
            next_money_win(side, money_after_actual), 0, ek_win))
            + (1 - actual_wp) * (LOSS_PENALTY_MDP + GAMMA * policy.value(
            next_money_loss(side, money_after_actual,
                            min(loss_streak + 1, MAX_LOSS_STREAK)),
            min(loss_streak + 1, MAX_LOSS_STREAK), 0)))
        optimal_val = policy.value(money, loss_streak, enemy_loss_streak)

        if not is_optimal and not note:
            if actual < optimal and EQUIP_COST[side][optimal] <= money:
                if optimal == FULL_BUY:
                    note = "Under-buying: had enough for full buy"
                else:
                    note = "Too passive: a force buy was viable"
            elif actual > optimal:
                if optimal == SAVE:
                    note = "Over-buying: should save for better buy next round"
                else:
                    note = "Full buy unnecessary, force would suffice"
            elif actual < optimal:
                note = "Could not afford optimal buy"

        # Enhanced economy evaluation
        hmm_pred = hmm_by_round.get(rd.round_num)
        enemy_buy_pred = _predict_enemy_buy(enemy_loss_streak, hmm_pred)
        enemy_tier = hmm_pred.get("predicted_tier", "") if hmm_pred else ""
        enemy_probs = hmm_pred.get("tier_probs", {}) if hmm_pred else {}
        enemy_money = hmm_pred.get("predicted_avg_money", 0) if hmm_pred else 0

        matchup_note = _weapon_matchup_note(p.primary_weapon, enemy_tier, side)
        team_avg, team_fb, player_fb = _team_economy_analysis(
            rd, target_player, side)
        is_drop = _detect_drop_or_pickup(p.primary_weapon, money, side)

        posthoc_detail = evaluate_posthoc(
            primary_weapon=p.primary_weapon,
            has_helmet=p.has_helmet,
            armor=p.armor,
            utilities=p.utilities,
            has_kit=getattr(p, "has_kit", False),
            money=money,
            side=side,
            enemy_predicted_tier=enemy_tier,
            actual_action=actual,
        )

        evaluations.append(BuyEvaluation(
            round_num=rd.round_num,
            side=side,
            money=money,
            loss_streak=loss_streak,
            enemy_loss_streak=enemy_loss_streak,
            actual_action=actual,
            optimal_action=optimal,
            actual_name=ACTION_NAMES[actual],
            optimal_name=ACTION_NAMES[optimal],
            is_optimal=is_optimal,
            actual_win_prob=actual_wp,
            optimal_win_prob=optimal_wp,
            win_prob_diff=optimal_wp - actual_wp,
            actual_value=actual_val,
            optimal_value=optimal_val,
            enemy_buy_prediction=enemy_buy_pred,
            note=note,
            enemy_predicted_tier=enemy_tier,
            enemy_tier_probs=enemy_probs,
            enemy_predicted_money=enemy_money,
            weapon_matchup_note=matchup_note,
            team_avg_money=team_avg,
            team_can_fullbuy_next=team_fb,
            player_can_fullbuy_next=player_fb,
            is_drop_or_pickup=is_drop,
            posthoc=posthoc_detail,
        ))

        won = (rd.winner == side)
        if won:
            loss_streak = 0
            enemy_loss_streak = min(enemy_loss_streak + 1, MAX_LOSS_STREAK)
        else:
            loss_streak = min(loss_streak + 1, MAX_LOSS_STREAK)
            enemy_loss_streak = 0
        prev_alive = p.alive_at_end

    # Post-process: weapon upgrade path annotations
    _annotate_upgrade_paths(evaluations, rounds, target_player)

    return evaluations


def _annotate_upgrade_paths(evaluations: list[BuyEvaluation],
                            rounds: list, target_player: str) -> None:
    """Annotate consecutive eco/force rounds with weapon upgrade path notes."""
    buy_tier_names = {SAVE: "eco", FORCE: "half-buy", FULL_BUY: "full buy"}

    weapon_tier_order = {"pistol": 0, "smg": 1, "shotgun": 1,
                         "rifle": 2, "awp": 3, "lmg": 2}
    player_weapons = {}
    for rd in rounds:
        p = rd.get_player(target_player)
        if p and p.primary_weapon:
            player_weapons[rd.round_num] = p.primary_weapon
        elif p:
            player_weapons[rd.round_num] = None

    # Find consecutive non-fullbuy streaks
    i = 0
    while i < len(evaluations):
        if evaluations[i].actual_action == FULL_BUY:
            i += 1
            continue

        streak = [evaluations[i]]
        j = i + 1
        while j < len(evaluations) and evaluations[j].actual_action != FULL_BUY:
            streak.append(evaluations[j])
            j += 1

        if len(streak) >= 2:
            path_parts = []
            for ev in streak:
                wpn = player_weapons.get(ev.round_num)
                w_tier = _WEAPON_TIER.get(wpn, "pistol") if wpn else "pistol"
                buy_label = buy_tier_names.get(ev.actual_action, "?")
                path_parts.append(f"R{ev.round_num} {buy_label} (${ev.money})")

            tiers = []
            for ev in streak:
                wpn = player_weapons.get(ev.round_num)
                tiers.append(weapon_tier_order.get(
                    _WEAPON_TIER.get(wpn, "pistol") if wpn else "pistol", 0))

            if all(tiers[k] <= tiers[k+1] for k in range(len(tiers)-1)):
                note = f"Good upgrade path: {' → '.join(path_parts)}"
            else:
                note = f"Upgrade path: {' → '.join(path_parts)}"

            for ev in streak:
                ev.upgrade_path_note = note

        i = j


def economy_summary(evaluations: list[BuyEvaluation]) -> dict:
    """Aggregate economy evaluation into a summary (no letter grade)."""
    if not evaluations:
        return {}

    n = len(evaluations)
    n_optimal = sum(1 for e in evaluations if e.is_optimal)
    mistakes = [e for e in evaluations if not e.is_optimal]
    avg_wp_loss = (sum(e.win_prob_diff for e in mistakes) / len(mistakes)
                   if mistakes else 0.0)

    over_buys = sum(1 for e in mistakes
                    if e.actual_action > e.optimal_action)
    under_buys = sum(1 for e in mistakes
                     if e.actual_action < e.optimal_action)

    fresh_buys = [e for e in evaluations if not e.note.startswith("(carried")]
    fresh_n = len(fresh_buys)
    fresh_optimal = sum(1 for e in fresh_buys if e.is_optimal)
    team_drops = n - fresh_n

    vs_eco_mistakes = sum(1 for e in mistakes if e.enemy_loss_streak == 1)
    vs_buy_mistakes = sum(1 for e in mistakes if e.enemy_loss_streak == 0)

    total_waste = sum(e.posthoc.waste for e in evaluations
                      if e.posthoc is not None)
    missing_kit_rounds = sum(1 for e in evaluations
                             if e.posthoc and e.posthoc.kit_note)
    weapon_mismatches = sum(1 for e in evaluations
                            if e.posthoc and not e.posthoc.weapon_appropriate)
    no_util_on_fullbuy = sum(1 for e in evaluations
                             if e.posthoc and not e.posthoc.utility_sufficient)

    return {
        "total_rounds": n,
        "fresh_buy_rounds": fresh_n,
        "team_drop_rounds": team_drops,
        "optimal_decisions": n_optimal,
        "fresh_buy_accuracy": fresh_optimal / fresh_n if fresh_n > 0 else 1.0,
        "overall_accuracy": n_optimal / n,
        "mistakes": len(mistakes),
        "over_buys": over_buys,
        "under_buys": under_buys,
        "mistakes_vs_enemy_eco": vs_eco_mistakes,
        "mistakes_vs_enemy_buy": vs_buy_mistakes,
        "avg_wp_loss_per_mistake": avg_wp_loss,
        "total_waste": total_waste,
        "missing_kit_rounds": missing_kit_rounds,
        "weapon_mismatches": weapon_mismatches,
        "no_util_on_fullbuy": no_util_on_fullbuy,
    }


def print_policy(policy: EconomyPolicy, enemy_streaks: list[int] | None = None) -> None:
    """Print the optimal policy as readable tables.

    Parameters
    ----------
    policy : EconomyPolicy
        Solved policy to display.
    enemy_streaks : list[int] | None
        Which enemy streak slices to print. Defaults to [0, 1, 3].
    """
    if enemy_streaks is None:
        enemy_streaks = [0, 1, 3]

    for ek in enemy_streaks:
        ek_label = "Enemy just won" if ek == 0 else f"Enemy lost {ek}"
        print(f"\nOptimal Buy Policy ({policy.side} side, {ek_label})")
        print(f"{'Money':>8s}", end="")
        for k in range(N_STREAKS):
            label = f"L{k}" if k > 0 else "W"
            print(f"  {label:>8s}", end="")
        print()
        print("-" * (8 + 10 * N_STREAKS))

        for b in range(N_MONEY_BINS):
            money = _bin_to_money(b)
            print(f"${money:>6d}", end="")
            for k in range(N_STREAKS):
                act = int(policy.policy[b, k, ek])
                print(f"  {ACTION_NAMES[act]:>8s}", end="")
            print()


if __name__ == "__main__":
    for side in ("T", "CT"):
        policy = solve_economy_mdp(side)
        print_policy(policy)
        print(f"\n{'='*70}")
        print(f"  {policy.side} side — State space: {policy.V.shape} "
              f"({policy.V.size} states)")
        print(f"{'='*70}")
