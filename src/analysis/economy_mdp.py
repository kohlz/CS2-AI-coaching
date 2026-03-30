"""
economy_mdp.py

Model CS2 economy as a Markov Decision Process and solve for the optimal
buy policy using Value Iteration.  Then evaluate a target player's actual
buy decisions against the optimal policy to produce coaching feedback.

CS2 Economy Rules (as of 2025-07 update)
-----------------------------------------
Starting money:         $800 (resets at half)
Money cap:              $16,000

Round win rewards:
  Elimination / time:   $3,250   (both sides)
  Bomb explode (T):     $3,500
  Bomb defuse  (CT):    $3,500

Bomb plant bonus (T):   $800 per T player (even on loss)
Planter individual:     $300 extra to the planter
Defuser individual:     $300 extra to the defuser

Loss bonus (consecutive losses):
  1st loss: $1,400     2nd: $1,900     3rd: $2,400
  4th: $2,900          5th+: $3,400
  Resets to $1,400 after a win.

CT kill sharing bonus:  $50 per CT player for each T killed (max $250/CT)

Kill rewards by weapon:
  Knife        $1,500      Shotguns (excl. XM1014)   $900
  SMGs (excl. P90)  $600   XM1014 / P90              $300
  Pistols / rifles / LMGs / grenades                  $300
  AWP / Zeus                                          $100
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# CS2 Economy Constants
# ---------------------------------------------------------------------------

MONEY_CAP = 16_000
STARTING_MONEY = 800

LOSS_BONUS = [1_400, 1_900, 2_400, 2_900, 3_400]  # indexed by streak-1

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

# ---------------------------------------------------------------------------
# MDP Configuration
# ---------------------------------------------------------------------------

MONEY_STEP = 500
N_MONEY_BINS = MONEY_CAP // MONEY_STEP + 1    # 0, 500, ..., 16000 → 33
MAX_LOSS_STREAK = 5                            # 0..5
N_STREAKS = MAX_LOSS_STREAK + 1                # 6

# Actions
SAVE = 0
FORCE = 1
FULL_BUY = 2
ACTIONS = [SAVE, FORCE, FULL_BUY]
ACTION_NAMES = {SAVE: "SAVE", FORCE: "FORCE", FULL_BUY: "FULL_BUY"}

# Approximate equipment cost for each action (what you spend)
EQUIP_COST = {
    "T":  {SAVE: 200,  FORCE: 2_600, FULL_BUY: 4_700},
    "CT": {SAVE: 200,  FORCE: 2_800, FULL_BUY: 5_500},
}

# Equipment tier index (maps action → tier for win-prob lookup)
# ECO=0, FORCE=1, FULL=2
ACTION_TIER = {SAVE: 0, FORCE: 1, FULL_BUY: 2}

# Win probability matrix: P(win | my_tier, opp_tier)
# Rows = my tier (0=eco, 1=force, 2=full), Cols = opponent tier
WIN_PROB = {
    "T": np.array([
        [0.45, 0.28, 0.12],   # eco  vs eco/force/full
        [0.65, 0.45, 0.30],   # force
        [0.82, 0.65, 0.48],   # full buy
    ]),
    "CT": np.array([
        [0.55, 0.30, 0.15],   # eco  (CT has slight inherent advantage)
        [0.70, 0.52, 0.35],   # force
        [0.88, 0.68, 0.52],   # full buy
    ]),
}

# Average kill income per round (approximate)
AVG_KILL_INCOME_WIN = 600    # ~2 kills * $300 avg
AVG_KILL_INCOME_LOSS = 200   # ~0.6 kills * $300
T_PLANT_RATE_ON_LOSS = 0.35  # bomb planted in ~35% of T round losses
CT_KILL_BONUS_WIN = 200      # ~4 T kills * $50
CT_KILL_BONUS_LOSS = 75      # ~1.5 T kills * $50

# Reward structure
WIN_REWARD_MDP = 1.0        # reward for winning a round
LOSS_PENALTY_MDP = -0.3     # penalty for losing (money in bank = wasted opportunity)

# Discount factor for infinite-horizon MDP
GAMMA = 0.85

# Value Iteration convergence threshold
VI_EPSILON = 1e-6
VI_MAX_ITER = 500


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _money_to_bin(money: int) -> int:
    return min(max(0, money // MONEY_STEP), N_MONEY_BINS - 1)


def _bin_to_money(b: int) -> int:
    return b * MONEY_STEP


def _loss_bonus(streak: int) -> int:
    """Loss bonus received after accumulating `streak` consecutive losses."""
    if streak <= 0:
        return 0
    idx = min(streak, MAX_LOSS_STREAK) - 1
    return LOSS_BONUS[idx]


def _opponent_equip_dist(my_loss_streak: int) -> np.ndarray:
    """Estimate opponent equipment distribution based on my loss streak.

    If I've been losing, the opponent has been winning → they're rich.
    If I just won, the opponent lost → they may eco.
    """
    if my_loss_streak == 0:
        return np.array([0.45, 0.30, 0.25])
    elif my_loss_streak == 1:
        return np.array([0.10, 0.20, 0.70])
    else:
        return np.array([0.05, 0.10, 0.85])


# ---------------------------------------------------------------------------
# MDP Transition Model
# ---------------------------------------------------------------------------

def _expected_win_prob(side: str, my_tier: int, my_loss_streak: int) -> float:
    """Expected win probability given my equipment tier and loss streak."""
    opp_dist = _opponent_equip_dist(my_loss_streak)
    return float(WIN_PROB[side][my_tier] @ opp_dist)


def _next_money_win(side: str, money_after_buy: int) -> int:
    """Money at the start of next round after winning."""
    income = WIN_REWARD_ELIM + AVG_KILL_INCOME_WIN
    if side == "CT":
        income += CT_KILL_BONUS_WIN
        income += (WIN_REWARD_DEFUSE - WIN_REWARD_ELIM) * 0.5
    else:
        income += (WIN_REWARD_BOMB - WIN_REWARD_ELIM) * 0.5
    return min(money_after_buy + int(income), MONEY_CAP)


def _next_money_loss(side: str, money_after_buy: int, new_streak: int) -> int:
    """Money at the start of next round after losing."""
    income = _loss_bonus(new_streak) + AVG_KILL_INCOME_LOSS
    if side == "T":
        income += int(T_BOMB_PLANT_BONUS * T_PLANT_RATE_ON_LOSS)
    else:
        income += CT_KILL_BONUS_LOSS
    return min(money_after_buy + int(income), MONEY_CAP)


# ---------------------------------------------------------------------------
# Value Iteration
# ---------------------------------------------------------------------------

@dataclass
class EconomyPolicy:
    """Solved MDP policy for one side (T or CT)."""
    side: str
    V: np.ndarray            # shape (N_MONEY_BINS, N_STREAKS)
    policy: np.ndarray       # shape (N_MONEY_BINS, N_STREAKS), values in {0,1,2}

    def recommend(self, money: int, loss_streak: int) -> int:
        b = _money_to_bin(money)
        k = min(loss_streak, MAX_LOSS_STREAK)
        return int(self.policy[b, k])

    def value(self, money: int, loss_streak: int) -> float:
        b = _money_to_bin(money)
        k = min(loss_streak, MAX_LOSS_STREAK)
        return float(self.V[b, k])


def solve_economy_mdp(side: str, gamma: float = GAMMA) -> EconomyPolicy:
    """Solve the economy MDP for one side using Value Iteration.

    Returns an EconomyPolicy with the optimal value function and policy.
    """
    V = np.zeros((N_MONEY_BINS, N_STREAKS))
    policy = np.zeros((N_MONEY_BINS, N_STREAKS), dtype=int)

    costs = EQUIP_COST[side]

    for iteration in range(VI_MAX_ITER):
        V_new = np.zeros_like(V)

        for b in range(N_MONEY_BINS):
            money = _bin_to_money(b)
            for k in range(N_STREAKS):
                best_val = -1e9
                best_act = SAVE

                for a in ACTIONS:
                    cost = costs[a]
                    actual_cost = min(cost, money)  # can't spend more than you have

                    if a != SAVE and cost > money:
                        continue  # can't afford this buy tier

                    money_after = money - actual_cost
                    tier = ACTION_TIER[a]
                    p_win = _expected_win_prob(side, tier, k)

                    # Win branch
                    next_m_win = _next_money_win(side, money_after)
                    b_win = _money_to_bin(next_m_win)
                    k_win = 0

                    # Loss branch
                    k_loss = min(k + 1, MAX_LOSS_STREAK)
                    next_m_loss = _next_money_loss(side, money_after, k_loss)
                    b_loss = _money_to_bin(next_m_loss)

                    q = (p_win * (WIN_REWARD_MDP + gamma * V[b_win, k_win]) +
                         (1 - p_win) * (LOSS_PENALTY_MDP + gamma * V[b_loss, k_loss]))

                    if q > best_val:
                        best_val = q
                        best_act = a

                V_new[b, k] = best_val
                policy[b, k] = best_act

        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < VI_EPSILON:
            break

    return EconomyPolicy(side=side, V=V, policy=policy)


# ---------------------------------------------------------------------------
# Player Buy-Decision Classifier
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Round-by-Round Evaluation
# ---------------------------------------------------------------------------

@dataclass
class BuyEvaluation:
    """Evaluation of a single round's buy decision."""
    round_num: int
    side: str
    money: int
    loss_streak: int
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
    note: str = ""


def _is_knife_or_pistol(round_num: int) -> bool:
    """Rounds 0-1 are knife + pistol in competitive/Faceit."""
    return round_num <= 1


def evaluate_player_economy(
    rounds: list,  # list[RoundData]
    target_player: str,
) -> list[BuyEvaluation]:
    """Evaluate the target player's buy decisions across all rounds.

    Accounts for equipment carry-over (surviving players keep their gear),
    skips knife round (round 0) and pistol rounds, and resets loss streak
    at halftime.

    Parameters
    ----------
    rounds : list[RoundData]
        Parsed round data from demo_parser.
    target_player : str
        In-game name to analyze.

    Returns
    -------
    list[BuyEvaluation]
        Per-round evaluation against optimal policy.
    """
    t_policy = solve_economy_mdp("T")
    ct_policy = solve_economy_mdp("CT")

    evaluations: list[BuyEvaluation] = []
    loss_streak = 0
    prev_alive = False
    prev_side: Optional[str] = None
    is_pistol = False  # next round after side switch is pistol

    for rd in rounds:
        p = rd.get_player(target_player)
        if p is None:
            prev_alive = False
            continue

        side = p.side
        if side not in ("T", "CT"):
            prev_alive = False
            continue

        # Detect half-time side switch → reset economy state
        if prev_side is not None and prev_side != side:
            loss_streak = 0
            prev_alive = False
            is_pistol = True   # first round after side switch = pistol

        prev_side = side
        money = p.start_money

        # Skip knife round, pistol rounds (first after each half), and
        # any round where the player had <$1000 (economy reset / pistol)
        skip = _is_knife_or_pistol(rd.round_num) or is_pistol or money < 1000
        if skip:
            is_pistol = False
            won = (rd.winner == side)
            loss_streak = 0 if won else min(loss_streak + 1, MAX_LOSS_STREAK)
            prev_alive = p.alive_at_end
            continue

        policy = t_policy if side == "T" else ct_policy

        # Classify what equipment tier the player ACTUALLY HAS
        actual_equip = classify_buy_decision(
            p.primary_weapon, p.secondary_weapon,
            p.has_helmet, p.armor,
            p.equipment_value, p.utilities,
        )

        # Determine what the player could afford to buy from scratch
        best_affordable = SAVE
        for a in [FULL_BUY, FORCE]:
            if EQUIP_COST[side][a] <= money:
                best_affordable = a
                break

        # What the optimal policy says to do with this money + streak
        optimal = policy.recommend(money, loss_streak)

        note = ""

        # Handle equipment carry-over:
        # If player survived last round, they may carry equipment worth
        # more than what they'd buy fresh. The effective tier is what they
        # HAVE, not what they spent. Their buy decision only matters for
        # the gap between what they carry and what they could upgrade to.
        if prev_alive and actual_equip > best_affordable:
            actual = actual_equip
            is_optimal = True
            note = "(carried equipment)"
        elif EQUIP_COST[side][actual_equip] > money and actual_equip > best_affordable:
            # Equipment value exceeds what money could buy → must be carried
            # or picked up (possible even after dying due to demo tick timing)
            actual = actual_equip
            is_optimal = (actual_equip >= optimal)
            if is_optimal:
                note = "(carried/picked up)"
        else:
            # Fresh buy decision — this is what the MDP evaluates
            actual = actual_equip
            is_optimal = (actual == optimal)

        actual_tier = ACTION_TIER[actual]
        optimal_tier = ACTION_TIER[optimal]
        actual_wp = _expected_win_prob(side, actual_tier, loss_streak)
        optimal_wp = _expected_win_prob(side, optimal_tier, loss_streak)

        cost_actual = min(EQUIP_COST[side][actual], money)
        money_after_actual = money - cost_actual
        actual_val = (actual_wp * (WIN_REWARD_MDP + GAMMA * policy.value(
            _next_money_win(side, money_after_actual), 0))
            + (1 - actual_wp) * (LOSS_PENALTY_MDP + GAMMA * policy.value(
            _next_money_loss(side, money_after_actual,
                             min(loss_streak + 1, MAX_LOSS_STREAK)),
            min(loss_streak + 1, MAX_LOSS_STREAK))))
        optimal_val = policy.value(money, loss_streak)

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

        evaluations.append(BuyEvaluation(
            round_num=rd.round_num,
            side=side,
            money=money,
            loss_streak=loss_streak,
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
            note=note,
        ))

        won = (rd.winner == side)
        loss_streak = 0 if won else min(loss_streak + 1, MAX_LOSS_STREAK)
        prev_alive = p.alive_at_end

    return evaluations


def economy_summary(evaluations: list[BuyEvaluation]) -> dict:
    """Aggregate economy evaluation into a summary."""
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

    # Distinguish fresh buy decisions from carried/team-dropped rounds
    fresh_buys = [e for e in evaluations if not e.note.startswith("(carried")]
    fresh_n = len(fresh_buys)
    fresh_optimal = sum(1 for e in fresh_buys if e.is_optimal)
    team_drops = n - fresh_n

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
        "avg_wp_loss_per_mistake": avg_wp_loss,
        "grade": _grade(n_optimal / n),
    }


def _grade(accuracy: float) -> str:
    if accuracy >= 0.90:
        return "A"
    if accuracy >= 0.75:
        return "B"
    if accuracy >= 0.60:
        return "C"
    if accuracy >= 0.45:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Debug / CLI
# ---------------------------------------------------------------------------

def print_policy(policy: EconomyPolicy) -> None:
    """Print the optimal policy as a readable table."""
    print(f"\nOptimal Buy Policy ({policy.side} side)")
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
            act = int(policy.policy[b, k])
            print(f"  {ACTION_NAMES[act]:>8s}", end="")
        print()


if __name__ == "__main__":
    for side in ("T", "CT"):
        policy = solve_economy_mdp(side)
        print_policy(policy)
