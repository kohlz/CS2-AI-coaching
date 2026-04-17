"""
generate_report.py

End-to-end coaching report generator. Combines demo parsing, economy MDP,
NN/LSTM predictions, RL suggestions, and engagement analysis into a
structured per-player report.
"""

from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "demo"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "report"))


from demo_parser import parse_demo, MatchData
from callouts_mirage import get_zone
from economy_mdp import evaluate_player_economy, economy_summary
from engagement import analyze_engagement, engagement_summary
from info_model import predict_enemy_economy

try:
    from visualize import generate_all_charts
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False

_NN_AVAILABLE = False
_RL_AVAILABLE = False

try:
    from strategy_nn import (
        load_models, PreRoundFormation, PreRoundAttack,
        FormationClassifier_T, FormationClassifier_CT,
    )
    _NN_AVAILABLE = True
except ImportError:
    pass

try:
    from tactical_rl import (
        TacticalQLearner, ACTION_NAMES,
        TacticalQLearnerV2, ACTION_NAMES_V2,
        TacticalQLearner_T, TacticalQLearner_CT,
    )
    _RL_AVAILABLE = True
except ImportError:
    pass


@dataclass
class CoachingReport:
    """Complete coaching report for one demo."""
    demo_file: str
    player_name: str
    map_name: str
    match_score: str           # "T 13 - 9 CT"
    total_rounds: int

    economy: dict = field(default_factory=dict)
    engagement: dict = field(default_factory=dict)
    game_sense: dict = field(default_factory=dict)
    nn_predictions: dict = field(default_factory=dict)

    round_details: list[dict] = field(default_factory=list)
    coaching_tips: list[str] = field(default_factory=list)

    generation_time_sec: float = 0.0


def _generate_tips(econ: dict, eng: dict, _gs=None) -> list[str]:
    """Generate natural-language coaching tips from summary stats."""
    tips: list[str] = []

    # Economy tips
    acc = econ.get("fresh_buy_accuracy", 1.0)
    if acc < 0.60:
        tips.append(
            f"Economy: Your buy decisions matched the optimal policy only "
            f"{acc:.0%} of the time.  Focus on saving when you should eco "
            f"and buying up when the economy allows.")
    elif acc < 0.80:
        tips.append(
            f"Economy: Decent economy management ({acc:.0%} optimal), "
            f"but room for improvement.")

    obuys = econ.get("over_buys", 0)
    ubuys = econ.get("under_buys", 0)
    if obuys > ubuys and obuys >= 3:
        tips.append(
            f"Economy: You over-bought {obuys} times — consider saving more "
            f"when the team needs a full buy next round.")
    elif ubuys > obuys and ubuys >= 3:
        tips.append(
            f"Economy: You under-bought {ubuys} times — you had money to "
            f"upgrade but didn't.  Don't sit on savings unnecessarily.")

    vs_eco = econ.get("mistakes_vs_enemy_eco", 0)
    if vs_eco >= 2:
        tips.append(
            f"Economy: {vs_eco} buy mistakes against eco opponents — "
            f"you can save more when the enemy is on an eco round.")

    # Engagement tips
    adr = eng.get("adr", 0)
    if adr < 55:
        tips.append(
            f"Impact: Your ADR of {adr:.0f} is below average.  Try to "
            f"deal more damage each round through better positioning.")

    iso_rate = eng.get("isolated_death_rate", 0)
    if iso_rate >= 0.40:
        tips.append(
            f"Positioning: {iso_rate:.0%} of your deaths were isolated "
            f"(no teammate could trade).  Stay closer to teammates "
            f"or hold positions where trades are possible.")

    trade = eng.get("trade_rate", 0.5)
    if trade < 0.30:
        tips.append(
            f"Teamplay: Your trade rate is only {trade:.0%} — when a "
            f"teammate dies, try to refrag within 5 seconds.")

    opening = eng.get("opening_success_rate", 0.5)
    o_duels = eng.get("opening_duels", 0)
    if o_duels >= 5 and opening < 0.35:
        tips.append(
            f"Dueling: You took {o_duels} opening duels but only won "
            f"{opening:.0%}.  Consider changing your peek angles or "
            f"using utility before engaging.")

    util = eng.get("avg_util_on_rifle_round", 0)
    if util < 1.5:
        tips.append(
            f"Utility: On rifle rounds you only used ~{util:.1f} nades. "
            f"Buy and use more utility to gain map control safely.")

    if not tips:
        tips.append("Solid performance overall — keep it up!")

    return tips


def _resolve_bomb_site(rd) -> str:
    """Return the bomb site as 'A' or 'B' for a round, or 'no_plant'/'unknown'."""
    if not rd.bomb_planted:
        return "no_plant"
    if rd.bomb_site in ("A", "B"):
        return rd.bomb_site

    TICK_RATE = 64
    POS_STEP = 2 * TICK_RATE

    for ev in rd.events:
        if ev.event_type == "bomb_plant":
            plant_tick = ev.tick
            planter = ev.data.get("planter", "")
            for p in rd.t_players:
                if p.name == planter and p.positions:
                    idx = round((plant_tick - rd.tick_freeze_end) / POS_STEP)
                    idx = max(0, min(idx, len(p.positions) - 1))
                    z = get_zone(p.positions[idx].x, p.positions[idx].y)
                    if z in ("A", "B"):
                        return z
                    for off in (1, -1, 2, -2):
                        ni = idx + off
                        if 0 <= ni < len(p.positions):
                            z = get_zone(p.positions[ni].x, p.positions[ni].y)
                            if z in ("A", "B"):
                                return z
    return "unknown"


def _resolve_plant_site(rd) -> str:
    """Return 'A', 'B', or 'no_plant' for a round."""
    for ev in rd.events:
        if ev.event_type == "bomb_plant":
            site = _resolve_bomb_site(rd)
            if site in ("A", "B"):
                return site
    return "no_plant"


def _compute_prior_features(match: MatchData) -> dict[int, dict]:
    """Compute prior-round tendency features for each round_num."""
    out: dict[int, dict] = {}
    rounds = sorted(match.rounds, key=lambda r: r.round_num)

    rounds_since_A = 999
    rounds_since_B = 999
    streak_same = 0
    last_site = ""
    prev_rd = None

    def _equip_tier(players) -> float:
        if not players:
            return 0.0
        from callouts_mirage import get_zone  # noqa: F401
        RIFLES = {"ak47", "m4a1_silencer", "m4a1", "m4a4", "awp", "famas",
                  "galilar", "aug", "sg556", "scar20", "g3sg1"}
        SMGS = {"mp9", "mac10", "ump45", "p90", "bizon", "mp7", "mp5sd"}
        rifles = sum(1 for p in players
                     if (p.primary_weapon or "").lower() in RIFLES)
        smgs = sum(1 for p in players
                   if (p.primary_weapon or "").lower() in SMGS)
        return 2.0 if rifles >= 3 else (1.0 if rifles + smgs >= 2 else 0.0)

    for rd in rounds:
        round_in_half = rd.round_num % 12
        is_half_start = (round_in_half == 0)

        if is_half_start or prev_rd is None:
            feats = {
                "prev_plant_A": 0, "prev_plant_B": 0, "prev_plant_none": 0,
                "prev_no_history": 1, "prev_t_won": 0.5,
                "prev_t_tier": 0.0, "prev_ct_tier": 0.0,
                "rounds_since_plant_A": 1.0, "rounds_since_plant_B": 1.0,
                "streak_same_site": 0.0,
            }
            rounds_since_A = 999
            rounds_since_B = 999
            streak_same = 0
            last_site = ""
        else:
            prev_site = _resolve_plant_site(prev_rd)
            feats = {
                "prev_plant_A": int(prev_site == "A"),
                "prev_plant_B": int(prev_site == "B"),
                "prev_plant_none": int(prev_site == "no_plant"),
                "prev_no_history": 0,
                "prev_t_won": 1.0 if prev_rd.winner == "T" else 0.0,
                "prev_t_tier": _equip_tier(prev_rd.t_players) / 2.0,
                "prev_ct_tier": _equip_tier(prev_rd.ct_players) / 2.0,
                "rounds_since_plant_A": min(rounds_since_A, 6) / 6.0,
                "rounds_since_plant_B": min(rounds_since_B, 6) / 6.0,
                "streak_same_site": min(streak_same, 3) / 3.0,
            }

        out[rd.round_num] = feats

        cur_site = _resolve_plant_site(rd)
        rounds_since_A = 0 if cur_site == "A" else rounds_since_A + 1
        rounds_since_B = 0 if cur_site == "B" else rounds_since_B + 1
        if cur_site in ("A", "B"):
            streak_same = streak_same + 1 if cur_site == last_site else 1
            last_site = cur_site
        else:
            streak_same = 0
        prev_rd = rd

    return out


def _run_nn_predictions(match: MatchData, models: dict,
                        hmm_predictions: list[dict] | None = None) -> dict:
    """Run NN predictions for each round and return aggregate info."""
    prf = models.get("preround_formation")
    pra = models.get("preround_attack")

    predictions = {
        "available": True,
        "formation_predictions": [],
        "attack_predictions": [],
    }

    hmm_by_round: dict[int, dict] = {}
    if hmm_predictions:
        for pred in hmm_predictions:
            hmm_by_round[pred["round_num"]] = pred

    prior_by_round = _compute_prior_features(match)

    for rd in match.rounds:
        p = rd.get_player(match.target_player)
        if p is None:
            continue

        round_in_half = rd.round_num % 12
        is_second_half = 1 if rd.round_num >= 12 else 0
        prior = prior_by_round.get(rd.round_num, {})

        hmm_pred = hmm_by_round.get(rd.round_num)
        if hmm_pred:
            tier_probs = hmm_pred.get("tier_probs", {})
            avg_money = hmm_pred.get("predicted_avg_money", 4000)
        else:
            tier_probs = {"BROKE": 0.2, "LOW": 0.2, "MEDIUM": 0.2,
                          "HIGH": 0.2, "RICH": 0.2}
            avg_money = 4000

        # Pre-round formation
        if prf is not None and prf.trained:
            try:
                probs = prf.predict_single(
                    tier_probs=tier_probs,
                    predicted_avg_money=avg_money,
                    round_in_half=round_in_half,
                    is_second_half=is_second_half,
                    prior=prior,
                )
                top_formation = max(probs, key=probs.get)
                predictions["formation_predictions"].append({
                    "round": rd.round_num,
                    "predicted": top_formation,
                    "probs": probs,
                })
            except Exception:
                pass

        # Pre-round attack
        if pra is not None and pra.trained:
            try:
                t_hmm = hmm_by_round.get(rd.round_num, {})
                t_tier_probs = t_hmm.get("t_tier_probs", tier_probs)
                t_avg_money = t_hmm.get("t_predicted_avg_money", avg_money)
                a_probs = pra.predict_single(
                    tier_probs=t_tier_probs,
                    predicted_avg_money=t_avg_money,
                    round_in_half=round_in_half,
                    is_second_half=is_second_half,
                    prior=prior,
                )
                top_site = max(a_probs, key=a_probs.get)
                predictions["attack_predictions"].append({
                    "round": rd.round_num,
                    "predicted": top_site,
                    "probs": a_probs,
                })
            except Exception:
                pass

    return predictions


TEMPLATES = {
    "eco_over_buy": (
        "Over-bought (did {action} at ${money}, optimal was {optimal}). "
        "Save this round to guarantee a full buy next round."),
    "eco_under_buy": (
        "Under-bought (did {action} at ${money}, optimal was {optimal}). "
        "You had enough to upgrade — don't sit on extra cash."),
    "eco_correct": "Good buy decision ({action} at ${money}).",
    "eco_enemy_eco": (
        "{enemy_pred} — you can play more aggressively and save utility."),
    "eco_enemy_force": (
        "{enemy_pred} — expect a mix of pistols and SMGs, play safe angles."),
    "eco_enemy_full_buy": (
        "{enemy_pred} — play default setup, use all your utility."),

    "opening_kill": (
        "Great opening frag — first blood gives your team a huge advantage."),
    "opening_death": (
        "Died first this round. Use utility (flash/smoke) before peeking, "
        "or change your angle next time."),
    "isolated_death": (
        "Died with no teammate close enough to trade. "
        "Hold positions where a teammate can refrag within 5 seconds."),
    "traded_death": (
        "You died but a teammate traded the kill — acceptable if you "
        "gained info or map control."),
    "multi_kill": "Impact round with {kills} kills and {damage} damage!",
    "clutch_win": "Incredible 1v{n} clutch! Great composure under pressure.",
    "clutch_loss": (
        "1v{n} clutch attempt — tough situation. "
        "If the round is unwinnable, consider saving your weapon for next round."),
    "zero_impact": (
        "No kills and no damage dealt. Try to find at least one "
        "engagement — even chip damage helps your team."),
    "low_impact": (
        "Only {damage} damage this round. "
        "Look for opportunities to deal more damage through utility or repositioning."),

    "gs_correct": (
        "Good game sense — you read the {zone} attack correctly "
        "({prob:.0%} belief) and were in position."),
    "gs_wrong_site": (
        "Attack came to {zone} ({prob:.0%} belief) but you were at "
        "{player_zone}. Rotate faster when you hear contact or see "
        "utility at {zone}."),
    "gs_died_early": (
        "Info pointed to {zone} ({prob:.0%}) but you died before you "
        "could rotate — avoid early aggressive peeks when you need to "
        "anchor a site."),

    "nn_win_ct_favored": (
        "[NN] Round win probability: {p_win:.0%} for your side (CT favored) — "
        "you have economy/position advantage, play default and use utility."),
    "nn_win_ct_slight": (
        "[NN] Round win probability: {p_win:.0%} for your side — "
        "roughly even, execute your role cleanly."),
    "nn_win_ct_underdog": (
        "[NN] Round win probability: {p_win:.0%} for your side (T favored) — "
        "play for trades, avoid isolated fights."),
    "nn_win_t_favored": (
        "[NN] Round win probability: {p_win:.0%} for your side (T favored) — "
        "you have the economy advantage, execute confidently with full utility."),
    "nn_win_t_slight": (
        "[NN] Round win probability: {p_win:.0%} for your side — "
        "roughly even, focus on getting the bomb down."),
    "nn_win_t_underdog": (
        "[NN] Round win probability: {p_win:.0%} for your side (CT favored) — "
        "consider a default to look for picks before committing to a site."),

    "nn_attack_a_likely": (
        "[NN] T-side is {a_prob:.0%} likely to hit A vs {b_prob:.0%} B — "
        "consider anchoring A or having utility ready for A."),
    "nn_attack_b_likely": (
        "[NN] T-side is {b_prob:.0%} likely to hit B vs {a_prob:.0%} A — "
        "consider anchoring B or having utility ready for B."),
    "nn_attack_split": (
        "[NN] Attack probability is split (A: {a_prob:.0%}, B: {b_prob:.0%}) — "
        "stay in your default position and wait for info."),
    "nn_t_should_hit": (
        "[NN] CT defense looks weaker on {site} ({prob:.0%}) — "
        "focus utility and coordination for a {site} execute."),
    "nn_t_split": (
        "[NN] Attack probability is roughly even (A: {a_prob:.0%}, B: {b_prob:.0%}) — "
        "stay flexible and read the round before committing to a site."),

    "nn_ct_formation": (
        "[NN] CT likely running {formation} ({fmt_prob:.0%}) — "
        "{formation_advice}"),
    "nn_ct_formation_uncertain": (
        "[NN] CT formation unclear — play default and read "
        "their setup from utility/positions."),

    "lstm_prediction": (
        "[LSTM @ {time}s] After {n_events} events: "
        "A={a_prob:.0%}, B={b_prob:.0%} → {interpretation}"),

    "lstm_ct_formation": (
        "[Formation] CT running {formation} ({confidence:.0%}) — "
        "{formation_advice}"),

    "rl_suggest": (
        "[Q-learning] {state_desc} → recommended: {rl_action}."),
    "rl_ss_suggest": (
        "[RL] {state_desc} → best play: {rl_action} "
        "(kill-Q: {kill_q:+.2f}, win-Q: {win_q:+.2f})."),
    "rl_v2_suggest": (
        "[Q-learning v2] {state_desc} → best play: {rl_action} "
        "(kill-value: {kill_q:+.2f}, win-value: {win_q:+.2f})."),
    "rl_v2_trade_tip": (
        "[Q-learning v2] Teammate just died nearby — trading has high kill reward "
        "({kill_q:+.2f}). Move toward the fight."),
    "rl_v2_utility_tip": (
        "[Q-learning v2] Using utility here has high strategic value "
        "(win-Q: {win_q:+.2f}). Smoke or flash before committing."),

    "clean_win": "Clean round win. Keep it up.",
    "tough_loss": "Tough loss — nothing specific to change here.",
    "pistol_round": (
        "Pistol round — aim for headshots, stick together, "
        "and trade kills quickly."),
}


def _interpret_lstm(probs: dict, side: str) -> str:
    """Produce a human-readable interpretation of LSTM site probabilities."""
    a_p = probs.get("A", 0)
    b_p = probs.get("B", 0)
    top = max(probs, key=probs.get)

    if side == "CT":
        if top == "no_plant":
            return "no clear attack direction yet"
        if a_p >= 0.60:
            return "A attack likely — rotate/stack A"
        if b_p >= 0.60:
            return "B attack likely — rotate/stack B"
        if a_p + b_p >= 0.60:
            return "attack incoming, direction unclear — stay alert"
        return "likely default/slow round"
    else:
        if top == "no_plant":
            return "no clear execute yet — play default and wait for a call"
        if a_p >= 0.60:
            return "your team is executing A — commit to the push"
        if b_p >= 0.60:
            return "your team is executing B — commit to the push"
        if a_p + b_p >= 0.60:
            return "execute developing — stay with your team and read the call"
        return "slow round — look for picks or wait for a play call"


def _suggest_for_round(
    rd_detail: dict,
    econ_eval,
    eng_eval,
    gs_eval,
    nn_round: Optional[dict] = None,
    rl_suggestion: Optional[dict] = None,
    rl_v2_suggestion: Optional[dict] = None,
    lstm_preds: Optional[list] = None,
    rl_timeline: Optional[list] = None,
    ct_formation_preds: Optional[list] = None,
    round_events: Optional[list] = None,
) -> dict:
    """Generate per-round coaching output with event-driven timeline.

    Returns a dict with ``pre_round`` (list[str]), ``timeline`` (list[dict]),
    and ``outcome`` (str).
    """
    pre_round: list[str] = []
    side = rd_detail.get("side", "")
    won = rd_detail.get("won", False)
    kills = rd_detail.get("kills", 0)
    damage = rd_detail.get("damage", 0)
    rnd = rd_detail.get("round", 0)

    is_pistol = (rnd <= 1 or (rnd >= 10 and rnd <= 11))
    if is_pistol:
        pre_round.append(TEMPLATES["pistol_round"])

    if econ_eval is not None:
        money = econ_eval.money
        action = econ_eval.actual_name
        optimal = econ_eval.optimal_name
        enemy_pred = econ_eval.enemy_buy_prediction

        if not econ_eval.is_optimal:
            if econ_eval.actual_action > econ_eval.optimal_action:
                pre_round.append(TEMPLATES["eco_over_buy"].format(
                    action=action, optimal=optimal, money=money))
            else:
                pre_round.append(TEMPLATES["eco_under_buy"].format(
                    action=action, optimal=optimal, money=money))
        else:
            pre_round.append(TEMPLATES["eco_correct"].format(
                action=action, money=money))

        if enemy_pred:
            pre_round.append(f"[Economy HMM] {enemy_pred}")

        # Enhanced economy notes
        if econ_eval.weapon_matchup_note:
            pre_round.append(f"[Weapon Matchup] {econ_eval.weapon_matchup_note}")
        if econ_eval.is_drop_or_pickup:
            pre_round.append(
                f"Picked up/received weapon — good economy play, saved money")
        if not econ_eval.team_can_fullbuy_next:
            pre_round.append(
                f"[Team Econ] Team avg ${econ_eval.team_avg_money} — "
                f"team cannot full buy next round")
        if econ_eval.upgrade_path_note:
            pre_round.append(f"[Upgrade] {econ_eval.upgrade_path_note}")

    if eng_eval is not None:
        if eng_eval.is_opening_kill:
            pre_round.append(TEMPLATES["opening_kill"])
        elif eng_eval.is_opening_death:
            pre_round.append(TEMPLATES["opening_death"])

        if eng_eval.clutch_win:
            pre_round.append(TEMPLATES["clutch_win"].format(
                n=eng_eval.clutch_attempted))
        elif eng_eval.clutch_attempted > 0:
            pre_round.append(TEMPLATES["clutch_loss"].format(
                n=eng_eval.clutch_attempted))
        elif eng_eval.multi_kill >= 2:
            pre_round.append(TEMPLATES["multi_kill"].format(
                kills=kills, damage=damage))

        if eng_eval.was_isolated_death:
            pre_round.append(TEMPLATES["isolated_death"])
        elif eng_eval.deaths > 0 and eng_eval.was_traded:
            pre_round.append(TEMPLATES["traded_death"])

        if kills == 0 and damage == 0:
            pre_round.append(TEMPLATES["zero_impact"])
        elif kills == 0 and 0 < damage < 50:
            pre_round.append(TEMPLATES["low_impact"].format(damage=damage))

    if nn_round is not None:
        wp = nn_round.get("win_prob")
        if wp is not None:
            p_win = wp if side == "T" else (1.0 - wp)
            if side == "CT":
                if p_win >= 0.60:
                    pre_round.append(TEMPLATES["nn_win_ct_favored"].format(p_win=p_win))
                elif p_win >= 0.40:
                    pre_round.append(TEMPLATES["nn_win_ct_slight"].format(p_win=p_win))
                else:
                    pre_round.append(TEMPLATES["nn_win_ct_underdog"].format(p_win=p_win))
            else:
                if p_win >= 0.60:
                    pre_round.append(TEMPLATES["nn_win_t_favored"].format(p_win=p_win))
                elif p_win >= 0.40:
                    pre_round.append(TEMPLATES["nn_win_t_slight"].format(p_win=p_win))
                else:
                    pre_round.append(TEMPLATES["nn_win_t_underdog"].format(p_win=p_win))

        attack = nn_round.get("attack_pred")
        if attack and side == "CT":
            a_prob = attack.get("A", 0)
            b_prob = attack.get("B", 0)
            no_plant = attack.get("no_plant", 0)

            if no_plant >= 0.50 and no_plant > a_prob and no_plant > b_prob:
                pre_round.append(
                    f"[Pre-Round Attack] T likely to eco / no plant "
                    f"({no_plant:.0%}) — hunt for picks, save util for next.")
            else:
                total_site = a_prob + b_prob
                if total_site > 0.01:
                    a_rel = a_prob / total_site
                    b_rel = b_prob / total_site
                else:
                    a_rel, b_rel = 0.5, 0.5

                if a_rel >= 0.65:
                    pre_round.append(TEMPLATES["nn_attack_a_likely"].format(
                        a_prob=a_rel, b_prob=b_rel))
                elif b_rel >= 0.65:
                    pre_round.append(TEMPLATES["nn_attack_b_likely"].format(
                        a_prob=a_rel, b_prob=b_rel))
                else:
                    pre_round.append(TEMPLATES["nn_attack_split"].format(
                        a_prob=a_rel, b_prob=b_rel))

        formation = nn_round.get("formation_pred")
        if formation and side == "T":
            sorted_fmts = sorted(
                ((k, v) for k, v in formation.items() if k != "other"),
                key=lambda x: x[1], reverse=True)
            other_prob = formation.get("other", 0)

            if sorted_fmts and sorted_fmts[0][1] > other_prob * 0.5:
                top_name, top_prob = sorted_fmts[0]
                advice_map = {
                    "2-1-2": "both sites have 2 defenders — mid control is key",
                    "1-1-3": "one site has 3 defenders — hit the weak site",
                    "1-2-2": "B/mid is stacked — A might be weak",
                    "2-2-1": "A/mid is stacked — B might be weak",
                    "3-1-1": "A is heavily stacked — hit B or mid-to-B",
                    "1-1-2": "light setup — fast execute can overwhelm a site",
                }
                advice = advice_map.get(top_name,
                    f"adapt your execute based on what you see")
                pre_round.append(TEMPLATES["nn_ct_formation"].format(
                    formation=top_name, fmt_prob=top_prob,
                    formation_advice=advice))
            else:
                pre_round.append(TEMPLATES["nn_ct_formation_uncertain"])

    timeline: list[dict] = []

    if lstm_preds:
        for lp in lstm_preds:
            probs = lp["probs"]
            a_p = probs.get("A", 0)
            b_p = probs.get("B", 0)
            interp = _interpret_lstm(probs, side)
            timeline.append({
                "time_sec": lp.get("time_sec", 0),
                "tick": lp.get("tick", 0),
                "source": "LSTM",
                "trigger": lp.get("trigger_desc", ""),
                "text": f"A={a_p:.0%} B={b_p:.0%} — {interp}",
            })

    if rl_timeline:
        for entry in rl_timeline:
            if entry.get("type") == "player_death":
                timeline.append({
                    "time_sec": entry["time_sec"],
                    "tick": entry.get("tick", 0),
                    "source": "EVENT",
                    "trigger": entry.get("trigger_desc", ""),
                    "text": entry.get("note", "You died"),
                })
            elif entry.get("type") == "rl_suggestion":
                blend = entry.get("blend", 0)
                timeline.append({
                    "time_sec": entry["time_sec"],
                    "tick": entry.get("tick", 0),
                    "source": "RL",
                    "trigger": entry.get("trigger_desc", ""),
                    "text": (f"{entry['raw_action']} (Q={blend:+.2f}) "
                             f"— {entry['action']} [{entry.get('state_desc', '')}]"),
                })

    if ct_formation_preds and side == "T":
        _CT_ADVICE = {
            "2-1-2": "balanced — mid control is key before committing",
            "1-2-2": "B/mid stacked — A might be weak",
            "2-2-1": "A/mid stacked — B might be weak",
            "3-1-1": "A heavily stacked — hit B or mid-to-B",
            "1-1-3": "B heavily stacked — hit A",
            "1-1-1": "spread thin — fast execute can overwhelm",
            "2-1-1": "A favored — consider B",
            "1-1-2": "B favored — consider A",
        }
        for fp in ct_formation_preds:
            fmt = fp.get("formation", "unknown")
            conf = fp.get("confidence", 0)
            advice = _CT_ADVICE.get(fmt, "adapt based on what you see")
            timeline.append({
                "time_sec": fp.get("time_sec", 0),
                "tick": fp.get("tick", 0),
                "source": "Formation",
                "trigger": fp.get("trigger_desc", ""),
                "text": f"CT running {fmt} ({conf:.0%}) — {advice}",
            })

    # Surface enemy utility events as trigger-only rows
    enemy_side = "T" if side == "CT" else "CT"
    util_types = {1, 2, 3, 5}
    if round_events:
        for ev in round_events:
            if ev.get("type_idx") not in util_types:
                continue
            if ev.get("_thrower_side") != enemy_side:
                continue
            timeline.append({
                "time_sec": ev.get("_time_sec", 0),
                "tick": ev.get("_tick", 0),
                "source": "Util",
                "trigger": ev.get("_desc", ""),
                "text": "",
            })

    # Scrub trigger descriptions that name own-side utility
    if round_events:
        own_util_keys = {
            (round(ev.get("_time_sec", 0), 1), ev.get("_desc", ""))
            for ev in round_events
            if ev.get("type_idx") in util_types
            and ev.get("_thrower_side") == side
        }
        for entry in timeline:
            key = (round(entry.get("time_sec", 0), 1),
                   entry.get("trigger", ""))
            if key in own_util_keys:
                entry["trigger"] = ""

    timeline.sort(key=lambda e: (e["time_sec"], e["tick"]))

    outcome = "Round won." if won else "Round lost."
    if kills > 0:
        outcome += f" You had {kills}K {damage}DMG."
    elif damage > 0:
        outcome += f" You dealt {damage}DMG but no kills."
    else:
        outcome += " No impact this round."

    if not pre_round and not timeline:
        pre_round.append(
            TEMPLATES["clean_win"] if won else TEMPLATES["tough_loss"])

    return {
        "pre_round": pre_round,
        "timeline": timeline,
        "outcome": outcome,
    }


ZONE_TO_IDX = {"A": 0, "B": 1, "MID": 2, "CT_BASE": 3, "T_BASE": 4}

_RL_LABELS = {
    ("CT", False): {
        "ROTATE_A": "Rotate towards A — info suggests pressure there",
        "ROTATE_B": "Rotate towards B — info suggests pressure there",
        "HOLD": "Hold your position and wait for T contact",
        "PUSH": "Push for information or an aggressive peek",
        "FALL_BACK": "Fall back to a crossfire position with a teammate",
    },
    ("CT", True): {
        "ROTATE_A": "Retake through A side",
        "ROTATE_B": "Retake through B side",
        "HOLD": "Hold the angle and wait for teammates to group up for retake",
        "PUSH": "Push the site immediately for retake",
        "FALL_BACK": "Consider saving your weapon for next round",
    },
    ("T", False): {
        "ROTATE_A": "Set up for an A execute",
        "ROTATE_B": "Set up for a B execute",
        "HOLD": "Play default — wait for a pick before committing",
        "PUSH": "Take aggressive map control (e.g. mid or an entry frag)",
        "FALL_BACK": "Pull back and regroup with your team",
    },
    ("T", True): {
        "ROTATE_A": "Watch for CT retake from A/connector side",
        "ROTATE_B": "Watch for CT retake from B/market side",
        "HOLD": "Hold your post-plant angle and let the bomb timer work",
        "PUSH": "Take an aggressive off-angle to catch rotating CTs",
        "FALL_BACK": "Play safe in cover — time is on your side",
    },
}


def _interpret_rl_action(raw_action: str, side: str,
                         bomb_planted: bool, player_zone: str,
                         bomb_site: str) -> str:
    """Convert a raw Q-learning action into side- and phase-aware advice."""
    labels = _RL_LABELS.get((side, bomb_planted),
                            _RL_LABELS[("CT", False)])
    base = labels.get(raw_action, raw_action)

    if bomb_planted and side == "CT" and raw_action == "FALL_BACK":
        team_alive_hint = ""
        base = "Consider saving your weapon for next round"
    elif bomb_planted and side == "T" and raw_action == "HOLD":
        if bomb_site and player_zone == bomb_site:
            base = f"Hold your post-plant position on {bomb_site} site"
        elif bomb_site:
            base = f"Bomb is on {bomb_site} — rotate to help hold the site"

    return base


def _get_rl_suggestion(rd, target_player: str, q_learner) -> Optional[dict]:
    """Get Q-learning recommendations for the player at a mid-round snapshot."""
    p = rd.get_player(target_player)
    if p is None or q_learner is None:
        return None
    if not p.alive_at_end and p.death_tick is not None:
        if p.death_tick <= rd.tick_freeze_end:
            return None

    side = p.side

    death_or_end = p.death_tick if (p.death_tick and not p.alive_at_end) else rd.tick_end
    decision_tick = (rd.tick_freeze_end + death_or_end) // 2

    def _alive_at(players, tick):
        return sum(1 for pl in players
                   if pl.death_tick is None or pl.death_tick > tick)

    t_alive = max(_alive_at(rd.t_players, decision_tick), 1)
    ct_alive = max(_alive_at(rd.ct_players, decision_tick), 1)

    bomb_planted = False
    bomb_site = ""
    bomb_status = 0
    if rd.bomb_planted:
        plant_tick = None
        for ev in rd.events:
            if ev.event_type == "bomb_plant":
                plant_tick = ev.tick
                break
        if plant_tick and plant_tick <= decision_tick:
            bomb_planted = True
            resolved = _resolve_bomb_site(rd)
            bomb_site = resolved if resolved in ("A", "B") else ""
            bomb_status = 1 if resolved == "A" else 2

    elapsed = (decision_tick - rd.tick_freeze_end) / 64.0
    if bomb_status > 0:
        time_bucket = 3
    elif elapsed <= 30:
        time_bucket = 0
    elif elapsed <= 60:
        time_bucket = 1
    else:
        time_bucket = 2

    player_zone = "MID"
    if p.positions:
        pos_step = 2 * 64
        dec_idx = round((decision_tick - rd.tick_freeze_end) / pos_step)
        dec_idx = max(0, min(dec_idx, len(p.positions) - 1))
        z = get_zone(p.positions[dec_idx].x, p.positions[dec_idx].y)
        if z in ZONE_TO_IDX:
            player_zone = z
    zone_idx = ZONE_TO_IDX.get(player_zone, 2)

    if side == "CT":
        my_alive, enemy_alive = ct_alive, t_alive
    else:
        my_alive, enemy_alive = t_alive, ct_alive

    try:
        best_action, q_vals = q_learner.recommend(
            t_alive, ct_alive, bomb_status, time_bucket, zone_idx, side)

        raw_action = ACTION_NAMES[best_action]
        action_text = _interpret_rl_action(
            raw_action, side, bomb_planted, player_zone, bomb_site)

        phase = "post-plant" if bomb_planted else (
            "early round" if time_bucket == 0 else
            "mid-round" if time_bucket == 1 else "late round")
        bomb_desc = f", bomb on {bomb_site}" if bomb_site else ""
        state_desc = (f"{phase}, {my_alive} teammates vs {enemy_alive} enemies"
                      f"{bomb_desc}, you at {player_zone}")

        return {"action": action_text, "state_desc": state_desc,
                "q_values": q_vals}
    except Exception:
        return None


_V2_RECENT_LABELS = ["none", "teammate died", "enemy killed",
                     "grenade thrown", "bomb planted"]


def _get_rl_v2_suggestion(rd, target_player: str, q_v2) -> Optional[dict]:
    """Get Q-learning v2 micro-decision recommendation."""
    p = rd.get_player(target_player)
    if p is None or q_v2 is None:
        return None
    if not p.alive_at_end and p.death_tick is not None:
        if p.death_tick <= rd.tick_freeze_end:
            return None

    side = p.side
    side_idx = 0 if side == "T" else 1

    death_or_end = p.death_tick if (p.death_tick and not p.alive_at_end) else rd.tick_end
    decision_tick = (rd.tick_freeze_end + death_or_end) // 2

    def _alive_at(players, tick):
        return sum(1 for pl in players
                   if pl.death_tick is None or pl.death_tick > tick)

    t_alive = max(_alive_at(rd.t_players, decision_tick), 1)
    ct_alive = max(_alive_at(rd.ct_players, decision_tick), 1)

    my_alive = t_alive if side == "T" else ct_alive
    enemy_alive = ct_alive if side == "T" else t_alive
    alive_adv = max(-3, min(3, my_alive - enemy_alive))

    bomb_planted = False
    bomb_site = ""
    bomb_status = 0
    if rd.bomb_planted:
        for ev in rd.events:
            if ev.event_type == "bomb_plant" and ev.tick <= decision_tick:
                bomb_planted = True
                resolved = _resolve_bomb_site(rd)
                bomb_site = resolved if resolved in ("A", "B") else ""
                bomb_status = 1 if resolved == "A" else 2
                break

    elapsed = (decision_tick - rd.tick_freeze_end) / 64.0
    time_bucket = (3 if bomb_status > 0 else
                   0 if elapsed <= 30 else
                   1 if elapsed <= 60 else 2)

    player_zone = "MID"
    if p.positions:
        pos_step = 2 * 64
        dec_idx = round((decision_tick - rd.tick_freeze_end) / pos_step)
        dec_idx = max(0, min(dec_idx, len(p.positions) - 1))
        z = get_zone(p.positions[dec_idx].x, p.positions[dec_idx].y)
        if z in ZONE_TO_IDX:
            player_zone = z
    zone_idx = ZONE_TO_IDX.get(player_zone, 2)

    recent = 0
    lookback = 5 * 64
    for ev in reversed(rd.events):
        if ev.tick > decision_tick:
            continue
        if ev.tick < decision_tick - lookback:
            break
        if ev.event_type == "bomb_plant":
            recent = 4; break
        if ev.event_type == "kill":
            victim_side = ev.data.get("victim_side", "")
            if not victim_side:
                victim = ev.data.get("victim", "")
                for pl in (rd.t_players + rd.ct_players):
                    if pl.name == victim:
                        victim_side = pl.side; break
            if victim_side == side:
                recent = 1; break
            else:
                recent = 2; break
        if ev.event_type in ("smoke", "flash", "he_grenade"):
            recent = 3; break

    try:
        from tactical_rl import _v2_state_index

        best, blended, detail = q_v2.recommend(
            side_idx, alive_adv, bomb_status, time_bucket, zone_idx, recent)

        raw_action = ACTION_NAMES_V2[best]
        best_detail = detail[raw_action]

        v2_labels = {
            ("CT", False): {
                "PEEK": "Push for info or an aggressive peek",
                "HOLD": "Hold your position and wait for contact",
                "TRADE": "Move to refrag a fallen teammate",
                "FALL_BACK": "Fall back to a safer crossfire angle",
                "UTILITY": "Use utility (flash/smoke) to control space",
                "ROTATE": "Rotate toward the active site",
            },
            ("CT", True): {
                "PEEK": "Peek aggressively for the retake",
                "HOLD": "Wait for teammates to group for retake",
                "TRADE": "Push to trade the entry fragger",
                "FALL_BACK": "Save your weapon for next round",
                "UTILITY": "Smoke/flash the site for retake",
                "ROTATE": "Rotate to the bomb site",
            },
            ("T", False): {
                "PEEK": "Take an aggressive entry peek",
                "HOLD": "Play default and wait for a pick",
                "TRADE": "Follow up on teammate's contact for a trade",
                "FALL_BACK": "Pull back and regroup",
                "UTILITY": "Use utility before committing to a site",
                "ROTATE": "Shift to the other site",
            },
            ("T", True): {
                "PEEK": "Take an off-angle to catch retaking CTs",
                "HOLD": "Hold your post-plant position",
                "TRADE": "Push to trade — time is on your side",
                "FALL_BACK": "Play safe in cover, let the timer work",
                "UTILITY": "Delay the retake with utility",
                "ROTATE": "Rotate to watch a different approach",
            },
        }
        labels = v2_labels.get((side, bomb_planted), v2_labels[("CT", False)])
        action_text = labels.get(raw_action, raw_action)

        phase = "post-plant" if bomb_planted else (
            "early" if time_bucket == 0 else
            "mid-round" if time_bucket == 1 else "late")
        adv_text = (f"{my_alive}v{enemy_alive}"
                    f" ({'advantage' if alive_adv > 0 else 'disadvantage' if alive_adv < 0 else 'even'})")
        recent_text = _V2_RECENT_LABELS[recent]
        bomb_desc = f", bomb {bomb_site}" if bomb_site else ""
        state_desc = (f"{phase}, {adv_text}{bomb_desc}, "
                      f"you at {player_zone}, recent: {recent_text}")

        return {
            "action": action_text,
            "state_desc": state_desc,
            "raw_action": raw_action,
            "kill_q": best_detail["kill_q"],
            "win_q": best_detail["win_q"],
            "blend": best_detail["blend"],
            "detail": detail,
        }
    except Exception:
        return None


_V2_LABELS = {
    ("CT", False): {
        "PEEK": "Push for info or an aggressive peek",
        "HOLD": "Hold your position and wait for contact",
        "TRADE": "Move to refrag a fallen teammate",
        "FALL_BACK": "Fall back to a safer crossfire angle",
        "UTILITY": "Use utility (flash/smoke) to control space",
        "ROTATE": "Rotate toward the active site",
    },
    ("CT", True): {
        "PEEK": "Peek aggressively for the retake",
        "HOLD": "Wait for teammates to group for retake",
        "TRADE": "Push to trade the entry fragger",
        "FALL_BACK": "Save your weapon for next round",
        "UTILITY": "Smoke/flash the site for retake",
        "ROTATE": "Rotate to the bomb site",
    },
    ("T", False): {
        "PEEK": "Take an aggressive entry peek",
        "HOLD": "Play default and wait for a pick",
        "TRADE": "Follow up on teammate's contact for a trade",
        "FALL_BACK": "Pull back and regroup",
        "UTILITY": "Use utility before committing to a site",
        "ROTATE": "Shift to the other site",
    },
    ("T", True): {
        "PEEK": "Take an off-angle to catch retaking CTs",
        "HOLD": "Hold your post-plant position",
        "TRADE": "Push to trade — time is on your side",
        "FALL_BACK": "Play safe in cover, let the timer work",
        "UTILITY": "Delay the retake with utility",
        "ROTATE": "Rotate to watch a different approach",
    },
}


def _get_rl_v2_timeline(rd, target_player: str, q_v2, events=None) -> list[dict]:
    """Query Q-learning v2 at every event tick; emit only when the action changes."""
    p = rd.get_player(target_player)
    if p is None or q_v2 is None:
        return []

    if events is None:
        events = _build_round_events(rd)
    if not events:
        return []

    side = p.side
    side_idx = 0 if side == "T" else 1
    t_names = {pl.name for pl in rd.t_players}
    ct_names = {pl.name for pl in rd.ct_players}

    def _alive_at(players, tick):
        return sum(1 for pl in players
                   if pl.death_tick is None or pl.death_tick > tick)

    player_dead = False
    prev_action = None
    timeline = []
    bomb_planted = False
    bomb_site = ""

    for ev in events:
        tick = ev.get("_tick", 0)
        time_sec = ev.get("_time_sec", 0)
        desc = ev.get("_desc", "")

        if ev.get("_victim") == target_player:
            player_dead = True
            timeline.append({
                "time_sec": time_sec,
                "tick": tick,
                "trigger_desc": desc,
                "type": "player_death",
                "note": "You died — round over for you",
            })
            break

        if ev.get("type_idx") == 4:
            bomb_planted = True
            bs = _resolve_bomb_site(rd)
            bomb_site = bs if bs in ("A", "B") else ""

        t_alive = max(_alive_at(rd.t_players, tick), 1)
        ct_alive = max(_alive_at(rd.ct_players, tick), 1)
        my_alive = t_alive if side == "T" else ct_alive
        enemy_alive = ct_alive if side == "T" else t_alive
        alive_adv = max(-3, min(3, my_alive - enemy_alive))

        bomb_status = 0
        if bomb_planted:
            bomb_status = 1 if bomb_site == "A" else 2 if bomb_site == "B" else 0

        elapsed = time_sec
        time_bucket = (3 if bomb_status > 0 else
                       0 if elapsed <= 30 else
                       1 if elapsed <= 60 else 2)

        player_zone = "MID"
        if p.positions:
            pos_step = 2 * 64
            idx = round((tick - rd.tick_freeze_end) / pos_step)
            idx = max(0, min(idx, len(p.positions) - 1))
            z = get_zone(p.positions[idx].x, p.positions[idx].y)
            if z in ZONE_TO_IDX:
                player_zone = z
        zone_idx = ZONE_TO_IDX.get(player_zone, 2)

        recent_type = ev.get("type_idx", 0)
        if recent_type == 0:
            victim = ev.get("_victim", "")
            if victim in (t_names if side == "T" else ct_names):
                recent = 1
            else:
                recent = 2
        elif recent_type in (1, 2, 3):
            recent = 3
        elif recent_type == 4:
            recent = 4
        else:
            recent = 0

        try:
            best, blended, detail = q_v2.recommend(
                side_idx, alive_adv, bomb_status, time_bucket, zone_idx, recent)
            raw_action = ACTION_NAMES_V2[best]
        except Exception:
            continue

        if raw_action == prev_action:
            continue

        prev_action = raw_action
        labels = _V2_LABELS.get((side, bomb_planted), _V2_LABELS[("CT", False)])
        action_text = labels.get(raw_action, raw_action)

        best_detail = detail.get(raw_action, {})
        blend_val = best_detail.get("blend", 0)

        phase = "post-plant" if bomb_planted else (
            "early" if time_bucket == 0 else
            "mid-round" if time_bucket == 1 else "late")

        timeline.append({
            "time_sec": time_sec,
            "tick": tick,
            "trigger_desc": desc,
            "type": "rl_suggestion",
            "raw_action": raw_action,
            "action": action_text,
            "blend": blend_val,
            "state_desc": (f"{phase}, {my_alive}v{enemy_alive}, "
                           f"you at {player_zone}"),
        })

    return timeline


_SS_LABELS = {
    ("CT", False): {
        "HOLD": "Hold your position and wait for contact",
        "ROTATE": "Rotate toward the active site",
        "RETAKE": "Set up for a retake",
        "PUSH": "Push for info or an aggressive peek",
        "FALL_BACK": "Fall back to a safer crossfire angle",
        "UTILITY": "Use utility (flash/smoke) to control space",
    },
    ("CT", True): {
        "HOLD": "Wait for teammates to group for retake",
        "ROTATE": "Rotate to the bomb site",
        "RETAKE": "Group for retake — push with utility",
        "PUSH": "Peek aggressively for the retake",
        "FALL_BACK": "Save your weapon for next round",
        "UTILITY": "Smoke/flash the site for retake",
    },
    ("T", False): {
        "EXECUTE": "Execute onto the site — go now",
        "PEEK": "Take an aggressive entry peek",
        "TRADE": "Follow up on teammate's contact for a trade",
        "FALL_BACK": "Pull back and regroup",
        "UTILITY": "Use utility before committing to a site",
        "LURK": "Lurk and play for a late-round pick",
    },
    ("T", True): {
        "EXECUTE": "Push to secure post-plant control",
        "PEEK": "Take an off-angle to catch retaking CTs",
        "TRADE": "Push to trade — time is on your side",
        "FALL_BACK": "Play safe in cover, let the timer work",
        "UTILITY": "Delay the retake with utility",
        "LURK": "Lurk and watch the flank",
    },
}


def _get_ss_rl_timeline(rd, target_player: str, ql_model,
                        events=None) -> list[dict]:
    """Query side-specific Q-learner at every event tick; emit only when the action changes."""
    p = rd.get_player(target_player)
    if p is None or ql_model is None or not ql_model.trained:
        return []

    if events is None:
        events = _build_round_events(rd)
    if not events:
        return []

    side = p.side
    t_names = {pl.name for pl in rd.t_players}
    ct_names = {pl.name for pl in rd.ct_players}

    def _alive_at(players, tick):
        return sum(1 for pl in players
                   if pl.death_tick is None or pl.death_tick > tick)

    def _team_support_at(rd, target_player, side, tick, pos_step=2*64,
                         freeze_end=0):
        my_zone = None
        p = rd.get_player(target_player)
        if p and p.positions:
            idx = round((tick - freeze_end) / pos_step)
            idx = max(0, min(idx, len(p.positions) - 1))
            my_zone = get_zone(p.positions[idx].x, p.positions[idx].y)

        teammates = rd.ct_players if side == "CT" else rd.t_players
        nearby = 0
        for tm in teammates:
            if tm.name == target_player:
                continue
            if tm.death_tick is not None and tm.death_tick <= tick:
                continue
            if tm.positions:
                ti = round((tick - freeze_end) / pos_step)
                ti = max(0, min(ti, len(tm.positions) - 1))
                tz = get_zone(tm.positions[ti].x, tm.positions[ti].y)
                if tz == my_zone:
                    nearby += 1
        return min(nearby, 2)

    player_dead = False
    prev_action = None
    timeline = []
    bomb_planted = False
    bomb_site = ""

    for ev in events:
        tick = ev.get("_tick", 0)
        time_sec = ev.get("_time_sec", 0)
        desc = ev.get("_desc", "")

        if ev.get("_victim") == target_player:
            player_dead = True
            timeline.append({
                "time_sec": time_sec, "tick": tick,
                "trigger_desc": desc, "type": "player_death",
                "note": "You died — round over for you",
            })
            break

        if ev.get("type_idx") == 4:
            bomb_planted = True
            bs = _resolve_bomb_site(rd)
            bomb_site = bs if bs in ("A", "B") else ""

        t_alive = max(_alive_at(rd.t_players, tick), 1)
        ct_alive = max(_alive_at(rd.ct_players, tick), 1)
        my_alive = t_alive if side == "T" else ct_alive
        enemy_alive = ct_alive if side == "T" else t_alive
        alive_adv = max(-3, min(3, my_alive - enemy_alive))

        bomb_status = 0
        if bomb_planted:
            bomb_status = 1 if bomb_site == "A" else 2 if bomb_site == "B" else 0

        elapsed = time_sec
        time_bucket = (3 if bomb_status > 0 else
                       0 if elapsed <= 30 else
                       1 if elapsed <= 60 else 2)

        player_zone = "MID"
        if p.positions:
            pos_step = 2 * 64
            idx = round((tick - rd.tick_freeze_end) / pos_step)
            idx = max(0, min(idx, len(p.positions) - 1))
            z = get_zone(p.positions[idx].x, p.positions[idx].y)
            if z in ZONE_TO_IDX:
                player_zone = z
        zone_idx = ZONE_TO_IDX.get(player_zone, 2)

        recent_type = ev.get("type_idx", 0)
        if recent_type == 0:
            victim = ev.get("_victim", "")
            if victim in (t_names if side == "T" else ct_names):
                recent = 1
            else:
                recent = 2
        elif recent_type in (1, 2, 3):
            recent = 3
        elif recent_type == 4:
            recent = 4
        else:
            recent = 0

        team_support = _team_support_at(
            rd, target_player, side, tick,
            freeze_end=rd.tick_freeze_end)

        try:
            best, blended, detail = ql_model.recommend(
                alive_adv, bomb_status, time_bucket, zone_idx, recent,
                team_support)
            raw_action = ql_model.action_names[best]
        except Exception:
            continue

        if raw_action == prev_action:
            continue

        prev_action = raw_action
        labels = _SS_LABELS.get((side, bomb_planted), _SS_LABELS[("CT", False)])
        action_text = labels.get(raw_action, raw_action)

        best_detail = detail.get(raw_action, {})
        blend_val = best_detail.get("blend", 0)

        phase = "post-plant" if bomb_planted else (
            "early" if time_bucket == 0 else
            "mid-round" if time_bucket == 1 else "late")

        timeline.append({
            "time_sec": time_sec, "tick": tick,
            "trigger_desc": desc, "type": "rl_suggestion",
            "raw_action": raw_action, "action": action_text,
            "blend": blend_val,
            "state_desc": (f"{phase}, {my_alive}v{enemy_alive}, "
                           f"you at {player_zone}"),
        })

    return timeline


def _get_ct_formation_predictions(rd, fc_ct_model, events=None,
                                  prior: list[float] | None = None) -> list[dict]:
    """Run FormationClassifier_CT at event checkpoints; emit on delta.

    ``prior`` is the pre-round formation distribution used to seed the LSTM.
    """
    if fc_ct_model is None or not fc_ct_model.trained:
        return []

    if events is None:
        events = _build_round_events(rd)
    if len(events) < 2:
        return []

    def _alive_at(players, tick):
        return sum(1 for pl in players
                   if pl.death_tick is None or pl.death_tick > tick)

    DELTA_THRESHOLD = 0.10
    results = []
    prev_formation = ""

    for i in range(1, len(events) + 1):
        sub_events = events[:i]
        clean = [{k: v for k, v in e.items() if not k.startswith("_")}
                 for e in sub_events]
        ct_alive_list = []
        for e in sub_events:
            tick = e.get("_tick", 0)
            ca = max(_alive_at(rd.ct_players, tick), 1)
            ct_alive_list.append(ca)

        pred = fc_ct_model.predict_readable(clean, ct_alive_list,
                                            prior=prior)
        formation = pred.get("formation", "unknown")
        confidence = pred.get("confidence", 0)

        if formation != prev_formation or i == len(events):
            trigger = events[i - 1]
            results.append({
                "n_events": i,
                "time_sec": trigger.get("_time_sec", 0),
                "tick": trigger.get("_tick", 0),
                "formation": formation,
                "confidence": confidence,
                "ct_alive": pred.get("ct_alive", 5),
                "trigger_desc": trigger.get("_desc", ""),
            })
            prev_formation = formation

    return results


_ZONE_IDX_MAP = {"A": 0, "B": 1, "MID": 2, "CT_BASE": 3, "T_BASE": 4}
_IDX_ZONE_MAP = {v: k for k, v in _ZONE_IDX_MAP.items()}
_UTIL_NAMES = {"smoke": "smoke", "flash": "flash", "he_grenade": "HE",
               "molotov": "molotov"}


def _build_round_events(rd) -> list[dict]:
    """Extract significant events with LSTM-compatible encoding and a readable description."""
    t_names = {p.name for p in rd.t_players}
    ct_names = {p.name for p in rd.ct_players}
    events = []

    for ev in rd.events:
        if ev.event_type == "kill":
            attacker = ev.data.get("attacker", "")
            victim = ev.data.get("victim", "")
            victim_side = "T" if victim in t_names else "CT" if victim in ct_names else ""
            attacker_side = "T" if attacker in t_names else "CT" if attacker in ct_names else ""
            actor_is_t = 1 if attacker_side == "T" else 0
            vx, vy = ev.data.get("victim_x", 0), ev.data.get("victim_y", 0)
            zone_str = get_zone(vx, vy) if vx else "MID"
            zone_clean = zone_str if zone_str in _ZONE_IDX_MAP else "MID"
            weapon = ev.data.get("weapon", "")
            desc = f"{attacker_side} kills {victim_side} @ {zone_clean}"
            if weapon:
                desc += f" ({weapon})"
            events.append({
                "type_idx": 0,
                "actor_side_is_t": actor_is_t,
                "zone_idx": _ZONE_IDX_MAP.get(zone_clean, 2),
                "time_norm": min(ev.time_in_round / 120.0, 1.0) if ev.time_in_round else 0,
                "is_headshot": int(ev.data.get("headshot", False)),
                "_time_sec": ev.time_in_round or 0,
                "_tick": ev.tick,
                "_desc": desc,
                "_victim": victim,
                "_attacker": attacker,
            })

        elif ev.event_type in ("smoke", "flash", "he_grenade", "molotov"):
            thrower = ev.data.get("thrower", "")
            thrower_side = "T" if thrower in t_names else "CT" if thrower in ct_names else "?"
            thrower_is_t = 1 if thrower_side == "T" else 0
            gx, gy = ev.data.get("x", 0), ev.data.get("y", 0)
            zone_str = get_zone(gx, gy) if gx else "MID"
            zone_clean = zone_str if zone_str in _ZONE_IDX_MAP else "MID"
            type_map = {"smoke": 1, "flash": 2, "he_grenade": 3, "molotov": 5}
            util_name = _UTIL_NAMES.get(ev.event_type, ev.event_type)
            desc = f"{thrower_side} {util_name} @ {zone_clean}"
            events.append({
                "type_idx": type_map.get(ev.event_type, 1),
                "actor_side_is_t": thrower_is_t,
                "zone_idx": _ZONE_IDX_MAP.get(zone_clean, 2),
                "time_norm": min(ev.time_in_round / 120.0, 1.0) if ev.time_in_round else 0,
                "is_headshot": 0,
                "_time_sec": ev.time_in_round or 0,
                "_tick": ev.tick,
                "_desc": desc,
                "_thrower_side": thrower_side,
                "_util_name": util_name,
            })

        elif ev.event_type == "bomb_plant":
            site = _resolve_bomb_site(rd)
            site_clean = site if site in ("A", "B") else "?"
            desc = f"Bomb planted {site_clean}"
            events.append({
                "type_idx": 4,
                "actor_side_is_t": 1,
                "zone_idx": _ZONE_IDX_MAP.get(site, 2),
                "time_norm": min(ev.time_in_round / 120.0, 1.0) if ev.time_in_round else 0,
                "is_headshot": 0,
                "_time_sec": ev.time_in_round or 0,
                "_tick": ev.tick,
                "_desc": desc,
            })

    return events


def _get_lstm_predictions(rd, lstm_model, events=None) -> list[dict]:
    """Run LSTM predictions after every significant event with delta filtering."""
    if lstm_model is None or not lstm_model.trained:
        return []

    if events is None:
        events = _build_round_events(rd)

    if len(events) < 2:
        return []

    DELTA_THRESHOLD = 0.10
    results = []
    prev_probs = {"A": 0.0, "B": 0.0, "no_plant": 1.0}

    for i in range(1, len(events) + 1):
        sub = events[:i]
        clean = [{k: v for k, v in e.items() if not k.startswith("_")}
                 for e in sub]
        pred = lstm_model.predict(clean)

        delta = max(abs(pred.get(k, 0) - prev_probs.get(k, 0))
                    for k in ("A", "B", "no_plant"))

        if delta >= DELTA_THRESHOLD or i == len(events):
            trigger = events[i - 1]
            results.append({
                "n_events": i,
                "time_sec": trigger.get("_time_sec", 0),
                "tick": trigger.get("_tick", 0),
                "probs": pred,
                "trigger_desc": trigger.get("_desc", ""),
            })
            prev_probs = dict(pred)

    return results


def _build_round_details(
    match: MatchData,
    econ_evals: list,
    eng_evals: list,
    nn_preds: Optional[dict] = None,
    q_learner=None,
    q_v2=None,
    lstm_model=None,
    ql_t=None,
    ql_ct=None,
    fc_ct_model=None,
) -> list[dict]:
    """Combine per-round info from all modules into unified round details."""
    econ_by_round = {e.round_num: e for e in econ_evals}
    eng_by_round = {e.round_num: e for e in eng_evals}

    nn_by_round: dict[int, dict] = {}
    if nn_preds and nn_preds.get("available"):
        for f in nn_preds.get("formation_predictions", []):
            nn_by_round.setdefault(f["round"], {})["formation_pred"] = f["probs"]
        for a in nn_preds.get("attack_predictions", []):
            nn_by_round.setdefault(a["round"], {})["attack_pred"] = a["probs"]

    details = []
    for rd in match.rounds:
        p = rd.get_player(match.target_player)
        if p is None:
            continue

        entry = {
            "round": rd.round_num,
            "side": p.side,
            "winner": rd.winner,
            "won": rd.winner == p.side,
            "kills": p.kills,
            "deaths": p.deaths,
            "damage": p.damage,
        }

        ec = econ_by_round.get(rd.round_num)
        if ec:
            econ_detail = {
                "money": ec.money,
                "action": ec.actual_name,
                "optimal": ec.optimal_name,
                "is_optimal": ec.is_optimal,
                "enemy_prediction": ec.enemy_buy_prediction,
                "enemy_tier": ec.enemy_predicted_tier,
                "enemy_money": ec.enemy_predicted_money,
                "weapon_matchup": ec.weapon_matchup_note,
                "team_avg_money": ec.team_avg_money,
                "is_drop": ec.is_drop_or_pickup,
            }
            if ec.posthoc:
                econ_detail["posthoc"] = {
                    "weapon_tier": ec.posthoc.weapon_tier,
                    "weapon_appropriate": ec.posthoc.weapon_appropriate,
                    "utility_level": ec.posthoc.utility_level,
                    "utility_sufficient": ec.posthoc.utility_sufficient,
                    "has_armor": ec.posthoc.has_armor,
                    "has_helmet": ec.posthoc.has_helmet,
                    "has_kit": ec.posthoc.has_kit,
                    "kit_note": ec.posthoc.kit_note,
                    "waste": ec.posthoc.waste,
                    "waste_note": ec.posthoc.waste_note,
                }
            entry["economy"] = econ_detail

        en = eng_by_round.get(rd.round_num)
        if en:
            entry["engagement"] = {
                "opening_kill": en.is_opening_kill,
                "opening_death": en.is_opening_death,
                "isolated_death": en.was_isolated_death,
                "multi_kill": en.multi_kill,
                "clutch": en.clutch_win,
                "notes": en.notes,
            }

        nn_round = nn_by_round.get(rd.round_num)

        round_events = _build_round_events(rd)

        # Enforce round termination: filter past team elimination or player death
        filtered_events = []
        for ev in round_events:
            filtered_events.append(ev)
            if ev.get("type_idx") == 0:
                tick = ev.get("_tick", 0)
                def _alive_at_tick(players, t):
                    return sum(1 for pl in players
                               if pl.death_tick is None or pl.death_tick > t)
                ta = _alive_at_tick(rd.t_players, tick)
                ca = _alive_at_tick(rd.ct_players, tick)
                if ta == 0 or ca == 0:
                    break
                if ev.get("_victim") == match.target_player:
                    break
        round_events = filtered_events

        # Side-aware model routing
        lstm_preds = []
        if p.side == "CT" and lstm_model is not None:
            lstm_preds = _get_lstm_predictions(
                rd, lstm_model, events=round_events)

        ct_fmt_preds = None
        if p.side == "T" and fc_ct_model is not None:
            from strategy_nn import FORMATION_CLASSES
            nn_round_data = nn_by_round.get(rd.round_num, {})
            fmt_pred = nn_round_data.get("formation_pred")
            ct_prior = None
            if fmt_pred:
                ct_prior = [fmt_pred.get(c, 0.0) for c in FORMATION_CLASSES]
            ct_fmt_preds = _get_ct_formation_predictions(
                rd, fc_ct_model, events=round_events, prior=ct_prior)

        # RL timeline: prefer side-specific, fall back to V2, then V1
        rl_tl = []
        rl_suggestion = None
        rl_v2_suggestion = None

        side_ql = ql_t if p.side == "T" else ql_ct
        if side_ql is not None and side_ql.trained:
            rl_tl = _get_ss_rl_timeline(
                rd, match.target_player, side_ql, events=round_events)
        elif q_v2 is not None:
            rl_tl = _get_rl_v2_timeline(
                rd, match.target_player, q_v2, events=round_events)

        if not rl_tl:
            if q_v2 is not None:
                rl_v2_suggestion = _get_rl_v2_suggestion(
                    rd, match.target_player, q_v2)
            if rl_v2_suggestion is None:
                rl_suggestion = _get_rl_suggestion(
                    rd, match.target_player, q_learner)

        entry["suggestions"] = _suggest_for_round(
            entry, ec, en, None, nn_round, rl_suggestion,
            rl_v2_suggestion, lstm_preds, rl_tl, ct_fmt_preds,
            round_events=round_events)

        details.append(entry)

    return details


def generate_full_report(
    demo_path: str,
    target_player: str,
    models_dir: str = "models",
) -> CoachingReport:
    """Generate a complete coaching report for one demo file.

    Args:
        demo_path: Path to a .dem file.
        target_player: Exact in-game name.
        models_dir: Directory containing saved NN weights and Q-table.
    """
    t0 = time.time()

    print(f"[1/5] Parsing demo: {demo_path}")
    match = parse_demo(demo_path, target_player)
    print(f"       Map: {match.map_name}, Rounds: {len(match.rounds)}, "
          f"Score: T {match.t_score} - {match.ct_score} CT")

    print("[2/5] Running economy analysis (HMM + MDP)...")
    hmm_preds = predict_enemy_economy(match.rounds, target_player)
    print(f"       Economy HMM: {len(hmm_preds)} rounds predicted")
    econ_evals = evaluate_player_economy(match.rounds, target_player, hmm_preds)
    econ_sum = economy_summary(econ_evals)
    print(f"       Economy accuracy: "
          f"{econ_sum.get('overall_accuracy', 0):.0%} optimal")

    print("[3/5] Running engagement analysis...")
    eng_evals = analyze_engagement(match.rounds, target_player)
    eng_sum = engagement_summary(eng_evals)
    print(f"       K/D: {eng_sum.get('kd_ratio', 0):.2f}, "
          f"ADR: {eng_sum.get('adr', 0):.0f}")

    print("[4/5] Loading NN/LSTM models...")
    nn_preds = {"available": False}
    models = {}

    if _NN_AVAILABLE:
        try:
            models = load_models(models_dir)
            print("       NN models loaded — running predictions...")
            nn_preds = _run_nn_predictions(match, models, hmm_preds)
        except Exception as e:
            print(f"       NN models not available: {e}")

    # Load Q-learners (side-specific preferred, v2 fallback, v1 last)
    q_learner = None
    q_v2 = None
    ql_t = None
    ql_ct = None
    if _RL_AVAILABLE:
        try:
            t_path = os.path.join(models_dir, "tactical_ql_t.npz")
            ct_path = os.path.join(models_dir, "tactical_ql_ct.npz")
            if os.path.exists(t_path) and os.path.exists(ct_path):
                ql_t = TacticalQLearner_T()
                ql_t.load(t_path)
                ql_ct = TacticalQLearner_CT()
                ql_ct.load(ct_path)
                print("       Side-specific Q-learners loaded (T + CT).")
            else:
                q2_path = os.path.join(models_dir, "tactical_ql_v2.npz")
                if os.path.exists(q2_path):
                    q2 = TacticalQLearnerV2()
                    q2.load(q2_path)
                    q_v2 = q2
                    print("       Q-learner v2 loaded (dual reward: kill + win).")
                else:
                    ql = TacticalQLearner()
                    q1_path = os.path.join(models_dir, "tactical_ql.npz")
                    if os.path.exists(q1_path):
                        ql.load(q1_path)
                        q_learner = ql
                        print("       Q-learner v1 loaded (fallback).")
                    else:
                        print("       No Q-table found, skipping RL suggestions.")
        except Exception as e:
            print(f"       Q-learner not available: {e}")

    lstm_model = models.get("formation_classifier_t") if _NN_AVAILABLE else None
    if lstm_model is None:
        lstm_model = models.get("event_sequence_predictor") if _NN_AVAILABLE else None
    if lstm_model and lstm_model.trained:
        print("       FormationClassifier_T (LSTM) loaded.")
    else:
        lstm_model = None

    fc_ct_model = models.get("formation_classifier_ct") if _NN_AVAILABLE else None
    if fc_ct_model and fc_ct_model.trained:
        print("       FormationClassifier_CT (LSTM, alive-aware) loaded.")
    else:
        fc_ct_model = None

    print("[5/5] Generating coaching tips...")
    round_details = _build_round_details(
        match, econ_evals, eng_evals, nn_preds,
        q_learner, q_v2, lstm_model,
        ql_t=ql_t, ql_ct=ql_ct, fc_ct_model=fc_ct_model)
    tips = _generate_tips(econ_sum, eng_sum, None)

    elapsed = time.time() - t0

    report = CoachingReport(
        demo_file=demo_path,
        player_name=target_player,
        map_name=match.map_name,
        match_score=f"T {match.t_score} - {match.ct_score} CT",
        total_rounds=len(match.rounds),
        economy=econ_sum,
        engagement=eng_sum,
        game_sense={},
        nn_predictions=nn_preds,
        round_details=round_details,
        coaching_tips=tips,
        generation_time_sec=round(elapsed, 2),
    )

    return report


def print_report(report: CoachingReport) -> None:
    """Print a human-readable coaching report to stdout."""

    print()
    print("=" * 70)
    print("  CS2 AI COACHING REPORT")
    print("=" * 70)
    print(f"  Demo:     {report.demo_file}")
    print(f"  Player:   {report.player_name}")
    print(f"  Map:      {report.map_name}")
    print(f"  Score:    {report.match_score}")
    print(f"  Rounds:   {report.total_rounds}")
    print(f"  Generated in {report.generation_time_sec:.1f}s")
    print("=" * 70)

    ec = report.economy
    if ec:
        print(f"\n--- Economy ---")
        print(f"  Buy accuracy:      {ec.get('overall_accuracy', 0):.0%} "
              f"({ec.get('optimal_decisions', 0)}/{ec.get('total_rounds', 0)} optimal)")
        print(f"  Fresh-buy accuracy:{ec.get('fresh_buy_accuracy', 0):.0%}")
        print(f"  Over-buys:         {ec.get('over_buys', 0)}")
        print(f"  Under-buys:        {ec.get('under_buys', 0)}")
        print(f"  Mistakes vs eco:   {ec.get('mistakes_vs_enemy_eco', 0)}")
        print(f"  Avg WP lost/mistake: {ec.get('avg_wp_loss_per_mistake', 0):.1%}")

    en = report.engagement
    if en:
        print(f"\n--- Engagement ---")
        print(f"  K/D:               {en.get('kd_ratio', 0):.2f} "
              f"({en.get('kills', 0)}K / {en.get('deaths', 0)}D)")
        print(f"  ADR:               {en.get('adr', 0):.0f}")
        print(f"  Opening duels:     {en.get('opening_kills', 0)}W / "
              f"{en.get('opening_deaths', 0)}L "
              f"({en.get('opening_success_rate', 0):.0%})")
        print(f"  Trade rate:        {en.get('trade_rate', 0):.0%} "
              f"({en.get('successful_trades', 0)}/"
              f"{en.get('trade_opportunities', 0)})")
        if en.get("avg_trade_time_sec"):
            print(f"  Avg trade time:    {en['avg_trade_time_sec']:.1f}s")
        print(f"  Isolated deaths:   {en.get('isolated_deaths', 0)} "
              f"({en.get('isolated_death_rate', 0):.0%})")
        print(f"  Multi-kills (2k+): {en.get('multi_kills_2k', 0)}")
        print(f"  Clutches won:      {en.get('clutches_won', 0)}/"
              f"{en.get('clutches_attempted', 0)}")
        print(f"  Util/rifle round:  {en.get('avg_util_on_rifle_round', 0):.1f}")

    nn = report.nn_predictions
    if nn.get("available"):
        print(f"\n--- Neural Network Predictions ---")
        if "win_pred_accuracy" in nn:
            print(f"  Win prediction accuracy:    {nn['win_pred_accuracy']:.0%} "
                  f"({len(nn.get('win_prob_samples', []))} rounds)")
        if "attack_accuracy" in nn:
            print(f"  Attack site accuracy:       {nn['attack_accuracy']:.0%} "
                  f"({len(nn.get('attack_predictions', []))} rounds)")
        formations = nn.get("formation_predictions", [])
        if formations:
            print(f"  Formation predictions:      {len(formations)} rounds analyzed")

    print(f"\n{'='*70}")
    print("  COACHING TIPS")
    print(f"{'='*70}")
    for i, tip in enumerate(report.coaching_tips, 1):
        print(f"  {i}. {tip}")

    print(f"\n{'='*70}")
    print("  ROUND-BY-ROUND BREAKDOWN")
    print(f"{'='*70}")
    for rd in report.round_details:
        w = "W" if rd.get("won") else "L"
        print(f"\n  R{rd['round']:2d} [{rd['side']}] {w}  "
              f"K:{rd['kills']} D:{rd['deaths']} DMG:{rd['damage']:3d}")

        en_info = rd.get("engagement", {})
        highlights = list(en_info.get("notes", []))

        ec_info = rd.get("economy", {})
        if ec_info:
            if not ec_info.get("is_optimal"):
                highlights.append(
                    f"Buy: {ec_info['action']} (optimal: {ec_info['optimal']})")
            ep = ec_info.get("enemy_prediction", "")
            if ep:
                highlights.append(ep)

        if highlights:
            print(f"    Highlights: {' | '.join(highlights)}")

        sugg = rd.get("suggestions", {})
        if isinstance(sugg, list):
            for s in sugg:
                print(f"    -> {s}")
        elif isinstance(sugg, dict):
            pre = sugg.get("pre_round", [])
            if pre:
                print("    ── Pre-Round (Economy + Formation) ──")
                for s in pre:
                    print(f"      -> {s}")

            tl = sugg.get("timeline", [])
            if tl:
                print("    ── Event Timeline (LSTM + RL) ──")
                # Group entries sharing the same trigger under one header
                groups: list[dict] = []
                for ev in tl:
                    t = round(ev.get("time_sec", 0), 1)
                    trig = ev.get("trigger", "") or ""
                    key = (t, trig)
                    if groups and groups[-1]["key"] == key:
                        groups[-1]["entries"].append(ev)
                    else:
                        groups.append({"key": key, "entries": [ev]})

                for g in groups:
                    t, trig = g["key"]
                    header = f"    [{t:5.1f}s]"
                    if trig:
                        header += f" {trig}"
                    print(header)
                    for ev in g["entries"]:
                        src = ev.get("source", "")
                        text = ev.get("text", "")
                        if src == "Util" and not text:
                            continue
                        print(f"        [{src}] {text}")

            outcome = sugg.get("outcome", "")
            if outcome:
                print("    ── Outcome ──")
                print(f"      >> {outcome}")

    print()


def save_report_json(report: CoachingReport, output_path: str) -> dict:
    """Save the report as a JSON file and return the serialized dict."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = {
        "demo_file": report.demo_file,
        "player_name": report.player_name,
        "map_name": report.map_name,
        "match_score": report.match_score,
        "total_rounds": report.total_rounds,
        "generation_time_sec": report.generation_time_sec,
        "economy": report.economy,
        "engagement": report.engagement,
        "game_sense": report.game_sense,
        "nn_predictions": {
            k: v for k, v in report.nn_predictions.items()
            if k != "win_prob_samples"
        },
        "coaching_tips": report.coaching_tips,
        "round_details": report.round_details,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Report saved to {output_path}")
    return data


def generate_report_charts(report: CoachingReport,
                           output_dir: str,
                           models_dir: str = "models") -> list[str]:
    """Generate accuracy + fun-fact charts alongside the JSON report."""
    if not _VIZ_AVAILABLE:
        print("  visualize module not available — skipping charts")
        return []

    data = {
        "demo_file": report.demo_file,
        "player_name": report.player_name,
        "map_name": report.map_name,
        "match_score": report.match_score,
        "total_rounds": report.total_rounds,
        "economy": report.economy,
        "engagement": report.engagement,
        "game_sense": report.game_sense,
        "nn_predictions": {
            k: v for k, v in report.nn_predictions.items()
            if k != "win_prob_samples"
        },
        "round_details": report.round_details,
    }
    try:
        return generate_all_charts(data, models_dir, output_dir)
    except Exception as e:
        print(f"  Chart generation failed: {e}")
        return []


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_report.py <demo.dem> <player_name> [models_dir]")
        sys.exit(1)

    demo = sys.argv[1]
    player = sys.argv[2]
    mdir = sys.argv[3] if len(sys.argv) > 3 else "models"

    report = generate_full_report(demo, player, mdir)
    print_report(report)

    out_dir = "reports"
    out_name = Path(demo).stem + f"_{player}_report.json"
    save_report_json(report, os.path.join(out_dir, out_name))

    print("\nGenerating visualization charts...")
    charts = generate_report_charts(report, out_dir, mdir)
    for c in charts:
        print(f"  Saved: {c}")
