"""
training_data.py

Batch-extract training features from CS2 demo files for the round-level NN,
attack-site NN, CT formation classifier, and tactical Q-learning models.

Public functions:
  extract_all        — parse all demos under a directory into training DataFrames.
  discover_demos     — find all .dem files recursively under a base directory.
"""

from __future__ import annotations

import os
import glob
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
from callouts_mirage import get_zone, ZONES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICK_RATE = 64
ZONE_TO_IDX = {z: i for i, z in enumerate(ZONES)}

RIFLE_WEAPONS = {
    "AK-47", "M4A4", "M4A1-S", "SG 553", "AUG", "AWP",
    "SSG 08", "G3SG1", "SCAR-20",
}
SMG_WEAPONS = {
    "MP9", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "MAC-10",
}
UTILITY_ITEMS = {
    "Smoke Grenade", "Flashbang", "High Explosive Grenade",
    "Molotov", "Incendiary Grenade", "Decoy Grenade",
}

# RL v1 actions (legacy)
RL_ACTIONS = ["ROTATE_A", "ROTATE_B", "HOLD", "PUSH", "FALL_BACK"]
RL_ACTION_IDX = {a: i for i, a in enumerate(RL_ACTIONS)}

# RL v2 actions (legacy, unified)
RL_ACTIONS_V2 = ["PEEK", "HOLD", "TRADE", "FALL_BACK", "UTILITY", "ROTATE"]
RL_ACTION_V2_IDX = {a: i for i, a in enumerate(RL_ACTIONS_V2)}

# Side-specific RL actions (new split Q-learners)
T_ACTIONS = ["EXECUTE", "PEEK", "TRADE", "FALL_BACK", "UTILITY", "LURK"]
T_ACTION_IDX = {a: i for i, a in enumerate(T_ACTIONS)}
CT_ACTIONS = ["HOLD", "ROTATE", "RETAKE", "PUSH", "FALL_BACK", "UTILITY"]
CT_ACTION_IDX = {a: i for i, a in enumerate(CT_ACTIONS)}

# Team support bins
TEAM_SUPPORT_ALONE = 0
TEAM_SUPPORT_SUPPORTED = 1
TEAM_SUPPORT_GROUPED = 2

REWARD_NORMALIZER = 5.0

# Zone aggression: how deep into enemy territory (0=safest, 3=deepest)
_CT_AGGRESSION = {"CT_BASE": 0, "A": 1, "B": 1, "MID": 2, "T_BASE": 3}
_T_AGGRESSION = {"T_BASE": 0, "MID": 1, "A": 2, "B": 2, "CT_BASE": 3}

# Recent event categories for v2 state
RECENT_EVENTS = ["none", "teammate_died", "enemy_killed", "grenade", "bomb_planted"]
RECENT_EVENT_IDX = {e: i for i, e in enumerate(RECENT_EVENTS)}

# Event types for sequence model
SEQ_EVENT_TYPES = ["kill", "smoke", "flash", "he", "plant", "molotov"]
SEQ_EVENT_IDX = {e: i for i, e in enumerate(SEQ_EVENT_TYPES)}
SEQ_ZONES = ["A", "B", "MID", "CT_BASE", "T_BASE"]
SEQ_ZONE_IDX = {z: i for i, z in enumerate(SEQ_ZONES)}


def _team_str(team_num) -> str:
    if team_num == 2:
        return "T"
    if team_num == 3:
        return "CT"
    return "?"


# ---------------------------------------------------------------------------
# Knife-round detection
# ---------------------------------------------------------------------------

def _is_knife_round(round_deaths_df: pd.DataFrame) -> bool:
    """Return True if this round looks like a knife/side-selection round."""
    if round_deaths_df is None or len(round_deaths_df) == 0:
        return False
    if "weapon" not in round_deaths_df.columns:
        return False
    weapons = [str(w).lower() for w in round_deaths_df["weapon"].tolist()]
    weapons = [w for w in weapons if w and w != "nan"]
    if not weapons:
        return False
    knife_kills = sum(1 for w in weapons if "knife" in w or w == "taser")
    return knife_kills == len(weapons)


def _knife_round_mask(parser: DemoParser, freeze_ends: list[int],
                      round_ends: list[int]) -> list[bool]:
    """Return a boolean mask, one per round, True if it's a knife round."""
    deaths = _safe_parse_event(parser, "player_death",
                               player=["team_num"])
    n = min(len(freeze_ends), len(round_ends))
    mask: list[bool] = []
    for i in range(n):
        fe = freeze_ends[i]
        end = round_ends[i]
        if deaths.empty or "tick" not in deaths.columns:
            mask.append(False)
            continue
        rd = deaths[(deaths["tick"] >= fe) & (deaths["tick"] <= end)]
        mask.append(_is_knife_round(rd))
    return mask


# ---------------------------------------------------------------------------
# Round-level feature extraction  (for NN training)
# ---------------------------------------------------------------------------

def _safe_parse_event(parser: DemoParser, event_name: str, **kwargs) -> pd.DataFrame:
    """parse_event that always returns a DataFrame with a 'tick' column."""
    result = parser.parse_event(event_name, **kwargs)
    if isinstance(result, list) or not hasattr(result, "columns"):
        return pd.DataFrame({"tick": pd.Series(dtype=int)})
    if "tick" not in result.columns:
        return pd.DataFrame({"tick": pd.Series(dtype=int)})
    return result


def _extract_round_features(parser: DemoParser, demo_path: str) -> pd.DataFrame:
    """Extract one row per round with features for all three NN models."""

    freeze_ends = sorted(_safe_parse_event(parser, "round_freeze_end")
                         .get("tick", pd.Series(dtype=int)).tolist())
    round_ends = sorted(set(
        _safe_parse_event(parser, "round_officially_ended")
        .get("tick", pd.Series(dtype=int)).tolist()
    ))
    bomb_plants = _safe_parse_event(parser, "bomb_planted",
                                    player=["X", "Y", "Z"],
                                    other=["total_rounds_played"])
    bomb_defuses = _safe_parse_event(parser, "bomb_defused",
                                     other=["total_rounds_played"])
    bomb_explodes = _safe_parse_event(parser, "bomb_exploded",
                                      other=["total_rounds_played"])
    deaths = _safe_parse_event(parser, "player_death",
                               player=["X", "Y", "Z", "team_num"])

    bt_df = _safe_parse_event(parser, "buytime_ended")
    buytime_ended_ticks = sorted(
        bt_df["tick"].tolist() if "tick" in bt_df.columns else []
    )

    # Economy at freeze end
    econ_ticks = []
    for fe in freeze_ends:
        econ_ticks.extend([fe, fe + 1])

    econ_df = parser.parse_ticks(
        ["balance", "current_equip_value", "team_num", "is_alive",
         "active_weapon_name", "health"],
        ticks=econ_ticks,
    )

    # Inventory at buytime ended (utility counts)
    inv_df = None
    if buytime_ended_ticks:
        try:
            inv_df = parser.parse_ticks(
                ["inventory", "has_helmet", "armor_value", "team_num"],
                ticks=buytime_ended_ticks,
            )
        except Exception:
            pass

    # Positions ~5s after freeze end (CT formation)
    pos_ticks = [fe + 5 * TICK_RATE for fe in freeze_ends]
    pos_df = parser.parse_ticks(
        ["X", "Y", "Z", "team_num", "is_alive"],
        ticks=pos_ticks,
    )

    # Grenade events
    smokes = parser.parse_event("smokegrenade_detonate", player=["X", "Y", "Z"])
    flashes = parser.parse_event("flashbang_detonate", player=["X", "Y", "Z"])

    n_rounds = min(len(freeze_ends), len(round_ends))
    knife_mask = _knife_round_mask(parser, freeze_ends[:n_rounds],
                                    round_ends[:n_rounds])
    rows = []
    t_loss_streak = 0
    ct_loss_streak = 0

    for i in range(n_rounds):
        if knife_mask[i]:
            continue  # skip knife rounds
        fe_tick = freeze_ends[i]
        end_tick = round_ends[i]

        # Economy snapshot
        snap = econ_df[econ_df["tick"] == fe_tick]
        if snap.empty:
            snap = econ_df[econ_df["tick"] == fe_tick + 1]
        if snap.empty:
            continue

        t_snap = snap[snap["team_num"] == 2]
        ct_snap = snap[snap["team_num"] == 3]

        t_avg_money = float(t_snap["balance"].mean()) if len(t_snap) else 0
        ct_avg_money = float(ct_snap["balance"].mean()) if len(ct_snap) else 0
        t_avg_equip = float(t_snap["current_equip_value"].mean()) if len(t_snap) else 0
        ct_avg_equip = float(ct_snap["current_equip_value"].mean()) if len(ct_snap) else 0

        t_n_alive = int(t_snap["is_alive"].sum()) if len(t_snap) else 5
        ct_n_alive = int(ct_snap["is_alive"].sum()) if len(ct_snap) else 5

        t_rifles = sum(1 for _, r in t_snap.iterrows()
                       if str(r.get("active_weapon_name", "")) in RIFLE_WEAPONS)
        t_smgs = sum(1 for _, r in t_snap.iterrows()
                     if str(r.get("active_weapon_name", "")) in SMG_WEAPONS)
        ct_rifles = sum(1 for _, r in ct_snap.iterrows()
                        if str(r.get("active_weapon_name", "")) in RIFLE_WEAPONS)
        ct_smgs = sum(1 for _, r in ct_snap.iterrows()
                      if str(r.get("active_weapon_name", "")) in SMG_WEAPONS)

        t_equip_tier = 2 if t_rifles >= 3 else (1 if t_smgs + t_rifles >= 2 else 0)
        ct_equip_tier = 2 if ct_rifles >= 3 else (1 if ct_smgs + ct_rifles >= 2 else 0)

        # Utility counts
        t_util_count = 0
        ct_util_count = 0
        if inv_df is not None:
            next_fe = freeze_ends[i + 1] if i + 1 < len(freeze_ends) else end_tick
            bt_tick = None
            for bt in buytime_ended_ticks:
                if fe_tick <= bt <= next_fe:
                    bt_tick = bt
                    break
            if bt_tick is not None:
                inv_snap = inv_df[inv_df["tick"] == bt_tick]
                for _, row in inv_snap.iterrows():
                    raw_inv = row.get("inventory")
                    if raw_inv is None:
                        raw_inv = []
                    utils = sum(1 for item in raw_inv if item in UTILITY_ITEMS)
                    if row["team_num"] == 2:
                        t_util_count += utils
                    elif row["team_num"] == 3:
                        ct_util_count += utils

        # CT formation from positions
        ct_zone_counts = {z: 0 for z in ZONES}
        pos_tick = fe_tick + 5 * TICK_RATE
        pos_snap = pos_df[(pos_df["tick"] == pos_tick) &
                          (pos_df["team_num"] == 3) &
                          (pos_df["is_alive"] == True)]
        for _, row in pos_snap.iterrows():
            zone = get_zone(row["X"], row["Y"])
            if zone in ZONES:
                ct_zone_counts[zone] += 1

        ct_a = ct_zone_counts.get("A", 0)
        ct_b = ct_zone_counts.get("B", 0)
        ct_mid = ct_zone_counts.get("MID", 0)

        # Round winner
        rd_bombs = bomb_plants[(bomb_plants["tick"] >= fe_tick) &
                               (bomb_plants["tick"] <= end_tick)]
        rd_defuses = bomb_defuses[(bomb_defuses["tick"] >= fe_tick) &
                                  (bomb_defuses["tick"] <= end_tick)]
        rd_explodes = bomb_explodes[(bomb_explodes["tick"] >= fe_tick) &
                                    (bomb_explodes["tick"] <= end_tick)]
        rd_deaths_df = deaths[(deaths["tick"] >= fe_tick) &
                              (deaths["tick"] <= end_tick)]

        bomb_planted = len(rd_bombs) > 0
        bomb_site = ""
        if bomb_planted:
            bx = rd_bombs.iloc[0].get("user_X")
            by = rd_bombs.iloc[0].get("user_Y")
            if bx is not None and by is not None:
                bzone = get_zone(float(bx), float(by))
                bomb_site = "A" if bzone == "A" else ("B" if bzone == "B" else bzone)

        if len(rd_explodes) > 0:
            winner = "T"
        elif len(rd_defuses) > 0:
            winner = "CT"
        else:
            t_dead = sum(1 for _, r in rd_deaths_df.iterrows()
                         if _team_str(r.get("user_team_num", 0)) == "T")
            ct_dead = sum(1 for _, r in rd_deaths_df.iterrows()
                          if _team_str(r.get("user_team_num", 0)) == "CT")
            if t_dead >= t_n_alive and ct_dead < ct_n_alive:
                winner = "CT"
            elif ct_dead >= ct_n_alive and t_dead < t_n_alive:
                winner = "T"
            else:
                winner = "CT"

        # Smoke zones (first 25s)
        gren_cutoff = fe_tick + 25 * TICK_RATE
        rd_smokes = smokes[(smokes["tick"] >= fe_tick) &
                           (smokes["tick"] <= gren_cutoff)]
        smoke_zones = set()
        for _, row in rd_smokes.iterrows():
            x = row.get("user_X", row.get("x"))
            y = row.get("user_Y", row.get("y"))
            if x is not None and y is not None:
                z = get_zone(float(x), float(y))
                if z in ZONES:
                    smoke_zones.add(z)

        # Attack site label
        if bomb_planted and bomb_site in ("A", "B"):
            attack_site = bomb_site
        else:
            attack_site = "no_plant"

        t_won = int(winner == "T")

        rows.append({
            "demo": os.path.basename(demo_path),
            "round_num": i,
            "t_avg_money": t_avg_money,
            "ct_avg_money": ct_avg_money,
            "t_avg_equip": t_avg_equip,
            "ct_avg_equip": ct_avg_equip,
            "t_equip_tier": t_equip_tier,
            "ct_equip_tier": ct_equip_tier,
            "t_rifles": t_rifles,
            "ct_rifles": ct_rifles,
            "t_smgs": t_smgs,
            "ct_smgs": ct_smgs,
            "t_util_count": t_util_count,
            "ct_util_count": ct_util_count,
            "t_loss_streak": t_loss_streak,
            "ct_loss_streak": ct_loss_streak,
            "round_in_half": i % 12,
            "is_second_half": int(i >= 12),
            "winner": winner,
            "t_won": t_won,
            "bomb_planted": int(bomb_planted),
            "bomb_site": bomb_site,
            "attack_site": attack_site,
            "ct_a": ct_a,
            "ct_b": ct_b,
            "ct_mid": ct_mid,
            "ct_formation": f"{ct_a}-{ct_mid}-{ct_b}",
            "smoke_A": int("A" in smoke_zones),
            "smoke_B": int("B" in smoke_zones),
            "smoke_MID": int("MID" in smoke_zones),
        })

        if t_won:
            t_loss_streak = 0
            ct_loss_streak = min(ct_loss_streak + 1, 5)
        else:
            ct_loss_streak = 0
            t_loss_streak = min(t_loss_streak + 1, 5)
        if i == 11:
            t_loss_streak = 0
            ct_loss_streak = 0

    # Compute prior-round tendency features
    rounds_since_plant_A = 999
    rounds_since_plant_B = 999
    last_plant_site = ""
    streak_same_site = 0
    prev_row: dict | None = None
    for idx, row in enumerate(rows):
        is_half_start = (row["round_in_half"] == 0)

        if is_half_start or prev_row is None:
            row["prev_plant_A"] = 0
            row["prev_plant_B"] = 0
            row["prev_plant_none"] = 0
            row["prev_no_history"] = 1
            row["prev_t_won"] = 0.5
            row["prev_t_tier"] = 0.0
            row["prev_ct_tier"] = 0.0
            row["rounds_since_plant_A"] = 1.0
            row["rounds_since_plant_B"] = 1.0
            row["streak_same_site"] = 0.0
            rounds_since_plant_A = 999
            rounds_since_plant_B = 999
            last_plant_site = ""
            streak_same_site = 0
        else:
            prev_site = prev_row.get("attack_site", "no_plant")
            row["prev_plant_A"] = int(prev_site == "A")
            row["prev_plant_B"] = int(prev_site == "B")
            row["prev_plant_none"] = int(prev_site == "no_plant")
            row["prev_no_history"] = 0
            row["prev_t_won"] = float(prev_row.get("t_won", 0))
            row["prev_t_tier"] = prev_row.get("t_equip_tier", 0) / 2.0
            row["prev_ct_tier"] = prev_row.get("ct_equip_tier", 0) / 2.0
            row["rounds_since_plant_A"] = min(rounds_since_plant_A, 6) / 6.0
            row["rounds_since_plant_B"] = min(rounds_since_plant_B, 6) / 6.0
            row["streak_same_site"] = min(streak_same_site, 3) / 3.0

        cur_site = row.get("attack_site", "no_plant")
        rounds_since_plant_A = 0 if cur_site == "A" else rounds_since_plant_A + 1
        rounds_since_plant_B = 0 if cur_site == "B" else rounds_since_plant_B + 1
        if cur_site in ("A", "B"):
            if cur_site == last_plant_site:
                streak_same_site += 1
            else:
                streak_same_site = 1
            last_plant_site = cur_site
        else:
            streak_same_site = 0

        prev_row = row

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RL time-step extraction  (for Q-learning)
# ---------------------------------------------------------------------------

def _extract_rl_transitions(parser: DemoParser, demo_path: str) -> pd.DataFrame:
    """Extract state-action-reward tuples sampled every 5 seconds per round."""

    freeze_ends = sorted(_safe_parse_event(parser, "round_freeze_end")
                         .get("tick", pd.Series(dtype=int)).tolist())
    round_ends = sorted(set(
        _safe_parse_event(parser, "round_officially_ended")
        .get("tick", pd.Series(dtype=int)).tolist()
    ))
    bomb_plants = _safe_parse_event(parser, "bomb_planted",
                                    player=["X", "Y", "Z"],
                                    other=["total_rounds_played"])
    bomb_defuses = _safe_parse_event(parser, "bomb_defused",
                                     other=["total_rounds_played"])
    bomb_explodes = _safe_parse_event(parser, "bomb_exploded",
                                      other=["total_rounds_played"])
    deaths = _safe_parse_event(parser, "player_death",
                               player=["X", "Y", "Z", "team_num"])

    n_rounds = min(len(freeze_ends), len(round_ends))
    knife_mask = _knife_round_mask(parser, freeze_ends[:n_rounds],
                                    round_ends[:n_rounds])

    # Build sample ticks (every 5 seconds)
    sample_ticks_by_round: dict[int, list[int]] = {}
    all_sample_ticks: list[int] = []
    for i in range(n_rounds):
        if knife_mask[i]:
            continue
        fe = freeze_ends[i]
        end = round_ends[i]
        ticks = list(range(fe, end + 1, 5 * TICK_RATE))
        sample_ticks_by_round[i] = ticks
        all_sample_ticks.extend(ticks)

    if not all_sample_ticks:
        return pd.DataFrame()

    pos_df = parser.parse_ticks(
        ["X", "Y", "Z", "team_num", "is_alive", "health"],
        ticks=all_sample_ticks,
    )

    # Round winners
    round_winners: dict[int, str] = {}
    bomb_info: dict[int, tuple] = {}
    for i in range(n_rounds):
        fe = freeze_ends[i]
        end = round_ends[i]
        rd_exp = bomb_explodes[(bomb_explodes["tick"] >= fe) & (bomb_explodes["tick"] <= end)]
        rd_def = bomb_defuses[(bomb_defuses["tick"] >= fe) & (bomb_defuses["tick"] <= end)]
        rd_deaths_df = deaths[(deaths["tick"] >= fe) & (deaths["tick"] <= end)]
        rd_bombs = bomb_plants[(bomb_plants["tick"] >= fe) & (bomb_plants["tick"] <= end)]

        if len(rd_exp) > 0:
            round_winners[i] = "T"
        elif len(rd_def) > 0:
            round_winners[i] = "CT"
        else:
            t_dead = sum(1 for _, r in rd_deaths_df.iterrows()
                         if _team_str(r.get("user_team_num", 0)) == "T")
            ct_dead = sum(1 for _, r in rd_deaths_df.iterrows()
                          if _team_str(r.get("user_team_num", 0)) == "CT")
            round_winners[i] = "T" if ct_dead > t_dead else "CT"

        if len(rd_bombs) > 0:
            bx = rd_bombs.iloc[0].get("user_X")
            by = rd_bombs.iloc[0].get("user_Y")
            site = ""
            if bx is not None and by is not None:
                bzone = get_zone(float(bx), float(by))
                site = "A" if bzone == "A" else ("B" if bzone == "B" else "")
            bomb_info[i] = (int(rd_bombs.iloc[0]["tick"]), site)
        else:
            bomb_info[i] = (None, "")

    rows = []
    for i in range(n_rounds):
        if knife_mask[i]:
            continue
        ticks = sample_ticks_by_round.get(i, [])
        fe = freeze_ends[i]
        winner = round_winners.get(i, "CT")
        plant_tick, plant_site = bomb_info[i]

        for t_idx in range(len(ticks) - 1):
            tick = ticks[t_idx]
            next_tick = ticks[t_idx + 1]

            snap = pos_df[pos_df["tick"] == tick]
            next_snap = pos_df[pos_df["tick"] == next_tick]

            t_alive = int(snap[(snap["team_num"] == 2) &
                               (snap["is_alive"] == True)].shape[0])
            ct_alive = int(snap[(snap["team_num"] == 3) &
                                (snap["is_alive"] == True)].shape[0])

            time_elapsed = max(0.0, (tick - fe) / TICK_RATE)
            time_bucket = (0 if time_elapsed < 30
                           else 1 if time_elapsed < 60
                           else 2)
            if plant_tick is not None and tick >= plant_tick:
                time_bucket = 3

            bomb_status = 0
            if plant_tick is not None and tick >= plant_tick:
                bomb_status = 1 if plant_site == "A" else 2

            is_terminal = (t_idx == len(ticks) - 2)

            alive_players = snap[snap["is_alive"] == True]
            next_alive = next_snap[next_snap["is_alive"] == True]

            for _, player_row in alive_players.iterrows():
                name = player_row["name"]
                side = _team_str(player_row["team_num"])
                if side not in ("T", "CT"):
                    continue

                zone = get_zone(player_row["X"], player_row["Y"])
                zone_idx = ZONE_TO_IDX.get(zone, -1)
                if zone_idx == -1:
                    continue

                next_player = next_alive[next_alive["name"] == name]
                if next_player.empty:
                    action = 2  # HOLD (died)
                    next_zone_idx = zone_idx
                else:
                    nr = next_player.iloc[0]
                    next_zone = get_zone(nr["X"], nr["Y"])
                    next_zone_idx = ZONE_TO_IDX.get(next_zone, zone_idx)

                    if next_zone_idx == zone_idx:
                        action = 2  # HOLD
                    elif next_zone == "A":
                        action = 0  # ROTATE_A
                    elif next_zone == "B":
                        action = 1  # ROTATE_B
                    elif ((side == "CT" and next_zone == "T_BASE") or
                          (side == "T" and next_zone == "CT_BASE")):
                        action = 3  # PUSH
                    elif ((side == "CT" and next_zone == "CT_BASE") or
                          (side == "T" and next_zone == "T_BASE")):
                        action = 4  # FALL_BACK
                    else:
                        action = 3  # REPOSITION → PUSH bucket

                reward = 0.0
                if is_terminal:
                    reward = 1.0 if winner == side else 0.0

                rows.append({
                    "demo": os.path.basename(demo_path),
                    "round_num": i,
                    "tick": tick,
                    "time_elapsed": time_elapsed,
                    "player": name,
                    "side": side,
                    "t_alive": t_alive,
                    "ct_alive": ct_alive,
                    "bomb_status": bomb_status,
                    "time_bucket": time_bucket,
                    "zone_idx": zone_idx,
                    "action": action,
                    "next_t_alive": int(next_snap[(next_snap["team_num"] == 2) &
                                                   (next_snap["is_alive"] == True)].shape[0]),
                    "next_ct_alive": int(next_snap[(next_snap["team_num"] == 3) &
                                                    (next_snap["is_alive"] == True)].shape[0]),
                    "next_zone_idx": next_zone_idx,
                    "reward": reward,
                    "is_terminal": int(is_terminal),
                    "round_won": int(winner == side),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RL v2: micro-decision extraction  (PEEK / HOLD / TRADE / FALL_BACK / UTILITY / ROTATE)
# ---------------------------------------------------------------------------

def _classify_action_v2(
    side: str, zone: str, next_zone: str,
    player_name: str, tick: int, next_tick: int,
    kills_window: list[dict],
    teammate_deaths_window: list[dict],
    grenades_window: list[dict],
) -> int:
    """Classify a player's micro-action in a 5s window.

    Priority: UTILITY > TRADE > zone-based (PEEK / FALL_BACK / ROTATE) > HOLD.
    """
    for g in grenades_window:
        if g["thrower"] == player_name:
            return RL_ACTION_V2_IDX["UTILITY"]

    if teammate_deaths_window:
        for k in kills_window:
            if k["attacker"] == player_name:
                return RL_ACTION_V2_IDX["TRADE"]

    agg = _CT_AGGRESSION if side == "CT" else _T_AGGRESSION
    cur_agg = agg.get(zone, 1)
    nxt_agg = agg.get(next_zone, 1)

    if zone == next_zone:
        return RL_ACTION_V2_IDX["HOLD"]
    elif nxt_agg > cur_agg:
        return RL_ACTION_V2_IDX["PEEK"]
    elif nxt_agg < cur_agg:
        return RL_ACTION_V2_IDX["FALL_BACK"]
    else:
        return RL_ACTION_V2_IDX["ROTATE"]


def _recent_event_category(
    side: str, tick: int,
    all_kills: list[dict],
    all_grenades: list[dict],
    bomb_plant_tick: int | None,
    lookback_ticks: int = 5 * TICK_RATE,
) -> int:
    """Categorize the most salient event in the last 5s for this player's side."""
    window_start = tick - lookback_ticks

    if bomb_plant_tick is not None and window_start < bomb_plant_tick <= tick:
        return RECENT_EVENT_IDX["bomb_planted"]

    for k in reversed(all_kills):
        if k["tick"] < window_start:
            break
        if k["tick"] > tick:
            continue
        if k["victim_side"] == side:
            return RECENT_EVENT_IDX["teammate_died"]
        elif k["attacker_side"] == side:
            return RECENT_EVENT_IDX["enemy_killed"]

    for g in reversed(all_grenades):
        if g["tick"] < window_start:
            break
        if g["tick"] > tick:
            continue
        return RECENT_EVENT_IDX["grenade"]

    return RECENT_EVENT_IDX["none"]


# Adjacent zones for team_support calculation
_ADJACENT_ZONES = {
    "A":       {"A", "MID"},
    "B":       {"B", "MID"},
    "MID":     {"MID", "A", "B", "CT_BASE", "T_BASE"},
    "CT_BASE": {"CT_BASE", "MID", "A", "B"},
    "T_BASE":  {"T_BASE", "MID"},
}


def _compute_team_support(
    player_zone: str,
    player_name: str,
    side: str,
    snap: pd.DataFrame,
) -> int:
    """Count teammates in same or adjacent zone and return support bin."""
    adj = _ADJACENT_ZONES.get(player_zone, {player_zone})
    team_num = 2 if side == "T" else 3
    teammates = snap[(snap["team_num"] == team_num) &
                     (snap["is_alive"] == True) &
                     (snap["name"] != player_name)]
    count = 0
    for _, row in teammates.iterrows():
        tz = get_zone(row["X"], row["Y"])
        if tz in adj:
            count += 1
    if count == 0:
        return TEAM_SUPPORT_ALONE
    if count == 1:
        return TEAM_SUPPORT_SUPPORTED
    return TEAM_SUPPORT_GROUPED


def _classify_action_side_specific(
    side: str, zone: str, next_zone: str,
    player_name: str, tick: int, next_tick: int,
    kills_window: list[dict],
    teammate_deaths_window: list[dict],
    grenades_window: list[dict],
    bomb_planted: bool,
) -> int:
    """Classify a player's micro-action using side-specific action spaces."""
    # Utility thrown
    for g in grenades_window:
        if g["thrower"] == player_name:
            return T_ACTION_IDX["UTILITY"] if side == "T" else CT_ACTION_IDX["UTILITY"]

    # Trade: teammate died AND player got a kill
    if teammate_deaths_window:
        for k in kills_window:
            if k["attacker"] == player_name:
                return T_ACTION_IDX["TRADE"] if side == "T" else CT_ACTION_IDX["PUSH"]

    agg = _CT_AGGRESSION if side == "CT" else _T_AGGRESSION
    cur_agg = agg.get(zone, 1)
    nxt_agg = agg.get(next_zone, 1)

    if side == "T":
        if zone == next_zone:
            if bomb_planted:
                return T_ACTION_IDX["EXECUTE"]
            return T_ACTION_IDX["LURK"] if cur_agg <= 1 else T_ACTION_IDX["PEEK"]
        if nxt_agg > cur_agg:
            return T_ACTION_IDX["EXECUTE"] if bomb_planted else T_ACTION_IDX["PEEK"]
        if nxt_agg < cur_agg:
            return T_ACTION_IDX["FALL_BACK"]
        return T_ACTION_IDX["LURK"]
    else:  # CT
        if zone == next_zone:
            return CT_ACTION_IDX["HOLD"]
        if bomb_planted and next_zone in ("A", "B"):
            return CT_ACTION_IDX["RETAKE"]
        if nxt_agg > cur_agg:
            return CT_ACTION_IDX["PUSH"]
        if nxt_agg < cur_agg:
            return CT_ACTION_IDX["FALL_BACK"]
        return CT_ACTION_IDX["ROTATE"]


# ---------------------------------------------------------------------------
# CT formation label extraction (for FormationClassifier_CT LSTM)
# ---------------------------------------------------------------------------

# All valid formations per alive count
CT_FORMATIONS_BY_ALIVE = {
    5: ["2-1-2", "1-2-2", "3-1-1", "1-1-3", "2-2-1", "0-2-3", "2-0-3", "other"],
    4: ["2-1-1", "1-2-1", "1-1-2", "2-0-2", "0-2-2", "3-1-0", "0-1-3", "other"],
    3: ["1-1-1", "2-1-0", "0-1-2", "2-0-1", "1-0-2", "0-2-1", "1-2-0", "other"],
    2: ["1-0-1", "2-0-0", "0-0-2", "0-1-1", "1-1-0", "0-2-0", "other"],
    1: ["1-0-0", "0-1-0", "0-0-1"],
}

ALL_CT_FORMATIONS = []
CT_FORMATION_TO_IDX = {}
for alive in sorted(CT_FORMATIONS_BY_ALIVE.keys()):
    for fmt in CT_FORMATIONS_BY_ALIVE[alive]:
        label = f"{alive}_{fmt}"
        if label not in CT_FORMATION_TO_IDX:
            CT_FORMATION_TO_IDX[label] = len(ALL_CT_FORMATIONS)
            ALL_CT_FORMATIONS.append(label)
N_CT_FORMATIONS = len(ALL_CT_FORMATIONS)

# Build alive-mask: for each alive count, which formation indices are valid
CT_ALIVE_MASK = {}
for alive, fmts in CT_FORMATIONS_BY_ALIVE.items():
    mask = [False] * N_CT_FORMATIONS
    for fmt in fmts:
        label = f"{alive}_{fmt}"
        if label in CT_FORMATION_TO_IDX:
            mask[CT_FORMATION_TO_IDX[label]] = True
    CT_ALIVE_MASK[alive] = mask


def _classify_ct_formation(ct_a: int, ct_mid: int, ct_b: int,
                           ct_alive: int) -> str:
    """Map zone counts to a formation label like '5_2-1-2'."""
    fmt_str = f"{ct_a}-{ct_mid}-{ct_b}"
    label = f"{ct_alive}_{fmt_str}"
    if label in CT_FORMATION_TO_IDX:
        return label
    return f"{ct_alive}_other"


def _extract_ct_formation_sequences(parser: DemoParser,
                                    demo_path: str) -> list[dict]:
    """Extract per-round event sequences with CT formation labels at each event."""
    freeze_ends = sorted(_safe_parse_event(parser, "round_freeze_end")
                         .get("tick", pd.Series(dtype=int)).tolist())
    round_ends = sorted(set(
        _safe_parse_event(parser, "round_officially_ended")
        .get("tick", pd.Series(dtype=int)).tolist()
    ))
    deaths_df = _safe_parse_event(parser, "player_death",
                                  player=["X", "Y", "Z", "team_num"])
    smokes_df = _safe_parse_event(parser, "smokegrenade_detonate",
                                  player=["X", "Y", "Z", "team_num"])
    flashes_df = _safe_parse_event(parser, "flashbang_detonate",
                                   player=["X", "Y", "Z", "team_num"])
    he_df = _safe_parse_event(parser, "hegrenade_detonate",
                              player=["X", "Y", "Z", "team_num"])
    molotov_df = _safe_parse_event(parser, "inferno_startburn",
                                   player=["X", "Y", "Z", "team_num"])
    bomb_plants = _safe_parse_event(parser, "bomb_planted",
                                    player=["X", "Y", "Z"],
                                    other=["total_rounds_played"])

    n_rounds = min(len(freeze_ends), len(round_ends))
    knife_mask = _knife_round_mask(parser, freeze_ends[:n_rounds],
                                    round_ends[:n_rounds])

    # Collect all event ticks to sample CT positions
    all_event_ticks: list[int] = []
    event_ticks_by_round: dict[int, list[int]] = {}
    for i in range(n_rounds):
        if knife_mask[i]:
            continue
        fe = freeze_ends[i]
        end = round_ends[i]
        ticks = set()
        for df in [deaths_df, smokes_df, flashes_df, he_df, molotov_df,
                   bomb_plants]:
            rd_df = df[(df["tick"] >= fe) & (df["tick"] <= end)]
            ticks.update(rd_df["tick"].tolist())
        sorted_ticks = sorted(ticks)
        event_ticks_by_round[i] = sorted_ticks
        all_event_ticks.extend(sorted_ticks)

    if not all_event_ticks:
        return []

    pos_df = parser.parse_ticks(
        ["X", "Y", "Z", "team_num", "is_alive"],
        ticks=sorted(set(all_event_ticks)),
    )

    results = []
    for i in range(n_rounds):
        if knife_mask[i]:
            continue
        fe = freeze_ends[i]
        end = round_ends[i]
        round_duration = max((end - fe) / TICK_RATE, 1.0)

        rd_deaths = deaths_df[(deaths_df["tick"] >= fe) & (deaths_df["tick"] <= end)]
        rd_events: list[dict] = []

        # Build event list (same format as _extract_event_sequences)
        for _, row in rd_deaths.iterrows():
            victim_side = _team_str(row.get("user_team_num", 0))
            attacker_is_t = 1 if victim_side == "CT" else 0
            vx, vy = row.get("user_X", 0), row.get("user_Y", 0)
            zone = get_zone(float(vx), float(vy)) if vx else "MID"
            rd_events.append({
                "tick": int(row["tick"]),
                "type_idx": SEQ_EVENT_IDX["kill"],
                "actor_side_is_t": attacker_is_t,
                "zone_idx": SEQ_ZONE_IDX.get(zone, 2),
                "time_norm": min((int(row["tick"]) - fe) / TICK_RATE / 120.0, 1.0),
                "is_headshot": int(row.get("headshot", 0)) if "headshot" in row.index else 0,
            })

        for gdf, gtype in [(smokes_df, "smoke"), (flashes_df, "flash"),
                           (he_df, "he"), (molotov_df, "molotov")]:
            rd_g = gdf[(gdf["tick"] >= fe) & (gdf["tick"] <= end)]
            for _, row in rd_g.iterrows():
                thrower_side = _team_str(row.get("user_team_num", 0))
                gx, gy = row.get("user_X", 0), row.get("user_Y", 0)
                zone = get_zone(float(gx), float(gy)) if gx else "MID"
                rd_events.append({
                    "tick": int(row["tick"]),
                    "type_idx": SEQ_EVENT_IDX[gtype],
                    "actor_side_is_t": 1 if thrower_side == "T" else 0,
                    "zone_idx": SEQ_ZONE_IDX.get(zone, 2),
                    "time_norm": min((int(row["tick"]) - fe) / TICK_RATE / 120.0, 1.0),
                    "is_headshot": 0,
                })

        rd_bombs = bomb_plants[(bomb_plants["tick"] >= fe) &
                               (bomb_plants["tick"] <= end)]
        if len(rd_bombs) > 0:
            bx = rd_bombs.iloc[0].get("user_X")
            by = rd_bombs.iloc[0].get("user_Y")
            bzone = "MID"
            if bx is not None and by is not None:
                bzone = get_zone(float(bx), float(by))
            rd_events.append({
                "tick": int(rd_bombs.iloc[0]["tick"]),
                "type_idx": SEQ_EVENT_IDX["plant"],
                "actor_side_is_t": 1,
                "zone_idx": SEQ_ZONE_IDX.get(bzone, 2),
                "time_norm": min((int(rd_bombs.iloc[0]["tick"]) - fe) / TICK_RATE / 120.0, 1.0),
                "is_headshot": 0,
            })

        rd_events.sort(key=lambda e: e["tick"])

        # At each event, compute CT formation label
        formation_labels = []
        ct_alive_list = []
        for ev in rd_events:
            ev_tick = ev["tick"]
            snap = pos_df[(pos_df["tick"] == ev_tick) &
                          (pos_df["team_num"] == 3) &
                          (pos_df["is_alive"] == True)]
            ct_alive = len(snap)
            if ct_alive == 0:
                break  # round over — all CTs eliminated

            ct_a = sum(1 for _, r in snap.iterrows()
                       if get_zone(r["X"], r["Y"]) == "A")
            ct_mid = sum(1 for _, r in snap.iterrows()
                         if get_zone(r["X"], r["Y"]) == "MID")
            ct_b = sum(1 for _, r in snap.iterrows()
                       if get_zone(r["X"], r["Y"]) == "B")

            label = _classify_ct_formation(ct_a, ct_mid, ct_b, ct_alive)
            formation_labels.append(label)
            ct_alive_list.append(ct_alive)

        # Trim events to match labels (stops at elimination)
        trimmed_events = rd_events[:len(formation_labels)]
        for e in trimmed_events:
            del e["tick"]

        if trimmed_events:
            results.append({
                "demo": os.path.basename(demo_path),
                "round_num": i,
                "events": trimmed_events,
                "formation_labels": formation_labels,
                "ct_alive_at_event": ct_alive_list,
            })

    return results


def _extract_rl_v2(parser: DemoParser, demo_path: str) -> pd.DataFrame:
    """Extract v2 state-action-reward tuples with micro-decision actions
    and dual reward signals (kill + win)."""

    freeze_ends = sorted(_safe_parse_event(parser, "round_freeze_end")
                         .get("tick", pd.Series(dtype=int)).tolist())
    round_ends = sorted(set(
        _safe_parse_event(parser, "round_officially_ended")
        .get("tick", pd.Series(dtype=int)).tolist()
    ))
    bomb_plants = _safe_parse_event(parser, "bomb_planted",
                                    player=["X", "Y", "Z"],
                                    other=["total_rounds_played"])
    bomb_defuses = _safe_parse_event(parser, "bomb_defused",
                                     other=["total_rounds_played"])
    bomb_explodes = _safe_parse_event(parser, "bomb_exploded",
                                      other=["total_rounds_played"])

    deaths_df = _safe_parse_event(parser, "player_death",
                                  player=["X", "Y", "Z", "team_num"])
    smokes_df = _safe_parse_event(parser, "smokegrenade_detonate",
                                  player=["X", "Y", "Z", "team_num"])
    flashes_df = _safe_parse_event(parser, "flashbang_detonate",
                                   player=["X", "Y", "Z", "team_num"])
    he_df = _safe_parse_event(parser, "hegrenade_detonate",
                              player=["X", "Y", "Z", "team_num"])
    molotov_df = _safe_parse_event(parser, "inferno_startburn",
                                   player=["X", "Y", "Z", "team_num"])

    n_rounds = min(len(freeze_ends), len(round_ends))
    knife_mask = _knife_round_mask(parser, freeze_ends[:n_rounds],
                                    round_ends[:n_rounds])
    sample_ticks_by_round: dict[int, list[int]] = {}
    all_sample_ticks: list[int] = []
    for i in range(n_rounds):
        fe = freeze_ends[i]
        end = round_ends[i]
        ticks = list(range(fe, end + 1, 5 * TICK_RATE))
        sample_ticks_by_round[i] = ticks
        all_sample_ticks.extend(ticks)

    if not all_sample_ticks:
        return pd.DataFrame()

    pos_df = parser.parse_ticks(
        ["X", "Y", "Z", "team_num", "is_alive", "health"],
        ticks=all_sample_ticks,
    )

    # Pre-process kills and grenades into structured lists per round
    def _build_kill_list(df, fe, end):
        kills = []
        rd = df[(df["tick"] >= fe) & (df["tick"] <= end)]
        for _, row in rd.iterrows():
            victim_side = _team_str(row.get("user_team_num", 0))
            attacker_side = "CT" if victim_side == "T" else "T"
            vx = row.get("user_X", 0)
            vy = row.get("user_Y", 0)
            kills.append({
                "tick": int(row["tick"]),
                "attacker": str(row.get("attacker_name", "")),
                "victim": str(row.get("user_name", "")),
                "attacker_side": attacker_side,
                "victim_side": victim_side,
                "zone": get_zone(float(vx), float(vy)) if vx else "MID",
            })
        return sorted(kills, key=lambda x: x["tick"])

    def _build_grenade_list(smoke_df, flash_df, he_df_, molotov_df_, fe, end):
        grenades = []
        for gdf, gtype in [(smoke_df, "smoke"), (flash_df, "flash"),
                           (he_df_, "he"), (molotov_df_, "molotov")]:
            rd = gdf[(gdf["tick"] >= fe) & (gdf["tick"] <= end)]
            for _, row in rd.iterrows():
                gx = row.get("user_X", 0)
                gy = row.get("user_Y", 0)
                grenades.append({
                    "tick": int(row["tick"]),
                    "thrower": str(row.get("user_name", "")),
                    "thrower_side": _team_str(row.get("user_team_num", 0)),
                    "type": gtype,
                    "zone": get_zone(float(gx), float(gy)) if gx else "MID",
                })
        return sorted(grenades, key=lambda x: x["tick"])

    # Round winners and bomb info
    round_winners: dict[int, str] = {}
    bomb_info: dict[int, tuple] = {}
    for i in range(n_rounds):
        fe = freeze_ends[i]
        end = round_ends[i]
        rd_exp = bomb_explodes[(bomb_explodes["tick"] >= fe) & (bomb_explodes["tick"] <= end)]
        rd_def = bomb_defuses[(bomb_defuses["tick"] >= fe) & (bomb_defuses["tick"] <= end)]
        rd_deaths = deaths_df[(deaths_df["tick"] >= fe) & (deaths_df["tick"] <= end)]
        rd_bombs = bomb_plants[(bomb_plants["tick"] >= fe) & (bomb_plants["tick"] <= end)]

        if len(rd_exp) > 0:
            round_winners[i] = "T"
        elif len(rd_def) > 0:
            round_winners[i] = "CT"
        else:
            t_dead = sum(1 for _, r in rd_deaths.iterrows()
                         if _team_str(r.get("user_team_num", 0)) == "T")
            ct_dead = sum(1 for _, r in rd_deaths.iterrows()
                          if _team_str(r.get("user_team_num", 0)) == "CT")
            round_winners[i] = "T" if ct_dead > t_dead else "CT"

        if len(rd_bombs) > 0:
            bx = rd_bombs.iloc[0].get("user_X")
            by = rd_bombs.iloc[0].get("user_Y")
            site = ""
            if bx is not None and by is not None:
                bzone = get_zone(float(bx), float(by))
                site = "A" if bzone == "A" else ("B" if bzone == "B" else "")
            bomb_info[i] = (int(rd_bombs.iloc[0]["tick"]), site)
        else:
            bomb_info[i] = (None, "")

    rows = []
    for i in range(n_rounds):
        if knife_mask[i]:
            continue  # skip knife rounds
        ticks = sample_ticks_by_round[i]
        fe = freeze_ends[i]
        end = round_ends[i]
        winner = round_winners.get(i, "CT")
        plant_tick, plant_site = bomb_info[i]

        round_kills = _build_kill_list(deaths_df, fe, end)
        round_grenades = _build_grenade_list(smokes_df, flashes_df, he_df,
                                             molotov_df, fe, end)

        # Determine attacked site once per round
        attacked_site = plant_site if plant_site else ""
        if not attacked_site:
            t_kill_zones = {"A": 0, "B": 0}
            for k in round_kills:
                if k["attacker_side"] == "T":
                    kz = k.get("zone", "")
                    if kz in t_kill_zones:
                        t_kill_zones[kz] += 1
            if t_kill_zones["A"] > t_kill_zones["B"]:
                attacked_site = "A"
            elif t_kill_zones["B"] > t_kill_zones["A"]:
                attacked_site = "B"

        round_terminated = False

        for t_idx in range(len(ticks) - 1):
            if round_terminated:
                break

            tick = ticks[t_idx]
            next_tick = ticks[t_idx + 1]

            snap = pos_df[pos_df["tick"] == tick]
            next_snap = pos_df[pos_df["tick"] == next_tick]

            t_alive = int(snap[(snap["team_num"] == 2) &
                               (snap["is_alive"] == True)].shape[0])
            ct_alive = int(snap[(snap["team_num"] == 3) &
                                (snap["is_alive"] == True)].shape[0])

            # Round termination: if either side is fully eliminated, stop
            if t_alive == 0 or ct_alive == 0:
                round_terminated = True
                break

            time_elapsed = max(0.0, (tick - fe) / TICK_RATE)
            bomb_planted_now = (plant_tick is not None and tick >= plant_tick)
            bomb_status = 0
            if bomb_planted_now:
                bomb_status = 1 if plant_site == "A" else 2
            time_bucket = (3 if bomb_status > 0 else
                           0 if time_elapsed < 30 else
                           1 if time_elapsed < 60 else 2)

            # Check if next tick has a team wipe — if so, this is terminal
            next_t_alive = int(next_snap[(next_snap["team_num"] == 2) &
                                         (next_snap["is_alive"] == True)].shape[0])
            next_ct_alive = int(next_snap[(next_snap["team_num"] == 3) &
                                          (next_snap["is_alive"] == True)].shape[0])
            is_terminal = (t_idx == len(ticks) - 2 or
                           next_t_alive == 0 or next_ct_alive == 0)
            if next_t_alive == 0 or next_ct_alive == 0:
                round_terminated = True

            kills_window = [k for k in round_kills
                            if tick <= k["tick"] < next_tick]
            grenades_window = [g for g in round_grenades
                               if tick <= g["tick"] < next_tick]

            alive_players = snap[snap["is_alive"] == True]
            next_alive = next_snap[next_snap["is_alive"] == True]

            for _, player_row in alive_players.iterrows():
                name = player_row["name"]
                side = _team_str(player_row["team_num"])
                if side not in ("T", "CT"):
                    continue

                zone = get_zone(player_row["X"], player_row["Y"])
                zone_idx = ZONE_TO_IDX.get(zone, -1)
                if zone_idx == -1:
                    continue

                next_player = next_alive[next_alive["name"] == name]
                if next_player.empty:
                    next_zone = zone
                    next_zone_idx = zone_idx
                else:
                    nr = next_player.iloc[0]
                    next_zone = get_zone(nr["X"], nr["Y"])
                    next_zone_idx = ZONE_TO_IDX.get(next_zone, zone_idx)

                teammate_deaths = [k for k in kills_window
                                   if k["victim_side"] == side
                                   and k["victim"] != name]

                # Side-specific action classification
                action_ss = _classify_action_side_specific(
                    side, zone, next_zone, name, tick, next_tick,
                    kills_window, teammate_deaths, grenades_window,
                    bomb_planted_now)

                # Legacy v2 action (for backward compat)
                action = _classify_action_v2(
                    side, zone, next_zone, name, tick, next_tick,
                    kills_window, teammate_deaths, grenades_window)

                # Team support
                team_support = _compute_team_support(
                    zone, name, side, snap)

                # Dual rewards with site-alignment + normalization
                player_got_kill = any(
                    k["attacker"] == name for k in kills_window)
                player_died = next_player.empty

                at_attacked = (zone in ("A", "B") and zone == attacked_site)
                at_wrong = (zone in ("A", "B") and attacked_site
                            and zone != attacked_site)
                rotating_toward = (next_zone == attacked_site
                                   and zone != attacked_site)

                kill_reward = 0.0
                if player_got_kill:
                    if at_attacked:
                        kill_reward += 3.0
                    else:
                        kill_reward += 1.0
                if player_died:
                    if at_wrong:
                        kill_reward -= 2.0
                    elif not player_got_kill:
                        kill_reward -= 0.5

                site_reward = 0.0
                if attacked_site:
                    if at_attacked:
                        site_reward += 2.0
                    elif rotating_toward:
                        site_reward += 1.5
                    elif at_wrong:
                        site_reward -= 1.0

                raw_combined = kill_reward + site_reward
                normalized_reward = raw_combined / REWARD_NORMALIZER

                win_reward = 0.0
                if is_terminal:
                    win_reward = 1.0 if winner == side else 0.0
                    if side == "CT" and len(bomb_defuses[
                            (bomb_defuses["tick"] >= fe) &
                            (bomb_defuses["tick"] <= end)]) > 0:
                        win_reward += 5.0
                    if side == "T" and plant_tick is not None:
                        win_reward += 3.0

                my_alive = t_alive if side == "T" else ct_alive
                enemy_alive = ct_alive if side == "T" else t_alive
                alive_adv = max(-3, min(3, my_alive - enemy_alive))

                recent_event = _recent_event_category(
                    side, tick, round_kills, round_grenades, plant_tick)

                rows.append({
                    "demo": os.path.basename(demo_path),
                    "round_num": i,
                    "tick": tick,
                    "time_elapsed": time_elapsed,
                    "player": name,
                    "side": side,
                    "side_idx": 0 if side == "T" else 1,
                    "alive_adv": alive_adv,
                    "bomb_status": bomb_status,
                    "time_bucket": time_bucket,
                    "zone_idx": zone_idx,
                    "recent_event": recent_event,
                    "team_support": team_support,
                    "action": action,
                    "action_ss": action_ss,
                    "kill_reward": normalized_reward,
                    "site_reward": 0.0,
                    "win_reward": win_reward,
                    "is_terminal": int(is_terminal),
                    "round_won": int(winner == side),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Event sequence extraction  (for LSTM / sequence NN)
# ---------------------------------------------------------------------------

def _extract_event_sequences(parser: DemoParser, demo_path: str) -> list[dict]:
    """Extract time-ordered event sequences per round for sequence model training."""
    freeze_ends = sorted(_safe_parse_event(parser, "round_freeze_end")
                         .get("tick", pd.Series(dtype=int)).tolist())
    round_ends = sorted(set(
        _safe_parse_event(parser, "round_officially_ended")
        .get("tick", pd.Series(dtype=int)).tolist()
    ))
    deaths_df = _safe_parse_event(parser, "player_death",
                                  player=["X", "Y", "Z", "team_num"])
    smokes_df = _safe_parse_event(parser, "smokegrenade_detonate",
                                  player=["X", "Y", "Z", "team_num"])
    flashes_df = _safe_parse_event(parser, "flashbang_detonate",
                                   player=["X", "Y", "Z", "team_num"])
    he_df = _safe_parse_event(parser, "hegrenade_detonate",
                              player=["X", "Y", "Z", "team_num"])
    molotov_df = _safe_parse_event(parser, "inferno_startburn",
                                   player=["X", "Y", "Z", "team_num"])
    bomb_plants = _safe_parse_event(parser, "bomb_planted",
                                    player=["X", "Y", "Z"],
                                    other=["total_rounds_played"])

    n_rounds = min(len(freeze_ends), len(round_ends))
    knife_mask = _knife_round_mask(parser, freeze_ends[:n_rounds],
                                    round_ends[:n_rounds])
    results = []

    for i in range(n_rounds):
        if knife_mask[i]:
            continue
        fe = freeze_ends[i]
        end = round_ends[i]
        round_duration = max((end - fe) / TICK_RATE, 1.0)
        events: list[dict] = []

        # Kills
        rd_deaths = deaths_df[(deaths_df["tick"] >= fe) & (deaths_df["tick"] <= end)]
        for _, row in rd_deaths.iterrows():
            victim_side = _team_str(row.get("user_team_num", 0))
            attacker_is_t = 1 if victim_side == "CT" else 0
            vx, vy = row.get("user_X", 0), row.get("user_Y", 0)
            zone = get_zone(float(vx), float(vy)) if vx else "MID"
            hs = int(row.get("headshot", 0)) if "headshot" in row.index else 0
            events.append({
                "tick": int(row["tick"]),
                "type_idx": SEQ_EVENT_IDX["kill"],
                "actor_side_is_t": attacker_is_t,
                "zone_idx": SEQ_ZONE_IDX.get(zone, 2),
                "time_norm": min((int(row["tick"]) - fe) / TICK_RATE / 120.0, 1.0),
                "is_headshot": hs,
            })

        # Grenades
        for gdf, gtype in [(smokes_df, "smoke"), (flashes_df, "flash"),
                           (he_df, "he"), (molotov_df, "molotov")]:
            rd_g = gdf[(gdf["tick"] >= fe) & (gdf["tick"] <= end)]
            for _, row in rd_g.iterrows():
                thrower_side = _team_str(row.get("user_team_num", 0))
                gx, gy = row.get("user_X", 0), row.get("user_Y", 0)
                zone = get_zone(float(gx), float(gy)) if gx else "MID"
                events.append({
                    "tick": int(row["tick"]),
                    "type_idx": SEQ_EVENT_IDX[gtype],
                    "actor_side_is_t": 1 if thrower_side == "T" else 0,
                    "zone_idx": SEQ_ZONE_IDX.get(zone, 2),
                    "time_norm": min((int(row["tick"]) - fe) / TICK_RATE / 120.0, 1.0),
                    "is_headshot": 0,
                })

        # Bomb plant
        rd_bombs = bomb_plants[(bomb_plants["tick"] >= fe) &
                               (bomb_plants["tick"] <= end)]
        bomb_site = ""
        if len(rd_bombs) > 0:
            bx = rd_bombs.iloc[0].get("user_X")
            by = rd_bombs.iloc[0].get("user_Y")
            if bx is not None and by is not None:
                bzone = get_zone(float(bx), float(by))
                bomb_site = "A" if bzone == "A" else ("B" if bzone == "B" else bzone)
            events.append({
                "tick": int(rd_bombs.iloc[0]["tick"]),
                "type_idx": SEQ_EVENT_IDX["plant"],
                "actor_side_is_t": 1,
                "zone_idx": SEQ_ZONE_IDX.get(bomb_site, 2),
                "time_norm": min((int(rd_bombs.iloc[0]["tick"]) - fe) / TICK_RATE / 120.0, 1.0),
                "is_headshot": 0,
            })

        events.sort(key=lambda e: e["tick"])
        for e in events:
            del e["tick"]

        attack_site = bomb_site if bomb_site in ("A", "B") else "no_plant"

        results.append({
            "demo": os.path.basename(demo_path),
            "round_num": i,
            "events": events,
            "attack_site": attack_site,
        })

    return results


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def discover_demos(base_dir: str) -> list[str]:
    """Find all .dem files recursively under base_dir."""
    demos = glob.glob(os.path.join(base_dir, "**", "*.dem"), recursive=True)
    demos.extend(glob.glob(os.path.join(base_dir, "*.dem")))
    return sorted(set(demos))


def extract_all(
    demo_dir: str = "src/demo",
    include_rl: bool = True,
    include_rl_v2: bool = False,
    include_sequences: bool = False,
    include_ct_formations: bool = False,
    verbose: bool = True,
) -> dict:
    """Parse all demos under ``demo_dir`` and return training DataFrames/sequences
    keyed by ``rounds``, ``rl_transitions``, ``rl_v2``, ``event_sequences``,
    and ``ct_formation_sequences``."""
    demos = discover_demos(demo_dir)
    if verbose:
        print(f"Found {len(demos)} demo files")

    all_rounds: list[pd.DataFrame] = []
    all_rl: list[pd.DataFrame] = []
    all_rl_v2: list[pd.DataFrame] = []
    all_sequences: list[dict] = []
    all_ct_formations: list[dict] = []

    for idx, path in enumerate(demos):
        name = os.path.basename(path)
        if verbose:
            print(f"  [{idx+1}/{len(demos)}] Parsing {name}...", end=" ", flush=True)

        try:
            p = DemoParser(path)

            rdf = _extract_round_features(p, path)
            all_rounds.append(rdf)
            if verbose:
                print(f"{len(rdf)} rounds", end="")

            if include_rl:
                rl_df = _extract_rl_transitions(p, path)
                all_rl.append(rl_df)
                if verbose:
                    print(f", {len(rl_df)} RL transitions", end="")

            if include_rl_v2:
                rl2_df = _extract_rl_v2(p, path)
                all_rl_v2.append(rl2_df)
                if verbose:
                    print(f", {len(rl2_df)} RLv2 transitions", end="")

            if include_sequences:
                seqs = _extract_event_sequences(p, path)
                all_sequences.extend(seqs)
                n_events = sum(len(s["events"]) for s in seqs)
                if verbose:
                    print(f", {len(seqs)} seqs ({n_events} events)", end="")

            if include_ct_formations:
                ct_seqs = _extract_ct_formation_sequences(p, path)
                all_ct_formations.extend(ct_seqs)
                if verbose:
                    print(f", {len(ct_seqs)} CT-form seqs", end="")

            if verbose:
                print(" OK")

        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
                traceback.print_exc()

    rounds_df = pd.concat(all_rounds, ignore_index=True) if all_rounds else pd.DataFrame()
    rl_df = pd.concat(all_rl, ignore_index=True) if all_rl else pd.DataFrame()
    rl_v2_df = pd.concat(all_rl_v2, ignore_index=True) if all_rl_v2 else pd.DataFrame()

    if verbose:
        print(f"\nTotal: {len(rounds_df)} rounds, {len(rl_df)} RL transitions, "
              f"{len(rl_v2_df)} RLv2 transitions, "
              f"{len(all_sequences)} event sequences, "
              f"{len(all_ct_formations)} CT formation sequences "
              f"from {len(demos)} demos")

    return {
        "rounds": rounds_df,
        "rl_transitions": rl_df,
        "rl_v2": rl_v2_df,
        "event_sequences": all_sequences,
        "ct_formation_sequences": all_ct_formations,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = extract_all("src/demo")
    rdf = data["rounds"]
    rl = data["rl_transitions"]

    print(f"\n{'='*60}")
    print(f"  Round Features: {len(rdf)} rows")
    print(f"{'='*60}")
    if not rdf.empty:
        print(f"  Demos: {rdf['demo'].nunique()}")
        print(f"  T wins: {rdf['t_won'].sum()} / {len(rdf)} "
              f"({rdf['t_won'].mean():.1%})")
        print(f"  Attack sites: "
              f"{rdf['attack_site'].value_counts().to_dict()}")
        print(f"  Top CT formations:")
        for fmt, cnt in rdf["ct_formation"].value_counts().head(6).items():
            print(f"    {fmt}: {cnt}")

    print(f"\n{'='*60}")
    print(f"  RL Transitions: {len(rl)} rows")
    print(f"{'='*60}")
    if not rl.empty:
        print(f"  Unique players: {rl['player'].nunique()}")
        act_map = {0: "ROTATE_A", 1: "ROTATE_B", 2: "HOLD",
                   3: "PUSH", 4: "FALL_BACK"}
        for a_idx, count in rl["action"].value_counts().sort_index().items():
            print(f"    {act_map.get(a_idx, '?'):12s}: {count}")
