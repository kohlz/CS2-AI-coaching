"""
training_data.py

Batch-extract training features from multiple CS2 demo files for:
  1. Round win probability estimator  (NN)
  2. T-side attack site predictor     (NN)
  3. CT defensive formation classifier (NN)
  4. Mid-round tactical Q-learning    (RL)

Usage
-----
    from training_data import extract_all

    data = extract_all("src/demo")
    round_df  = data["rounds"]           # one row per round
    rl_df     = data["rl_transitions"]   # one row per time-step per player
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

# RL v2 actions (micro-decision level)
RL_ACTIONS_V2 = ["PEEK", "HOLD", "TRADE", "FALL_BACK", "UTILITY", "ROTATE"]
RL_ACTION_V2_IDX = {a: i for i, a in enumerate(RL_ACTIONS_V2)}

# Zone aggression: how deep into enemy territory (0=safest, 3=deepest)
_CT_AGGRESSION = {"CT_BASE": 0, "A": 1, "B": 1, "MID": 2, "T_BASE": 3}
_T_AGGRESSION = {"T_BASE": 0, "MID": 1, "A": 2, "B": 2, "CT_BASE": 3}

# Recent event categories for v2 state
RECENT_EVENTS = ["none", "teammate_died", "enemy_killed", "grenade", "bomb_planted"]
RECENT_EVENT_IDX = {e: i for i, e in enumerate(RECENT_EVENTS)}

# Event types for sequence model
SEQ_EVENT_TYPES = ["kill", "smoke", "flash", "he", "plant"]
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
    rows = []
    t_loss_streak = 0
    ct_loss_streak = 0

    for i in range(n_rounds):
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

    # Build sample ticks (every 5 seconds)
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
        ticks = sample_ticks_by_round[i]
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

    Priority: UTILITY > TRADE > zone-based (PEEK / FALL_BACK / ROTATE) > HOLD
    """
    # 1. UTILITY — player threw a grenade in this window
    for g in grenades_window:
        if g["thrower"] == player_name:
            return RL_ACTION_V2_IDX["UTILITY"]

    # 2. TRADE — a teammate died in this window AND player got a kill
    if teammate_deaths_window:
        for k in kills_window:
            if k["attacker"] == player_name:
                return RL_ACTION_V2_IDX["TRADE"]

    # 3. Zone-change-based actions
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

    n_rounds = min(len(freeze_ends), len(round_ends))
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

    def _build_grenade_list(smoke_df, flash_df, he_df_, fe, end):
        grenades = []
        for gdf, gtype in [(smoke_df, "smoke"), (flash_df, "flash"), (he_df_, "he")]:
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
        ticks = sample_ticks_by_round[i]
        fe = freeze_ends[i]
        end = round_ends[i]
        winner = round_winners.get(i, "CT")
        plant_tick, plant_site = bomb_info[i]

        round_kills = _build_kill_list(deaths_df, fe, end)
        round_grenades = _build_grenade_list(smokes_df, flashes_df, he_df, fe, end)

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
            bomb_status = 0
            if plant_tick is not None and tick >= plant_tick:
                bomb_status = 1 if plant_site == "A" else 2
            time_bucket = (3 if bomb_status > 0 else
                           0 if time_elapsed < 30 else
                           1 if time_elapsed < 60 else 2)

            is_terminal = (t_idx == len(ticks) - 2)

            # Events in this 5s window for action classification
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

                # Teammate deaths in this window
                teammate_deaths = [k for k in kills_window
                                   if k["victim_side"] == side
                                   and k["victim"] != name]

                action = _classify_action_v2(
                    side, zone, next_zone, name, tick, next_tick,
                    kills_window, teammate_deaths, grenades_window)

                # --- Determine attacked site for this round ---
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

                # Dual rewards with site-alignment
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

                win_reward = 0.0
                if is_terminal:
                    win_reward = 1.0 if winner == side else 0.0
                    # Objective bonuses
                    if side == "CT" and len(bomb_defuses[
                            (bomb_defuses["tick"] >= fe) &
                            (bomb_defuses["tick"] <= end)]) > 0:
                        win_reward += 5.0
                    if side == "T" and plant_tick is not None:
                        win_reward += 3.0

                # Alive advantage from this player's perspective
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
                    "action": action,
                    "kill_reward": kill_reward,
                    "site_reward": site_reward,
                    "win_reward": win_reward,
                    "is_terminal": int(is_terminal),
                    "round_won": int(winner == side),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Event sequence extraction  (for LSTM / sequence NN)
# ---------------------------------------------------------------------------

def _extract_event_sequences(parser: DemoParser, demo_path: str) -> list[dict]:
    """Extract time-ordered event sequences per round for sequence model training.

    Returns a list of dicts, one per round:
      {"demo", "round_num", "events": list[dict], "attack_site": str}

    Each event dict has keys:
      type_idx, actor_side_is_t, zone_idx, time_norm, is_headshot
    """
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
    bomb_plants = _safe_parse_event(parser, "bomb_planted",
                                    player=["X", "Y", "Z"],
                                    other=["total_rounds_played"])

    n_rounds = min(len(freeze_ends), len(round_ends))
    results = []

    for i in range(n_rounds):
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
        for gdf, gtype in [(smokes_df, "smoke"), (flashes_df, "flash"), (he_df, "he")]:
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
    verbose: bool = True,
) -> dict:
    """Parse all demos and return training DataFrames.

    Returns dict with keys:
      "rounds"           — round-level features (NN)
      "rl_transitions"   — v1 RL transitions (legacy)
      "rl_v2"            — v2 RL transitions (micro-decisions)
      "event_sequences"  — event sequences per round (LSTM)
    """
    demos = discover_demos(demo_dir)
    if verbose:
        print(f"Found {len(demos)} demo files")

    all_rounds: list[pd.DataFrame] = []
    all_rl: list[pd.DataFrame] = []
    all_rl_v2: list[pd.DataFrame] = []
    all_sequences: list[dict] = []

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
              f"{len(all_sequences)} event sequences "
              f"from {len(demos)} demos")

    return {
        "rounds": rounds_df,
        "rl_transitions": rl_df,
        "rl_v2": rl_v2_df,
        "event_sequences": all_sequences,
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
