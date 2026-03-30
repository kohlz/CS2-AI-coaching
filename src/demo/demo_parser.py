"""
demo_parser.py

Parse a CS2 demo file (.dem) into structured per-round match data.

Usage
-----
    from demo_parser import parse_demo

    match = parse_demo("path/to/demo.dem", target_player="PlayerName")
    for r in match.rounds:
        print(r.round_num, r.winner, r.events)

The target_player parameter specifies which player's perspective the
coaching system should analyze.  All round data is structured to make
it easy to answer questions like "what did the target player know at
tick T?" and "was their buy decision optimal?"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from demoparser2 import DemoParser


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

TICK_RATE = 64  # CS2 standard demo tick rate


@dataclass
class Position:
    x: float
    y: float
    z: float
    yaw: float = 0.0
    pitch: float = 0.0

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class GameEvent:
    tick: int
    time_in_round: float
    event_type: str
    data: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"GameEvent({self.time_in_round:.1f}s {self.event_type} {self.data})"


@dataclass
class PlayerRound:
    name: str
    steamid: str
    side: str               # "T" or "CT"
    start_money: int = 0
    equipment_value: int = 0
    weapon: str = ""         # active weapon at freeze end (legacy)
    primary_weapon: Optional[str] = None
    secondary_weapon: Optional[str] = None
    utilities: list[str] = field(default_factory=list)
    has_kit: bool = False
    has_bomb: bool = False
    has_helmet: bool = False
    armor: int = 0
    kills: int = 0
    assists: int = 0
    deaths: int = 0
    damage: int = 0
    alive_at_end: bool = True
    death_tick: Optional[int] = None
    positions: list[Position] = field(default_factory=list)


@dataclass
class RoundData:
    round_num: int
    tick_freeze_end: int
    tick_end: int
    winner: str                         # "T" or "CT"
    win_reason: str                     # "elimination", "bomb_explode", "bomb_defuse", "time"
    t_players: list[PlayerRound] = field(default_factory=list)
    ct_players: list[PlayerRound] = field(default_factory=list)
    events: list[GameEvent] = field(default_factory=list)
    bomb_planted: bool = False
    bomb_site: Optional[str] = None

    @property
    def duration(self) -> float:
        return (self.tick_end - self.tick_freeze_end) / TICK_RATE

    def get_player(self, name: str) -> Optional[PlayerRound]:
        for p in self.t_players + self.ct_players:
            if p.name == name:
                return p
        return None


@dataclass
class MatchData:
    map_name: str
    target_player: str
    target_steamid: str
    players: dict[str, dict]            # name → {steamid, team_num}
    rounds: list[RoundData] = field(default_factory=list)

    @property
    def t_score(self) -> int:
        return sum(1 for r in self.rounds if r.winner == "T")

    @property
    def ct_score(self) -> int:
        return sum(1 for r in self.rounds if r.winner == "CT")

    def target_side(self, round_num: int) -> str:
        rd = self.rounds[round_num]
        p = rd.get_player(self.target_player)
        return p.side if p else "unknown"


# ---------------------------------------------------------------------------
# Team number mapping
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Weapon / item classification
# ---------------------------------------------------------------------------

PRIMARY_WEAPONS = {
    # Rifles
    "AK-47", "M4A4", "M4A1-S", "SG 553", "AUG", "Galil AR", "FAMAS",
    # Snipers
    "AWP", "SSG 08", "G3SG1", "SCAR-20",
    # SMGs
    "MP9", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "MAC-10",
    # Shotguns
    "XM1014", "Nova", "MAG-7", "Sawed-Off",
    # LMGs
    "M249", "Negev",
}

SECONDARY_WEAPONS = {
    "Glock-18", "USP-S", "P2000", "P250", "Five-SeveN", "Tec-9",
    "Dual Berettas", "Desert Eagle", "CZ75-Auto", "R8 Revolver",
}

UTILITY_ITEMS = {
    "Smoke Grenade", "Flashbang", "High Explosive Grenade",
    "Molotov", "Incendiary Grenade", "Decoy Grenade",
}

BOMB_ITEM = "C4 Explosive"


def _classify_inventory(
    inventory: list[str],
    has_defuser: bool,
    has_helmet: bool,
    armor_value: int,
) -> dict:
    """Classify a raw inventory list into structured loadout fields."""
    primary = None
    secondary = None
    utilities: list[str] = []
    has_bomb = False

    for item in inventory:
        if item in PRIMARY_WEAPONS:
            primary = item
        elif item in SECONDARY_WEAPONS:
            secondary = item
        elif item in UTILITY_ITEMS:
            utilities.append(item)
        elif item == BOMB_ITEM:
            has_bomb = True
        # else: knife or unknown skin name — skip

    return {
        "primary_weapon": primary,
        "secondary_weapon": secondary,
        "utilities": utilities,
        "has_kit": has_defuser,
        "has_bomb": has_bomb,
        "has_helmet": has_helmet,
        "armor": armor_value,
    }


def _team_str(team_num: int | float) -> str:
    if team_num == 2:
        return "T"
    if team_num == 3:
        return "CT"
    return "unknown"


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def parse_demo(demo_path: str, target_player: str) -> MatchData:
    """
    Parse a CS2 demo file and return structured match data.

    Parameters
    ----------
    demo_path : str
        Path to the .dem file.
    target_player : str
        Name of the player to analyze.  Must match the in-game name
        exactly as it appears in the demo.

    Returns
    -------
    MatchData
        Complete match data segmented into rounds.
    """
    parser = DemoParser(demo_path)
    header = parser.parse_header()

    # ── Player info ──────────────────────────────────────────────────
    player_info = parser.parse_player_info()
    players = {}
    target_steamid = None
    for _, row in player_info.iterrows():
        name = row["name"]
        sid = row["steamid"]
        players[name] = {"steamid": sid, "team_number": int(row["team_number"])}
        if name == target_player:
            target_steamid = sid

    if target_steamid is None:
        available = list(players.keys())
        raise ValueError(
            f"Player '{target_player}' not found in demo. "
            f"Available players: {available}"
        )

    # ── Events ───────────────────────────────────────────────────────
    deaths = parser.parse_event(
        "player_death",
        player=["X", "Y", "Z", "team_num"],
        other=["total_rounds_played"],
    )
    hurts = parser.parse_event(
        "player_hurt",
        other=["total_rounds_played"],
    )
    bomb_plants = parser.parse_event(
        "bomb_planted",
        other=["total_rounds_played"],
    )
    bomb_defuses = parser.parse_event("bomb_defused", other=["total_rounds_played"])
    bomb_explodes = parser.parse_event("bomb_exploded", other=["total_rounds_played"])

    smokes = parser.parse_event(
        "smokegrenade_detonate",
        player=["X", "Y", "Z"],
        other=["total_rounds_played"],
    )
    flashes = parser.parse_event(
        "flashbang_detonate",
        player=["X", "Y", "Z"],
        other=["total_rounds_played"],
    )
    he_grenades = parser.parse_event(
        "hegrenade_detonate",
        player=["X", "Y", "Z"],
        other=["total_rounds_played"],
    )

    freeze_ends = sorted(parser.parse_event("round_freeze_end")["tick"].tolist())
    round_ends_raw = sorted(parser.parse_event("round_officially_ended")["tick"].tolist())
    round_ends = sorted(set(round_ends_raw))

    buytime_ended_ticks = sorted(
        parser.parse_event("buytime_ended")["tick"].tolist()
    )

    # ── Tick data at round boundaries (for economy) ──────────────────
    economy_ticks = []
    for fe_tick in freeze_ends:
        economy_ticks.append(fe_tick)
        economy_ticks.append(fe_tick + 1)

    econ_df = parser.parse_ticks(
        ["balance", "current_equip_value", "team_num",
         "active_weapon_name", "is_alive", "health",
         "total_rounds_played"],
        ticks=economy_ticks,
    )

    # ── Inventory snapshot at buytime_ended (post-buy loadout) ────────
    inv_df = None
    if buytime_ended_ticks:
        inv_df = parser.parse_ticks(
            ["inventory", "has_defuser", "has_helmet", "armor_value",
             "team_num"],
            ticks=buytime_ended_ticks,
        )

    # ── Position snapshots (sample every 2 seconds during rounds) ────
    position_ticks = []
    for i, fe_tick in enumerate(freeze_ends):
        end_tick = round_ends[i] if i < len(round_ends) else fe_tick + 115 * TICK_RATE
        t = fe_tick
        while t <= end_tick:
            position_ticks.append(t)
            t += 2 * TICK_RATE  # every 2 seconds

    pos_df = parser.parse_ticks(
        ["X", "Y", "Z", "pitch", "yaw", "team_num", "is_alive",
         "total_rounds_played"],
        ticks=position_ticks,
    )

    # ── Build rounds ─────────────────────────────────────────────────
    n_rounds = min(len(freeze_ends), len(round_ends))
    match = MatchData(
        map_name=header.get("map_name", "unknown"),
        target_player=target_player,
        target_steamid=target_steamid,
        players=players,
    )

    for i in range(n_rounds):
        fe_tick = freeze_ends[i]
        end_tick = round_ends[i]
        next_fe = freeze_ends[i + 1] if i + 1 < len(freeze_ends) else end_tick

        # Find the buytime_ended tick for this round
        bt_tick = None
        for bt in buytime_ended_ticks:
            if fe_tick <= bt <= next_fe:
                bt_tick = bt
                break

        rd = _build_round(
            round_num=i,
            tick_freeze_end=fe_tick,
            tick_end=end_tick,
            deaths=deaths,
            hurts=hurts,
            bomb_plants=bomb_plants,
            bomb_defuses=bomb_defuses,
            bomb_explodes=bomb_explodes,
            smokes=smokes,
            flashes=flashes,
            he_grenades=he_grenades,
            econ_df=econ_df,
            pos_df=pos_df,
            fe_tick=fe_tick,
            inv_df=inv_df,
            bt_tick=bt_tick,
        )
        match.rounds.append(rd)

    return match


# ---------------------------------------------------------------------------
# Round builder
# ---------------------------------------------------------------------------

def _build_round(
    round_num: int,
    tick_freeze_end: int,
    tick_end: int,
    deaths: pd.DataFrame,
    hurts: pd.DataFrame,
    bomb_plants: pd.DataFrame,
    bomb_defuses: pd.DataFrame,
    bomb_explodes: pd.DataFrame,
    smokes: pd.DataFrame,
    flashes: pd.DataFrame,
    he_grenades: pd.DataFrame,
    econ_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    fe_tick: int,
    inv_df: Optional[pd.DataFrame] = None,
    bt_tick: Optional[int] = None,
) -> RoundData:

    # Filter events to this round's tick range
    def _in_round(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["tick"] >= tick_freeze_end) & (df["tick"] <= tick_end)]

    rd_deaths = _in_round(deaths)
    rd_hurts = _in_round(hurts)
    rd_bombs = _in_round(bomb_plants)
    rd_defuses = _in_round(bomb_defuses)
    rd_explodes = _in_round(bomb_explodes)
    rd_smokes = _in_round(smokes)
    rd_flashes = _in_round(flashes)
    rd_hes = _in_round(he_grenades)

    # ── Economy snapshot at freeze end ───────────────────────────────
    econ_snap = econ_df[econ_df["tick"] == fe_tick]

    t_players: dict[str, PlayerRound] = {}
    ct_players: dict[str, PlayerRound] = {}

    for _, row in econ_snap.iterrows():
        name = row["name"]
        sid = row["steamid"]
        side = _team_str(row["team_num"])
        pr = PlayerRound(
            name=name,
            steamid=sid,
            side=side,
            start_money=int(row.get("balance", 0)),
            equipment_value=int(row.get("current_equip_value", 0)),
            weapon=str(row.get("active_weapon_name", "")),
        )
        if side == "T":
            t_players[name] = pr
        elif side == "CT":
            ct_players[name] = pr

    # ── Inventory snapshot at buytime_ended (post-buy loadout) ───────
    if inv_df is not None and bt_tick is not None:
        inv_snap = inv_df[inv_df["tick"] == bt_tick]
        for _, row in inv_snap.iterrows():
            name = row["name"]
            raw_inv = row.get("inventory", [])
            if raw_inv is None:
                raw_inv = []
            loadout = _classify_inventory(
                raw_inv,
                bool(row.get("has_defuser", False)),
                bool(row.get("has_helmet", False)),
                int(row.get("armor_value", 0)),
            )
            for pool in (t_players, ct_players):
                if name in pool:
                    pr = pool[name]
                    pr.primary_weapon = loadout["primary_weapon"]
                    pr.secondary_weapon = loadout["secondary_weapon"]
                    pr.utilities = loadout["utilities"]
                    pr.has_kit = loadout["has_kit"]
                    pr.has_bomb = loadout["has_bomb"]
                    pr.has_helmet = loadout["has_helmet"]
                    pr.armor = loadout["armor"]

    # ── Events timeline ──────────────────────────────────────────────
    events: list[GameEvent] = []

    for _, row in rd_deaths.iterrows():
        t = float(row["tick"] - fe_tick) / TICK_RATE
        victim = row.get("user_name", "")
        attacker = row.get("attacker_name", "")
        weapon = row.get("weapon", "")
        headshot = bool(row.get("headshot", False))

        events.append(GameEvent(
            tick=int(row["tick"]),
            time_in_round=t,
            event_type="kill",
            data={
                "attacker": attacker,
                "victim": victim,
                "weapon": weapon,
                "headshot": headshot,
                "victim_x": row.get("user_X"),
                "victim_y": row.get("user_Y"),
                "victim_z": row.get("user_Z"),
                "attacker_x": row.get("attacker_X"),
                "attacker_y": row.get("attacker_Y"),
                "attacker_z": row.get("attacker_Z"),
            },
        ))

        # Update player kill/death stats
        victim_side = _team_str(row.get("user_team_num", 0))
        pool = t_players if victim_side == "T" else ct_players
        if victim in pool:
            pool[victim].deaths += 1
            pool[victim].alive_at_end = False
            pool[victim].death_tick = int(row["tick"])

        att_side = _team_str(row.get("attacker_team_num", 0))
        att_pool = t_players if att_side == "T" else ct_players
        if attacker in att_pool:
            att_pool[attacker].kills += 1

    # Damage totals
    for _, row in rd_hurts.iterrows():
        attacker = row.get("attacker_name", "")
        dmg = int(row.get("dmg_health", 0))
        for pool in (t_players, ct_players):
            if attacker in pool:
                pool[attacker].damage += dmg

    # Bomb events
    bomb_planted = len(rd_bombs) > 0
    bomb_site = None
    if bomb_planted:
        first_plant = rd_bombs.iloc[0]
        events.append(GameEvent(
            tick=int(first_plant["tick"]),
            time_in_round=float(first_plant["tick"] - fe_tick) / TICK_RATE,
            event_type="bomb_plant",
            data={
                "planter": first_plant.get("user_name", ""),
                "site": str(first_plant.get("site", "")),
            },
        ))
        bomb_site = str(first_plant.get("site", ""))

    bomb_defused = len(rd_defuses) > 0
    if bomb_defused:
        d = rd_defuses.iloc[0]
        events.append(GameEvent(
            tick=int(d["tick"]),
            time_in_round=float(d["tick"] - fe_tick) / TICK_RATE,
            event_type="bomb_defuse",
            data={"defuser": d.get("user_name", "")},
        ))

    bomb_exploded = len(rd_explodes) > 0
    if bomb_exploded:
        e = rd_explodes.iloc[0]
        events.append(GameEvent(
            tick=int(e["tick"]),
            time_in_round=float(e["tick"] - fe_tick) / TICK_RATE,
            event_type="bomb_explode",
            data={},
        ))

    # Grenade events
    for label, df in [("smoke", rd_smokes), ("flash", rd_flashes), ("he_grenade", rd_hes)]:
        for _, row in df.iterrows():
            events.append(GameEvent(
                tick=int(row["tick"]),
                time_in_round=float(row["tick"] - fe_tick) / TICK_RATE,
                event_type=label,
                data={
                    "thrower": row.get("user_name", ""),
                    "x": row.get("x", row.get("user_X")),
                    "y": row.get("y", row.get("user_Y")),
                    "z": row.get("z", row.get("user_Z")),
                },
            ))

    events.sort(key=lambda e: e.tick)

    # ── Positions (sampled every 2s) ─────────────────────────────────
    rd_pos = pos_df[
        (pos_df["tick"] >= fe_tick) & (pos_df["tick"] <= tick_end)
    ]
    for _, row in rd_pos.iterrows():
        name = row["name"]
        if not row.get("is_alive", True):
            continue
        pos = Position(
            x=float(row.get("X", 0)),
            y=float(row.get("Y", 0)),
            z=float(row.get("Z", 0)),
            yaw=float(row.get("yaw", 0)),
            pitch=float(row.get("pitch", 0)),
        )
        for pool in (t_players, ct_players):
            if name in pool:
                pool[name].positions.append(pos)

    # ── Determine round winner ───────────────────────────────────────
    if bomb_exploded:
        winner, win_reason = "T", "bomb_explode"
    elif bomb_defused:
        winner, win_reason = "CT", "bomb_defuse"
    else:
        t_alive = sum(1 for p in t_players.values() if p.alive_at_end)
        ct_alive = sum(1 for p in ct_players.values() if p.alive_at_end)
        if t_alive == 0 and ct_alive > 0:
            winner, win_reason = "CT", "elimination"
        elif ct_alive == 0 and t_alive > 0:
            winner, win_reason = "T", "elimination"
        else:
            winner, win_reason = "CT", "time"

    return RoundData(
        round_num=round_num,
        tick_freeze_end=fe_tick,
        tick_end=tick_end,
        winner=winner,
        win_reason=win_reason,
        t_players=list(t_players.values()),
        ct_players=list(ct_players.values()),
        events=events,
        bomb_planted=bomb_planted,
        bomb_site=bomb_site,
    )


# ---------------------------------------------------------------------------
# CLI: quick demo inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python demo_parser.py <demo.dem> <player_name>")
        sys.exit(1)

    match = parse_demo(sys.argv[1], sys.argv[2])
    print(f"Map: {match.map_name}")
    print(f"Target: {match.target_player} ({match.target_steamid})")
    print(f"Players: {list(match.players.keys())}")
    print(f"Score: T {match.t_score} - {match.ct_score} CT")
    print(f"Rounds: {len(match.rounds)}")
    print()

    for rd in match.rounds:
        target = rd.get_player(match.target_player)
        side = target.side if target else "?"
        money = target.start_money if target else 0
        equip = target.equipment_value if target else 0
        kills = target.kills if target else 0
        deaths = target.deaths if target else 0
        dmg = target.damage if target else 0
        alive = target.alive_at_end if target else False

        kill_events = [e for e in rd.events if e.event_type == "kill"]
        bomb_ev = [e for e in rd.events if "bomb" in e.event_type]

        print(f"  R{rd.round_num:2d} [{side}] "
              f"W:{rd.winner}({rd.win_reason:14s}) "
              f"${money:5d} equip=${equip:5d} "
              f"K:{kills} D:{deaths} DMG:{dmg:3d} "
              f"{'alive' if alive else 'DEAD ':5s} "
              f"kills={len(kill_events)} "
              f"{'BOMB' if rd.bomb_planted else '    '} "
              f"{rd.duration:.0f}s")
