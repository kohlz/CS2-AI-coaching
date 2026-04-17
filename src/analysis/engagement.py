"""
engagement.py

Heuristic engagement analysis for a target player across a match,
covering trades, isolated deaths, opening duels, utility usage, and
impact kills. Operates on structured RoundData from demo_parser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
from callouts_mirage import get_zone

TICK_RATE = 64
TRADE_WINDOW_SEC = 5.0


@dataclass
class RoundEngagement:
    """Engagement metrics for one round."""
    round_num: int
    side: str
    kills: int = 0
    deaths: int = 0
    damage: int = 0
    adr_equiv: float = 0.0

    is_opening_kill: bool = False
    is_opening_death: bool = False
    traded_teammates: int = 0
    untraded_teammate_deaths: int = 0
    was_traded: bool = False
    was_isolated_death: bool = False

    trade_time: Optional[float] = None
    utility_thrown: int = 0
    has_rifle: bool = False

    multi_kill: int = 0        # 2k, 3k, 4k, ace
    clutch_win: bool = False   # won 1vN
    clutch_attempted: int = 0  # N in 1vN

    notes: list[str] = field(default_factory=list)


def _distance_2d(x1, y1, x2, y2) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def analyze_engagement(
    rounds: list,
    target_player: str,
) -> list[RoundEngagement]:
    """Analyze engagement for every round of a match."""

    results: list[RoundEngagement] = []

    for rd in rounds:
        p = rd.get_player(target_player)
        if p is None:
            continue

        eng = RoundEngagement(
            round_num=rd.round_num,
            side=p.side,
            kills=p.kills,
            deaths=min(p.deaths, 1),
            damage=p.damage,
            adr_equiv=p.damage,
        )

        eng.utility_thrown = len(p.utilities)
        eng.has_rifle = p.primary_weapon is not None and any(
            w in (p.primary_weapon or "")
            for w in ("AK-47", "M4A4", "M4A1-S", "AUG", "SG 553",
                      "AWP", "SSG 08", "Galil AR", "FAMAS")
        )

        kill_events = sorted(
            [e for e in rd.events if e.event_type == "kill"],
            key=lambda e: e.tick,
        )

        if not kill_events:
            results.append(eng)
            continue

        first_kill = kill_events[0]
        if first_kill.data.get("attacker") == target_player:
            eng.is_opening_kill = True
            eng.notes.append("Won opening duel")
        elif first_kill.data.get("victim") == target_player:
            eng.is_opening_death = True
            eng.notes.append("Lost opening duel")

        if p.kills >= 2:
            eng.multi_kill = p.kills
            eng.notes.append(f"{p.kills}K round")

        teammates = (
            [pl.name for pl in rd.t_players if pl.name != target_player]
            if p.side == "T" else
            [pl.name for pl in rd.ct_players if pl.name != target_player]
        )

        target_kill_ticks = [
            e.tick for e in kill_events
            if e.data.get("attacker") == target_player
        ]

        for e in kill_events:
            victim = e.data.get("victim", "")
            if victim not in teammates:
                continue

            traded = False
            for kt in target_kill_ticks:
                if 0 < (kt - e.tick) <= TRADE_WINDOW_SEC * TICK_RATE:
                    traded = True
                    eng.traded_teammates += 1
                    time_diff = (kt - e.tick) / TICK_RATE
                    if eng.trade_time is None or time_diff < eng.trade_time:
                        eng.trade_time = time_diff
                    break
            if not traded:
                eng.untraded_teammate_deaths += 1

        if p.deaths > 0 and p.death_tick is not None:
            traded_back = False
            enemy_names = set(
                pl.name for pl in (rd.ct_players if p.side == "T" else rd.t_players)
            )
            for e in kill_events:
                attacker = e.data.get("attacker", "")
                if attacker in teammates:
                    if 0 < (e.tick - p.death_tick) <= TRADE_WINDOW_SEC * TICK_RATE:
                        traded_back = True
                        break
            eng.was_traded = traded_back
            if not traded_back:
                eng.was_isolated_death = True
                eng.notes.append("Died without trade")

        if p.side in ("T", "CT"):
            my_team = rd.t_players if p.side == "T" else rd.ct_players
            enemy_team = rd.ct_players if p.side == "T" else rd.t_players

            teammate_deaths_before_target = 0
            for tm in my_team:
                if tm.name != target_player and tm.death_tick is not None:
                    if p.death_tick is None or tm.death_tick < p.death_tick:
                        teammate_deaths_before_target += 1

            if teammate_deaths_before_target >= 4:
                enemies_alive_at_clutch = sum(
                    1 for e in enemy_team
                    if e.alive_at_end or (e.death_tick is not None and
                        (p.death_tick is None or e.death_tick > min(
                            tm.death_tick for tm in my_team
                            if tm.name != target_player and tm.death_tick is not None
                        )))
                )
                eng.clutch_attempted = max(1, enemies_alive_at_clutch)
                winner = rd.winner
                if winner == p.side:
                    eng.clutch_win = True
                    eng.notes.append(f"1v{eng.clutch_attempted} clutch WIN")
                else:
                    eng.notes.append(f"1v{eng.clutch_attempted} clutch attempt")

        results.append(eng)

    return results


def engagement_summary(evaluations: list[RoundEngagement]) -> dict:
    """Aggregate engagement stats into a summary dict."""
    if not evaluations:
        return {}

    n = len(evaluations)
    total_kills = sum(e.kills for e in evaluations)
    total_deaths = sum(e.deaths for e in evaluations)
    total_damage = sum(e.damage for e in evaluations)

    opening_kills = sum(1 for e in evaluations if e.is_opening_kill)
    opening_deaths = sum(1 for e in evaluations if e.is_opening_death)
    opening_duels = opening_kills + opening_deaths

    traded = sum(e.traded_teammates for e in evaluations)
    untraded = sum(e.untraded_teammate_deaths for e in evaluations)
    trade_opportunities = traded + untraded

    isolated = sum(1 for e in evaluations if e.was_isolated_death)
    death_rounds = sum(1 for e in evaluations if e.deaths > 0)

    multi_2k = sum(1 for e in evaluations if e.multi_kill >= 2)
    multi_3k = sum(1 for e in evaluations if e.multi_kill >= 3)
    clutches_won = sum(1 for e in evaluations if e.clutch_win)
    clutches_attempted = sum(1 for e in evaluations if e.clutch_attempted > 0)

    trade_times = [e.trade_time for e in evaluations if e.trade_time is not None]
    avg_trade_time = sum(trade_times) / len(trade_times) if trade_times else None

    rifle_rounds = [e for e in evaluations if e.has_rifle]
    rifle_util = sum(e.utility_thrown for e in rifle_rounds)
    avg_util_on_rifle = rifle_util / len(rifle_rounds) if rifle_rounds else 0

    return {
        "rounds": n,
        "kills": total_kills,
        "deaths": total_deaths,
        "kd_ratio": total_kills / max(total_deaths, 1),
        "adr": total_damage / n,
        "opening_duels": opening_duels,
        "opening_kills": opening_kills,
        "opening_deaths": opening_deaths,
        "opening_success_rate": opening_kills / opening_duels if opening_duels else 0.0,
        "trade_opportunities": trade_opportunities,
        "successful_trades": traded,
        "trade_rate": traded / trade_opportunities if trade_opportunities else 0.0,
        "avg_trade_time_sec": avg_trade_time,
        "isolated_deaths": isolated,
        "isolated_death_rate": isolated / death_rounds if death_rounds else 0.0,
        "multi_kills_2k": multi_2k,
        "multi_kills_3k": multi_3k,
        "clutches_won": clutches_won,
        "clutches_attempted": clutches_attempted,
        "avg_util_on_rifle_round": avg_util_on_rifle,
        "grade": _engagement_grade(
            total_kills / max(total_deaths, 1),
            total_damage / n,
            traded / trade_opportunities if trade_opportunities else 0.5,
        ),
    }


def _engagement_grade(kd: float, adr: float, trade_rate: float) -> str:
    score = 0.0
    score += min(kd / 1.5, 1.0) * 35
    score += min(adr / 90.0, 1.0) * 35
    score += trade_rate * 30

    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    if score >= 35:
        return "D"
    return "F"
