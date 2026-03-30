"""Quick test: extract k_z_ loadout and movements in a real buy round."""
import sys
sys.path.insert(0, "src/demo")
from demo_parser import parse_demo
from callouts_mirage import get_callout, get_zone

match = parse_demo("src/demo/260319mirage.dem", "k_z_")
print(f"Map: {match.map_name}")
print(f"Player: {match.target_player} ({match.target_steamid})")
print(f"Score: T {match.t_score} - {match.ct_score} CT")
print(f"Rounds: {len(match.rounds)}\n")

# Skip knife (round 0) and pistol (round 1); find a buy round with kills
for rd in match.rounds:
    if rd.round_num < 3:
        continue
    p = rd.get_player("k_z_")
    if not p or p.primary_weapon is None or p.kills == 0 or len(p.positions) < 4:
        continue

    print(f"=== Round {rd.round_num} ({p.side} side) ===")
    print(f"Economy:  start=${p.start_money}  equip=${p.equipment_value}")
    print(f"Primary:  {p.primary_weapon or '(none)'}")
    print(f"Secondary:{p.secondary_weapon or '(none)'}")
    print(f"Utilities:{p.utilities if p.utilities else '(none)'}")
    if p.side == "CT":
        print(f"Kit:      {'Yes' if p.has_kit else 'No'}")
    else:
        print(f"Bomb:     {'Yes' if p.has_bomb else 'No'}")
    print(f"Armor:    {p.armor}  Helmet: {'Yes' if p.has_helmet else 'No'}")
    print(f"Stats:    K:{p.kills} D:{p.deaths} DMG:{p.damage}  alive_at_end={p.alive_at_end}")
    print(f"Winner:   {rd.winner} ({rd.win_reason})  duration={rd.duration:.0f}s")
    print()

    # Show all teammate loadouts for context
    team_pool = rd.ct_players if p.side == "CT" else rd.t_players
    print(f"Team loadout ({p.side}):")
    for tp in team_pool:
        marker = " <<<" if tp.name == "k_z_" else ""
        kit_str = " [KIT]" if tp.has_kit else ""
        bomb_str = " [BOMB]" if tp.has_bomb else ""
        helmet_str = f"  helmet={'Y' if tp.has_helmet else 'N'}  armor={tp.armor}"
        utils = ", ".join(tp.utilities) if tp.utilities else "-"
        print(f"  {tp.name:>14s}  ${tp.start_money:5d}  "
              f"pri={tp.primary_weapon or '-':>10s}  "
              f"sec={tp.secondary_weapon or '-':>10s}  "
              f"util=[{utils}]"
              f"{helmet_str}{kit_str}{bomb_str}{marker}")
    print()

    # Events in the round
    print("Round events:")
    for e in rd.events:
        d = e.data
        if e.event_type == "kill":
            att = d.get("attacker") or "world"
            vic = d.get("victim") or "?"
            hs = " (HS)" if d.get("headshot") else ""
            loc = ""
            if d.get("victim_x") is not None:
                loc = get_callout(d["victim_x"], d["victim_y"])
            marker = " <<<" if att == "k_z_" or vic == "k_z_" else ""
            print(f"  {e.time_in_round:6.1f}s  {att:>14s} killed {vic:<14s} "
                  f"[{d.get('weapon','?'):>12s}]{hs:5s}  @ {loc}{marker}")
        elif e.event_type in ("smoke", "flash", "he_grenade"):
            thrower = d.get("thrower", "?")
            marker = " <<<" if thrower == "k_z_" else ""
            loc = ""
            if d.get("x") is not None:
                loc = get_callout(d["x"], d["y"])
            print(f"  {e.time_in_round:6.1f}s  {thrower:>14s} threw {e.event_type:<14s} @ {loc}{marker}")
        elif e.event_type == "bomb_plant":
            print(f"  {e.time_in_round:6.1f}s  bomb_plant  (site={d.get('site','')})")
        elif e.event_type in ("bomb_defuse", "bomb_explode"):
            print(f"  {e.time_in_round:6.1f}s  {e.event_type}")
    print()

    # Position timeline (only while alive)
    print("k_z_ position timeline:")
    print(f"{'time':>6s}  {'callout':>16s}  {'zone':>10s}  {'facing':>8s}  {'vert':>6s}")
    print("-" * 60)
    for i, pos in enumerate(p.positions):
        t = i * 2.0
        callout = get_callout(pos.x, pos.y)
        zone = get_zone(pos.x, pos.y)

        yaw = pos.yaw % 360
        dirs = [
            (0, "East"), (45, "NE"), (90, "North"), (135, "NW"),
            (180, "West"), (225, "SW"), (270, "South"), (315, "SE"), (360, "East"),
        ]
        look_dir = min(dirs, key=lambda d: abs(d[0] - yaw))[1]
        vert = "down" if pos.pitch > 10 else ("up" if pos.pitch < -10 else "level")
        print(f"{t:5.0f}s  {callout:>16s}  {zone:>10s}  {look_dir:>8s}  {vert:>6s}")

    print()
    break
else:
    print("No suitable buy round found with kills for k_z_")
