"""Test economy MDP evaluation against k_z_'s actual buy decisions."""
import sys
sys.path.insert(0, "src/demo")
sys.path.insert(0, "src/analysis")

from demo_parser import parse_demo
from economy_mdp import (
    evaluate_player_economy, economy_summary,
    ACTION_NAMES, EQUIP_COST,
)

match = parse_demo("src/demo/260319mirage.dem", "k_z_")
print(f"Map: {match.map_name}")
print(f"Player: {match.target_player}")
print(f"Score: T {match.t_score} - {match.ct_score} CT")
print(f"Rounds: {len(match.rounds)}\n")

# Show raw loadout data for context
print("Raw round data for k_z_:")
print(f"{'R':>3s} {'Side':>4s} {'$':>6s} {'Equip':>6s} {'Primary':>12s} {'Alive':>6s} {'Winner':>6s}")
print("-" * 55)
for rd in match.rounds:
    p = rd.get_player("k_z_")
    if p is None:
        continue
    print(f"R{rd.round_num:2d} {p.side:>4s} ${p.start_money:>5d} ${p.equipment_value:>5d} "
          f"{(p.primary_weapon or '-'):>12s} {'alive' if p.alive_at_end else 'DEAD':>6s} "
          f"{rd.winner:>6s}")
print()

# Run evaluation
evals = evaluate_player_economy(match.rounds, "k_z_")

print(f"{'Rnd':>3s} {'Side':>4s} {'$':>6s} {'Strk':>4s} "
      f"{'Actual':>10s} {'Optimal':>10s} {'OK?':>4s} "
      f"{'WP_a':>5s} {'WP_o':>5s} {'Note'}")
print("-" * 95)

for e in evals:
    mark = "OK" if e.is_optimal else "BAD"
    print(f"R{e.round_num:2d} {e.side:>4s} ${e.money:>5d} "
          f"L{e.loss_streak:>1d}  "
          f"{e.actual_name:>10s} {e.optimal_name:>10s} {mark:>4s} "
          f"{e.actual_win_prob:4.0%} {e.optimal_win_prob:4.0%}  "
          f"{e.note}")

print()
summary = economy_summary(evals)
print("=== Economy Summary ===")
for k, v in summary.items():
    if isinstance(v, float):
        if "accuracy" in k or "wp" in k:
            print(f"  {k:30s}: {v:.1%}")
        else:
            print(f"  {k:30s}: {v:.3f}")
    else:
        print(f"  {k:30s}: {v}")
