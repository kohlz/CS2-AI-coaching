"""
tactical_rl.py

Tabular Q-learning for mid-round tactical suggestions.

State:  (t_alive, ct_alive, bomb_status, time_bucket, player_zone)
        6 x 6 x 3 x 4 x 5 = 2,160 discrete states

Actions: ROTATE_A, ROTATE_B, HOLD, PUSH, FALL_BACK  (5 actions)

Reward:  +1.0 if team wins the round (terminal), 0 otherwise

Trained offline from demo replay data extracted by training_data.py.

Usage
-----
    from tactical_rl import TacticalQLearner

    ql = TacticalQLearner()
    ql.train_from_demos("src/demo")

    action, q_vals = ql.recommend(t_alive=3, ct_alive=2,
                                  bomb_status=0, time_bucket=1,
                                  zone_idx=2, side="CT")
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# State / action space
# ---------------------------------------------------------------------------

N_T_ALIVE = 6     # 0..5
N_CT_ALIVE = 6    # 0..5
N_BOMB_STATUS = 3  # 0=not_planted, 1=planted_A, 2=planted_B
N_TIME_BUCKET = 4  # 0=early(<30s), 1=mid(30-60s), 2=late(>60s), 3=post_plant
N_ZONES = 5       # A=0, B=1, MID=2, CT_BASE=3, T_BASE=4

STATE_DIMS = (N_T_ALIVE, N_CT_ALIVE, N_BOMB_STATUS, N_TIME_BUCKET, N_ZONES)
N_STATES = N_T_ALIVE * N_CT_ALIVE * N_BOMB_STATUS * N_TIME_BUCKET * N_ZONES

N_ACTIONS = 5
ACTION_NAMES = ["ROTATE_A", "ROTATE_B", "HOLD", "PUSH", "FALL_BACK"]

ZONE_NAMES = ["A", "B", "MID", "CT_BASE", "T_BASE"]


def _state_index(t_alive: int, ct_alive: int, bomb: int,
                 time_b: int, zone: int) -> tuple:
    return (min(t_alive, 5), min(ct_alive, 5),
            min(bomb, 2), min(time_b, 3), min(zone, 4))


# ---------------------------------------------------------------------------
# Q-Learning agent
# ---------------------------------------------------------------------------

class TacticalQLearner:
    """Tabular Q-learning for mid-round tactical decisions."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 0.15):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((*STATE_DIMS, N_ACTIONS))
        self.visit_count = np.zeros((*STATE_DIMS, N_ACTIONS), dtype=int)
        self.trained = False

    def _get_q(self, state: tuple) -> np.ndarray:
        return self.Q[state]

    def recommend(self, t_alive: int, ct_alive: int, bomb_status: int,
                  time_bucket: int, zone_idx: int,
                  side: str = "CT") -> tuple[int, dict]:
        """Return (best_action_idx, {action_name: q_value})."""
        state = _state_index(t_alive, ct_alive, bomb_status,
                             time_bucket, zone_idx)
        q_vals = self._get_q(state)
        best = int(np.argmax(q_vals))
        q_dict = {ACTION_NAMES[i]: float(q_vals[i]) for i in range(N_ACTIONS)}
        return best, q_dict

    def recommend_named(self, t_alive: int, ct_alive: int, bomb_status: int,
                        time_bucket: int, zone_idx: int,
                        side: str = "CT") -> str:
        best, _ = self.recommend(t_alive, ct_alive, bomb_status,
                                 time_bucket, zone_idx, side)
        return ACTION_NAMES[best]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _update(self, state: tuple, action: int, reward: float,
                next_state: tuple, done: bool):
        q_sa = self.Q[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (target - q_sa)
        self.visit_count[state][action] += 1

    def train_from_dataframe(self, df: pd.DataFrame,
                             n_passes: int = 30,
                             verbose: bool = True) -> dict:
        """Train Q-table from RL transition DataFrame.

        Multiple passes over the data simulate replay-buffer-style training.
        """
        if df.empty:
            return {}

        rounds = df.groupby(["demo", "round_num", "player"])

        transitions = []
        for (demo, rnd, player), group in rounds:
            group = group.sort_values("tick")
            rows = group.to_dict("records")
            for j in range(len(rows)):
                r = rows[j]
                state = _state_index(r["t_alive"], r["ct_alive"],
                                     r["bomb_status"], r["time_bucket"],
                                     r["zone_idx"])
                action = int(r["action"])
                reward = float(r["reward"])
                done = bool(r["is_terminal"])

                if j + 1 < len(rows):
                    nr = rows[j + 1]
                    next_state = _state_index(
                        nr["t_alive"], nr["ct_alive"],
                        nr["bomb_status"], nr["time_bucket"],
                        nr["zone_idx"])
                else:
                    next_state = state

                transitions.append((state, action, reward, next_state, done))

        n_trans = len(transitions)
        if verbose:
            print(f"  Q-learning: {n_trans} transitions, "
                  f"{n_passes} passes")

        rng = np.random.default_rng(42)

        for p in range(n_passes):
            order = rng.permutation(n_trans)
            for idx in order:
                s, a, r, ns, d = transitions[idx]
                self._update(s, a, r, ns, d)

            if verbose and (p + 1) % 10 == 0:
                coverage = (self.visit_count > 0).sum()
                total_cells = self.Q.size
                mean_q = self.Q[self.visit_count > 0].mean() if coverage > 0 else 0
                print(f"    Pass {p+1}/{n_passes}: "
                      f"coverage={coverage}/{total_cells} "
                      f"({coverage/total_cells:.1%}), "
                      f"mean Q={mean_q:.3f}")

        self.trained = True

        visited_states = (self.visit_count.sum(axis=-1) > 0).sum()
        total_states = np.prod(STATE_DIMS)

        return {
            "transitions": n_trans,
            "passes": n_passes,
            "visited_states": int(visited_states),
            "total_states": int(total_states),
            "state_coverage": visited_states / total_states,
            "visited_sa_pairs": int((self.visit_count > 0).sum()),
        }

    def train_from_demos(self, demo_dir: str = "src/demo",
                         n_passes: int = 30,
                         verbose: bool = True) -> dict:
        """Extract RL data from demos and train."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from training_data import extract_all

        data = extract_all(demo_dir, include_rl=True, verbose=verbose)
        rl_df = data["rl_transitions"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Training Q-learning on {len(rl_df)} transitions")
            print(f"{'='*60}\n")

        stats = self.train_from_dataframe(rl_df, n_passes=n_passes,
                                          verbose=verbose)
        return stats

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str = "models/tactical_ql.npz"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, Q=self.Q, visits=self.visit_count)

    def load(self, path: str = "models/tactical_ql.npz"):
        data = np.load(path)
        self.Q = data["Q"]
        self.visit_count = data["visits"]
        self.trained = True

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def print_policy_slice(self, bomb_status: int = 0,
                           time_bucket: int = 1):
        """Print the optimal action for common alive-count + zone combos."""
        bomb_names = ["Pre-plant", "Planted A", "Planted B"]
        time_names = ["Early (<30s)", "Mid (30-60s)", "Late (>60s)", "Post-plant"]

        print(f"\nQ-Policy: {bomb_names[bomb_status]}, "
              f"{time_names[time_bucket]}")
        print(f"{'Alive':>10s}", end="")
        for z in range(N_ZONES):
            print(f"  {ZONE_NAMES[z]:>10s}", end="")
        print()
        print("-" * (10 + 12 * N_ZONES))

        for t_a in range(1, 6):
            for ct_a in range(1, 6):
                label = f"{t_a}T v {ct_a}CT"
                print(f"{label:>10s}", end="")
                for z in range(N_ZONES):
                    state = _state_index(t_a, ct_a, bomb_status,
                                         time_bucket, z)
                    q = self._get_q(state)
                    if self.visit_count[state].sum() == 0:
                        print(f"  {'---':>10s}", end="")
                    else:
                        best = int(np.argmax(q))
                        print(f"  {ACTION_NAMES[best]:>10s}", end="")
                print()

    def evaluate_player_round(self, transitions: pd.DataFrame) -> list[dict]:
        """Evaluate a player's actions in one round against Q-policy."""
        results = []
        for _, row in transitions.iterrows():
            state = _state_index(
                int(row["t_alive"]), int(row["ct_alive"]),
                int(row["bomb_status"]), int(row["time_bucket"]),
                int(row["zone_idx"]))

            actual_action = int(row["action"])
            q_vals = self._get_q(state)
            optimal_action = int(np.argmax(q_vals))

            results.append({
                "time": row.get("time_elapsed", 0),
                "state": f"{row['t_alive']}Tv{row['ct_alive']}CT "
                         f"bomb={row['bomb_status']} "
                         f"zone={ZONE_NAMES[int(row['zone_idx'])]}",
                "actual": ACTION_NAMES[actual_action],
                "optimal": ACTION_NAMES[optimal_action],
                "match": actual_action == optimal_action,
                "actual_q": float(q_vals[actual_action]),
                "optimal_q": float(q_vals[optimal_action]),
                "q_diff": float(q_vals[optimal_action] - q_vals[actual_action]),
            })
        return results


# ===========================================================================
# V2: Micro-decision Q-learning with dual reward
# ===========================================================================

N_SIDE = 2         # T=0, CT=1
N_ALIVE_ADV = 7    # -3..+3  → index 0..6
N_BOMB_V2 = 3      # same as v1
N_TIME_V2 = 4      # same as v1
N_ZONES_V2 = 5     # same as v1
N_RECENT = 5       # none, teammate_died, enemy_killed, grenade, bomb_planted

V2_STATE_DIMS = (N_SIDE, N_ALIVE_ADV, N_BOMB_V2, N_TIME_V2, N_ZONES_V2, N_RECENT)

N_ACTIONS_V2 = 6
ACTION_NAMES_V2 = ["PEEK", "HOLD", "TRADE", "FALL_BACK", "UTILITY", "ROTATE"]


def _v2_state_index(side_idx: int, alive_adv: int, bomb: int,
                    time_b: int, zone: int, recent: int) -> tuple:
    return (min(max(side_idx, 0), 1),
            min(max(alive_adv + 3, 0), 6),
            min(max(bomb, 0), 2),
            min(max(time_b, 0), 3),
            min(max(zone, 0), 4),
            min(max(recent, 0), 4))


class TacticalQLearnerV2:
    """Dual Q-table learner for micro-decision evaluation.

    Maintains two separate Q-tables:
      Q_kill  — trained with immediate kill/death reward
      Q_win   — trained with terminal round-win reward

    Recommendation blends both: Q = alpha * Q_kill + (1-alpha) * Q_win
    """

    def __init__(self, alpha_lr: float = 0.1, gamma: float = 0.9,
                 blend_alpha: float = 0.4):
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.blend_alpha = blend_alpha

        self.Q_kill = np.zeros((*V2_STATE_DIMS, N_ACTIONS_V2))
        self.Q_win = np.zeros((*V2_STATE_DIMS, N_ACTIONS_V2))
        self.visits_kill = np.zeros((*V2_STATE_DIMS, N_ACTIONS_V2), dtype=int)
        self.visits_win = np.zeros((*V2_STATE_DIMS, N_ACTIONS_V2), dtype=int)
        self.trained = False

    def _update(self, Q, visits, state, action, reward, next_state, done):
        q_sa = Q[state][action]
        target = reward if done else reward + self.gamma * np.max(Q[next_state])
        Q[state][action] += self.alpha_lr * (target - q_sa)
        visits[state][action] += 1

    def recommend(self, side_idx: int, alive_adv: int, bomb_status: int,
                  time_bucket: int, zone_idx: int, recent_event: int,
                  ) -> tuple[int, dict, dict]:
        """Return (best_action_idx, {action: blended_q}, {action: {kill_q, win_q}})."""
        state = _v2_state_index(side_idx, alive_adv, bomb_status,
                                time_bucket, zone_idx, recent_event)
        q_k = self.Q_kill[state]
        q_w = self.Q_win[state]
        q_blend = self.blend_alpha * q_k + (1 - self.blend_alpha) * q_w

        best = int(np.argmax(q_blend))
        blended = {ACTION_NAMES_V2[i]: float(q_blend[i])
                   for i in range(N_ACTIONS_V2)}
        detail = {ACTION_NAMES_V2[i]: {"kill_q": float(q_k[i]),
                                        "win_q": float(q_w[i]),
                                        "blend": float(q_blend[i])}
                  for i in range(N_ACTIONS_V2)}
        return best, blended, detail

    def train_from_dataframe(self, df: pd.DataFrame,
                             n_passes: int = 30,
                             verbose: bool = True) -> dict:
        if df.empty:
            return {}

        groups = df.groupby(["demo", "round_num", "player"])

        kill_transitions = []
        win_transitions = []

        for (demo, rnd, player), group in groups:
            group = group.sort_values("tick")
            rows = group.to_dict("records")
            for j in range(len(rows)):
                r = rows[j]
                state = _v2_state_index(
                    r["side_idx"], r["alive_adv"], r["bomb_status"],
                    r["time_bucket"], r["zone_idx"], r["recent_event"])
                action = int(r["action"])
                done = bool(r["is_terminal"])

                if j + 1 < len(rows):
                    nr = rows[j + 1]
                    next_state = _v2_state_index(
                        nr["side_idx"], nr["alive_adv"], nr["bomb_status"],
                        nr["time_bucket"], nr["zone_idx"], nr["recent_event"])
                else:
                    next_state = state

                kr = float(r["kill_reward"]) + float(r.get("site_reward", 0))
                kill_transitions.append(
                    (state, action, kr, next_state, done))
                win_transitions.append(
                    (state, action, float(r["win_reward"]), next_state, done))

        n_trans = len(kill_transitions)
        if verbose:
            print(f"  Q-learning v2: {n_trans} transitions, {n_passes} passes")

        rng = np.random.default_rng(42)

        for p in range(n_passes):
            order = rng.permutation(n_trans)
            for idx in order:
                s, a, r_k, ns, d = kill_transitions[idx]
                self._update(self.Q_kill, self.visits_kill, s, a, r_k, ns, d)

                s, a, r_w, ns, d = win_transitions[idx]
                self._update(self.Q_win, self.visits_win, s, a, r_w, ns, d)

            if verbose and (p + 1) % 10 == 0:
                cov_k = (self.visits_kill > 0).sum()
                cov_w = (self.visits_win > 0).sum()
                total = self.Q_kill.size
                mean_k = self.Q_kill[self.visits_kill > 0].mean() if cov_k else 0
                mean_w = self.Q_win[self.visits_win > 0].mean() if cov_w else 0
                print(f"    Pass {p+1}/{n_passes}: "
                      f"kill coverage={cov_k}/{total} ({cov_k/total:.1%}), "
                      f"mean Q_kill={mean_k:.3f}, mean Q_win={mean_w:.3f}")

        self.trained = True

        v_kill = (self.visits_kill.sum(axis=-1) > 0).sum()
        v_win = (self.visits_win.sum(axis=-1) > 0).sum()
        total_states = int(np.prod(V2_STATE_DIMS))

        return {
            "transitions": n_trans,
            "passes": n_passes,
            "visited_states_kill": int(v_kill),
            "visited_states_win": int(v_win),
            "total_states": total_states,
            "state_coverage": int(v_kill) / total_states,
            "kill_sa_pairs": int((self.visits_kill > 0).sum()),
            "win_sa_pairs": int((self.visits_win > 0).sum()),
        }

    def train_from_demos(self, demo_dir: str = "src/demo",
                         n_passes: int = 30,
                         verbose: bool = True) -> dict:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from training_data import extract_all

        data = extract_all(demo_dir, include_rl=False, include_rl_v2=True,
                           verbose=verbose)
        rl_df = data["rl_v2"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Training Q-learning v2 on {len(rl_df)} transitions")
            print(f"{'='*60}\n")

            if not rl_df.empty:
                print("  Action distribution:")
                for a_idx, cnt in rl_df["action"].value_counts().sort_index().items():
                    print(f"    {ACTION_NAMES_V2[a_idx]:12s}: {cnt}")
                print()

        stats = self.train_from_dataframe(rl_df, n_passes=n_passes,
                                          verbose=verbose)
        return stats

    def save(self, path: str = "models/tactical_ql_v2.npz"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path,
                            Q_kill=self.Q_kill, Q_win=self.Q_win,
                            visits_kill=self.visits_kill,
                            visits_win=self.visits_win)

    def load(self, path: str = "models/tactical_ql_v2.npz"):
        data = np.load(path)
        self.Q_kill = data["Q_kill"]
        self.Q_win = data["Q_win"]
        self.visits_kill = data["visits_kill"]
        self.visits_win = data["visits_win"]
        self.trained = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ql = TacticalQLearner()
    stats = ql.train_from_demos("src/demo", n_passes=30)

    print(f"\n{'='*60}")
    print("  Q-Learning Training Summary")
    print(f"{'='*60}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k:25s}: {v:.2%}")
        else:
            print(f"    {k:25s}: {v}")

    ql.save()
    print("\nQ-table saved to models/tactical_ql.npz")

    print("\n--- Policy Slices ---")
    ql.print_policy_slice(bomb_status=0, time_bucket=1)
    ql.print_policy_slice(bomb_status=1, time_bucket=3)

    # Sample recommendations
    print("\n--- Sample Recommendations ---")
    scenarios = [
        (3, 3, 0, 0, 2, "CT", "3v3 pre-plant early, CT at MID"),
        (2, 4, 0, 2, 0, "CT", "2v4 pre-plant late, CT at A"),
        (4, 2, 1, 3, 1, "T",  "4v2 bomb planted A, T at B"),
        (1, 3, 2, 3, 2, "CT", "1v3 bomb planted B, CT at MID"),
    ]
    for t_a, ct_a, bomb, time_b, zone, side, desc in scenarios:
        action, q_dict = ql.recommend(t_a, ct_a, bomb, time_b, zone, side)
        top = sorted(q_dict.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {desc}")
        print(f"    Best: {ACTION_NAMES[action]}  "
              f"(Q: {', '.join(f'{n}={v:.3f}' for n,v in top)})")
