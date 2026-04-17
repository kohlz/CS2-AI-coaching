"""
tactical_rl.py

Side-specific tabular Q-learning for mid-round tactical suggestions.
Trains separate T and CT Q-learners with dual kill/win Q-tables.
See docs/models/qlearning.md for state space and hyperparameter details.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

N_T_ALIVE = 6
N_CT_ALIVE = 6
N_BOMB_STATUS = 3
N_TIME_BUCKET = 4
N_ZONES = 5

STATE_DIMS = (N_T_ALIVE, N_CT_ALIVE, N_BOMB_STATUS, N_TIME_BUCKET, N_ZONES)
N_STATES = N_T_ALIVE * N_CT_ALIVE * N_BOMB_STATUS * N_TIME_BUCKET * N_ZONES

N_ACTIONS = 5
ACTION_NAMES = ["ROTATE_A", "ROTATE_B", "HOLD", "PUSH", "FALL_BACK"]

ZONE_NAMES = ["A", "B", "MID", "CT_BASE", "T_BASE"]


def _state_index(t_alive, ct_alive, bomb, time_b, zone):
    return (min(t_alive, 5), min(ct_alive, 5),
            min(bomb, 2), min(time_b, 3), min(zone, 4))


N_SIDE = 2
N_ALIVE_ADV = 7
N_BOMB_V2 = 3
N_TIME_V2 = 4
N_ZONES_V2 = 5
N_RECENT = 5

V2_STATE_DIMS = (N_SIDE, N_ALIVE_ADV, N_BOMB_V2, N_TIME_V2, N_ZONES_V2, N_RECENT)
N_ACTIONS_V2 = 6
ACTION_NAMES_V2 = ["PEEK", "HOLD", "TRADE", "FALL_BACK", "UTILITY", "ROTATE"]


def _v2_state_index(side_idx, alive_adv, bomb, time_b, zone, recent):
    return (min(max(side_idx, 0), 1),
            min(max(alive_adv + 3, 0), 6),
            min(max(bomb, 0), 2),
            min(max(time_b, 0), 3),
            min(max(zone, 0), 4),
            min(max(recent, 0), 4))


N_ALIVE_ADV_SS = 7
N_BOMB_SS = 3
N_TIME_SS = 4
N_ZONES_SS = 5
N_RECENT_SS = 5
N_TEAM_SUPPORT = 3

SS_STATE_DIMS = (N_ALIVE_ADV_SS, N_BOMB_SS, N_TIME_SS, N_ZONES_SS,
                 N_RECENT_SS, N_TEAM_SUPPORT)

T_ACTIONS = ["EXECUTE", "PEEK", "TRADE", "FALL_BACK", "UTILITY", "LURK"]
CT_ACTIONS = ["HOLD", "ROTATE", "RETAKE", "PUSH", "FALL_BACK", "UTILITY"]
N_ACTIONS_SS = 6


def _ss_state_index(alive_adv, bomb, time_b, zone, recent, team_support):
    return (min(max(alive_adv + 3, 0), 6),
            min(max(bomb, 0), 2),
            min(max(time_b, 0), 3),
            min(max(zone, 0), 4),
            min(max(recent, 0), 4),
            min(max(team_support, 0), 2))


class _SideQLearner:
    """Dual Q-table learner for one side (T or CT).

    Blends a kill/site-reward Q-table with a terminal win-reward Q-table.
    """

    def __init__(self, side: str, alpha_lr: float = 0.1,
                 gamma: float = 0.95, blend_alpha: float = 0.25):
        self.side = side
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.blend_alpha = blend_alpha
        self.action_names = T_ACTIONS if side == "T" else CT_ACTIONS

        self.Q_kill = np.zeros((*SS_STATE_DIMS, N_ACTIONS_SS))
        self.Q_win = np.zeros((*SS_STATE_DIMS, N_ACTIONS_SS))
        self.visits_kill = np.zeros((*SS_STATE_DIMS, N_ACTIONS_SS), dtype=int)
        self.visits_win = np.zeros((*SS_STATE_DIMS, N_ACTIONS_SS), dtype=int)
        self.trained = False

    def _update(self, Q, visits, state, action, reward, next_state, done,
                lr: float):
        q_sa = Q[state][action]
        target = reward if done else reward + self.gamma * np.max(Q[next_state])
        Q[state][action] += lr * (target - q_sa)
        visits[state][action] += 1

    def recommend(self, alive_adv: int, bomb_status: int,
                  time_bucket: int, zone_idx: int, recent_event: int,
                  team_support: int) -> tuple[int, dict, dict]:
        """Return (best_action_idx, {action: blended_q}, {action: detail})."""
        state = _ss_state_index(alive_adv, bomb_status, time_bucket,
                                zone_idx, recent_event, team_support)
        q_k = self.Q_kill[state]
        q_w = self.Q_win[state]
        q_blend = self.blend_alpha * q_k + (1 - self.blend_alpha) * q_w

        best = int(np.argmax(q_blend))
        blended = {self.action_names[i]: float(q_blend[i])
                   for i in range(N_ACTIONS_SS)}
        detail = {self.action_names[i]: {
            "kill_q": float(q_k[i]),
            "win_q": float(q_w[i]),
            "blend": float(q_blend[i])
        } for i in range(N_ACTIONS_SS)}
        return best, blended, detail

    def train_from_dataframe(self, df: pd.DataFrame,
                             n_passes: int = 50,
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
                state = _ss_state_index(
                    r["alive_adv"], r["bomb_status"],
                    r["time_bucket"], r["zone_idx"], r["recent_event"],
                    r.get("team_support", 0))
                action = int(r["action_ss"])
                done = bool(r["is_terminal"])

                if j + 1 < len(rows):
                    nr = rows[j + 1]
                    next_state = _ss_state_index(
                        nr["alive_adv"], nr["bomb_status"],
                        nr["time_bucket"], nr["zone_idx"], nr["recent_event"],
                        nr.get("team_support", 0))
                else:
                    next_state = state
                    done = True

                kr = float(r["kill_reward"])
                kill_transitions.append(
                    (state, action, kr, next_state, done))
                win_transitions.append(
                    (state, action, float(r["win_reward"]), next_state, done))

        n_trans = len(kill_transitions)
        if verbose:
            print(f"  {self.side} Q-learner: {n_trans} transitions, "
                  f"{n_passes} passes")

        rng = np.random.default_rng(42)

        for p in range(n_passes):
            lr = self.alpha_lr * (1.0 / (1.0 + (9.0 / (n_passes - 1)) * p)
                                 if n_passes > 1 else 1.0)

            order = rng.permutation(n_trans)
            for idx in order:
                s, a, r_k, ns, d = kill_transitions[idx]
                self._update(self.Q_kill, self.visits_kill, s, a, r_k, ns, d, lr)

                s, a, r_w, ns, d = win_transitions[idx]
                self._update(self.Q_win, self.visits_win, s, a, r_w, ns, d, lr)

            if verbose and (p + 1) % 10 == 0:
                cov = (self.visits_kill > 0).sum()
                total = self.Q_kill.size
                mean_k = self.Q_kill[self.visits_kill > 0].mean() if cov else 0
                mean_w = self.Q_win[self.visits_win > 0].mean() if cov else 0
                print(f"    Pass {p+1}/{n_passes} (lr={lr:.4f}): "
                      f"coverage={cov}/{total} ({cov/total:.1%}), "
                      f"Q_kill={mean_k:.3f}, Q_win={mean_w:.3f}")

        self.trained = True

        v_states = (self.visits_kill.sum(axis=-1) > 0).sum()
        total_states = int(np.prod(SS_STATE_DIMS))

        return {
            "side": self.side,
            "transitions": n_trans,
            "passes": n_passes,
            "visited_states": int(v_states),
            "total_states": total_states,
            "state_coverage": int(v_states) / total_states,
            "sa_pairs": int((self.visits_kill > 0).sum()),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path,
                            Q_kill=self.Q_kill, Q_win=self.Q_win,
                            visits_kill=self.visits_kill,
                            visits_win=self.visits_win)

    def load(self, path: str):
        data = np.load(path)
        self.Q_kill = data["Q_kill"]
        self.Q_win = data["Q_win"]
        self.visits_kill = data["visits_kill"]
        self.visits_win = data["visits_win"]
        self.trained = True


class TacticalQLearner_T(_SideQLearner):
    """T-side Q-learner: EXECUTE, PEEK, TRADE, FALL_BACK, UTILITY, LURK."""
    def __init__(self, **kwargs):
        super().__init__("T", **kwargs)


class TacticalQLearner_CT(_SideQLearner):
    """CT-side Q-learner: HOLD, ROTATE, RETAKE, PUSH, FALL_BACK, UTILITY."""
    def __init__(self, **kwargs):
        super().__init__("CT", **kwargs)


class TacticalQLearner:
    """Legacy tabular Q-learning (v1)."""

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.15):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((*STATE_DIMS, N_ACTIONS))
        self.visit_count = np.zeros((*STATE_DIMS, N_ACTIONS), dtype=int)
        self.trained = False

    def _get_q(self, state):
        return self.Q[state]

    def recommend(self, t_alive, ct_alive, bomb_status,
                  time_bucket, zone_idx, side="CT"):
        state = _state_index(t_alive, ct_alive, bomb_status,
                             time_bucket, zone_idx)
        q_vals = self._get_q(state)
        best = int(np.argmax(q_vals))
        q_dict = {ACTION_NAMES[i]: float(q_vals[i]) for i in range(N_ACTIONS)}
        return best, q_dict

    def recommend_named(self, t_alive, ct_alive, bomb_status,
                        time_bucket, zone_idx, side="CT"):
        best, _ = self.recommend(t_alive, ct_alive, bomb_status,
                                 time_bucket, zone_idx, side)
        return ACTION_NAMES[best]

    def _update(self, state, action, reward, next_state, done):
        q_sa = self.Q[state][action]
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (target - q_sa)
        self.visit_count[state][action] += 1

    def train_from_dataframe(self, df, n_passes=30, verbose=True):
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

        rng = np.random.default_rng(42)
        for p in range(n_passes):
            order = rng.permutation(len(transitions))
            for idx in order:
                s, a, r, ns, d = transitions[idx]
                self._update(s, a, r, ns, d)
        self.trained = True
        visited = (self.visit_count.sum(axis=-1) > 0).sum()
        total = np.prod(STATE_DIMS)
        return {"transitions": len(transitions), "visited_states": int(visited),
                "total_states": int(total), "state_coverage": visited / total}

    def train_from_demos(self, demo_dir="src/demo", n_passes=30, verbose=True):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from training_data import extract_all
        data = extract_all(demo_dir, include_rl=True, verbose=verbose)
        return self.train_from_dataframe(data["rl_transitions"],
                                         n_passes=n_passes, verbose=verbose)

    def save(self, path="models/tactical_ql.npz"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, Q=self.Q, visits=self.visit_count)

    def load(self, path="models/tactical_ql.npz"):
        data = np.load(path)
        self.Q = data["Q"]
        self.visit_count = data["visits"]
        self.trained = True

    def print_policy_slice(self, bomb_status=0, time_bucket=1):
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
                    state = _state_index(t_a, ct_a, bomb_status, time_bucket, z)
                    q = self._get_q(state)
                    if self.visit_count[state].sum() == 0:
                        print(f"  {'---':>10s}", end="")
                    else:
                        best = int(np.argmax(q))
                        print(f"  {ACTION_NAMES[best]:>10s}", end="")
                print()

    def evaluate_player_round(self, transitions):
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
                "actual": ACTION_NAMES[actual_action],
                "optimal": ACTION_NAMES[optimal_action],
                "match": actual_action == optimal_action,
            })
        return results


class TacticalQLearnerV2:
    """Legacy dual Q-table learner (v2, unified side)."""

    def __init__(self, alpha_lr=0.1, gamma=0.9, blend_alpha=0.4):
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

    def recommend(self, side_idx, alive_adv, bomb_status,
                  time_bucket, zone_idx, recent_event):
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

    def train_from_dataframe(self, df, n_passes=30, verbose=True):
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
                kill_transitions.append((state, action, kr, next_state, done))
                win_transitions.append(
                    (state, action, float(r["win_reward"]), next_state, done))

        rng = np.random.default_rng(42)
        for p in range(n_passes):
            order = rng.permutation(len(kill_transitions))
            for idx in order:
                s, a, r_k, ns, d = kill_transitions[idx]
                self._update(self.Q_kill, self.visits_kill, s, a, r_k, ns, d)
                s, a, r_w, ns, d = win_transitions[idx]
                self._update(self.Q_win, self.visits_win, s, a, r_w, ns, d)
        self.trained = True
        v = (self.visits_kill.sum(axis=-1) > 0).sum()
        total = int(np.prod(V2_STATE_DIMS))
        return {"transitions": len(kill_transitions),
                "visited_states": int(v), "total_states": total,
                "state_coverage": int(v) / total}

    def train_from_demos(self, demo_dir="src/demo", n_passes=30, verbose=True):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from training_data import extract_all
        data = extract_all(demo_dir, include_rl=False, include_rl_v2=True,
                           verbose=verbose)
        return self.train_from_dataframe(data["rl_v2"],
                                         n_passes=n_passes, verbose=verbose)

    def save(self, path="models/tactical_ql_v2.npz"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path,
                            Q_kill=self.Q_kill, Q_win=self.Q_win,
                            visits_kill=self.visits_kill,
                            visits_win=self.visits_win)

    def load(self, path="models/tactical_ql_v2.npz"):
        data = np.load(path)
        self.Q_kill = data["Q_kill"]
        self.Q_win = data["Q_win"]
        self.visits_kill = data["visits_kill"]
        self.visits_win = data["visits_win"]
        self.trained = True


def _eval_ql(learner: _SideQLearner, df: pd.DataFrame) -> dict:
    """Evaluate a Q-learner on held-out data.

    Returns dict with agreement rate (how often the learner's recommendation
    matches the action actually taken) and avg Q-value for taken actions.
    """
    if df.empty:
        return {}
    agree, total, q_vals = 0, 0, []
    for _, r in df.iterrows():
        state = _ss_state_index(
            r["alive_adv"], r["bomb_status"],
            r["time_bucket"], r["zone_idx"], r["recent_event"],
            r.get("team_support", 0))
        action = int(r["action_ss"])
        best, blended, _ = learner.recommend(
            r["alive_adv"], r["bomb_status"],
            r["time_bucket"], r["zone_idx"], r["recent_event"],
            r.get("team_support", 0))
        if best == action:
            agree += 1
        q_blend = learner.blend_alpha * learner.Q_kill[state] + \
                  (1 - learner.blend_alpha) * learner.Q_win[state]
        q_vals.append(float(q_blend[action]))
        total += 1
    return {
        "agreement": agree / max(total, 1),
        "avg_q_taken": float(np.mean(q_vals)) if q_vals else 0.0,
        "n_samples": total,
    }


def train_side_specific(
    demo_dir: str = "src/demo",
    n_passes: int = 50,
    verbose: bool = True,
    *,
    train_data: dict | None = None,
    val_data: dict | None = None,
    alpha_lr: float = 0.1,
    gamma: float = 0.95,
    blend_alpha: float = 0.25,
) -> dict:
    """Train both T and CT Q-learners.

    If *train_data* is provided (dict with key rl_v2 as DataFrame),
    it is used directly — skipping demo extraction.
    If *val_data* is also provided, validation metrics are reported.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if train_data is not None:
        rl_df = train_data.get("rl_v2", pd.DataFrame())
    else:
        from training_data import extract_all
        data = extract_all(demo_dir, include_rl=False, include_rl_v2=True,
                           verbose=verbose)
        rl_df = data["rl_v2"]

    if rl_df.empty:
        print("No RL data extracted.")
        return {}

    t_df = rl_df[rl_df["side"] == "T"].copy()
    ct_df = rl_df[rl_df["side"] == "CT"].copy()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training side-specific Q-learners")
        print(f"  γ={gamma}, α_lr={alpha_lr}, blend={blend_alpha}, "
              f"passes={n_passes}")
        print(f"  T transitions: {len(t_df)}, CT transitions: {len(ct_df)}")
        print(f"{'='*60}\n")

    ql_t = TacticalQLearner_T(alpha_lr=alpha_lr, gamma=gamma,
                               blend_alpha=blend_alpha)
    stats_t = ql_t.train_from_dataframe(t_df, n_passes=n_passes,
                                         verbose=verbose)
    ql_t.save("models/tactical_ql_t.npz")

    if verbose:
        print()

    ql_ct = TacticalQLearner_CT(alpha_lr=alpha_lr, gamma=gamma,
                                 blend_alpha=blend_alpha)
    stats_ct = ql_ct.train_from_dataframe(ct_df, n_passes=n_passes,
                                           verbose=verbose)
    ql_ct.save("models/tactical_ql_ct.npz")

    all_stats = {"t": stats_t, "ct": stats_ct}

    # Validation evaluation
    if val_data is not None:
        val_rl = val_data.get("rl_v2", pd.DataFrame())
        if not val_rl.empty:
            val_t = val_rl[val_rl["side"] == "T"].copy()
            val_ct = val_rl[val_rl["side"] == "CT"].copy()
            val_stats_t = _eval_ql(ql_t, val_t)
            val_stats_ct = _eval_ql(ql_ct, val_ct)
            all_stats["t_val"] = val_stats_t
            all_stats["ct_val"] = val_stats_ct

            if verbose:
                print(f"\n  Validation:")
                if val_stats_t:
                    print(f"    T: agreement={val_stats_t['agreement']:.1%}, "
                          f"avg_Q={val_stats_t['avg_q_taken']:.3f} "
                          f"(n={val_stats_t['n_samples']})")
                if val_stats_ct:
                    print(f"    CT: agreement={val_stats_ct['agreement']:.1%}, "
                          f"avg_Q={val_stats_ct['avg_q_taken']:.3f} "
                          f"(n={val_stats_ct['n_samples']})")

    return all_stats


def tune_ql_hyperparameters(train_data: dict, val_data: dict,
                            verbose: bool = True) -> dict:
    """Grid search over Q-learner hyperparameters using validation set.

    Returns best hyperparams and all trial results.
    """
    gammas = [0.90, 0.95, 0.99]
    blend_alphas = [0.15, 0.25, 0.35]
    n_passes_list = [50, 80]
    alpha_lrs = [0.1, 0.15]

    train_rl = train_data.get("rl_v2", pd.DataFrame())
    val_rl = val_data.get("rl_v2", pd.DataFrame())

    if train_rl.empty or val_rl.empty:
        print("Need both train and val RL data for tuning.")
        return {}

    train_t = train_rl[train_rl["side"] == "T"].copy()
    train_ct = train_rl[train_rl["side"] == "CT"].copy()
    val_t = val_rl[val_rl["side"] == "T"].copy()
    val_ct = val_rl[val_rl["side"] == "CT"].copy()

    trials = []
    best_score = -1.0
    best_params = {}

    total = len(gammas) * len(blend_alphas) * len(n_passes_list) * len(alpha_lrs)
    trial_num = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Q-Learner Hyperparameter Search ({total} trials)")
        print(f"{'='*60}\n")

    for gamma in gammas:
        for blend_alpha in blend_alphas:
            for n_passes in n_passes_list:
                for alpha_lr in alpha_lrs:
                    trial_num += 1

                    ql_t = TacticalQLearner_T(alpha_lr=alpha_lr,
                                              gamma=gamma,
                                              blend_alpha=blend_alpha)
                    ql_t.train_from_dataframe(train_t, n_passes=n_passes,
                                              verbose=False)

                    ql_ct = TacticalQLearner_CT(alpha_lr=alpha_lr,
                                                gamma=gamma,
                                                blend_alpha=blend_alpha)
                    ql_ct.train_from_dataframe(train_ct, n_passes=n_passes,
                                               verbose=False)

                    ev_t = _eval_ql(ql_t, val_t)
                    ev_ct = _eval_ql(ql_ct, val_ct)

                    avg_agreement = (ev_t.get("agreement", 0) +
                                     ev_ct.get("agreement", 0)) / 2

                    trial = {
                        "gamma": gamma,
                        "blend_alpha": blend_alpha,
                        "n_passes": n_passes,
                        "alpha_lr": alpha_lr,
                        "val_t_agreement": ev_t.get("agreement", 0),
                        "val_ct_agreement": ev_ct.get("agreement", 0),
                        "avg_agreement": avg_agreement,
                    }
                    trials.append(trial)

                    if avg_agreement > best_score:
                        best_score = avg_agreement
                        best_params = trial.copy()

                    if verbose:
                        flag = " *BEST*" if avg_agreement >= best_score else ""
                        print(f"  [{trial_num:2d}/{total}] "
                              f"γ={gamma} α_lr={alpha_lr} "
                              f"blend={blend_alpha} passes={n_passes} "
                              f"-> T={ev_t.get('agreement', 0):.1%} "
                              f"CT={ev_ct.get('agreement', 0):.1%} "
                              f"avg={avg_agreement:.1%}{flag}")

    if verbose:
        print(f"\n  Best: γ={best_params['gamma']}, "
              f"α_lr={best_params['alpha_lr']}, "
              f"blend={best_params['blend_alpha']}, "
              f"passes={best_params['n_passes']} "
              f"-> avg agreement={best_score:.1%}")

    return {"best": best_params, "trials": trials}


if __name__ == "__main__":
    stats = train_side_specific("src/demo", n_passes=50)

    print(f"\n{'='*60}")
    print("  Side-Specific Q-Learning Summary")
    print(f"{'='*60}")
    for side, s in stats.items():
        print(f"\n  {side.upper()} side:")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"    {k:25s}: {v:.2%}")
            else:
                print(f"    {k:25s}: {v}")
