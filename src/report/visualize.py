"""
visualize.py

Per-report matplotlib charts: model accuracy (LSTM, Q-learner, pre-round NN)
and fun facts (kill heatmap, economy flow, attack rate by tier, utility
usage, win rate by side).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper: extract LSTM final prediction and actual site per round
# ---------------------------------------------------------------------------

def _extract_lstm_accuracy(report: dict) -> tuple[list, list]:
    """Return (predicted_sites, actual_sites) for rounds with LSTM data."""
    predicted = []
    actual = []
    for rd in report.get("round_details", []):
        sugg = rd.get("suggestions", {})
        if not isinstance(sugg, dict):
            continue
        timeline = sugg.get("timeline", [])
        lstm_entries = [e for e in timeline if e.get("source") == "LSTM"]
        if not lstm_entries:
            continue
        last = lstm_entries[-1]
        text = last.get("text", "")
        a_prob = b_prob = 0.0
        for part in text.split():
            if part.startswith("A=") and part.endswith("%"):
                try:
                    a_prob = float(part[2:-1]) / 100.0
                except ValueError:
                    pass
            elif part.startswith("B=") and part.endswith("%"):
                try:
                    b_prob = float(part[2:-1]) / 100.0
                except ValueError:
                    pass
        if a_prob > b_prob:
            pred_site = "A"
        elif b_prob > a_prob:
            pred_site = "B"
        else:
            pred_site = "no_plant"

        actual_site = "no_plant"
        for ev in timeline:
            trig = ev.get("trigger", "") or ""
            if "Bomb planted A" in trig:
                actual_site = "A"
                break
            elif "Bomb planted B" in trig:
                actual_site = "B"
                break

        predicted.append(pred_site)
        actual.append(actual_site)

    return predicted, actual


# ---------------------------------------------------------------------------
# Model accuracy charts
# ---------------------------------------------------------------------------

def plot_lstm_accuracy(report: dict, output_dir: str) -> str | None:
    """Plot LSTM prediction accuracy: final prediction vs actual bomb site."""
    if not _MPL_AVAILABLE:
        return None

    predicted, actual = _extract_lstm_accuracy(report)
    if len(predicted) < 3:
        return None

    classes = ["A", "B", "no_plant"]
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    for p, a in zip(predicted, actual):
        if p in classes and a in classes:
            matrix[classes.index(a)][classes.index(p)] += 1

    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    total = len(predicted)
    acc = correct / total if total else 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Confusion matrix
    ax = axes[0]
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(n))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"LSTM T-Attack Prediction\nAccuracy: {acc:.0%} ({correct}/{total})")
    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color=color, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Per-round correctness
    ax2 = axes[1]
    correct_list = [1 if p == a else 0 for p, a in zip(predicted, actual)]
    ax2.bar(range(len(correct_list)), correct_list, color=["#2ECC71" if c else "#E74C3C" for c in correct_list], alpha=0.8)
    ax2.set_title("LSTM Correct Per Round")
    ax2.set_xlabel("Round (with LSTM data)")
    ax2.set_ylabel("Correct (1) / Wrong (0)")
    ax2.set_ylim(-0.1, 1.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "lstm_accuracy.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_preround_formation_accuracy(report: dict, output_dir: str) -> str | None:
    """Plot pre-round formation prediction accuracy across rounds."""
    if not _MPL_AVAILABLE:
        return None

    rounds = report.get("round_details", [])
    nn = report.get("nn_predictions", {})
    formations = nn.get("formation_predictions", [])
    if not formations:
        return None

    formation_by_round = {f["round"]: f for f in formations}

    pred_list = []
    for rd in rounds:
        rnd = rd["round"]
        fp = formation_by_round.get(rnd)
        if fp:
            pred_list.append({
                "round": rnd,
                "predicted": max(fp["probs"], key=fp["probs"].get),
                "confidence": max(fp["probs"].values()),
            })

    if len(pred_list) < 3:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Confidence over rounds
    ax = axes[0]
    rounds_x = [p["round"] for p in pred_list]
    confs = [p["confidence"] for p in pred_list]
    ax.plot(rounds_x, confs, marker="o", markersize=4, color="#9B59B6",
            linewidth=1.5)
    ax.fill_between(rounds_x, confs, alpha=0.15, color="#9B59B6")
    ax.set_title("Pre-Round Formation Prediction Confidence")
    ax.set_xlabel("Round")
    ax.set_ylabel("Top Formation Probability")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)

    # Formation distribution
    ax2 = axes[1]
    from collections import Counter
    fmt_counts = Counter(p["predicted"] for p in pred_list)
    labels = sorted(fmt_counts.keys(), key=lambda x: fmt_counts[x], reverse=True)
    counts = [fmt_counts[l] for l in labels]
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    ax2.barh(labels, counts, color=colors, edgecolor="white")
    ax2.set_title("Predicted Formation Distribution")
    ax2.set_xlabel("Count")
    for i, v in enumerate(counts):
        ax2.text(v + 0.2, i, str(v), va="center", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "preround_formation.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_ql_coverage(models_dir: str, output_dir: str) -> str | None:
    """Plot Q-learner state coverage and Q-value distribution."""
    if not _MPL_AVAILABLE:
        return None

    charts = []
    for name, label in [("tactical_ql_t.npz", "T Q-Learner"),
                         ("tactical_ql_ct.npz", "CT Q-Learner"),
                         ("tactical_ql_v2.npz", "V2 Q-Learner (legacy)")]:
        path = os.path.join(models_dir, name)
        if not os.path.exists(path):
            continue
        data = np.load(path)
        for key in ["Q_kill", "Q_win", "Q"]:
            if key in data:
                q = data[key]
                charts.append((label + f" ({key})", q))
                break

    if not charts:
        return None

    fig, axes = plt.subplots(1, len(charts), figsize=(5 * len(charts), 4))
    if len(charts) == 1:
        axes = [axes]

    for ax, (label, q) in zip(axes, charts):
        flat = q.flatten()
        nonzero = flat[flat != 0]
        total = len(flat)
        coverage = len(nonzero) / total if total else 0

        ax.hist(nonzero, bins=50, color="#FF6B6B", alpha=0.8, edgecolor="white")
        ax.set_title(f"{label}\n{coverage:.1%} non-zero ({len(nonzero):,}/{total:,})")
        ax.set_xlabel("Q-value")
        ax.set_ylabel("Count")

    plt.tight_layout()
    path = os.path.join(output_dir, "ql_coverage.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Fun fact charts
# ---------------------------------------------------------------------------

def plot_economy_flow(report: dict, output_dir: str) -> str | None:
    """Plot player money over rounds with buy decision annotations."""
    if not _MPL_AVAILABLE:
        return None

    rounds = report.get("round_details", [])
    if not rounds:
        return None

    round_nums = []
    player_money = []
    buy_colors = []
    for rd in rounds:
        econ = rd.get("economy", {})
        if econ:
            round_nums.append(rd["round"])
            player_money.append(econ.get("money", 0))
            if econ.get("is_optimal", True):
                buy_colors.append("#2ECC71")
            else:
                buy_colors.append("#E74C3C")

    if not round_nums:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(round_nums, player_money, color="#2ECC71", linewidth=1.5, alpha=0.5)
    ax.scatter(round_nums, player_money, c=buy_colors, s=30, zorder=5,
               edgecolors="white", linewidths=0.5)
    ax.fill_between(round_nums, player_money, alpha=0.1, color="#2ECC71")
    ax.set_title("Economy Flow (green = optimal buy, red = sub-optimal)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Starting Money ($)")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "economy_flow.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_kill_heatmap(report: dict, output_dir: str) -> str | None:
    """Plot kills by zone as horizontal bar chart."""
    if not _MPL_AVAILABLE:
        return None

    rounds = report.get("round_details", [])
    zone_kills = {"A": 0, "B": 0, "MID": 0, "CT_BASE": 0, "T_BASE": 0}

    for rd in rounds:
        sugg = rd.get("suggestions", {})
        if not isinstance(sugg, dict):
            continue
        timeline = sugg.get("timeline", [])
        for ev in timeline:
            desc = ev.get("trigger", "") or ev.get("text", "")
            for zone in zone_kills:
                if f"@ {zone}" in desc:
                    zone_kills[zone] += 1

    if sum(zone_kills.values()) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    zones = list(zone_kills.keys())
    counts = [zone_kills[z] for z in zones]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    ax.barh(zones, counts, color=colors, edgecolor="white")
    ax.set_title("Kills by Zone")
    ax.set_xlabel("Kill Count")
    for i, v in enumerate(counts):
        ax.text(v + 0.2, i, str(v), va="center", fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "kill_heatmap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_win_rate_by_side(report: dict, output_dir: str) -> str | None:
    """Plot win rate by side."""
    if not _MPL_AVAILABLE:
        return None

    rounds = report.get("round_details", [])
    side_stats = {"T": {"wins": 0, "total": 0}, "CT": {"wins": 0, "total": 0}}
    for rd in rounds:
        side = rd.get("side", "")
        if side in side_stats:
            side_stats[side]["total"] += 1
            if rd.get("won"):
                side_stats[side]["wins"] += 1

    if not any(s["total"] for s in side_stats.values()):
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    sides = ["T", "CT"]
    win_rates = [side_stats[s]["wins"] / max(side_stats[s]["total"], 1)
                 for s in sides]
    totals = [side_stats[s]["total"] for s in sides]
    colors = ["#F39C12", "#3498DB"]

    bars = ax.bar(sides, win_rates, color=colors, edgecolor="white", width=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate by Side")
    for bar, wr, n in zip(bars, win_rates, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{wr:.0%} ({n} rounds)", ha="center", fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "win_rate_by_side.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_attack_patterns_by_tier(report: dict, output_dir: str) -> str | None:
    """Plot attack rate on A vs B grouped by enemy economy tier."""
    if not _MPL_AVAILABLE:
        return None

    rounds = report.get("round_details", [])
    tier_sites = {}

    for rd in rounds:
        econ = rd.get("economy", {})
        tier = econ.get("enemy_tier", "")
        if not tier:
            continue

        site = "no_plant"
        sugg = rd.get("suggestions", {})
        if isinstance(sugg, dict):
            for ev in sugg.get("timeline", []):
                trig = ev.get("trigger", "") or ""
                if "Bomb planted A" in trig:
                    site = "A"
                    break
                elif "Bomb planted B" in trig:
                    site = "B"
                    break

        tier_sites.setdefault(tier, {"A": 0, "B": 0, "no_plant": 0})
        tier_sites[tier][site] += 1

    if not tier_sites:
        return None

    tier_order = ["BROKE", "LOW", "MEDIUM", "HIGH", "RICH"]
    tiers = [t for t in tier_order if t in tier_sites]
    if not tiers:
        return None

    a_counts = [tier_sites[t]["A"] for t in tiers]
    b_counts = [tier_sites[t]["B"] for t in tiers]
    np_counts = [tier_sites[t]["no_plant"] for t in tiers]

    x = np.arange(len(tiers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width, a_counts, width, label="A site", color="#FF6B6B")
    ax.bar(x, b_counts, width, label="B site", color="#4ECDC4")
    ax.bar(x + width, np_counts, width, label="No plant", color="#95A5A6")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_xlabel("Enemy Economy Tier")
    ax.set_ylabel("Round Count")
    ax.set_title("Attack Patterns by Enemy Economy Tier")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    path = os.path.join(output_dir, "attack_by_tier.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_utility_usage(report: dict, output_dir: str) -> str | None:
    """Plot utility usage patterns (smokes, flashes, HE) by zone."""
    if not _MPL_AVAILABLE:
        return None

    rounds = report.get("round_details", [])
    util_zones = {"smoke": {}, "flash": {}, "HE": {}}
    zone_order = ["A", "B", "MID", "CT_BASE", "T_BASE"]

    for rd in rounds:
        sugg = rd.get("suggestions", {})
        if not isinstance(sugg, dict):
            continue
        for ev in sugg.get("timeline", []):
            desc = ev.get("trigger", "") or ""
            for utype in util_zones:
                if utype.lower() in desc.lower():
                    for zone in zone_order:
                        if f"@ {zone}" in desc:
                            util_zones[utype][zone] = util_zones[utype].get(zone, 0) + 1
                            break

    total = sum(sum(v.values()) for v in util_zones.values())
    if total == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(zone_order))
    width = 0.25
    colors = {"smoke": "#95A5A6", "flash": "#F1C40F", "HE": "#E74C3C"}

    for i, (utype, zone_counts) in enumerate(util_zones.items()):
        counts = [zone_counts.get(z, 0) for z in zone_order]
        ax.bar(x + i * width, counts, width, label=utype, color=colors[utype])

    ax.set_xticks(x + width)
    ax.set_xticklabels(zone_order)
    ax.set_xlabel("Zone")
    ax.set_ylabel("Usage Count")
    ax.set_title("Utility Usage by Zone")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    path = os.path.join(output_dir, "utility_usage.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Generate all charts
# ---------------------------------------------------------------------------

def generate_all_charts(report: dict, models_dir: str = "models",
                        output_dir: str = "reports") -> list[str]:
    """Generate all visualization charts and return list of saved file paths."""
    os.makedirs(output_dir, exist_ok=True)

    paths = []

    for fn in [plot_lstm_accuracy, plot_preround_formation_accuracy]:
        try:
            p = fn(report, output_dir)
            if p:
                paths.append(p)
        except Exception as e:
            print(f"  Chart {fn.__name__} failed: {e}")

    try:
        p = plot_ql_coverage(models_dir, output_dir)
        if p:
            paths.append(p)
    except Exception as e:
        print(f"  Chart plot_ql_coverage failed: {e}")

    for fn in [plot_economy_flow, plot_kill_heatmap, plot_win_rate_by_side,
               plot_attack_patterns_by_tier, plot_utility_usage]:
        try:
            p = fn(report, output_dir)
            if p:
                paths.append(p)
        except Exception as e:
            print(f"  Chart {fn.__name__} failed: {e}")

    if paths:
        print(f"  Generated {len(paths)} charts in {output_dir}/")
    else:
        print("  No charts generated (matplotlib may not be available)")

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python visualize.py <report.json> [models_dir]")
        sys.exit(1)

    report_path = sys.argv[1]
    mdir = sys.argv[2] if len(sys.argv) > 2 else "models"

    with open(report_path, "r") as f:
        report = json.load(f)

    out_dir = os.path.dirname(report_path) or "reports"
    charts = generate_all_charts(report, mdir, out_dir)
    for c in charts:
        print(f"  Saved: {c}")
