"""Generate publication-ready training-result charts for the final report.

Reads ``data/training_results.json`` and writes accuracy, loss-curve, F1,
and Q-learner agreement charts into ``--out-dir``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLOR_TRAIN = "#2E7BB8"
COLOR_VAL = "#E8822F"
COLOR_TEST = "#8A8A8A"
COLOR_LINES = ["#2E7BB8", "#E8822F", "#8A8A8A", "#6AA84F"]

MODEL_ORDER = [
    ("preround_formation", "Pre-Round\nFormation"),
    ("preround_attack", "Pre-Round\nAttack"),
    ("formation_classifier_t", "Formation\nClassifier_T"),
    ("formation_classifier_ct", "Formation\nClassifier_CT"),
]


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_EPOCH_PATTERNS = {
    "preround_formation":
        re.compile(r"PreRoundFormation\s+epoch\s+(\d+)/\d+:\s+loss=([\d.]+),\s+acc=([\d.]+)%"),
    "formation_classifier_t":
        re.compile(r"FormationClassifier_T\s+epoch\s+(\d+)/\d+:\s+loss=([\d.]+),\s+acc=([\d.]+)%"),
    "formation_classifier_ct":
        re.compile(r"FormationClassifier_CT\s+epoch\s+(\d+)/\d+:\s+loss=([\d.]+),\s+acc=([\d.]+)%"),
}


def _scrape_log(log_path: str) -> dict[str, dict]:
    """Pull sparse (epoch, loss, acc) points from a training log."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Split PRF shared-prefix points into PRF and PRA by section headers
    prf_pts: list[tuple[int, float, float]] = []
    pra_pts: list[tuple[int, float, float]] = []
    cur = prf_pts
    for line in text.splitlines():
        if "Pre-Round Attack" in line:
            cur = pra_pts
            continue
        if "FormationClassifier_" in line:
            cur = None
        m = _EPOCH_PATTERNS["preround_formation"].search(line)
        if m and cur is not None:
            cur.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

    def _extract(model_key: str) -> list[tuple[int, float, float]]:
        pts = []
        for line in text.splitlines():
            m = _EPOCH_PATTERNS[model_key].search(line)
            if m:
                pts.append((int(m.group(1)), float(m.group(2)),
                            float(m.group(3))))
        return pts

    return {
        "preround_formation": prf_pts,
        "preround_attack": pra_pts,
        "formation_classifier_t": _extract("formation_classifier_t"),
        "formation_classifier_ct": _extract("formation_classifier_ct"),
    }


def _get_history(results: dict, model_key: str,
                 scraped: dict[str, list[tuple[int, float, float]]]
                 ) -> tuple[list[int], list[float], list[float]]:
    """Return (epochs, loss, acc) for a model, preferring full history, else scraped points."""
    stats = (results.get("nn", {}).get(model_key, {})
             .get("train", {}))
    loss_hist = stats.get("loss_history")
    acc_hist = stats.get("acc_history")
    if loss_hist and acc_hist:
        return (list(range(1, len(loss_hist) + 1)),
                list(loss_hist),
                [a * 100 for a in acc_hist])

    pts = scraped.get(model_key, [])
    if not pts:
        return [], [], []
    pts.sort()
    ep = [p[0] for p in pts]
    ls = [p[1] for p in pts]
    ac = [p[2] for p in pts]
    return ep, ls, ac


def plot_accuracy(results: dict, out_path: str) -> None:
    nn = results.get("nn", {})
    test = results.get("test", {})

    labels = []
    train_vals = []
    val_vals = []
    test_vals = []

    for key, pretty in MODEL_ORDER:
        m = nn.get(key, {})
        if not m:
            continue
        train_acc = (m.get("train", {}) or {}).get("accuracy")
        val_acc = (m.get("val", {}) or {}).get("accuracy")
        test_acc = (test.get(key, {}) or {}).get("test_accuracy")

        labels.append(pretty)
        train_vals.append(100 * train_acc if train_acc is not None else 0)
        val_vals.append(100 * val_acc if val_acc is not None else 0)
        test_vals.append(100 * test_acc if test_acc is not None else 0)

    x = np.arange(len(labels))
    w = 0.27

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    b1 = ax.bar(x - w, train_vals, w, label="Train", color=COLOR_TRAIN)
    b2 = ax.bar(x,     val_vals,   w, label="Validation", color=COLOR_VAL)
    b3 = ax.bar(x + w, test_vals,  w, label="Test", color=COLOR_TEST)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0,
                        f"{h:.0f}%", ha="center", va="bottom",
                        fontsize=8, color="#333")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy by Split")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(100, max(train_vals + val_vals + test_vals) + 8))
    ax.yaxis.grid(True, linestyle=":", color="#ccc", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(results: dict, scraped: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)

    any_curve = False
    for (key, pretty), color in zip(MODEL_ORDER, COLOR_LINES):
        epochs, losses, _ = _get_history(results, key, scraped)
        if not epochs:
            continue
        label = pretty.replace("\n", " ")
        ax.plot(epochs, losses, label=label, color=color,
                linewidth=2.0, marker="o" if len(epochs) <= 30 else None,
                markersize=4)
        any_curve = True

    if not any_curve:
        ax.text(0.5, 0.5, "No training history available.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Convergence")
    ax.yaxis.grid(True, linestyle=":", color="#ccc", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


COLOR_F1 = "#6AA84F"
COLOR_BASELINE = "#C0392B"


def plot_classifier_metrics(results: dict, out_path: str) -> None:
    """Grouped bar: test Accuracy + Macro-F1, with majority-baseline marker."""
    test = results.get("test", {})

    labels, acc, f1, baseline = [], [], [], []
    for key, pretty in MODEL_ORDER:
        block = test.get(key) or {}
        a = block.get("accuracy", block.get("test_accuracy"))
        f = block.get("macro_f1")
        b = block.get("majority_baseline")
        if a is None and f is None:
            continue
        labels.append(pretty)
        acc.append(100 * a if a is not None else 0)
        f1.append(100 * f if f is not None else 0)
        baseline.append(100 * b if b is not None else 0)

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    b1 = ax.bar(x - w / 2, acc, w, label="Test Accuracy", color=COLOR_TRAIN)
    b2 = ax.bar(x + w / 2, f1, w, label="Test Macro-F1 (×100)",
                color=COLOR_F1)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2,
                        f"{h:.0f}", ha="center", va="bottom",
                        fontsize=8, color="#333")

    for xi, bval in zip(x, baseline):
        ax.hlines(bval, xi - w, xi + w, colors=COLOR_BASELINE,
                  linestyles="--", linewidth=1.6, zorder=3)
    ax.plot([], [], color=COLOR_BASELINE, linestyle="--",
            linewidth=1.6, label="Majority-class baseline")

    ax.set_ylabel("Score (%)")
    ax.set_title("Test-Set Classifier Performance — Accuracy vs Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(100, max(acc + f1 + baseline) + 10))
    ax.yaxis.grid(True, linestyle=":", color="#ccc", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


COLOR_MACRO = "#6AA84F"
COLOR_WEIGHTED = "#3D85C6"


def plot_f1_scores(results: dict, out_path: str) -> None:
    """Macro-F1 vs Weighted-F1 side-by-side for each classifier."""
    test = results.get("test", {})

    labels, macro, weighted = [], [], []
    for key, pretty in MODEL_ORDER:
        block = test.get(key) or {}
        mf = block.get("macro_f1")
        wf = block.get("weighted_f1")
        if mf is None and wf is None:
            continue
        labels.append(pretty)
        macro.append(mf if mf is not None else 0)
        weighted.append(wf if wf is not None else 0)

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    b1 = ax.bar(x - w / 2, macro, w, label="Macro-F1", color=COLOR_MACRO)
    b2 = ax.bar(x + w / 2, weighted, w, label="Weighted-F1",
                color=COLOR_WEIGHTED)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                        f"{h:.2f}", ha="center", va="bottom",
                        fontsize=8, color="#333")

    ax.set_ylabel("F1 Score")
    ax.set_title("Test-Set F1 Score by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, linestyle=":", color="#ccc", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(results: dict, out_path: str) -> None:
    """One subplot per classifier showing F1 for every class (shade = support)."""
    test = results.get("test", {})

    plottable = []
    for key, pretty in MODEL_ORDER:
        block = test.get(key) or {}
        f1s = block.get("per_class_f1") or []
        names = block.get("class_names") or []
        support = block.get("per_class_support") or []
        if not f1s or not names:
            continue
        plottable.append((pretty, f1s, names, support))

    if not plottable:
        return

    n = len(plottable)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.6), dpi=150,
                             squeeze=False)
    axes = axes[0]

    max_support = max((max(s) for _, _, _, s in plottable if s),
                      default=1)

    for ax, (title, f1s, names, support) in zip(axes, plottable):
        keep = [i for i, s in enumerate(support) if s > 0]
        if not keep:
            keep = list(range(len(support)))
        f1s = [f1s[i] for i in keep]
        names = [names[i] for i in keep]
        support = [support[i] for i in keep]

        order = np.argsort(-np.array(support))
        f1s_s = [f1s[i] for i in order]
        names_s = [names[i] for i in order]
        support_s = [support[i] for i in order]

        MAX_ROWS = 10
        truncated = False
        if len(f1s_s) > MAX_ROWS:
            f1s_s = f1s_s[:MAX_ROWS]
            names_s = names_s[:MAX_ROWS]
            support_s = support_s[:MAX_ROWS]
            truncated = True

        norm_support = [min(1.0, s / max_support) for s in support_s]
        colors = [plt.cm.Blues(0.35 + 0.55 * ns) for ns in norm_support]

        ypos = np.arange(len(f1s_s))
        ax.barh(ypos, f1s_s, color=colors, edgecolor="#333", linewidth=0.5)

        for i, (v, s) in enumerate(zip(f1s_s, support_s)):
            label = f"{v:.2f} (n={s})"
            ax.text(min(v + 0.02, 0.98), i, label,
                    va="center", fontsize=8, color="#333")

        ax.set_yticks(ypos)
        ax.set_yticklabels(names_s, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("F1", fontsize=9)
        clean_title = title.replace("\n", " ")
        if truncated:
            clean_title += f"  (top {MAX_ROWS} by support)"
        ax.set_title(clean_title, fontsize=10)
        ax.xaxis.grid(True, linestyle=":", color="#ccc", alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Per-Class F1 on Test Set (bar shade = class support)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ql_agreement(results: dict, out_path: str) -> None:
    ql = results.get("ql", {})
    test = results.get("test", {})

    t_val = (ql.get("t_val", {}) or {}).get("agreement")
    ct_val = (ql.get("ct_val", {}) or {}).get("agreement")
    t_test = (test.get("ql_t", {}) or {}).get("agreement")
    ct_test = (test.get("ql_ct", {}) or {}).get("agreement")

    vals = {
        "Validation": [100 * (t_val or 0), 100 * (ct_val or 0)],
        "Test":       [100 * (t_test or 0), 100 * (ct_test or 0)],
    }

    x = np.arange(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)

    b1 = ax.bar(x - w / 2, vals["Validation"], w, label="Validation",
                color=COLOR_VAL)
    b2 = ax.bar(x + w / 2, vals["Test"], w, label="Test", color=COLOR_TEST)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                        f"{h:.0f}%", ha="center", va="bottom",
                        fontsize=9, color="#333")

    ax.set_ylabel("Agreement with Expert Action (%)")
    ax.set_title("Tactical Q-Learner Agreement by Side")
    ax.set_xticks(x)
    ax.set_xticklabels(["T-side", "CT-side"], fontsize=10)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle=":", color="#ccc", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Random-baseline reference line
    ax.axhline(100 / 7, color="#c0392b", linestyle="--", linewidth=1,
               alpha=0.7)
    ax.text(1.4, 100 / 7 + 1.5, "random baseline (14%)",
            color="#c0392b", fontsize=8, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _find_default_log() -> str | None:
    """Pick the most recent training terminal log in the Cursor terminals dir."""
    candidates = [
        Path(os.path.expanduser("~")) / ".cursor" / "projects",
    ]
    for root in candidates:
        if not root.exists():
            continue
        best = None
        best_mtime = 0.0
        for term_dir in root.rglob("terminals"):
            for p in term_dir.glob("*.txt"):
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        head = f.read(2048)
                except Exception:
                    continue
                if "train_pipeline.py train" not in head:
                    continue
                mt = p.stat().st_mtime
                if mt > best_mtime:
                    best_mtime = mt
                    best = p
        if best is not None:
            return str(best)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/training_results.json")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--log-file", default=None,
                    help="Optional training log to scrape loss points from "
                         "when training_results.json doesn't carry history.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = load_results(args.results)

    log_path = args.log_file or _find_default_log()
    if log_path and os.path.exists(log_path):
        print(f"Scraping training log: {log_path}")
        scraped = _scrape_log(log_path)
    else:
        scraped = {}

    outputs = []

    p_acc = os.path.join(args.out_dir, "training_accuracy.png")
    plot_accuracy(results, p_acc)
    outputs.append(p_acc)

    p_loss = os.path.join(args.out_dir, "training_loss_curves.png")
    plot_loss_curves(results, scraped, p_loss)
    outputs.append(p_loss)

    p_metrics = os.path.join(args.out_dir, "classifier_metrics.png")
    plot_classifier_metrics(results, p_metrics)
    outputs.append(p_metrics)

    p_f1 = os.path.join(args.out_dir, "f1_scores.png")
    plot_f1_scores(results, p_f1)
    outputs.append(p_f1)

    p_f1c = os.path.join(args.out_dir, "f1_per_class.png")
    plot_per_class_f1(results, p_f1c)
    outputs.append(p_f1c)

    p_ql = os.path.join(args.out_dir, "qlearner_agreement.png")
    plot_ql_agreement(results, p_ql)
    outputs.append(p_ql)

    print("Wrote:")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
