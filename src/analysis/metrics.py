"""Classifier evaluation metrics (accuracy, macro-F1, weighted-F1, majority
baseline) for the four NN classifiers in the pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategy_nn import (  # noqa: E402
    ALL_CT_FORMATIONS,
    ATTACK_SITE_CLASSES,
    CT_PRIOR_DIM,
    FORMATION_CLASSES,
    _build_pra_df,
    _build_prf_df,
    load_models,
)


def classifier_metrics(y_true, y_pred, n_classes: int,
                       class_names: list[str] | None = None) -> dict:
    """Top-1 accuracy, macro-F1, weighted-F1, per-class F1, per-class support,
    and majority-class baseline."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return dict(accuracy=0.0, macro_f1=0.0, weighted_f1=0.0,
                    majority_baseline=0.0, n_samples=0,
                    per_class_f1=[], per_class_support=[],
                    class_names=class_names or [])

    labels = list(range(n_classes))
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro",
                              labels=labels, zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted",
                                 labels=labels, zero_division=0))
    per_class = f1_score(y_true, y_pred, average=None,
                         labels=labels, zero_division=0)
    support = [int((y_true == k).sum()) for k in labels]
    vals, counts = np.unique(y_true, return_counts=True)
    majority_class = int(vals[counts.argmax()])
    baseline_acc = float((y_true == majority_class).mean())
    return dict(accuracy=acc, macro_f1=macro_f1, weighted_f1=weighted_f1,
                majority_baseline=baseline_acc, n_samples=int(len(y_true)),
                per_class_f1=[float(x) for x in per_class],
                per_class_support=support,
                class_names=list(class_names) if class_names else
                [str(i) for i in labels])


def _eval_preround_fnn(model, df: pd.DataFrame, n_classes: int,
                       class_names: list[str] | None = None) -> dict:
    X, y = model._prepare(df)
    model.model.eval()
    device = next(model.model.parameters()).device
    with torch.no_grad():
        yhat = model.model(X.to(device)).argmax(dim=1).cpu().numpy()
    return classifier_metrics(y.cpu().numpy(), yhat, n_classes, class_names)


def _eval_lstm_t_metrics(model, sequences: list, n_classes: int,
                         class_names: list[str] | None = None) -> dict:
    """T-side LSTM: labels stored under 'attack_site' as strings."""
    cls_to_idx = {c: i for i, c in enumerate(ATTACK_SITE_CLASSES)}
    y_true, y_pred = [], []
    for seq in sequences:
        events = seq.get("events", [])
        label = seq.get("attack_site", "")
        if not events or label not in cls_to_idx:
            continue
        pred = model.predict(events)
        if not pred:
            continue
        top = max(pred, key=pred.get)
        y_pred.append(cls_to_idx[top])
        y_true.append(cls_to_idx[label])
    return classifier_metrics(y_true, y_pred, n_classes, class_names)


def _eval_lstm_ct_metrics(model, sequences: list, n_classes: int,
                          class_names: list[str] | None = None) -> dict:
    """CT-side LSTM: labels under 'formation_labels' (list), last item is the
    ground truth; alive counts under 'ct_alive_at_event'."""
    cls_to_idx = {c: i for i, c in enumerate(ALL_CT_FORMATIONS)}
    y_true, y_pred = [], []
    for seq in sequences:
        events = seq.get("events", [])
        labels = seq.get("formation_labels", [])
        alive_counts = seq.get("ct_alive_at_event", [])
        if not events or not labels or not alive_counts:
            continue
        last_label = labels[-1]
        if last_label not in cls_to_idx:
            continue
        prior = seq.get("pre_round_prior")
        raw = model.predict(events, alive_counts, prior=prior)
        if not raw:
            continue
        top = max(raw, key=raw.get)
        y_pred.append(cls_to_idx[top])
        y_true.append(cls_to_idx[last_label])
    return classifier_metrics(y_true, y_pred, n_classes, class_names)


def evaluate_all(model_dir: str = "models",
                 data_dir: str = "data") -> dict:
    models = load_models(model_dir)
    rounds_test = pd.read_csv(f"{data_dir}/rounds_test.csv")

    out: dict = {}

    prf = models.get("preround_formation")
    if prf and prf.trained:
        out["preround_formation"] = _eval_preround_fnn(
            prf, _build_prf_df(rounds_test), len(FORMATION_CLASSES),
            class_names=list(FORMATION_CLASSES))

    pra = models.get("preround_attack")
    if pra and pra.trained:
        out["preround_attack"] = _eval_preround_fnn(
            pra, _build_pra_df(rounds_test), len(ATTACK_SITE_CLASSES),
            class_names=list(ATTACK_SITE_CLASSES))

    fc_t = models.get("formation_classifier_t")
    if fc_t and fc_t.trained:
        with open(f"{data_dir}/event_sequences_test.json", "r",
                  encoding="utf-8") as f:
            test_seqs = json.load(f)
        out["formation_classifier_t"] = _eval_lstm_t_metrics(
            fc_t, test_seqs, len(ATTACK_SITE_CLASSES),
            class_names=list(ATTACK_SITE_CLASSES))

    fc_ct = models.get("formation_classifier_ct")
    if fc_ct and fc_ct.trained:
        with open(f"{data_dir}/ct_formations_test.json", "r",
                  encoding="utf-8") as f:
            ct_test = json.load(f)
        out["formation_classifier_ct"] = _eval_lstm_ct_metrics(
            fc_ct, ct_test, len(ALL_CT_FORMATIONS),
            class_names=list(ALL_CT_FORMATIONS))

    return out


def _backfill_and_print():
    metrics = evaluate_all()
    results_path = Path("data/training_results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    test_block = results.setdefault("test", {})
    for name, m in metrics.items():
        block = test_block.setdefault(name, {})
        block.update({
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "majority_baseline": m["majority_baseline"],
            "n_samples": m["n_samples"],
            "per_class_f1": m["per_class_f1"],
            "per_class_support": m["per_class_support"],
            "class_names": m["class_names"],
        })
        if "test_accuracy" not in block:
            block["test_accuracy"] = m["accuracy"]

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    header = (f"{'Model':<28s} {'Acc':>7s} {'MacroF1':>8s} "
             f"{'WF1':>7s} {'Baseline':>9s}  n")
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(f"{name:<28s} {m['accuracy']:>6.1%} {m['macro_f1']:>8.3f} "
              f"{m['weighted_f1']:>7.3f} {m['majority_baseline']:>8.1%}  "
              f"{m['n_samples']}")


if __name__ == "__main__":
    _backfill_and_print()
