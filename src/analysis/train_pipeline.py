"""Master training pipeline with two stages: extract demos into splits, then
train and evaluate all models on those splits."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def stage_extract(demo_dir: str = "src/demo/train_demos",
                  data_dir: str = "data",
                  verbose: bool = True) -> dict:
    """Stage 1: extract demos and save train/val/test splits to disk."""
    from dataset_builder import build_and_save_datasets
    return build_and_save_datasets(demo_dir, data_dir, verbose=verbose)


def stage_train(data_dir: str = "data",
                model_dir: str = "models",
                verbose: bool = True) -> dict:
    """Stage 2: load saved splits, train all models, evaluate on val & test."""
    from dataset_builder import load_split, load_split_info

    info = load_split_info(data_dir)
    if not info:
        print("ERROR: No saved datasets found. Run 'extract' stage first.")
        return {}

    if verbose:
        print("=" * 60)
        print("  TRAINING PIPELINE — Load → Train → Evaluate")
        print("=" * 60)
        print(f"\n  Data dir : {data_dir}")
        print(f"  Model dir: {model_dir}")
        print(f"  Splits   : {info.get('total_rounds', '?')} rounds, "
              f"{info.get('total_demos', '?')} demos total "
              f"({info.get('split_strategy', 'by_demo')})")
        for name in ["train", "val", "test"]:
            s = info["sizes"][name]
            print(f"    {name:5s}: {s['demos']} demos, "
                  f"{s['rounds']} rounds, "
                  f"{s['rl_v2_transitions']} RL transitions")
        print()

    # Load splits
    if verbose:
        print("[1/4] Loading saved datasets...")
    t0 = time.time()
    train_data = load_split(data_dir, "train")
    val_data = load_split(data_dir, "val")
    test_data = load_split(data_dir, "test")
    if verbose:
        print(f"  Loaded in {time.time()-t0:.1f}s\n")

    results = {}
    os.makedirs(model_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("  [2/4] Training Neural Networks")
        print("=" * 60)
        print()

    from strategy_nn import train_all_models
    t0 = time.time()
    nn_result = train_all_models(
        save_dir=model_dir,
        verbose=verbose,
        train_data=train_data,
        val_data=val_data,
    )
    nn_time = time.time() - t0
    results["nn"] = nn_result.get("stats", {})
    results["nn"]["train_time_s"] = nn_time

    if verbose:
        print()
        print("=" * 60)
        print("  [3/5] Q-Learner Hyperparameter Search")
        print("=" * 60)

    from tactical_rl import train_side_specific, tune_ql_hyperparameters
    t0 = time.time()
    tune_result = tune_ql_hyperparameters(train_data, val_data,
                                           verbose=verbose)
    tune_time = time.time() - t0
    results["ql_tuning"] = tune_result.get("best", {})
    results["ql_tuning"]["tune_time_s"] = tune_time

    best = tune_result.get("best", {})
    best_gamma = best.get("gamma", 0.95)
    best_blend = best.get("blend_alpha", 0.25)
    best_passes = best.get("n_passes", 50)
    best_alpha = best.get("alpha_lr", 0.1)

    if verbose:
        print()
        print("=" * 60)
        print(f"  [4/5] Training Q-Learners (best: γ={best_gamma}, "
              f"blend={best_blend}, α_lr={best_alpha}, "
              f"passes={best_passes})")
        print("=" * 60)

    t0 = time.time()
    ql_result = train_side_specific(
        n_passes=best_passes,
        verbose=verbose,
        train_data=train_data,
        val_data=val_data,
        alpha_lr=best_alpha,
        gamma=best_gamma,
        blend_alpha=best_blend,
    )
    ql_time = time.time() - t0
    results["ql"] = ql_result
    results["ql"]["train_time_s"] = ql_time
    results["ql"]["best_hyperparams"] = {
        "gamma": best_gamma, "blend_alpha": best_blend,
        "n_passes": best_passes, "alpha_lr": best_alpha,
    }

    if verbose:
        print()
        print("=" * 60)
        print("  [5/5] Test Set Evaluation")
        print("=" * 60)
        print()

    test_results = _evaluate_all(model_dir, test_data, verbose=verbose)
    results["test"] = test_results

    summary_path = os.path.join(data_dir, "training_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_make_serializable(results), f, indent=2)

    if verbose:
        print()
        print("=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        print(f"  NN training time:  {nn_time:.1f}s")
        print(f"  QL training time:  {ql_time:.1f}s")
        print(f"  Results saved to:  {summary_path}")
        _print_summary(results)
        print("=" * 60)

    return results


def _evaluate_all(model_dir: str,
                  test_data: dict,
                  verbose: bool = True) -> dict:
    """Evaluate all trained models on test data."""
    from strategy_nn import (
        PreRoundFormation, PreRoundAttack,
        FormationClassifier_T, FormationClassifier_CT,
        load_models, _eval_lstm_t, _eval_lstm_ct, _build_prf_df, _build_pra_df,
        FORMATION_CLASSES, ATTACK_SITE_CLASSES, ALL_CT_FORMATIONS,
    )
    from tactical_rl import (
        TacticalQLearner_T, TacticalQLearner_CT, _eval_ql,
    )
    from metrics import (
        _eval_preround_fnn, _eval_lstm_t_metrics, _eval_lstm_ct_metrics,
    )

    results = {}

    # Load trained models
    models = load_models(model_dir)

    # PreRoundFormation
    prf = models.get("preround_formation")
    if prf and prf.trained:
        prf_df = _build_prf_df(test_data["rounds"])
        if not prf_df.empty:
            m = _eval_preround_fnn(prf, prf_df, len(FORMATION_CLASSES))
            results["preround_formation"] = {
                "test_accuracy": m["accuracy"], **m,
            }
            if verbose:
                print(f"  PreRoundFormation   test acc: {m['accuracy']:.1%}  "
                      f"macroF1={m['macro_f1']:.3f}  "
                      f"baseline={m['majority_baseline']:.1%}")

    # PreRoundAttack
    pra = models.get("preround_attack")
    if pra and pra.trained:
        pra_df = _build_pra_df(test_data["rounds"])
        if not pra_df.empty:
            m = _eval_preround_fnn(pra, pra_df, len(ATTACK_SITE_CLASSES))
            results["preround_attack"] = {
                "test_accuracy": m["accuracy"], **m,
            }
            if verbose:
                print(f"  PreRoundAttack      test acc: {m['accuracy']:.1%}  "
                      f"macroF1={m['macro_f1']:.3f}  "
                      f"baseline={m['majority_baseline']:.1%}")

    # FormationClassifier_T
    fc_t = models.get("formation_classifier_t")
    if fc_t and fc_t.trained:
        test_seqs = test_data.get("event_sequences", [])
        if test_seqs:
            m = _eval_lstm_t_metrics(fc_t, test_seqs,
                                     len(ATTACK_SITE_CLASSES))
            results["formation_classifier_t"] = {
                "test_accuracy": m["accuracy"], **m,
                "n_sequences": len(test_seqs),
            }
            if verbose:
                print(f"  FormationClassifier_T test acc: {m['accuracy']:.1%}  "
                      f"macroF1={m['macro_f1']:.3f}  "
                      f"baseline={m['majority_baseline']:.1%}  "
                      f"(n={len(test_seqs)})")

    # FormationClassifier_CT
    fc_ct = models.get("formation_classifier_ct")
    if fc_ct and fc_ct.trained:
        test_ct = test_data.get("ct_formation_sequences", [])
        if test_ct:
            m = _eval_lstm_ct_metrics(fc_ct, test_ct, len(ALL_CT_FORMATIONS))
            results["formation_classifier_ct"] = {
                "test_accuracy": m["accuracy"], **m,
                "n_sequences": len(test_ct),
            }
            if verbose:
                print(f"  FormationClassifier_CT test acc: {m['accuracy']:.1%}  "
                      f"macroF1={m['macro_f1']:.3f}  "
                      f"baseline={m['majority_baseline']:.1%}  "
                      f"(n={len(test_ct)})")

    # Q-Learners
    ql_t = TacticalQLearner_T()
    ql_ct = TacticalQLearner_CT()
    t_path = os.path.join(model_dir, "tactical_ql_t.npz")
    ct_path = os.path.join(model_dir, "tactical_ql_ct.npz")

    rl_test = test_data.get("rl_v2", None)
    if rl_test is not None and not rl_test.empty:
        if os.path.exists(t_path):
            ql_t.load(t_path)
            test_t = rl_test[rl_test["side"] == "T"]
            ev_t = _eval_ql(ql_t, test_t)
            results["ql_t"] = ev_t
            if verbose and ev_t:
                print(f"  QL_T  test agreement: {ev_t['agreement']:.1%} "
                      f"(n={ev_t['n_samples']})")

        if os.path.exists(ct_path):
            ql_ct.load(ct_path)
            test_ct = rl_test[rl_test["side"] == "CT"]
            ev_ct = _eval_ql(ql_ct, test_ct)
            results["ql_ct"] = ev_ct
            if verbose and ev_ct:
                print(f"  QL_CT test agreement: {ev_ct['agreement']:.1%} "
                      f"(n={ev_ct['n_samples']})")

    return results


def _print_summary(results: dict):
    """Print a condensed summary table of accuracy, macro-F1, and baseline."""
    print()
    print(f"  {'Model':<30s} {'Train':>8s} {'Val':>8s} "
          f"{'Test':>8s} {'MacroF1':>8s} {'Baseline':>9s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")

    nn_stats = results.get("nn", {})
    for model_name in ["preround_formation", "preround_attack",
                       "formation_classifier_t", "formation_classifier_ct"]:
        ms = nn_stats.get(model_name, {})
        train_acc = ms.get("train", {}).get("accuracy", None)
        val_acc = ms.get("val", {}).get("accuracy", None)
        test_block = results.get("test", {}).get(model_name, {}) or {}
        test_acc = test_block.get("test_accuracy")
        macro_f1 = test_block.get("macro_f1")
        baseline = test_block.get("majority_baseline")

        t_str = f"{train_acc:.1%}" if train_acc is not None else "  —"
        v_str = f"{val_acc:.1%}" if val_acc is not None else "  —"
        ts_str = f"{test_acc:.1%}" if test_acc is not None else "  —"
        f_str = f"{macro_f1:.3f}" if macro_f1 is not None else "  —"
        b_str = f"{baseline:.1%}" if baseline is not None else "  —"
        print(f"  {model_name:<30s} {t_str:>8s} {v_str:>8s} {ts_str:>8s} "
              f"{f_str:>8s} {b_str:>9s}")

    ql = results.get("ql", {})
    for side, label in [("t", "QL_T"), ("ct", "QL_CT")]:
        train_cov = ql.get(side, {}).get("state_coverage", None)
        val_agr = ql.get(f"{side}_val", {}).get("agreement", None)
        test_agr = results.get("test", {}).get(f"ql_{side}", {}).get(
            "agreement", None)

        t_str = f"{train_cov:.1%}" if train_cov is not None else "  —"
        v_str = f"{val_agr:.1%}" if val_agr is not None else "  —"
        ts_str = f"{test_agr:.1%}" if test_agr is not None else "  —"
        print(f"  {label:<30s} {t_str:>8s} {v_str:>8s} {ts_str:>8s}")

    print()


def _make_serializable(obj):
    """Recursively convert numpy types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()
                if not callable(v) and k != "models"}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python train_pipeline.py extract [demo_dir] [data_dir]")
        print("  python train_pipeline.py train   [data_dir] [model_dir]")
        print("  python train_pipeline.py all     [demo_dir] [data_dir] [model_dir]")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "extract":
        demo_dir = sys.argv[2] if len(sys.argv) > 2 else "src/demo/train_demos"
        data_dir = sys.argv[3] if len(sys.argv) > 3 else "data"
        stage_extract(demo_dir, data_dir)

    elif cmd == "train":
        data_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
        model_dir = sys.argv[3] if len(sys.argv) > 3 else "models"
        stage_train(data_dir, model_dir)

    elif cmd == "all":
        demo_dir = sys.argv[2] if len(sys.argv) > 2 else "src/demo/train_demos"
        data_dir = sys.argv[3] if len(sys.argv) > 3 else "data"
        model_dir = sys.argv[4] if len(sys.argv) > 4 else "models"
        stage_extract(demo_dir, data_dir)
        print("\n\n")
        stage_train(data_dir, model_dir)

    else:
        print(f"Unknown command: {cmd}")
        print("Use: extract, train, or all")
        sys.exit(1)


if __name__ == "__main__":
    main()
