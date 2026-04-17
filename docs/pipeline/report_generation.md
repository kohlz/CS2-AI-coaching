# Report Generation & Visualization

**Source files:**

- [`src/report/generate_report.py`](../../src/report/generate_report.py)
- [`src/report/visualize.py`](../../src/report/visualize.py)
- [`src/report/training_charts.py`](../../src/report/training_charts.py)
- [`src/analysis/metrics.py`](../../src/analysis/metrics.py)

## Role in the Pipeline

The terminal stage of the pipeline. Loads every trained model,
re-parses one demo from the perspective of one player, runs every
model over each round, and emits:

1. A structured per-round coaching report (text + JSON).
2. Post-game visualization PNGs to `reports/`.
3. (Separately) training-time figures from
   `data/training_results.json`.

Returning to [main pipeline](../README.md#1-pipeline).

## Python Packages

- `numpy`, `pandas` — numerical glue
- `matplotlib` — all PNG visualizations
- `torch` — model loading
- `dataclasses`, `json` — structured report

## Entry Point

```bash
python src/report/generate_report.py <demo.dem> <player_name>
```

Internally `generate_full_report(demo_path, player_name)`:

1. **Parse** the demo via `demo_parser.parse_demo`.
2. **Load** all checkpoints via `strategy_nn.load_models()` and
   `tactical_rl.load_ql_t() / load_ql_ct()`.
3. **Run economy HMM + MDP** over the round sequence.
4. **Run pre-round NNs** (`PreRoundFormation` for T-side rounds,
   `PreRoundAttack` for CT-side rounds).
5. **Run LSTMs** at natural breakpoints in each round.
6. **Run Q-learners** at each event timestamp until the player dies.
7. **Format** everything into the per-round template documented in
   [README §2](../README.md#2-expected-output).
8. **Write** the JSON report to `reports/<demo>_<player>_report.json`.
9. **Render** post-game charts via `visualize.run_all(...)`.

## Side-Aware Routing

Model selection is gated by the player's current side:

| Player side | Pre-round | In-round LSTM | Q-learner |
|---|---|---|---|
| **T**  | [PreRoundFormation](../models/preround_formation.md) | [FormationClassifier_CT](../models/formation_classifier_ct.md) | T-side |
| **CT** | [PreRoundAttack](../models/preround_attack.md)       | [FormationClassifier_T](../models/formation_classifier_t.md)   | CT-side |

Utility events shown in the timeline are also side-filtered: a CT
player only sees enemy (T) utility, and vice versa.

## Visualization

`src/report/visualize.py` produces post-game PNGs:

- `kill_heatmap.png` — kills per Mirage zone
- `utility_usage.png` — smokes / flashes / mollys per zone per round
- `economy_flow.png` — both teams' money over rounds
- `win_rate_by_side.png` — per-half win rates
- `attack_by_tier.png` — A/B preference vs T economy tier
- `preround_formation.png` — confidence histogram
- `ql_coverage.png` — fraction of Q-table states visited
- `lstm_accuracy.png` — confusion matrices

`src/report/training_charts.py` reads `data/training_results.json`
and emits the model-evaluation figures referenced from
[README §3](../README.md#3-numerical-analysis):

- `training_accuracy.png` — train/val/test accuracy by model
- `training_loss_curves.png` — per-epoch loss for each NN
- `classifier_metrics.png` — Test Acc vs Macro-F1 vs majority baseline
- `f1_scores.png` — Macro-F1 vs Weighted-F1
- `f1_per_class.png` — per-class F1 breakdown (truncated to top 10
  classes by support)
- `qlearner_agreement.png` — expert-action agreement on val/test

## Metrics

`src/analysis/metrics.py` computes:

- Accuracy, Macro-F1, Weighted-F1
- Per-class F1 + per-class support + class names
- Majority-class baseline accuracy

It is the canonical source of evaluation numbers and back-fills
`training_results.json` so the chart and the README stay consistent.
Run `python src/analysis/metrics.py` after re-training to refresh
the metrics block in the JSON.
