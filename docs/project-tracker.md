# CS2 AI Coaching Project Tracker

## Project Stages
1. Repository setup and planning — Done
2. Resource collection and technical exploration — Done
3. Input pipeline selection and state representation — Done
4. First prototype implementation — Done
5. Recommendation generation and evaluation — Done
6. Final refinement and presentation preparation — In Progress

## Technical Direction Changes

### Input Pipeline
- **Original plan**: CV/YOLO-based visual input from game frames + CNN scene localization
- **Outcome**: HUD extractor pipeline was fully implemented (`src/hud/`) but processing speed was too slow for practical use
- **Final approach**: Demo file parsing via `demoparser2` for structured game data extraction

### Decision / Recommendation Layer
- **Original plan**: POMDP / belief-state model with rule-based updates
- **Outcome**: Rule-based belief update prototype completed (`src/belief/belief_update.py`) as v0
- **Final approach**: Tactical Q-learning (v1 + v2) for mid-round decisions, feedforward NNs for round-start predictions, LSTM for event-sequence prediction, HMM for enemy economy inference

## Final System Architecture

```
.dem file
   │
   ▼
demo_parser.py ──► MatchData (per-round structured data)
   │
   ├──► economy_mdp.py          Value Iteration optimal buy policy + player evaluation
   ├──► tactical_rl.py          Q-learning v1 (macro) + v2 (micro) tactical suggestions
   ├──► strategy_nn.py          WinPredictor, AttackPredictor, FormationClassifier (FFN)
   │                            EventSequencePredictor (LSTM)
   ├──► info_model.py           HMM enemy economy prediction (Bayes predict-update)
   ├──► engagement.py           Trade kills, opening duels, clutch detection, ADR
   ├──► training_data.py        Batch feature extraction from demos for all models
   │
   └──► generate_report.py      End-to-end coaching report (JSON + text output)
```

Supporting modules:
- `callouts_mirage.py` — game coordinates → Mirage map callout names
- `src/hud/` — HUD visual extraction pipeline (completed, not used in final system)
- `src/belief/belief_update.py` — rule-based belief update prototype (superseded by HMM + RL)

## Completed Modules

| Module | Location | Description | AI Method |
|---|---|---|---|
| Demo parser | `src/demo/demo_parser.py` | Parse .dem files into per-round structured data | — |
| Callout mapping | `src/demo/callouts_mirage.py` | Game coordinates → map region names | — |
| HUD extractor | `src/hud/hud_extractor.py` | OCR + pixel analysis of game frames | CNN (weapon), Tesseract OCR |
| Event extractor | `src/hud/event_extractor.py` | Event-driven frame processing | Change detection |
| Weapon classifier | `src/hud/weapon_classifier.py` | Classify weapon HUD icons | CNN (PyTorch) |
| Economy MDP | `src/analysis/economy_mdp.py` | Optimal buy policy + player evaluation | Value Iteration |
| Tactical RL v1 | `src/analysis/tactical_rl.py` | Mid-round macro decisions | Tabular Q-learning |
| Tactical RL v2 | `src/analysis/tactical_rl.py` | Mid-round micro decisions (dual reward) | Tabular Q-learning |
| Win predictor | `src/analysis/strategy_nn.py` | P(T wins round) from economy features | Feedforward NN |
| Attack predictor | `src/analysis/strategy_nn.py` | P(attack A / B / no plant) | Feedforward NN |
| Formation classifier | `src/analysis/strategy_nn.py` | CT defensive formation prediction | Feedforward NN |
| Event sequence predictor | `src/analysis/strategy_nn.py` | Dynamic attack site prediction from events | LSTM |
| Enemy economy HMM | `src/analysis/info_model.py` | Predict enemy team economy tier | Hidden Markov Model |
| Engagement analysis | `src/analysis/engagement.py` | Trade kills, opening duels, clutch, ADR | Heuristic |
| Training data extraction | `src/analysis/training_data.py` | Batch feature extraction for all models | — |
| Belief update (v0) | `src/belief/belief_update.py` | Rule-based enemy position belief | Rule-based prototype |
| Report generator | `src/report/generate_report.py` | End-to-end coaching report | — |

## Individual Contributions

- **Haozhe Zhu**: Implemented the complete system — demo parsing pipeline, HUD visual extraction pipeline (completed but deprecated due to speed), Mirage callout mapping, Economy MDP (Value Iteration), Tactical Q-learning (v1 + v2), WinPredictor / AttackPredictor / FormationClassifier (FFN), EventSequencePredictor (LSTM), Economy HMM, engagement analysis, training data extraction pipeline, end-to-end coaching report generator. Collected academic resources.
- **Zichen Mi**: Project documentation organization (README, check-in summary, initial project tracker), belief state / POMDP rule-based prototype (`belief_update.py`), labels and observation schema definition.
- **Bohan Shi**: Participated in early project direction discussion. No code contribution.
