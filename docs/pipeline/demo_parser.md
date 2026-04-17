# Demo Parser

**Source:** [`src/demo/demo_parser.py`](../../src/demo/demo_parser.py)

## Role in the Pipeline

Reads a single CS2 demo file (`.dem`) and produces a structured
`Match` object with one `RoundData` per round. Every downstream
component — feature extraction, model inference, report generation —
consumes data shaped by this module.

Returning to [main pipeline](../README.md#1-pipeline).

## Python Package

- [`demoparser2`](https://pypi.org/project/demoparser2/) — Rust-backed
  CS2 demo parser; the only third-party dependency for parsing.
- `numpy`, `pandas`, `dataclasses` — for restructuring parsed events
  into the project's data classes.

`demoparser2` exposes the demo as a wide DataFrame keyed by tick. We
project that into:

- `Position` (x, y, z, yaw, pitch)
- `PlayerSnapshot` (per-tick per-player state)
- `RoundEvent` (kill / smoke / flash / he / molotov / plant /
  defuse / round_start / round_end)
- `RoundData` (start_tick, end_tick, winner, score, all events,
  per-tick player snapshots)
- `Match` (sequence of RoundData + map name + scoreline)

## Usage

```python
from demo_parser import parse_demo

match = parse_demo("path/to/demo.dem", target_player="k_z_")
for r in match.rounds:
    print(r.round_num, r.winner, len(r.events))
```

The `target_player` argument is the *display* name (or steam ID) of
the player whose perspective the rest of the pipeline analyses. It
is propagated all the way through into the per-round coaching
report.

## Demo Conventions

- **Tick rate** is fixed at 64 Hz (CS2 standard).
- **Round numbering** is 0-indexed within the match. Half boundaries
  are tracked separately so `is_second_half` can be set on
  pre-round features.
- **Knife rounds** are detected and filtered downstream in
  `training_data.py` so they never enter the dataset (knife rounds
  do not award real money / streak bonuses, which would corrupt
  the economy HMM).

## Output Use

- `training_data.extract_all(...)` calls `parse_demo` for every demo
  in `src/demo/train_demos/` to build the training tables.
- `report.generate_report.generate_full_report(...)` calls
  `parse_demo` on the single demo whose report is being generated.
