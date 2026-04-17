"""Extract all demo data once, save to disk, and split into train/val/test.

Splits by (demo, round_num) pairs into 70/15/15 train/val/test by default.
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_data import extract_all, discover_demos


def _round_key(demo: str, round_num: int) -> str:
    return f"{demo}::{round_num}"


def _split_rounds(round_keys: list[str],
                  train_ratio: float = 0.70,
                  val_ratio: float = 0.15) -> dict[str, set[str]]:
    """Split (demo, round) keys deterministically into train/val/test."""
    sorted_keys = sorted(round_keys,
                         key=lambda k: hashlib.md5(k.encode()).hexdigest())
    n = len(sorted_keys)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train = sorted_keys[:n_train]
    val = sorted_keys[n_train:n_train + n_val]
    test = sorted_keys[n_train + n_val:]
    if not test and val:
        test = [val.pop()]

    return {"train": set(train), "val": set(val), "test": set(test)}


def _split_df_by_round(df: pd.DataFrame,
                       split: dict[str, set[str]]) -> dict[str, pd.DataFrame]:
    """Split a DataFrame by (demo, round_num) into train/val/test."""
    result = {}
    if df.empty or "demo" not in df.columns or "round_num" not in df.columns:
        for name in split:
            result[name] = df.iloc[0:0].copy()
        return result
    keys = df["demo"].astype(str) + "::" + df["round_num"].astype(str)
    for name, key_set in split.items():
        mask = keys.isin(key_set)
        result[name] = df[mask].reset_index(drop=True)
    return result


def _split_sequences_by_round(sequences: list[dict],
                              split: dict[str, set[str]]
                              ) -> dict[str, list[dict]]:
    """Split a list of sequence dicts by (demo, round_num)."""
    result = {name: [] for name in split}
    for seq in sequences:
        key = _round_key(seq.get("demo", ""), seq.get("round_num", 0))
        for name, key_set in split.items():
            if key in key_set:
                result[name].append(seq)
                break
    return result


def build_and_save_datasets(
    demo_dir: str = "src/demo/train_demos",
    output_dir: str = "data",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    verbose: bool = True,
) -> dict:
    """Extract all demos, split by round, and save to disk.

    Returns a dict with split info and dataset sizes.
    """
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("  DATASET BUILDER — Extract, Split, Save")
        print("=" * 60)
        print()

    # Extract everything
    if verbose:
        print("[1/3] Extracting data from all demos...")
    data = extract_all(
        demo_dir,
        include_rl=True,
        include_rl_v2=True,
        include_sequences=True,
        include_ct_formations=True,
        verbose=verbose,
    )

    rounds_df = data["rounds"]
    rl_df = data["rl_transitions"]
    rl_v2_df = data["rl_v2"]
    event_seqs = data["event_sequences"]
    ct_form_seqs = data["ct_formation_sequences"]

    if rounds_df.empty:
        print("ERROR: No data extracted from demos!")
        return {}

    # Split by round (demo + round_num pair)
    round_keys = [_round_key(d, rn)
                  for d, rn in zip(rounds_df["demo"], rounds_df["round_num"])]
    round_keys = sorted(set(round_keys))

    if verbose:
        n_demos = rounds_df["demo"].nunique()
        print(f"\n[2/3] Splitting {len(round_keys)} rounds "
              f"from {n_demos} demos "
              f"({train_ratio:.0%} / {val_ratio:.0%} / "
              f"{1 - train_ratio - val_ratio:.0%})...")

    split = _split_rounds(round_keys, train_ratio, val_ratio)

    if verbose:
        for name, keys in split.items():
            n_d = len({k.split("::")[0] for k in keys})
            print(f"  {name:5s}: {len(keys)} rounds across {n_d} demos")

    # Split DataFrames / sequences by (demo, round) key
    rounds_split = _split_df_by_round(rounds_df, split)
    rl_split = (_split_df_by_round(rl_df, split) if not rl_df.empty else
                {k: pd.DataFrame() for k in split})
    rl_v2_split = (_split_df_by_round(rl_v2_df, split) if not rl_v2_df.empty
                    else {k: pd.DataFrame() for k in split})
    seq_split = _split_sequences_by_round(event_seqs, split)
    ct_form_split = _split_sequences_by_round(ct_form_seqs, split)

    # Save to disk
    if verbose:
        print(f"\n[3/3] Saving datasets to {output_dir}/...")

    for name in ["train", "val", "test"]:
        r_path = os.path.join(output_dir, f"rounds_{name}.csv")
        rounds_split[name].to_csv(r_path, index=False)
        if verbose:
            print(f"  rounds_{name}.csv: {len(rounds_split[name])} rows")

        if not rl_split[name].empty:
            rl_path = os.path.join(output_dir, f"rl_v1_{name}.csv")
            rl_split[name].to_csv(rl_path, index=False)
            if verbose:
                print(f"  rl_v1_{name}.csv: {len(rl_split[name])} rows")

        if not rl_v2_split[name].empty:
            rl2_path = os.path.join(output_dir, f"rl_v2_{name}.csv")
            rl_v2_split[name].to_csv(rl2_path, index=False)
            if verbose:
                print(f"  rl_v2_{name}.csv: {len(rl_v2_split[name])} rows")

        seq_path = os.path.join(output_dir, f"event_sequences_{name}.json")
        with open(seq_path, "w", encoding="utf-8") as f:
            json.dump(seq_split[name], f)
        if verbose:
            n_events = sum(len(s["events"]) for s in seq_split[name])
            print(f"  event_sequences_{name}.json: "
                  f"{len(seq_split[name])} rounds, {n_events} events")

        ct_path = os.path.join(output_dir, f"ct_formations_{name}.json")
        with open(ct_path, "w", encoding="utf-8") as f:
            json.dump(ct_form_split[name], f)
        if verbose:
            print(f"  ct_formations_{name}.json: "
                  f"{len(ct_form_split[name])} rounds")

    split_info = {
        "split_strategy": "by_round",
        "split": {name: sorted(list(keys)) for name, keys in split.items()},
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1 - train_ratio - val_ratio, 2),
        "total_rounds": len(round_keys),
        "total_demos": int(rounds_df["demo"].nunique()),
        "demo_dir": demo_dir,
        "sizes": {
            name: {
                "rounds": len(rounds_split[name]),
                "demos": len({k.split("::")[0] for k in split[name]}),
                "rl_v2_transitions": len(rl_v2_split[name]),
                "event_sequences": len(seq_split[name]),
                "ct_formation_sequences": len(ct_form_split[name]),
            }
            for name in ["train", "val", "test"]
        },
    }

    info_path = os.path.join(output_dir, "split_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    if verbose:
        print(f"\n  Split info saved to {info_path}")
        print(f"\n{'='*60}")
        print("  Dataset Summary")
        print(f"{'='*60}")
        for name in ["train", "val", "test"]:
            s = split_info["sizes"][name]
            print(f"  {name:5s}: {s['demos']} demos, "
                  f"{s['rounds']} rounds, "
                  f"{s['rl_v2_transitions']} RL transitions, "
                  f"{s['event_sequences']} LSTM seqs, "
                  f"{s['ct_formation_sequences']} CT-form seqs")
        print(f"{'='*60}")

    return split_info


# ---------------------------------------------------------------------------
# Load saved datasets
# ---------------------------------------------------------------------------

def load_split(output_dir: str = "data",
               split_name: str = "train") -> dict:
    """Load a previously saved dataset split from disk.

    Returns dict with keys: rounds, rl_v1, rl_v2,
    event_sequences, ct_formation_sequences
    """
    result = {}

    r_path = os.path.join(output_dir, f"rounds_{split_name}.csv")
    result["rounds"] = pd.read_csv(r_path) if os.path.exists(r_path) else pd.DataFrame()

    rl1_path = os.path.join(output_dir, f"rl_v1_{split_name}.csv")
    result["rl_transitions"] = pd.read_csv(rl1_path) if os.path.exists(rl1_path) else pd.DataFrame()

    rl2_path = os.path.join(output_dir, f"rl_v2_{split_name}.csv")
    result["rl_v2"] = pd.read_csv(rl2_path) if os.path.exists(rl2_path) else pd.DataFrame()

    seq_path = os.path.join(output_dir, f"event_sequences_{split_name}.json")
    if os.path.exists(seq_path):
        with open(seq_path, "r", encoding="utf-8") as f:
            result["event_sequences"] = json.load(f)
    else:
        result["event_sequences"] = []

    ct_path = os.path.join(output_dir, f"ct_formations_{split_name}.json")
    if os.path.exists(ct_path):
        with open(ct_path, "r", encoding="utf-8") as f:
            result["ct_formation_sequences"] = json.load(f)
    else:
        result["ct_formation_sequences"] = []

    return result


def load_split_info(output_dir: str = "data") -> dict:
    """Load the split metadata."""
    info_path = os.path.join(output_dir, "split_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_dir = sys.argv[1] if len(sys.argv) > 1 else "src/demo/train_demos"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data"

    build_and_save_datasets(demo_dir, out_dir)
