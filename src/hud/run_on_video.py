"""
run_on_video.py

Run the HUD extractor on a recorded video file and print per-frame results.

Supports two extraction modes:

  Event-driven (default):
      Checks every frame for pixel-level changes (~2ms) and only runs OCR
      when a HUD field actually changes.  No --sample flag needed.

  Polling (legacy):
      Runs full OCR on every N-th frame.  Activate with --poll N.

Usage
-----
    python run_on_video.py <path_to_video.mp4>
    python run_on_video.py <path_to_video.mp4> --poll 10

Optional flags
--------------
    --poll N        Legacy mode: full OCR every N-th frame (disables event mode)
    --save-frames   Save a debug image for every processed frame to debug_frames/
    --no-ocr-skip   Print every frame even if state is unchanged

Output
------
Prints a formatted table to stdout and saves a CSV next to the video file.
"""

import sys
import csv
import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Allow running from the hud/ directory or from the repo root
sys.path.insert(0, str(Path(__file__).parent))
from hud_extractor import (
    FrameAdapter,
    extract_live_state,
    debug_live_state,
)
from event_extractor import EventDrivenExtractor


# ---------------------------------------------------------------------------
# Temporal smoother
# ---------------------------------------------------------------------------

class StateTracker:
    """
    Applies frame-to-frame consistency rules to raw extracted state.

    Problems addressed
    ------------------
    Phase hysteresis   : require PHASE_CONFIRM consecutive identical phase
                         readings before switching, to suppress rapid
                         freeze↔live flicker during round transitions.
    Player monotonicity: within a round the player lifecycle is strictly
                         alive → dead/observer.  Once the player leaves
                         "alive", they cannot return until a new round.
    Alive-count mono   : teammates_alive and enemies_alive can only decrease
                         within a live round (no respawns in competitive).
    Timer mono         : time_left should never increase during a live round.
    HP mono            : health cannot increase mid-round (no healing in comp).
    Armor stability    : reject single-frame OCR jumps > 60 in either direction.
    Helmet lock        : once helmet=True, stays True until armor breaks to 0.
    Weapon confirm     : require 2 consecutive agreeing reads to change weapon.
    None carry-forward : carry last valid value for intermittent None reads.
    """

    PHASE_CONFIRM = 2

    _CARRY_FIELDS = ("side", "money", "teammates_alive", "enemies_alive",
                     "weapon_class", "armor", "helmet", "has_kit")

    _NULL_FIELDS = ("time_left", "teammates_alive", "enemies_alive", "hp", "money",
                    "armor", "helmet", "has_kit", "weapon_class")

    _ROUND_FIELDS = ("time_left", "hp", "teammates_alive", "enemies_alive",
                     "armor", "helmet", "_pending_weapon", "_died_this_round",
                     "_pending_teammates_alive", "_pending_enemies_alive")

    def __init__(self) -> None:
        self._last: dict = {}
        self._phase_streak: int = 0
        self._phase_candidate: Optional[str] = None
        self._confirmed_phase: Optional[str] = None

    def _confirm_phase(self, raw_phase: Optional[str]) -> str:
        """Require PHASE_CONFIRM identical readings before switching phase."""
        if raw_phase == self._phase_candidate:
            self._phase_streak += 1
        else:
            self._phase_candidate = raw_phase
            self._phase_streak = 1

        if self._phase_streak >= self.PHASE_CONFIRM:
            self._confirmed_phase = raw_phase
        return self._confirmed_phase or raw_phase

    def update(self, raw: dict) -> dict:
        out = dict(raw)
        prev = self._last

        # --- Phase hysteresis: suppress freeze↔live flicker ---
        stable_phase = self._confirm_phase(raw.get("phase"))
        out["phase"] = stable_phase

        # --- Player state monotonicity ---
        # Once player transitions away from "alive" in a round, they cannot
        # return to "alive" until a new round starts (phase change).
        prev_confirmed_phase = prev.get("phase")
        phase_changed = (prev_confirmed_phase is not None
                         and prev_confirmed_phase != stable_phase)
        if phase_changed:
            for field in self._ROUND_FIELDS:
                self._last.pop(field, None)
            prev = self._last

        died_this_round = prev.get("_died_this_round", False)
        raw_state = raw.get("player_state", "alive")
        raw_alive = raw.get("player_alive", True)

        if died_this_round:
            out["player_state"] = raw_state if raw_state != "alive" else "observer"
            out["player_alive"] = False
        else:
            if raw_state in ("dead", "observer") or not raw_alive:
                died_this_round = True
                out["player_alive"] = False
            out["player_state"] = raw_state
        out["_died_this_round"] = died_this_round

        # ----- Player is NOT alive ----------------------------------------
        if not out.get("player_alive"):
            for field in self._NULL_FIELDS:
                out[field] = None
            for field in self._NULL_FIELDS:
                self._last.pop(field, None)
            self._last.update({k: v for k, v in out.items() if v is not None})
            for k in list(out):
                if k.startswith("_"):
                    out.pop(k)
            return out

        # ----- Player IS alive — normal smoothing -------------------------

        # --- time_left: should decrease; reject upward jumps ---
        tl = raw.get("time_left")
        prev_tl = prev.get("time_left")
        if tl is None:
            out["time_left"] = prev_tl
        elif prev_tl is not None and tl > prev_tl + 3:
            out["time_left"] = prev_tl
        else:
            out["time_left"] = tl

        # --- hp: should not increase during a live round ---
        hp = raw.get("hp")
        prev_hp = prev.get("hp")
        if hp is None:
            out["hp"] = prev_hp
        elif (stable_phase == "live"
              and prev_hp is not None
              and hp > prev_hp + 10):
            out["hp"] = prev_hp
        else:
            out["hp"] = hp

        # --- armor: reject implausible single-frame OCR jumps (>60) ---
        armor = raw.get("armor")
        prev_armor = prev.get("armor")
        if armor is None:
            out["armor"] = prev_armor
        elif (stable_phase == "live"
              and prev_armor is not None
              and abs(armor - prev_armor) > 60):
            out["armor"] = prev_armor
        else:
            out["armor"] = armor

        # --- helmet: lock True once seen (until armor breaks to 0) ---
        helmet = raw.get("helmet")
        prev_helmet = prev.get("helmet")
        if stable_phase == "live" and prev_helmet is True:
            cur_armor = out.get("armor")
            if cur_armor is not None and cur_armor > 0:
                out["helmet"] = True
            else:
                out["helmet"] = helmet
        else:
            out["helmet"] = helmet

        # --- weapon_class: require 2 consecutive agreeing reads ---
        wc = raw.get("weapon_class")
        prev_wc = prev.get("weapon_class")
        pending_wc = prev.get("_pending_weapon")
        if wc is None:
            out["weapon_class"] = prev_wc
        elif wc == prev_wc:
            out["weapon_class"] = wc
        elif wc == pending_wc:
            out["weapon_class"] = wc
        else:
            out["weapon_class"] = prev_wc if prev_wc is not None else wc
        out["_pending_weapon"] = wc

        # --- alive counts: accept upward corrections, confirm decreases ---
        # During a round, OCR may under-read digits.  We accept increases
        # immediately (correcting upward from earlier OCR errors) but require
        # two consecutive lower readings to confirm a genuine decrease.
        for field in ("teammates_alive", "enemies_alive"):
            val = raw.get(field)
            prev_val = prev.get(field)
            pending_key = f"_pending_{field}"
            pending_dec = prev.get(pending_key)

            if val is None:
                out[field] = prev_val
            elif prev_val is None or val >= prev_val:
                out[field] = val
            elif val == pending_dec:
                out[field] = val
            else:
                out[field] = prev_val
            out[pending_key] = val

        # --- other fields: carry forward when None ---
        for field in self._CARRY_FIELDS:
            if out.get(field) is None and prev.get(field) is not None:
                out[field] = prev[field]

        self._last = {k: v for k, v in out.items() if v is not None}

        for k in list(out):
            if k.startswith("_"):
                out.pop(k)
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HUD extractor on a CS2 video.")
    p.add_argument("video", help="Path to the input video file")
    p.add_argument("--poll", type=int, default=0, metavar="N",
                   help="Legacy polling mode: full OCR every N-th frame "
                        "(default: 0 = event-driven mode)")
    p.add_argument("--save-frames", action="store_true",
                   help="Save a debug image for each processed frame")
    p.add_argument("--no-ocr-skip", action="store_true",
                   help="Print every frame even if state is unchanged")
    return p.parse_args()


def _open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {path}")
        sys.exit(1)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, total, width, height


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    cap, fps, total, width, height = _open_video(video_path)
    duration = total / fps

    print(f"Video : {video_path.name}")
    print(f"Size  : {width}×{height}  |  {fps:.1f} fps  |  "
          f"{total} frames  |  {duration:.1f}s")
    mode_label = f"poll every {args.poll}" if args.poll else "event-driven"
    print(f"Mode  : {mode_label}")
    print()

    # --- FrameAdapter calibration -----------------------------------------
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame.")
        sys.exit(1)

    adapter = FrameAdapter()
    adapter.setup(first_frame)
    print(f"FrameAdapter: aspect={adapter.aspect}  "
          f"game_rect={adapter.game_rect}")
    print()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Output setup -----------------------------------------------------
    csv_path = video_path.with_suffix(".csv")
    frame_dir = video_path.parent / "debug_frames"
    if args.save_frames:
        frame_dir.mkdir(exist_ok=True)

    columns = [
        "frame", "time_s", "phase", "player_state", "player_alive", "side",
        "time_left", "teammates_alive", "enemies_alive", "hp", "money",
        "armor", "helmet", "has_kit", "weapon_class",
    ]

    print("  ".join(f"{c:>15}" for c in columns))
    print("-" * (17 * len(columns)))

    tracker = StateTracker()
    rows: list[dict] = []
    prev_state: dict | None = None
    frame_idx = 0
    processed = 0
    ocr_triggers = 0
    t_start = time.perf_counter()

    if not args.poll:
        # Event-driven mode: the event extractor already applies phase +
        # player-state hysteresis, so disable the StateTracker's own phase
        # hysteresis to avoid timing mismatches.
        tracker.PHASE_CONFIRM = 1

    if args.poll:
        # ── Legacy polling mode ──────────────────────────────────────────
        while True:
            ret, raw = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % args.poll != 0:
                continue

            game_frame = adapter.crop(raw)
            raw_state = extract_live_state(game_frame)
            state = tracker.update(raw_state)
            ocr_triggers += 1

            state_changed = (prev_state is None or state != prev_state)
            if not args.no_ocr_skip and not state_changed:
                processed += 1
                continue

            time_s = frame_idx / fps
            row = {c: state.get(c) for c in columns}
            row["frame"] = frame_idx
            row["time_s"] = round(time_s, 2)
            rows.append(row)

            values = [str(row[c]) for c in columns]
            print("  ".join(f"{v:>15}" for v in values))

            if args.save_frames:
                debug_live_state(
                    game_frame,
                    save_path=str(frame_dir / f"frame_{frame_idx:06d}.png"),
                )
            prev_state = state
            processed += 1
    else:
        # ── Event-driven mode ────────────────────────────────────────────
        extractor = EventDrivenExtractor(fps=fps)

        while True:
            ret, raw = cap.read()
            if not ret:
                break
            frame_idx += 1

            game_frame = adapter.crop(raw)
            raw_state, event_changed = extractor.process(game_frame, frame_idx)
            state = tracker.update(raw_state)
            processed += 1

            if event_changed:
                ocr_triggers += 1

            state_changed = (prev_state is None or state != prev_state)
            if not args.no_ocr_skip and not state_changed:
                continue

            time_s = frame_idx / fps
            row = {c: state.get(c) for c in columns}
            row["frame"] = frame_idx
            row["time_s"] = round(time_s, 2)
            rows.append(row)

            values = [str(row[c]) for c in columns]
            print("  ".join(f"{v:>15}" for v in values))

            if args.save_frames:
                debug_live_state(
                    game_frame,
                    save_path=str(frame_dir / f"frame_{frame_idx:06d}.png"),
                )
            prev_state = state

    cap.release()

    elapsed = time.perf_counter() - t_start
    print()
    print(f"Processed {processed} frames in {elapsed:.1f}s  "
          f"({processed / elapsed:.1f} frames/s)  "
          f"OCR triggers: {ocr_triggers}")

    # --- Save CSV ---------------------------------------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
