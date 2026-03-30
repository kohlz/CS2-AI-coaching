"""
event_extractor.py

Event-driven HUD extraction for CS2 game frames.

Instead of running expensive OCR on every sampled frame, this module checks
every frame for pixel-level changes (~1-2ms) and only triggers OCR when a
HUD field actually changes.  This allows processing every frame of a 30fps
video (or live stream) while keeping total OCR cost low.

Usage
-----
    from event_extractor import EventDrivenExtractor

    ext = EventDrivenExtractor(fps=30.0)
    for frame_idx, frame in enumerate(frames):
        state, changed = ext.process(frame, frame_idx)
        if changed:
            print(state)
"""

import cv2
import numpy as np
from typing import Optional

from hud_extractor import (
    _crop,
    detect_phase,
    detect_player_state,
    detect_side,
    extract_top_center,
    extract_hp,
    extract_money,
    extract_armor,
    detect_helmet,
    detect_kit,
    extract_weapon_class,
    _ocr_single_digit,
)


# ---------------------------------------------------------------------------
# Change detector
# ---------------------------------------------------------------------------

class ChangeDetector:
    """
    Monitors a single HUD ROI for pixel-level changes.

    Workflow per frame:
      1. Crop the ROI, convert to grayscale, threshold to binary.
      2. Compute mean absolute diff against the previous binary image.
      3. If diff exceeds *threshold* for *debounce* consecutive frames,
         report a change and reset the counter.

    The binary thresholding isolates bright HUD text from the dark
    semi-transparent panel, making the detector immune to game-world
    movement behind the HUD overlay.
    """

    def __init__(self, roi_key: str, threshold: float = 12.0,
                 debounce: int = 2, binary_thresh: int = 130):
        self.roi_key = roi_key
        self.threshold = threshold
        self.debounce = debounce
        self.binary_thresh = binary_thresh
        self._prev_binary: Optional[np.ndarray] = None
        self._dirty_count: int = 0

    def check(self, frame: np.ndarray) -> bool:
        roi = _crop(frame, self.roi_key)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.binary_thresh, 255,
                                  cv2.THRESH_BINARY)

        triggered = False
        if self._prev_binary is not None and binary.shape == self._prev_binary.shape:
            diff = float(cv2.absdiff(binary, self._prev_binary).mean())
            if diff > self.threshold:
                self._dirty_count += 1
            else:
                self._dirty_count = 0
            if self._dirty_count >= self.debounce:
                triggered = True
                self._dirty_count = 0

        self._prev_binary = binary
        return triggered

    def reset(self) -> None:
        self._prev_binary = None
        self._dirty_count = 0


# ---------------------------------------------------------------------------
# Event-driven extractor
# ---------------------------------------------------------------------------

_WEAPON_POLL_INTERVAL = 30    # frames between weapon CNN polls
_TIMER_RESYNC_INTERVAL = 300  # frames (~10s at 30fps) between timer OCR
_DEATH_CONFIRM = 8            # consecutive non-alive frames to confirm death
_PHASE_CONFIRM = 5            # consecutive identical phase reads to confirm switch


class EventDrivenExtractor:
    """
    Two-tier extraction engine.

    Tier 1 (every frame, ~2ms):
      - detect_phase, detect_player_state, detect_side  (pixel-only, no OCR)
      - pixel-diff change detectors on 6 monitored ROIs

    Tier 2 (on-demand, ~100-200ms per field):
      - Tesseract OCR on the specific ROI that changed
      - Full extraction once on phase transition (freeze -> live)
    """

    def __init__(self, fps: float = 30.0):
        self._fps = fps

        self._detectors = {
            "alive_left":  ChangeDetector("alive_left",  threshold=12, debounce=2),
            "alive_right": ChangeDetector("alive_right", threshold=12, debounce=2),
            "hp":          ChangeDetector("hp",          threshold=10, debounce=1,
                                          binary_thresh=180),
            "money":       ChangeDetector("money",       threshold=12, debounce=3,
                                          binary_thresh=180),
            "armor_value": ChangeDetector("armor_value", threshold=10, debounce=3,
                                          binary_thresh=180),
            "armor_icon":  ChangeDetector("armor_icon",  threshold=8,  debounce=2),
        }

        self._state: dict = {
            "phase": "freeze",
            "player_state": "alive",
            "player_alive": True,
            "side": "unknown",
            "time_left": None,
            "teammates_alive": None,
            "enemies_alive": None,
            "hp": None,
            "money": None,
            "armor": None,
            "helmet": None,
            "has_kit": None,
            "weapon_class": None,
        }

        self._prev_phase: str = "freeze"
        self._confirmed_phase: str = "freeze"
        self._phase_streak: int = 0
        self._phase_candidate: str = "freeze"

        self._not_alive_streak: int = 0

        # Sticky-fresh: re-emit OCR values for a few frames after trigger
        # so the StateTracker's decrease-confirmation logic can see two
        # consecutive identical readings.
        self._fresh_until: dict[str, int] = {}
        self._fresh_values: dict[str, object] = {}

        self._timer_anchor_frame: Optional[int] = None
        self._timer_anchor_value: Optional[int] = None
        self._last_timer_ocr_frame: int = -_TIMER_RESYNC_INTERVAL

    def process(self, frame: np.ndarray,
                frame_idx: int) -> tuple[dict, bool]:
        """
        Process a single game frame (already letterbox-stripped).

        Returns (state_dict, changed) where *changed* is True when at least
        one field was updated compared to the previous call.
        """
        changed = False

        # ── Tier 1: cheap checks (every frame) ──────────────────────────

        raw_phase = detect_phase(frame)
        raw_player_state = detect_player_state(frame)
        raw_side = detect_side(frame)

        # Phase hysteresis: require _PHASE_CONFIRM consecutive identical
        # readings before confirming a phase change.
        if raw_phase == self._phase_candidate:
            self._phase_streak += 1
        else:
            self._phase_candidate = raw_phase
            self._phase_streak = 1
        if self._phase_streak >= _PHASE_CONFIRM:
            self._confirmed_phase = raw_phase

        phase = self._confirmed_phase

        # Player-state hysteresis: require _DEATH_CONFIRM consecutive
        # non-alive readings to confirm death. Prevents a single bad
        # frame from locking the player as dead for the rest of the round.
        if raw_player_state == "alive":
            self._not_alive_streak = 0
            player_state = "alive"
        else:
            self._not_alive_streak += 1
            if self._not_alive_streak >= _DEATH_CONFIRM:
                player_state = raw_player_state
            else:
                player_state = self._state.get("player_state", "alive")

        if phase != self._state["phase"]:
            changed = True
        if player_state != self._state["player_state"]:
            changed = True

        side = raw_side
        if side != "unknown" and side != self._state["side"]:
            changed = True

        self._state["phase"] = phase
        self._state["player_state"] = player_state
        self._state["player_alive"] = (player_state == "alive")
        if side != "unknown":
            self._state["side"] = side

        # ── Phase transition: full extraction to seed the round ──────────

        phase_just_went_live = (phase == "live" and self._prev_phase != "live")
        if phase_just_went_live:
            self._not_alive_streak = 0
            self._state["player_state"] = "alive"
            self._state["player_alive"] = True
        self._prev_phase = phase

        if phase_just_went_live:
            self._full_extraction(frame, frame_idx)
            self._reset_detectors()
            return dict(self._state), True

        # ── Player not alive: null out game fields ───────────────────────

        if not self._state["player_alive"]:
            nulled = self._null_game_fields()
            if nulled:
                changed = True
            self._reset_detectors()
            return dict(self._state), changed

        # ── Phase is not live: nothing to extract ────────────────────────

        if phase != "live":
            return dict(self._state), changed

        # ── Tier 2: change detection + targeted OCR ──────────────────────
        # For change-detected fields, we only emit the freshly-OCR'd value
        # on the frame it was read.  On all other frames we emit None so the
        # StateTracker carries forward its own confirmed value (and its
        # alive-count confirmation logic isn't defeated by stale repeats).

        _OCR_FIELDS = ("teammates_alive", "enemies_alive", "hp",
                       "money", "armor", "helmet")
        fresh: dict[str, object] = {}

        triggered: set[str] = set()
        for name, det in self._detectors.items():
            if det.check(frame):
                triggered.add(name)

        if "alive_left" in triggered or "alive_right" in triggered:
            ta, ea = self._read_alive_counts(frame)
            if ta is not None:
                fresh["teammates_alive"] = ta
                self._state["teammates_alive"] = ta
            if ea is not None:
                fresh["enemies_alive"] = ea
                self._state["enemies_alive"] = ea
            changed = True

        if "hp" in triggered:
            val = extract_hp(frame)
            if val is not None:
                prev_hp = self._state.get("hp")
                if prev_hp is None or abs(val - prev_hp) <= 60:
                    fresh["hp"] = val
                    self._state["hp"] = val
                    changed = True

        if "money" in triggered:
            val = extract_money(frame)
            if val is not None:
                fresh["money"] = val
                self._state["money"] = val
                changed = True

        if "armor_value" in triggered:
            val = extract_armor(frame)
            if val is not None:
                prev_armor = self._state.get("armor")
                if prev_armor is None or abs(val - prev_armor) <= 60:
                    fresh["armor"] = val
                    self._state["armor"] = val
                    changed = True

        if "armor_icon" in triggered:
            helmet = detect_helmet(frame)
            if helmet is not None:
                fresh["helmet"] = helmet
                self._state["helmet"] = helmet
                changed = True

        # ── Timer: decrement by wall-clock, resync periodically ──────────

        timer_changed = self._update_timer(frame, frame_idx)
        if timer_changed:
            changed = True

        # ── Weapon: poll every N frames (CNN is cheap) ───────────────────

        if frame_idx % _WEAPON_POLL_INTERVAL == 0:
            wc = extract_weapon_class(frame)
            if wc is not None and wc != self._state["weapon_class"]:
                self._state["weapon_class"] = wc
                changed = True

        # ── Kit: cheap pixel check, poll every 30 frames ────────────────

        if frame_idx % _WEAPON_POLL_INTERVAL == 0:
            if self._state["side"] == "CT":
                self._state["has_kit"] = detect_kit(frame)
            else:
                self._state["has_kit"] = False

        # ── Update sticky-fresh tracking: keep emitting fresh values for
        #    _STICKY_FRAMES after the trigger so the StateTracker can
        #    confirm decreases (it needs 2 consecutive identical readings).
        _STICKY_FRAMES = 3
        for f, v in fresh.items():
            self._fresh_until[f] = frame_idx + _STICKY_FRAMES
            self._fresh_values[f] = v

        # ── Build output: fresh/sticky OCR values where available, else None
        out = dict(self._state)
        for f in _OCR_FIELDS:
            if f in fresh:
                out[f] = fresh[f]
            elif self._fresh_until.get(f, 0) >= frame_idx:
                out[f] = self._fresh_values[f]
            else:
                out[f] = None
        return out, changed

    # ── Internal helpers ─────────────────────────────────────────────────

    def _full_extraction(self, frame: np.ndarray, frame_idx: int) -> None:
        """Run OCR on all fields — used once at round start."""
        time_left, ct_alive, t_alive = extract_top_center(frame)
        side = self._state["side"]

        if side == "T":
            teammates, enemies = t_alive, ct_alive
        elif side == "CT":
            teammates, enemies = ct_alive, t_alive
        else:
            teammates, enemies = None, None

        self._state["time_left"] = time_left
        self._state["teammates_alive"] = teammates
        self._state["enemies_alive"] = enemies
        self._state["hp"] = extract_hp(frame)
        self._state["money"] = extract_money(frame)
        self._state["armor"] = extract_armor(frame)
        self._state["helmet"] = detect_helmet(frame)
        self._state["has_kit"] = detect_kit(frame) if side == "CT" else False
        self._state["weapon_class"] = extract_weapon_class(frame)

        if time_left is not None:
            self._timer_anchor_frame = frame_idx
            self._timer_anchor_value = time_left
            self._last_timer_ocr_frame = frame_idx

    def _read_alive_counts(self, frame: np.ndarray
                           ) -> tuple[Optional[int], Optional[int]]:
        """OCR both alive count digits and map to (teammates, enemies)."""
        ct_alive = _ocr_single_digit(_crop(frame, "alive_left"))
        t_alive = _ocr_single_digit(_crop(frame, "alive_right"))

        side = self._state["side"]
        if side == "T":
            return (t_alive, ct_alive)
        elif side == "CT":
            return (ct_alive, t_alive)
        return (None, None)

    def _update_timer(self, frame: np.ndarray, frame_idx: int) -> bool:
        """
        Maintain time_left by decrementing from the last known OCR anchor.
        Re-syncs with OCR every TIMER_RESYNC_INTERVAL frames.
        """
        needs_ocr = (
            self._timer_anchor_value is None
            or (frame_idx - self._last_timer_ocr_frame) >= _TIMER_RESYNC_INTERVAL
        )

        if needs_ocr:
            time_left, _, _ = extract_top_center(frame)
            if time_left is not None:
                self._timer_anchor_frame = frame_idx
                self._timer_anchor_value = time_left
                self._last_timer_ocr_frame = frame_idx
                old = self._state["time_left"]
                self._state["time_left"] = time_left
                return old != time_left

        if self._timer_anchor_value is not None and self._timer_anchor_frame is not None:
            elapsed_frames = frame_idx - self._timer_anchor_frame
            elapsed_seconds = int(elapsed_frames / self._fps)
            computed = self._timer_anchor_value - elapsed_seconds
            computed = max(computed, 0)
            old = self._state["time_left"]
            self._state["time_left"] = computed
            return old != computed

        return False

    def _null_game_fields(self) -> bool:
        """Set all gameplay fields to None. Returns True if anything changed."""
        fields = ("time_left", "teammates_alive", "enemies_alive", "hp",
                  "money", "armor", "helmet", "has_kit", "weapon_class")
        any_changed = False
        for f in fields:
            if self._state[f] is not None:
                self._state[f] = None
                any_changed = True
        self._timer_anchor_value = None
        self._timer_anchor_frame = None
        return any_changed

    def _reset_detectors(self) -> None:
        """Clear cached binary images in all change detectors."""
        for det in self._detectors.values():
            det.reset()
