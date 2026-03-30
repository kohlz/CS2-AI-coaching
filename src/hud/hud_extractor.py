"""
hud_extractor.py

HUD state extraction from CS2 game frames.

Resolution handling
-------------------
All ROI coordinates are stored as (y1_frac, y2_frac, x1_frac, x2_frac)
fractions of the *game content* dimensions, not the raw captured frame.

When the captured frame contains letterbox bars (e.g. a 4:3 game stream
embedded in a 16:9 Discord window), use FrameAdapter to strip the bars
before passing frames to the extractor functions:

    adapter = FrameAdapter()
    adapter.setup(first_frame)          # called once at startup

    for raw_frame in stream:
        frame = adapter.crop(raw_frame)  # strips black bars
        state = extract_live_state(frame)

ROI profiles are keyed by aspect ratio ('16:9', '4:3').  The 16:9 profile
is fully calibrated at 1920×1080.  The 4:3 profile uses the same values as
a starting point; run debug_live_state() on a 4:3 capture to verify the
boxes and tune the constants in _ROI_PROFILES['4:3'] as needed.

Public API
----------
  FrameAdapter                        letterbox + aspect-ratio detection
  detect_phase(frame)       -> "freeze" | "live"
  extract_weapon_class(frame) -> "awp" | "rifle" | "smg" | "pistol" | None
  detect_side(frame)        -> "T" | "CT" | "unknown"
  extract_top_center(frame) -> (time_left, ct_alive, t_alive)
  extract_hp(frame)         -> int | None
  extract_money(frame)      -> int | None
  extract_live_state(frame) -> dict

  debug_phase(frame, save_path)      -> dict + optional annotated image
  debug_live_state(frame, save_path) -> dict + optional annotated image
"""

import os
import re
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Tesseract setup
# ---------------------------------------------------------------------------

try:
    import pytesseract
    # On Windows, Tesseract is rarely on PATH after install.
    if not shutil.which("tesseract"):
        _WIN_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.isfile(_WIN_DEFAULT):
            pytesseract.pytesseract.tesseract_cmd = _WIN_DEFAULT
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False


# ---------------------------------------------------------------------------
# Frame adapter: letterbox stripping + aspect-ratio detection
# ---------------------------------------------------------------------------

def detect_letterbox(frame: np.ndarray,
                     black_threshold: int = 10) -> tuple[int, int, int, int]:
    """
    Find the bounding box of the actual game content inside a captured frame.

    Discord (and other stream viewers) may embed a 4:3 game image inside a
    16:9 window, adding black bars on the left and right.  This function
    scans inward from each edge until it finds non-black content.

    Parameters
    ----------
    frame           : raw captured BGR frame
    black_threshold : rows/columns with mean brightness below this are
                      treated as black bars (0–255)

    Returns
    -------
    (x1, y1, x2, y2) pixel coordinates of the game content region.
    Returns (0, 0, width, height) when no bars are detected.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _row_mean(y: int) -> float:
        return float(gray[y, :].mean())

    def _col_mean(x: int) -> float:
        return float(gray[:, x].mean())

    # Scan at most 30 % inward to avoid false positives on dark game scenes.
    limit_y = h // 3
    limit_x = w // 3

    y1 = 0
    while y1 < limit_y and _row_mean(y1) < black_threshold:
        y1 += 1

    y2 = h
    while y2 > h - limit_y and _row_mean(y2 - 1) < black_threshold:
        y2 -= 1

    x1 = 0
    while x1 < limit_x and _col_mean(x1) < black_threshold:
        x1 += 1

    x2 = w
    while x2 > w - limit_x and _col_mean(x2 - 1) < black_threshold:
        x2 -= 1

    return x1, y1, x2, y2


def _classify_aspect(width: int, height: int) -> str:
    """Return '16:9', '4:3', or 'unknown' for a given content size."""
    r = width / height
    if abs(r - 16 / 9) < 0.06:
        return "16:9"
    if abs(r - 4 / 3) < 0.06:
        return "4:3"
    return "unknown"


class FrameAdapter:
    """
    One-time calibration of letterbox bounds and aspect ratio.

    Call setup() once on a representative game frame (e.g. the first frame
    captured from the Discord stream).  After that, pass adapter.crop(frame)
    to all extractor functions instead of the raw captured frame.

    Example
    -------
    adapter = FrameAdapter()
    adapter.setup(first_raw_frame)

    for raw in stream:
        state = extract_live_state(adapter.crop(raw))
    """

    def __init__(self) -> None:
        self._rect: tuple[int, int, int, int] | None = None
        self._aspect: str = "unknown"

    def setup(self, frame: np.ndarray) -> None:
        """
        Detect letterbox bars and aspect ratio from a reference frame.
        Safe to call again if the stream resolution changes.
        """
        self._rect = detect_letterbox(frame)
        x1, y1, x2, y2 = self._rect
        self._aspect = _classify_aspect(x2 - x1, y2 - y1)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """
        Strip letterbox bars and return the active game content region.
        Falls back to the full frame if setup() has not been called.
        """
        if self._rect is None:
            return frame
        x1, y1, x2, y2 = self._rect
        return frame[y1:y2, x1:x2]

    @property
    def aspect(self) -> str:
        """Detected aspect ratio: '16:9', '4:3', or 'unknown'."""
        return self._aspect

    @property
    def game_rect(self) -> tuple[int, int, int, int] | None:
        """(x1, y1, x2, y2) of game content in the raw captured frame."""
        return self._rect

    def info(self) -> dict:
        """Return a summary dict useful for logging / debugging."""
        rect = self._rect
        if rect:
            x1, y1, x2, y2 = rect
            content_size = f"{x2 - x1}×{y2 - y1}"
            bars = f"L={x1} R={rect[2]-x2 if False else 0} T={y1}"  # simplified
        else:
            content_size = "uncalibrated"
            bars = "—"
        return {
            "aspect":       self._aspect,
            "game_rect":    self._rect,
            "content_size": content_size,
        }


# ---------------------------------------------------------------------------
# ROI definitions
# ---------------------------------------------------------------------------
# Each profile maps field name → (y1_frac, y2_frac, x1_frac, x2_frac).
# All fractions are relative to the *game content* region (after letterbox
# stripping), not the raw captured frame.
#
# '16:9' profile: calibrated at 1920×1080.
# '4:3'  profile: initialized to the same values as 16:9.
#                 Run debug_live_state() on a 4:3 capture and adjust as needed.

_ROI_PROFILES: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "16:9": {
        "phase_strip":  (0.000, 0.140, 0.250, 0.750),
        # Wide strip: timer + both alive count digits (psm-11 OCR)
        "top_center":   (0.002, 0.055, 0.435, 0.565),
        # Single-digit crops for alive counts (widened to avoid clipping digits)
        "alive_left":   (0.000, 0.040, 0.425, 0.460),
        "alive_right":  (0.000, 0.040, 0.543, 0.578),
        # Slightly wider x and deeper y to avoid dropping leading/trailing digits
        "money":        (0.944, 0.993, 0.001, 0.155),
        "hp":           (0.915, 0.972, 0.280, 0.370),
        "side_icon":    (0.912, 0.980, 0.462, 0.540),
        # Armor: icon + value between money and HP
        "armor_value":  (0.950, 0.982, 0.240, 0.290),
        # Armor icon (for helmet detection): full icon bounding box
        "armor_icon":   (0.940, 0.985, 0.230, 0.300),
        # Kit icon (CT only): scissors icon right of ammo display
        "kit_icon":     (0.940, 0.985, 0.780, 0.860),
    },
    # ── 4:3 profile ────────────────────────────────────────────────────────
    # Starting point: same as 16:9.  Verify with debug_live_state() on a
    # real 4:3 capture and tune the values that land in the wrong spot.
    "4:3": {
        "phase_strip":  (0.000, 0.140, 0.250, 0.750),
        "top_center":   (0.002, 0.055, 0.435, 0.565),
        "alive_left":   (0.000, 0.040, 0.425, 0.460),
        "alive_right":  (0.000, 0.040, 0.543, 0.578),
        "money":        (0.944, 0.993, 0.001, 0.155),
        "hp":           (0.915, 0.972, 0.280, 0.370),
        "side_icon":    (0.912, 0.980, 0.462, 0.540),
        "armor_value":  (0.950, 0.982, 0.240, 0.290),
        "armor_icon":   (0.940, 0.985, 0.230, 0.300),
        "kit_icon":     (0.940, 0.985, 0.780, 0.860),
    },
}

# Active profile — updated automatically by _set_profile() or manually.
_active_profile: str = "16:9"


def _set_profile(aspect: str) -> None:
    """Switch the active ROI profile.  Called by FrameAdapter implicitly."""
    global _active_profile
    if aspect in _ROI_PROFILES:
        _active_profile = aspect


def _crop(frame: np.ndarray, key: str) -> np.ndarray:
    """Crop a named ROI from a (letterbox-stripped) game frame."""
    y1f, y2f, x1f, x2f = _ROI_PROFILES[_active_profile][key]
    h, w = frame.shape[:2]
    return frame[int(h * y1f):int(h * y2f), int(w * x1f):int(w * x2f)]


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

# Calibrated edge-density threshold separating freeze from live/warmup:
#   freeze (10 player cards)  → density ≈ 0.12
#   warmup / live             → density ≤ 0.056
_FREEZE_EDGE_THRESHOLD = 0.06


def detect_phase(frame: np.ndarray) -> str:
    """
    Detect buy/freeze phase vs live round.

    Returns "freeze" or "live".
    """
    roi = _crop(frame, "phase_strip")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    density = np.count_nonzero(edges) / edges.size
    return "freeze" if density > _FREEZE_EDGE_THRESHOLD else "live"


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _preprocess_for_ocr(roi: np.ndarray, scale: int = 3,
                         thresh: int = 130) -> np.ndarray:
    """
    Upscale and threshold a HUD region for Tesseract.

    CS2 HUD text is bright (white / cyan) on a dark semi-transparent panel,
    so a brightness threshold isolates digits cleanly after upscaling.
    """
    h, w = roi.shape[:2]
    large = cv2.resize(roi, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
    _, out = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return out


def _ocr_raw(roi: np.ndarray, config: str) -> Optional[str]:
    """Run Tesseract and return stripped text, or None if unavailable."""
    if not _HAS_TESSERACT:
        return None
    return pytesseract.image_to_string(
        _preprocess_for_ocr(roi), config=config
    ).strip() or None


# ---------------------------------------------------------------------------
# Top-center strip (timer + alive counts)
# ---------------------------------------------------------------------------

_MAX_ROUND_TIME = 120   # seconds; CS2 competitive default is 115 (1:55)


def _ocr_single_digit(roi: np.ndarray) -> Optional[int]:
    """
    OCR a small region that should contain a single digit (0–5).

    Uses 5× upscale (instead of the default 3×) to give Tesseract more
    pixels to work with on the tiny alive-count ROIs, and tries psm 10
    (single char), psm 13 (sparse text), then psm 8 (single word).
    """
    h, w = roi.shape[:2]
    large = cv2.resize(roi, (w * 5, h * 5), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

    for psm in (10, 13, 8):
        text = pytesseract.image_to_string(
            binary,
            config=f"--psm {psm} -c tessedit_char_whitelist=012345",
        ).strip() if _HAS_TESSERACT else ""
        if text:
            digits = re.sub(r"\D", "", text)
            if digits:
                val = int(digits[-1])
                if 0 <= val <= 5:
                    return val
    return None


def extract_top_center(
    frame: np.ndarray,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Read the timer and both alive counts from the top-center HUD strip.

    Strategy (two-pass)
    -------------------
    Pass 1 – wide strip (psm 11):
        Tries to parse "N | M:SS | N" in one OCR call.  Fast when it works.
    Pass 2 – tight single-digit ROIs (psm 10/8):
        Falls back to separate crops for each alive-count digit when the
        pipe-pattern is not found in pass 1.

    Timer sanity: values above _MAX_ROUND_TIME are rejected (OCR artefact
    from the freeze→live transition where the scoreboard timer is briefly
    in the same ROI).

    Returns (time_left, ct_alive, t_alive); any value may be None.
    """
    raw = _ocr_raw(_crop(frame, "top_center"), config="--psm 11")

    # --- timer (pass 1) ---
    time_left: Optional[int] = None
    if raw:
        m = re.search(r"(\d+):(\d{2})", raw)
        if m:
            val = int(m.group(1)) * 60 + int(m.group(2))
            time_left = val if val <= _MAX_ROUND_TIME else None

    # --- alive counts (pass 1: pipe pattern) ---
    ct_alive: Optional[int] = None
    t_alive:  Optional[int] = None
    if raw:
        m2 = re.search(r"(\d)\s*\|[^|]*\|\s*(\d)", raw)
        if m2:
            left, right = int(m2.group(1)), int(m2.group(2))
            ct_alive = left  if 0 <= left  <= 5 else None
            t_alive  = right if 0 <= right <= 5 else None

    # --- alive counts (pass 2: tight single-digit ROIs) ---
    if ct_alive is None:
        ct_alive = _ocr_single_digit(_crop(frame, "alive_left"))
    if t_alive is None:
        t_alive  = _ocr_single_digit(_crop(frame, "alive_right"))

    return time_left, ct_alive, t_alive


# ---------------------------------------------------------------------------
# Bottom HUD bar fields
# ---------------------------------------------------------------------------

def extract_hp(frame: np.ndarray) -> Optional[int]:
    """
    Read the player health number (1–100) from the bottom HUD bar.

    When armor is shown, CS2 renders "[armor 100] [health 100]" in the same
    area.  OCR may concatenate them (e.g. "3100").  We resolve this by
    checking validity and, if needed, reading only the last three digits.
    """
    raw = _ocr_raw(
        _crop(frame, "hp"),
        config="--psm 7 -c tessedit_char_whitelist=0123456789",
    )
    if raw is None:
        return None

    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None

    val = int(digits)
    if 1 <= val <= 100:
        return val

    # Value > 100: likely armor prepended to health.
    # Take the last 3 digits, then the last 2.
    for n in (3, 2):
        if len(digits) >= n:
            tail = int(digits[-n:])
            if 1 <= tail <= 100:
                return tail

    return None


def extract_money(frame: np.ndarray) -> Optional[int]:
    """
    Read the player money from the bottom-left HUD display.

    Handles "$800", "$10400", "$16000" formats.
    Returns the dollar amount as an integer (0–16000), or None.
    """
    raw = _ocr_raw(
        _crop(frame, "money"),
        config="--psm 7 -c tessedit_char_whitelist=0123456789$,",
    )
    if raw is None:
        return None

    digits = re.sub(r"\D", "", raw)
    if digits:
        val = int(digits)
        if 0 <= val <= 16000:
            return val
    return None


# ---------------------------------------------------------------------------
# Side detection + player alive detection
# ---------------------------------------------------------------------------
# All three functions below operate on the same bottom-center icon ROI.
#
# Three HUD states are distinguished:
#
#   alive    — T-star (gold) or CT logo (blue) is visible.
#              The icon has a BRIGHT CENTRE relative to its full area
#              because the star / logo occupies the middle.
#              center/full brightness ratio ≈ 1.37 (T) – 1.64 (CT)
#
#   observer — Player died and is now spectating a teammate.
#              A circular PORTRAIT replaces the icon.
#              The portrait has a bright outer gold ring but a darker face
#              in the centre, so center/full ratio ≈ 1.03 (near uniform).
#              Colored-pixel ratio is still high (≈ 0.91) because of the ring.
#
#   dead     — Icon has just disappeared; very few saturated pixels remain.
#              Colored-pixel ratio < _ICON_COLORED_THRESHOLD.
#
# Calibrated thresholds:
#   _ICON_COLORED_THRESHOLD  = 0.08   (below → dead)
#   _ICON_CENTER_RATIO_MIN   = 1.15   (above → icon with bright centre → alive)
#                                      (at or below → portrait → observer)

_ICON_COLORED_THRESHOLD = 0.08
_ICON_CENTER_RATIO_MIN  = 1.19


def detect_player_state(frame: np.ndarray) -> str:
    """
    Detect the local player's current HUD state.

    Returns
    -------
    "alive"    — T/CT icon visible; player is in control of their own body.
    "observer" — Portrait thumbnail visible; player died and is spectating.
    "dead"     — Icon absent; transition frame immediately after death.
    """
    roi  = _crop(frame, "side_icon")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Step 1: is there any coloured content in the ROI?
    mask  = (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 55)
    ratio = float(np.sum(mask)) / mask.size
    if ratio < _ICON_COLORED_THRESHOLD:
        return "dead"

    # Step 2: does the bright content have a brighter CENTRE than edges?
    # Icon (star / logo) → bright centre; portrait → relatively uniform.
    rh, rw = gray.shape
    cy1, cy2 = rh // 3, 2 * rh // 3
    cx1, cx2 = rw // 3, 2 * rw // 3
    center_mean = float(gray[cy1:cy2, cx1:cx2].mean())
    full_mean   = float(gray.mean())
    center_ratio = center_mean / full_mean if full_mean > 1e-6 else 0.0

    if center_ratio >= _ICON_CENTER_RATIO_MIN:
        return "alive"
    return "observer"


def detect_player_alive(frame: np.ndarray) -> bool:
    """
    Convenience wrapper: returns True only when detect_player_state() == 'alive'.
    Both 'dead' and 'observer' return False.
    """
    return detect_player_state(frame) == "alive"


def detect_side(frame: np.ndarray) -> str:
    """
    Detect whether the local player is on the T or CT side.

    The bottom-center icon is a gold/orange star on T-side and a blue/purple
    emblem on CT-side.  Classified by median hue of saturated pixels (HSV).

    Returns "T", "CT", or "unknown".
    "unknown" is returned when the icon is absent (player dead) or when the
    hue does not match a known side.  Use detect_player_alive() first if you
    need to distinguish the two cases.

    Calibrated hue values (OpenCV range 0–179):
      T  gold star  → median hue ≈ 14  → threshold [8, 45]
      CT blue logo  → median hue ≈ 105 → threshold [85, 135]
    """
    roi = _crop(frame, "side_icon")
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 55)
    if not np.any(mask):
        return "unknown"

    median_hue = float(np.median(hsv[:, :, 0][mask]))

    if 8  <= median_hue <= 45:
        return "T"
    if 85 <= median_hue <= 135:
        return "CT"
    return "unknown"


# ---------------------------------------------------------------------------
# Armor value extraction (OCR)
# ---------------------------------------------------------------------------

_ARMOR_ICON_EDGE_THRESHOLD = 0.05   # below this → no armor icon present


def extract_armor(frame: np.ndarray) -> Optional[int]:
    """
    Read the armor value from the HUD icon.

    Returns 0–100, or None if the armor icon is not visible (no armor).
    """
    icon_roi = _crop(frame, "armor_icon")
    gray_icon = cv2.cvtColor(icon_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_icon, 60, 150)
    if np.sum(edges > 0) / edges.size < _ARMOR_ICON_EDGE_THRESHOLD:
        return 0   # no armor icon present → armor = 0

    roi = _crop(frame, "armor_value")
    up = cv2.resize(roi, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(
        thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789"
    ).strip()

    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    val = int(digits[-3:]) if len(digits) > 3 else int(digits)
    if 0 <= val <= 100:
        return val
    return None


# ---------------------------------------------------------------------------
# Helmet detection (pixel analysis on armor icon)
# ---------------------------------------------------------------------------

_HELMET_CYAN_THRESHOLD = 0.11   # top-third cyan ratio above → helmet present


def detect_helmet(frame: np.ndarray) -> Optional[bool]:
    """
    Detect whether the player has a helmet.

    The helmet adds a dome shape to the top of the armor icon.  When present,
    the top third of the icon contains more cyan-colored pixels.

    Returns True (helmet), False (vest only), or None (no armor at all).
    """
    icon_roi = _crop(frame, "armor_icon")
    gray_icon = cv2.cvtColor(icon_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_icon, 60, 150)
    if np.sum(edges > 0) / edges.size < _ARMOR_ICON_EDGE_THRESHOLD:
        return None   # no armor icon → helmet question is moot

    icon_h = icon_roi.shape[0]
    top_third = icon_roi[:icon_h // 3, :]
    hsv = cv2.cvtColor(top_third, cv2.COLOR_BGR2HSV)
    cyan_mask = (
        (hsv[:, :, 0] > 70) & (hsv[:, :, 0] < 110)
        & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 80)
    )
    cyan_ratio = float(np.sum(cyan_mask)) / cyan_mask.size
    return cyan_ratio >= _HELMET_CYAN_THRESHOLD


# ---------------------------------------------------------------------------
# Kit detection (CT-side only)
# ---------------------------------------------------------------------------

_KIT_CYAN_THRESHOLD = 0.02   # above → kit icon present


def detect_kit(frame: np.ndarray) -> bool:
    """
    Detect whether the player has a defuse kit.

    The kit icon (cyan scissors) appears right of the ammo display.
    Detected via cyan pixel ratio to avoid false positives from game-world edges.
    Only relevant on CT side; T-side players never have a kit.
    """
    roi = _crop(frame, "kit_icon")
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    cyan_mask = (
        (hsv[:, :, 0] > 75) & (hsv[:, :, 0] < 105)
        & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 100)
    )
    return float(np.sum(cyan_mask)) / cyan_mask.size >= _KIT_CYAN_THRESHOLD


# ---------------------------------------------------------------------------
# Weapon classification (CNN-based)
# ---------------------------------------------------------------------------

_weapon_predict = None   # lazy-loaded from weapon_classifier module


def extract_weapon_class(frame: np.ndarray) -> Optional[str]:
    """
    Classify the player's primary weapon from the slot-1 HUD icon.

    Returns 'awp', 'rifle', 'smg', 'pistol', or None if the model
    is not available or confidence is too low.
    """
    global _weapon_predict
    if _weapon_predict is None:
        try:
            from weapon_classifier import predict as _wp
            _wp(frame)   # warm up / verify model loads
            _weapon_predict = _wp
        except Exception:
            _weapon_predict = lambda _: None
    return _weapon_predict(frame)


# ---------------------------------------------------------------------------
# Composite extractor
# ---------------------------------------------------------------------------

def extract_live_state(frame: np.ndarray) -> dict:
    """
    Extract all available live-phase HUD fields from a single frame.

    Returns
    -------
    dict with keys:
        phase, player_alive, side,
        time_left, teammates_alive, enemies_alive,
        hp, money

    Any field that cannot be read will be None.

    player_alive
        Detected by the presence of the bottom-center side icon.
        False means the player is dead / spectating this round.
        When False, hp is forced to 0 and money is still read
        (CS2 keeps the money display visible after death).

    Side mapping (when alive):
        T-side  → teammates = T alive,  enemies = CT alive
        CT-side → teammates = CT alive, enemies = T alive
    """
    phase        = detect_phase(frame)
    player_state = detect_player_state(frame)
    player_alive = (player_state == "alive")
    side         = detect_side(frame)

    _NULL_STATE: dict = {
        "phase": phase, "player_state": player_state,
        "player_alive": player_alive, "side": side,
        "time_left": None, "teammates_alive": None,
        "enemies_alive": None, "hp": None, "money": None,
        "armor": None, "helmet": None, "has_kit": None,
        "weapon_class": None,
    }

    if phase != "live":
        return _NULL_STATE

    # After death every number on screen belongs to the spectated player,
    # not to us.  Stop reading HUD entirely — return all fields as None.
    if not player_alive:
        return _NULL_STATE

    time_left, ct_alive, t_alive = extract_top_center(frame)

    if side == "T":
        teammates_alive, enemies_alive = t_alive, ct_alive
    elif side == "CT":
        teammates_alive, enemies_alive = ct_alive, t_alive
    else:
        teammates_alive, enemies_alive = None, None

    return {
        "phase":           phase,
        "player_state":    player_state,
        "player_alive":    player_alive,
        "side":            side,
        "time_left":       time_left,
        "teammates_alive": teammates_alive,
        "enemies_alive":   enemies_alive,
        "hp":              extract_hp(frame),
        "money":           extract_money(frame),
        "armor":           extract_armor(frame),
        "helmet":          detect_helmet(frame),
        "has_kit":         detect_kit(frame) if side == "CT" else False,
        "weapon_class":    extract_weapon_class(frame),
    }


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def debug_adapter(frame: np.ndarray,
                  save_path: Optional[str] = None) -> dict:
    """
    Run FrameAdapter on a raw captured frame and return diagnostics.
    Optionally saves an annotated image showing the detected game region.

    Use this to verify that letterbox detection is working correctly before
    feeding frames to the extractor.
    """
    adapter = FrameAdapter()
    adapter.setup(frame)
    info = adapter.info()

    if save_path and adapter.game_rect:
        x1, y1, x2, y2 = adapter.game_rect
        out = frame.copy()
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 3)
        label = f"game region  aspect={adapter.aspect}  {x2-x1}x{y2-y1}px"
        cv2.putText(out, label, (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(save_path, out)

    return info


def debug_phase(frame: np.ndarray,
                save_path: Optional[str] = None) -> dict:
    """
    Run phase detection and return diagnostics.
    Optionally saves an annotated image showing the ROI box.
    """
    y1f, y2f, x1f, x2f = _ROI_PROFILES[_active_profile]["phase_strip"]
    h, w = frame.shape[:2]
    y1, y2, x1, x2 = int(h*y1f), int(h*y2f), int(w*x1f), int(w*x2f)

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    density = np.count_nonzero(edges) / edges.size
    phase = "freeze" if density > _FREEZE_EDGE_THRESHOLD else "live"

    if save_path:
        out = frame.copy()
        color = (0, 255, 0) if phase == "freeze" else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{phase}  density={density:.4f}",
                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(save_path, out)

    return {"phase": phase, "edge_density": round(density, 4),
            "threshold": _FREEZE_EDGE_THRESHOLD, "roi_rect": (x1, y1, x2, y2)}


# Colour per ROI for the debug overlay
_DEBUG_COLORS: dict[str, tuple[int, int, int]] = {
    "top_center": (0, 255, 255),    # cyan
    "money":      (0, 200, 255),    # orange
    "hp":         (0, 255, 0),      # green
    "side_icon":  (255, 0, 255),    # magenta
}


def debug_live_state(frame: np.ndarray,
                     save_path: Optional[str] = None) -> dict:
    """
    Run full live-state extraction and return diagnostics.
    Optionally saves an annotated image with every ROI box labelled.
    """
    state = extract_live_state(frame)

    if save_path:
        h, w = frame.shape[:2]
        out = frame.copy()
        for key, color in _DEBUG_COLORS.items():
            y1f, y2f, x1f, x2f = _ROI_PROFILES[_active_profile][key]
            y1, y2 = int(h * y1f), int(h * y2f)
            x1, x2 = int(w * x1f), int(w * x2f)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, key, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.imwrite(save_path, out)

    return state


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    script_dir = Path(__file__).parent

    if not _HAS_TESSERACT:
        print("[WARNING] pytesseract not installed — OCR fields will be None.")
        print("  pip install pytesseract")
        print("  Tesseract: https://github.com/UB-Mannheim/tesseract/wiki\n")

    # ---- FrameAdapter / letterbox detection test -------------------------
    # Build a synthetic letterboxed frame: paste the live-game screenshot
    # into a wider 16:9 black canvas with 160px bars on each side.
    live_path = script_dir / "example_T_game.png"
    src = cv2.imread(str(live_path))
    if src is not None:
        print("FrameAdapter / letterbox detection")
        print("-" * 60)

        # No bars — adapter should return full-frame rect
        a = FrameAdapter()
        a.setup(src)
        h, w = src.shape[:2]
        ok_no_bars = (a.game_rect == (0, 0, w, h))
        print(f"  [{'PASS' if ok_no_bars else 'FAIL'}] no bars detected on plain frame"
              f"  rect={a.game_rect}  aspect={a.aspect}")

        # Synthetic left+right bars (160px each side)
        bar = 160
        canvas = np.zeros((h, w + bar * 2, 3), dtype=np.uint8)
        canvas[:, bar:bar + w] = src
        a2 = FrameAdapter()
        a2.setup(canvas)
        expected_rect = (bar, 0, bar + w, h)
        ok_bars = (a2.game_rect == expected_rect)
        print(f"  [{'PASS' if ok_bars else 'FAIL'}] side bars stripped correctly"
              f"  rect={a2.game_rect}  expected={expected_rect}")

        # Verify that adapter.crop() returns the original game frame
        cropped = a2.crop(canvas)
        ok_crop = (cropped.shape == src.shape)
        print(f"  [{'PASS' if ok_crop else 'FAIL'}] crop() returns correct shape"
              f"  got={cropped.shape}  expected={src.shape}")
        print()

        debug_adapter(
            canvas,
            save_path=str(script_dir / "debug_adapter_letterbox.png"),
        )

    # ---- phase detection -------------------------------------------------
    phase_cases = [
        ("example_T_pre-game.png", "freeze"),
        ("example_T_game.png",     "live"),
        ("example_CT.png",         "live"),
        ("example_CT_2.png",       "live"),
    ]
    print("Phase detection")
    print("-" * 60)
    for fname, expected in phase_cases:
        frame = cv2.imread(str(script_dir / fname))
        if frame is None:
            print(f"  [SKIP] {fname}")
            continue
        info = debug_phase(frame, save_path=str(script_dir / f"debug_{fname}"))
        ok = info["phase"] == expected
        print(f"  [{'PASS' if ok else 'FAIL'}] {fname:<28} "
              f"detected={info['phase']:<7} expected={expected:<7} "
              f"density={info['edge_density']:.4f}")
    print()

    # ---- live-state extraction -------------------------------------------
    # Expected values from example_T_game.png:
    #   side=T, time_left=114 (1:54), teammates_alive=5,
    #   enemies_alive=5, hp=100, money=800
    live_path = script_dir / "example_T_game.png"
    frame = cv2.imread(str(live_path))
    if frame is not None:
        print("Live-state extraction  (example_T_game.png)")
        print("-" * 60)
        state = debug_live_state(
            frame,
            save_path=str(script_dir / "debug_live_state.png"),
        )
        expected = {
            "phase":           "live",
            "player_state":    "alive",
            "player_alive":    True,
            "side":            "T",
            "time_left":       114,
            "teammates_alive": 5,
            "enemies_alive":   5,
            "hp":              100,
            "money":           800,
            "weapon_class":    "rifle",
            # example_T_game.png has no armor → armor=0, helmet=None
        }
        for field, exp_val in expected.items():
            got = state.get(field)
            ok  = (got == exp_val)
            note = f"  (expected {exp_val})" if not ok else ""
            print(f"  [{'PASS' if ok else 'FAIL'}] {field:<18} = {str(got):<8}{note}")
    # ---- player state detection test ------------------------------------
    if frame is not None:
        print("Player state detection  (alive / dead / observer)")
        print("-" * 60)

        # alive: real example_T_game.png
        state_alive = detect_player_state(frame)
        print(f"  [{'PASS' if state_alive == 'alive' else 'FAIL'}] T-game frame      → '{state_alive}'  (expected 'alive')")

        # dead: black out the icon region
        dead_frame = frame.copy()
        h2, w2 = dead_frame.shape[:2]
        y1f, y2f, x1f, x2f = _ROI_PROFILES[_active_profile]["side_icon"]
        dead_frame[int(h2*y1f):int(h2*y2f), int(w2*x1f):int(w2*x2f)] = 0
        state_dead = detect_player_state(dead_frame)
        print(f"  [{'PASS' if state_dead == 'dead' else 'FAIL'}] icon erased       → '{state_dead}'  (expected 'dead')")

        # observer: the example_observer.png screenshot
        obs_path = script_dir / "example_observer.png"
        obs_frame = cv2.imread(str(obs_path))
        if obs_frame is not None:
            state_obs = detect_player_state(obs_frame)
            print(f"  [{'PASS' if state_obs == 'observer' else 'FAIL'}] observer frame    → '{state_obs}'  (expected 'observer')")

        # All game fields must be None when not alive
        dead_state = extract_live_state(dead_frame)
        obs_state  = extract_live_state(obs_frame) if obs_frame is not None else {}
        null_fields = ("time_left", "teammates_alive", "enemies_alive", "hp", "money", "armor", "helmet", "has_kit", "weapon_class")
        dead_all_none = all(dead_state.get(f) is None for f in null_fields)
        obs_all_none  = all(obs_state.get(f) is None for f in null_fields)
        print(f"  [{'PASS' if dead_all_none else 'FAIL'}] all None when dead → {{{', '.join(f'{f}={dead_state.get(f)}' for f in null_fields)}}}")
        print(f"  [{'PASS' if obs_all_none else 'FAIL'}] all None when obs  → {{{', '.join(f'{f}={obs_state.get(f)}' for f in null_fields)}}}")
    print()
    print("Debug images saved to src/hud/")


if __name__ == "__main__":
    _run_tests()
