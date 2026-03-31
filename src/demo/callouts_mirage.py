"""
callouts_mirage.py

Map XYZ game coordinates to named callout regions on de_mirage.

Coordinate system (CS2, verified via in-game screenshots):
  - X-axis: negative = CT side (left), positive = T side (right)
  - Y-axis: positive = B side (top of radar), negative = A side (bottom)
  - Z-axis: elevation (negative = ground level)

Regions are defined as axis-aligned bounding boxes [x_min, x_max, y_min, y_max].
When regions overlap, the first match in CALLOUTS wins (more specific regions
are listed before broader ones).

Usage
-----
    from callouts_mirage import get_callout, get_zone

    name = get_callout(-1969, 450)    # "B site"
    zone = get_zone(-1969, 450)       # "B"
    name = get_callout(-500, -2000)   # "A site"
    zone = get_zone(-500, -2000)      # "A"
"""

from __future__ import annotations
from typing import Optional

# ---------------------------------------------------------------------------
# Callout definitions: (x_min, x_max, y_min, y_max)
# Ordered from most specific to least specific.
# ---------------------------------------------------------------------------

CALLOUTS: list[tuple[str, tuple[float, float, float, float]]] = [
    # ── B site area (positive Y, upper portion of radar) ─────────────
    ("B site",          (-2600, -1600, 50, 650)),
    ("B apartments",    (-2200, -200, 650, 900)),
    ("B short",         (-1600, -900, 50, 500)),
    ("bench",           (-1700, -1300, -200, 50)),
    ("market",          (-2400, -1300, -750, 50)),
    ("connector",       (-1300, -700, -750, 100)),
    ("market door",     (-1300, -700, -900, -750)),

    # ── CT spawn + CT area ───────────────────────────────────────────
    ("CT spawn",        (-2100, -1200, -2300, -1600)),

    # ── A site area (negative Y, lower portion of radar) ─────────────
    ("A site",          (-800, -100, -2500, -1850)),
    ("palace",          (-150, 800, -2300, -1300)),
    ("A ramp",          (-800, -200, -1850, -1200)),
    ("jungle",          (-1200, -800, -2500, -1850)),
    ("stairs",          (-1200, -800, -1850, -1200)),
    ("A tetris",        (-1200, -800, -1200, -900)),

    # ── Mid ──────────────────────────────────────────────────────────
    ("window",          (-1610, -900, -1600, -700)),
    ("mid",             (-900, 300, -1250, -200)),
    ("top mid",         (200, 800, -700, 200)),
    ("underpass",       (-900, -200, -700, -200)),
    ("catwalk",         (-1100, -200, -200, 300)),

    # ── T side ───────────────────────────────────────────────────────
    ("T spawn",         (900, 1500, -600, 300)),
    ("T ramp",          (500, 900, -500, 300)),
    ("T apartments",    (300, 900, -1600, -500)),
]

# Coarse zones for POMDP / HMM (5 high-level regions)
ZONE_MAP: dict[str, str] = {
    "B site": "B",
    "B apartments": "B",
    "B short": "B",
    "bench": "B",
    "market": "B",
    "market door": "B",

    "A site": "A",
    "palace": "A",
    "A ramp": "A",
    "jungle": "A",
    "stairs": "A",
    "A tetris": "A",

    "mid": "MID",
    "window": "MID",
    "top mid": "MID",
    "underpass": "MID",
    "catwalk": "MID",
    "connector": "MID",

    "CT spawn": "CT_BASE",

    "T spawn": "T_BASE",
    "T ramp": "T_BASE",
    "T apartments": "T_BASE",
}

ZONES = ["A", "B", "MID", "CT_BASE", "T_BASE"]


def get_callout(x: float, y: float) -> str:
    """Return the named callout for game coordinates (x, y).

    When no bounding box matches, returns ``"unknown (x, y)"`` so the
    raw coordinates are preserved for training / downstream matching.
    """
    for name, (x_min, x_max, y_min, y_max) in CALLOUTS:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return name
    return f"unknown ({x:.0f}, {y:.0f})"


def get_zone(x: float, y: float) -> str:
    """Return the coarse zone (A, B, MID, CT_BASE, T_BASE) for (x, y).

    Unknown positions include raw coordinates for training use.
    """
    callout = get_callout(x, y)
    if callout.startswith("unknown"):
        return callout  # pass through with coordinates
    return ZONE_MAP.get(callout, f"unknown ({x:.0f}, {y:.0f})")


def get_callout_center(name: str) -> Optional[tuple[float, float]]:
    """Return the center (x, y) of a named callout, or None."""
    for n, (x_min, x_max, y_min, y_max) in CALLOUTS:
        if n == name:
            return ((x_min + x_max) / 2, (y_min + y_max) / 2)
    return None


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def coverage_report(positions: list[tuple[float, float]]) -> dict[str, int]:
    """Count how many positions map to each callout.  'unknown' = unmapped."""
    counts: dict[str, int] = {}
    for x, y in positions:
        c = get_callout(x, y)
        counts[c] = counts.get(c, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    # Quick validation: check known positions
    tests = [
        (1200, -100, "T spawn"),
        (-1700, -1900, "CT spawn"),
        (-1969, 450, "B site"),
        (-500, -2000, "A site"),
        (-200, -500, "mid"),
    ]
    for x, y, expected in tests:
        got = get_callout(x, y)
        ok = "OK" if got == expected else "MISMATCH"
        print(f"  ({x:6.0f}, {y:6.0f}) → {got:20s}  expected={expected:20s}  [{ok}]")

    # Validate against demo death positions
    try:
        from demoparser2 import DemoParser
        p = DemoParser("src/demo/260319mirage.dem")
        deaths = p.parse_event("player_death", player=["X", "Y"])
        positions = [(r["user_X"], r["user_Y"]) for _, r in deaths.iterrows()
                     if r["user_X"] is not None]
        report = coverage_report(positions)
        total = sum(report.values())
        unknown = report.get("unknown", 0)
        print(f"\nDeath coverage: {total - unknown}/{total} mapped "
              f"({100*(total-unknown)/total:.0f}%)")
        for name, count in report.items():
            print(f"  {name:20s}: {count}")
    except Exception as e:
        print(f"Could not validate against demo: {e}")
