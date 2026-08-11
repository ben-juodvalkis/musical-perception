"""
Beat-grid annotation tooling (agent-charter rung 1, EVAL-CHANGE).

Ground truth for stage-level pulse scoring is a per-clip *beat grid*:
beat times anchored at vowel onsets (P-centers), never at word starts
(Standing Lesson 1; review-1 §2.9). Grids are pre-annotated by the
peakRate tap-assist annotator and stay `provisional: true` until the
owner verifies them (rung 1.5) — provisional grids never gate anything.
"""

from musical_perception.annotation.grids import BeatGrid, load_grid, save_grid
from musical_perception.annotation.peakrate import PeakRateParams, peak_rate_events

__all__ = [
    "BeatGrid",
    "PeakRateParams",
    "load_grid",
    "peak_rate_events",
    "save_grid",
]
