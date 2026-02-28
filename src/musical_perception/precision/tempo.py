"""
Tempo calculation from beat timestamps.

KEEP — precision math that AI models won't replace.
Pure functions: timestamps in, BPM out. No I/O, no models.
"""

import numpy as np

from musical_perception.types import TempoResult


def normalize_tempo(
    bpm: float,
    low: float = 70.0,
    high: float = 140.0,
) -> tuple[float, int]:
    """
    Snap a BPM value into the target range by multiplying or dividing by 2 or 3.

    Ballet class tempos almost always fall in the 70-140 BPM range at the beat
    level. Values outside this range usually indicate the detector locked onto
    a subdivision level (too fast) or a measure level (too slow).

    The multiplier tracks how the original pulse relates to the normalized beat:
    - multiplier=1: already at beat level
    - multiplier=2: original was at measure level (doubled to reach beat)
    - multiplier=3: original was at measure level in triple meter
    - multiplier=-2: original was at subdivision level (halved to reach beat)
    - multiplier=-3: original was at triplet subdivision level
    - multiplier=0: could not normalize (BPM too extreme for any ×2/×3 transform)

    Args:
        bpm: Raw BPM value to normalize.
        low: Lower bound of the target range (inclusive).
        high: Upper bound of the target range (inclusive).

    Returns:
        (normalized_bpm, multiplier) tuple. When multiplier=0, the raw BPM is
        returned unchanged — the caller should treat it as unreliable.
    """
    if low <= bpm <= high:
        return round(bpm, 1), 1

    # Try multiplying up (measure → beat)
    for factor in (2, 3):
        candidate = bpm * factor
        if low <= candidate <= high:
            return round(candidate, 1), factor

    # Try dividing down (subdivision → beat)
    for factor in (2, 3):
        candidate = bpm / factor
        if low <= candidate <= high:
            return round(candidate, 1), -factor

    # Nothing fits — BPM is too extreme to normalize
    return round(bpm, 1), 0


def calculate_tempo(timestamps: list[float]) -> TempoResult | None:
    """
    Calculate tempo from a list of beat timestamps.

    Uses median interval for robustness to outliers.
    Confidence is based on coefficient of variation (lower CV = higher confidence).

    Args:
        timestamps: Times (in seconds) when beats occurred

    Returns:
        TempoResult with BPM and confidence, or None if insufficient data
    """
    if len(timestamps) < 2:
        return None

    intervals = []
    for i in range(1, len(timestamps)):
        intervals.append(timestamps[i] - timestamps[i - 1])

    if not intervals:
        return None

    median_interval = np.median(intervals)
    bpm = 60.0 / median_interval

    # Confidence: lower standard deviation = higher confidence
    std_interval = np.std(intervals)
    cv = std_interval / median_interval if median_interval > 0 else 1.0
    confidence = max(0.0, 1.0 - cv)

    return TempoResult(
        bpm=round(bpm, 1),
        confidence=round(confidence, 2),
        beat_count=len(timestamps),
        intervals=intervals,
    )
