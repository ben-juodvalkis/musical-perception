"""
Tempo calculation from beat timestamps.

KEEP — precision math that AI models won't replace.
Pure functions: timestamps in, BPM out. No I/O, no models.
"""

import numpy as np

from musical_perception.types import (
    Meter,
    NormalizedTempo,
    OnsetTempoResult,
    TempoResult,
)


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


def interpret_meter(
    onset_tempo: OnsetTempoResult | None,
    gemini_tempo: TempoResult | None,
    gemini_meter: Meter | None,
    gemini_subdivision: str | None,
) -> NormalizedTempo | None:
    """
    Produce a coherent metric interpretation from raw tempo signals.

    Picks the best raw BPM, normalizes it to 70-140, and derives meter
    and subdivision from how the BPM was scaled. The multiplier encodes
    the metric level of the raw pulse:

    - multiplier=1: raw was already at beat level → trust Gemini meter/subdivision
    - multiplier=2: raw was at measure level, doubled → 4/4, no subdivision
    - multiplier=3: raw was at measure level, tripled → 3/4, no subdivision
    - multiplier=-2: raw was at subdivision level, halved → duple subdivision
    - multiplier=-3: raw was at subdivision level, divided by 3 → triplet subdivision

    Args:
        onset_tempo: Classification-free tempo from word onsets.
        gemini_tempo: Tempo from Gemini-classified beat markers.
        gemini_meter: Gemini's meter guess (used as fallback).
        gemini_subdivision: Gemini's subdivision observation (used as fallback).

    Returns:
        NormalizedTempo with coherent BPM + meter + subdivision, or None
        if no usable tempo signal exists.
    """
    # Pick best raw BPM (same priority as before)
    raw_bpm = None
    confidence = 0.0
    if onset_tempo is not None and onset_tempo.confidence >= 0.3:
        raw_bpm = onset_tempo.bpm
        confidence = onset_tempo.confidence
    elif gemini_tempo is not None:
        raw_bpm = gemini_tempo.bpm
        confidence = gemini_tempo.confidence
    elif onset_tempo is not None:
        raw_bpm = onset_tempo.bpm
        confidence = onset_tempo.confidence

    if raw_bpm is None:
        return None

    normalized_bpm, multiplier = normalize_tempo(raw_bpm)

    if multiplier == 0:
        return None

    # Cross-signal check: when onset is in range but Gemini BPM is much
    # lower, the ratio tells us about meter. onset/gemini ≈ 3 → triple meter.
    if multiplier == 1 and onset_tempo is not None and gemini_tempo is not None:
        ratio = raw_bpm / gemini_tempo.bpm if gemini_tempo.bpm > 0 else 1.0
        if 2.5 <= ratio <= 3.5:
            # Onset at beat level, Gemini at measure level of triple meter.
            # Note: this overloads multiplier=3 — here raw_bpm was NOT tripled
            # (it was already in range), but we use 3 to signal triple meter.
            # raw_bpm is stored separately so consumers don't need to reverse it.
            multiplier = 3

    # Derive meter and subdivision from the multiplier
    if multiplier == 1:
        # BPM was already at beat level — trust Gemini's observations
        meter = gemini_meter or Meter(beats_per_measure=4, beat_unit=4)
        subdivision = gemini_subdivision or "none"
    elif multiplier == 2:
        # Raw was at measure level, doubled → duple meter, no subdivision
        meter = Meter(beats_per_measure=4, beat_unit=4)
        subdivision = "none"
    elif multiplier == 3:
        # Raw was at measure level, tripled → triple meter, no subdivision
        meter = Meter(beats_per_measure=3, beat_unit=4)
        subdivision = "none"
    elif multiplier == -2:
        # Raw was at subdivision level, halved → duple subdivision
        meter = gemini_meter or Meter(beats_per_measure=4, beat_unit=4)
        subdivision = "duple"
    elif multiplier == -3:
        # Raw was at subdivision level, divided by 3 → triplet subdivision
        meter = gemini_meter or Meter(beats_per_measure=4, beat_unit=4)
        subdivision = "triplet"
    else:
        # Unreachable: normalize_tempo() returns only {1,2,3,-2,-3,0}
        # and multiplier==0 triggers early return above.
        raise ValueError(f"unexpected multiplier {multiplier}")

    return NormalizedTempo(
        bpm=normalized_bpm,
        meter=meter,
        subdivision=subdivision,
        confidence=round(confidence, 2),
        raw_bpm=round(raw_bpm, 1),
        tempo_multiplier=multiplier,
    )
