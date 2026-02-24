"""
Counting signature computation from prosodic features.

KEEP — signal processing math. Computes weight ratios, category stats,
and intensity/pitch comparisons from per-word features.
"""

import numpy as np

from musical_perception.types import (
    WordFeatures,
    CategoryStats,
    CountingSignature,
    MarkerType,
)


def compute_category_stats(
    features: list[WordFeatures],
    marker_type: MarkerType,
    category_name: str,
) -> CategoryStats | None:
    """
    Compute aggregate statistics for a category of words.

    Args:
        features: All word features
        marker_type: Which marker type to filter for
        category_name: Human-readable name ("beat", "and", "ah")

    Returns:
        CategoryStats or None if no words of this type
    """
    matching = [f for f in features if f.marker_type == marker_type]

    if not matching:
        return None

    pitches = [f.pitch_hz for f in matching if not np.isnan(f.pitch_hz)]
    intensities = [f.intensity_db for f in matching if not np.isnan(f.intensity_db)]
    durations = [f.duration for f in matching]

    return CategoryStats(
        category=category_name,
        count=len(matching),
        avg_pitch_hz=float(np.mean(pitches)) if pitches else np.nan,
        avg_intensity_db=float(np.mean(intensities)) if intensities else np.nan,
        avg_duration=float(np.mean(durations)) if durations else np.nan,
        pitches=pitches,
        intensities=intensities,
        durations=durations,
    )


def compute_signature(features: list[WordFeatures]) -> CountingSignature:
    """
    Compute the full counting signature from word features.

    Args:
        features: List of WordFeatures extracted from audio

    Returns:
        CountingSignature with stats and weight ratios
    """
    beat_stats = compute_category_stats(features, MarkerType.BEAT, "beat")
    and_stats = compute_category_stats(features, MarkerType.AND, "and")
    ah_stats = compute_category_stats(features, MarkerType.AH, "ah")

    # Compute weight ratios
    beat_vs_and_intensity = None
    beat_vs_and_pitch = None
    beat_vs_ah_intensity = None
    beat_vs_ah_pitch = None

    if beat_stats and and_stats:
        if not np.isnan(beat_stats.avg_intensity_db) and not np.isnan(and_stats.avg_intensity_db):
            beat_vs_and_intensity = beat_stats.avg_intensity_db - and_stats.avg_intensity_db
        if not np.isnan(beat_stats.avg_pitch_hz) and not np.isnan(and_stats.avg_pitch_hz):
            beat_vs_and_pitch = beat_stats.avg_pitch_hz / and_stats.avg_pitch_hz

    if beat_stats and ah_stats:
        if not np.isnan(beat_stats.avg_intensity_db) and not np.isnan(ah_stats.avg_intensity_db):
            beat_vs_ah_intensity = beat_stats.avg_intensity_db - ah_stats.avg_intensity_db
        if not np.isnan(beat_stats.avg_pitch_hz) and not np.isnan(ah_stats.avg_pitch_hz):
            beat_vs_ah_pitch = beat_stats.avg_pitch_hz / ah_stats.avg_pitch_hz

    # Determine loudest category
    loudest = None
    candidates = []
    if beat_stats and not np.isnan(beat_stats.avg_intensity_db):
        candidates.append(("beat", beat_stats.avg_intensity_db))
    if and_stats and not np.isnan(and_stats.avg_intensity_db):
        candidates.append(("and", and_stats.avg_intensity_db))
    if ah_stats and not np.isnan(ah_stats.avg_intensity_db):
        candidates.append(("ah", ah_stats.avg_intensity_db))

    if candidates:
        loudest = max(candidates, key=lambda x: x[1])[0]

    # Determine weight placement
    weight_placement = None
    if loudest == "beat":
        weight_placement = "on_beat"
    elif loudest in ("and", "ah"):
        weight_placement = "after_beat"
    elif beat_vs_and_intensity is not None and abs(beat_vs_and_intensity) < 1.0:
        weight_placement = "even"

    return CountingSignature(
        words=features,
        beat_stats=beat_stats,
        and_stats=and_stats,
        ah_stats=ah_stats,
        beat_vs_and_intensity_db=beat_vs_and_intensity,
        beat_vs_and_pitch_ratio=beat_vs_and_pitch,
        beat_vs_ah_intensity_db=beat_vs_ah_intensity,
        beat_vs_ah_pitch_ratio=beat_vs_ah_pitch,
        loudest_category=loudest,
        weight_placement=weight_placement,
    )
