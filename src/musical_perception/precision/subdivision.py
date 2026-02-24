"""
Subdivision classification from marker sequences.

KEEP — measures actual timing ratios between beat markers.
Pure functions: markers in, subdivision type out.
"""

import numpy as np

from musical_perception.types import TimedMarker, MarkerType, SubdivisionResult


def analyze_subdivisions(markers: list[TimedMarker]) -> SubdivisionResult:
    """
    Analyze the subdivision pattern from a list of markers.

    Determines whether the counting is in duple (1-and-2-and),
    triplet (1-and-ah-2-and-ah), or has no subdivisions.

    Args:
        markers: List of TimedMarker objects with beat associations

    Returns:
        SubdivisionResult with classification and confidence
    """
    if not markers:
        return SubdivisionResult(
            subdivision_type="none",
            confidence=1.0,
            subdivisions_per_beat=None,
            avg_ratios=[],
        )

    # Group markers by beat number
    beat_groups: dict[int, list[TimedMarker]] = {}
    for marker in markers:
        if marker.beat_number is not None:
            if marker.beat_number not in beat_groups:
                beat_groups[marker.beat_number] = []
            beat_groups[marker.beat_number].append(marker)

    if not beat_groups:
        return SubdivisionResult(
            subdivision_type="none",
            confidence=1.0,
            subdivisions_per_beat=None,
            avg_ratios=[],
        )

    # Count subdivisions per beat (excluding the beat marker itself)
    subdivision_counts = []
    for beat_num, group in beat_groups.items():
        sub_count = sum(1 for m in group if m.marker_type != MarkerType.BEAT)
        subdivision_counts.append(sub_count)

    if not subdivision_counts or all(c == 0 for c in subdivision_counts):
        return SubdivisionResult(
            subdivision_type="none",
            confidence=1.0,
            subdivisions_per_beat=None,
            avg_ratios=[],
        )

    avg_subdivisions = np.mean(subdivision_counts)
    std_subdivisions = np.std(subdivision_counts)

    # Classify based on average subdivision count
    if avg_subdivisions < 0.5:
        subdivision_type = "none"
        subdivisions_per_beat = None
    elif avg_subdivisions < 1.5:
        subdivision_type = "duple"
        subdivisions_per_beat = 2
    elif avg_subdivisions < 2.5:
        subdivision_type = "triplet"
        subdivisions_per_beat = 3
    else:
        subdivision_type = "unknown"
        subdivisions_per_beat = round(avg_subdivisions) + 1

    # Confidence based on consistency of subdivision counts
    if len(subdivision_counts) > 1 and avg_subdivisions > 0:
        cv = std_subdivisions / avg_subdivisions
        confidence = max(0.0, min(1.0, 1.0 - cv))
    else:
        confidence = 0.5

    return SubdivisionResult(
        subdivision_type=subdivision_type,
        confidence=round(confidence, 2),
        subdivisions_per_beat=subdivisions_per_beat,
        avg_ratios=[],
    )
