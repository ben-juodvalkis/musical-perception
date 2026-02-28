"""
musical-perception: Extract structured musical parameters from audio.

Usage:
    from musical_perception import analyze
    result = analyze("path/to/audio.wav")
    print(result.tempo.bpm)

For precision math only (no model dependencies):
    from musical_perception.precision.tempo import calculate_tempo
    result = calculate_tempo([0.0, 0.5, 1.0, 1.5])
"""

__version__ = "0.1.0"

from musical_perception.types import (
    MusicalParameters,
    TempoResult,
    OnsetTempoResult,
    RhythmicSection,
    SubdivisionResult,
    CountingSignature,
    MarkerType,
    TimestampedWord,
    TimedMarker,
)
from musical_perception.analyze import analyze

__all__ = [
    "analyze",
    "MusicalParameters",
    "TempoResult",
    "OnsetTempoResult",
    "RhythmicSection",
    "SubdivisionResult",
    "CountingSignature",
    "MarkerType",
    "TimestampedWord",
    "TimedMarker",
]
