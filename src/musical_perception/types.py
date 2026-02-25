"""
Data types for musical perception.

This module defines all shared types. It has no dependencies beyond
the standard library and numpy.
"""

from dataclasses import dataclass, field
from enum import Enum


class MarkerType(Enum):
    """Type of rhythmic marker detected in speech."""
    BEAT = "beat"  # 1, 2, 3, etc.
    AND = "and"    # "and", "&" - half-beat in duple, 2nd of 3 in triplet
    AH = "ah"      # "ah", "a" - 3rd subdivision in triplet
    E = "e"        # "e" - for future 16th note support (1-e-and-a)


@dataclass
class TimestampedWord:
    """A word with its timing information from transcription."""
    word: str
    start: float
    end: float


@dataclass
class TimedMarker:
    """A rhythmic marker with timing and classification."""
    marker_type: MarkerType
    beat_number: int | None  # Which beat this belongs to (1-8), None if uncertain
    timestamp: float
    raw_word: str  # Original transcribed word


@dataclass
class TempoResult:
    """Result of tempo extraction."""
    bpm: float
    confidence: float  # 0-1, based on consistency of intervals
    beat_count: int    # number of beats detected
    intervals: list[float]  # raw intervals between beats


@dataclass
class SubdivisionResult:
    """Result of subdivision analysis."""
    subdivision_type: str  # "duple", "triplet", "none", "unknown"
    confidence: float  # 0-1
    subdivisions_per_beat: int | None  # 2, 3, or None if no subdivisions detected
    avg_ratios: list[float] = field(default_factory=list)


@dataclass
class WordFeatures:
    """Prosodic features extracted for a single word."""
    word: str
    start: float
    end: float
    marker_type: MarkerType | None
    pitch_hz: float  # Mean F0 in Hz (NaN if unvoiced)
    intensity_db: float  # Mean intensity in dB
    duration: float  # Duration in seconds


@dataclass
class CategoryStats:
    """Aggregate statistics for a category of words (beats, ands, ahs)."""
    category: str  # "beat", "and", "ah"
    count: int
    avg_pitch_hz: float
    avg_intensity_db: float
    avg_duration: float
    pitches: list[float] = field(default_factory=list)
    intensities: list[float] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)


@dataclass
class CountingSignature:
    """
    The complete prosodic signature extracted from a counting sample.

    Captures where emphasis falls — independent of subdivision type.
    """
    words: list[WordFeatures]
    beat_stats: CategoryStats | None
    and_stats: CategoryStats | None
    ah_stats: CategoryStats | None
    beat_vs_and_intensity_db: float | None  # +5 dB means beats louder
    beat_vs_and_pitch_ratio: float | None   # 0.85 means beats lower pitch
    beat_vs_ah_intensity_db: float | None
    beat_vs_ah_pitch_ratio: float | None
    loudest_category: str | None  # "beat", "and", or "ah"
    weight_placement: str | None  # "on_beat", "after_beat", "even"


@dataclass
class ExerciseMatch:
    """A detected exercise mention in the transcription."""
    exercise_type: str      # Canonical name (e.g., "plie")
    display_name: str       # Pretty name (e.g., "Plié")
    matched_text: str       # What was actually transcribed
    timestamp: float        # When it was spoken (seconds)
    confidence: float       # Match confidence (0-1)


@dataclass
class ExerciseDetectionResult:
    """Result of exercise detection analysis."""
    primary_exercise: str | None
    display_name: str | None
    confidence: float
    all_matches: list[ExerciseMatch]


# === Gemini bridge types (no timestamps) ===

@dataclass
class GeminiWord:
    """A word classified by Gemini, without timestamps."""
    word: str
    marker_type: MarkerType | None  # None for non-rhythmic words
    beat_number: int | None


@dataclass
class GeminiCountingStructure:
    """Gemini's qualitative observation of counting structure."""
    total_counts: int | None
    prep_counts: str | None  # e.g. "5, 6, 7, 8"
    subdivision_type: str | None  # "none", "duple", "triplet"
    estimated_bpm: float | None  # Unreliable — logged but not used


@dataclass
class GeminiAnalysisResult:
    """Raw result from Gemini API call. Bridge type before merging with timestamps."""
    words: list[GeminiWord]
    exercise: ExerciseDetectionResult | None
    counting_structure: GeminiCountingStructure | None
    model: str  # Which model was used


# === The stable interface ===

@dataclass
class MusicalParameters:
    """
    The contract between perception and downstream consumers.

    This is the stable interface. Everything upstream produces it.
    Everything downstream consumes it. The implementation of how
    these parameters are extracted will change; this schema should not.
    """
    tempo: TempoResult | None = None
    subdivision: SubdivisionResult | None = None
    meter: dict | None = None  # {beats_per_measure: int, beat_unit: int}
    exercise: ExerciseDetectionResult | None = None
    quality: dict | None = None  # {descriptors: list[str]}
    counting_signature: CountingSignature | None = None
    structure: dict | None = None  # {counts: int, sides: int}
    words: list[TimestampedWord] = field(default_factory=list)
    markers: list[TimedMarker] = field(default_factory=list)
    stress_labels: list[tuple[str, int]] | None = None  # (word, 0|1) from WhiStress
