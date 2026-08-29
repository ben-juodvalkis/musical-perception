"""
Data types for musical perception.

This module defines all shared types. It has no dependencies beyond
the standard library and numpy.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


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
class Meter:
    """Time signature of the exercise."""
    beats_per_measure: int   # 2, 3, 4, or 6
    beat_unit: int           # 4 (quarter note) or 8 (eighth note)


@dataclass
class PhraseStructure:
    """Phrase structure of the exercise."""
    # Counts in one full phrase (16, 32). None = the counts estimator
    # abstained: the evidence disagreed, and a wrong phrase length is
    # worse than asking (see precision/structure.py).
    counts: int | None
    sides: int               # 1 (one-sided) or 2 (both sides)


@dataclass
class PhraseQuality:
    """Quality ratings for a single phrase/section of a combination."""
    phrase_number: int
    description: str
    articulation: float  # 0 = staccato (sharp, detached), 1 = legato (smooth, flowing)
    weight: float        # 0 = light (buoyant, airy), 1 = heavy (grounded, pressing)
    energy: float        # 0 = calm (controlled, gentle), 1 = energetic (active, explosive)
    primary: bool        # True = core movement, False = transition/preparation


@dataclass
class QualityProfile:
    """
    Three numeric dimensions describing movement character.

    Each dimension is a float from 0.0 to 1.0. These are generic musical
    terms — any music generator (pianist, DJ, generative AI) can interpret
    them. They describe movement quality, not instrument-specific decisions.

    When per-phrase data is available, these values are computed from
    primary phrases only (filtering out transitions and preparations).
    """
    articulation: float  # 0 = staccato (sharp, detached), 1 = legato (smooth, flowing)
    weight: float        # 0 = light (buoyant, airy), 1 = heavy (grounded, pressing)
    energy: float        # 0 = calm (controlled, gentle), 1 = energetic (active, explosive)
    phrases: list[PhraseQuality] = field(default_factory=list)  # Per-phrase breakdown


@dataclass
class LandmarkTimeSeries:
    """
    Pose landmark positions over time.

    Source-agnostic container — works with any pose model that produces
    33 keypoints (MediaPipe indexing). The precision layer consumes this
    type without knowing which model produced it.
    """
    timestamps: np.ndarray   # (N,) seconds from video start
    landmarks: np.ndarray    # (N, 33, 3) x/y/z normalized coords per frame
    fps: float               # Source video frame rate
    detection_rate: float    # Fraction of frames with successful pose detection


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
    # Position in the ASR transcript Gemini was given (index-keyed
    # classification). None when Gemini transcribed freely (legacy path).
    index: int | None = None


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
    meter: Meter | None
    quality: QualityProfile | None
    structure: PhraseStructure | None
    model: str  # Which model was used
    # Verbatim JSON the model returned — what trace recording freezes, so
    # replay exercises the current parser (ADR-009 tier 1).
    raw_response: dict | None = None


@dataclass
class RhythmicSection:
    """A section of regularly-spaced speech detected from word onset timing."""
    start: float              # Window start time in seconds
    end: float                # Window end time in seconds
    bpm: float                # Tempo estimated from mean IOI in this section
    mean_ioi: float           # Mean inter-onset interval in seconds
    cv: float                 # Coefficient of variation of IOIs (lower = more regular)
    word_count: int           # Number of word onsets in this section
    words: list[str] = field(default_factory=list)  # Actual words (for display/debug)


@dataclass
class TempoCandidate:
    """
    One member of the metric-level family for a raw tempo reading (ADR-014).

    A raw pulse of N BPM is equally consistent with N, N×2, N×3, N/2 and N/3
    at the beat level — from onset regularity alone the levels are
    indistinguishable. Each candidate carries the meter/subdivision its
    multiplier implies, using the same derivation table as the primary.
    """
    bpm: float              # Candidate beat-level BPM
    meter: Meter            # Implied by this candidate's multiplier
    subdivision: str        # "none", "duple", "triplet"
    multiplier: int         # Relation to raw_bpm — same encoding as NormalizedTempo
    in_comfort_band: bool   # True when bpm sits in the 70-140 ballet-class range
    # Posterior mass of this candidate's ±8% tempo neighborhood (ADR-017).
    # None on candidates produced by the legacy heuristic path.
    weight: float | None = None


@dataclass
class GroupingLevel:
    """
    One rung of the grouping ladder (ADR-017): evidence that beats bundle
    in groups of `level` above the pulse. The bar is one rung, the count
    phrase another; a silent rung is simply absent from the list — that
    absence is the honest output for audio carrying no accent at that
    level.
    """
    level: int          # beats per group (2, 3, 4, 6, 8, ...)
    strength: float     # 0-1 evidence strength
    source: str         # "counting" | "gaps"


@dataclass
class NormalizedTempo:
    """
    Coherent metric interpretation: BPM, meter, and subdivision as one answer.

    The normalization step picks a metric level for the raw BPM and derives
    meter and subdivision that are consistent with that choice. This prevents
    contradictions like "4/4 triplet at 40 BPM" when the pulse is actually
    at 120 BPM in 3/4.
    """
    bpm: float                  # Beat-level BPM (soft-prior level selection
    #   since W9 — the answer may sit outside 70-140 when the measurement
    #   earns it)
    meter: Meter                # Label derived late — Gemini's claim is one
    #   vote, never a state variable (ADR-017)
    subdivision: str            # "none", "duple", "triplet"
    confidence: float           # 0-1; posterior-path answers report the
    #   posterior mass of the committed ±8% tempo neighborhood
    raw_bpm: float              # Original BPM before normalization
    tempo_multiplier: int       # Metric level of raw pulse:
    #   1 = raw was at beat level (bpm ≈ raw_bpm)
    #   2 = raw was at duple measure level (bpm ≈ raw_bpm × 2)
    #   3 = triple meter — either raw was tripled, OR cross-signal
    #       detection found onset/gemini ratio ≈ 3 (bpm ≈ raw_bpm)
    #  -2 = raw was at duple subdivision level (bpm ≈ raw_bpm / 2)
    #  -3 = raw was at triplet subdivision level (bpm ≈ raw_bpm / 3)
    # The other musically-sane readings of the same raw pulse (ADR-014).
    # Additive and informational: primary selection ignores this field.
    alternates: list[TempoCandidate] = field(default_factory=list)
    # The grouping ladder (ADR-017): per-level evidence above the beat.
    # Empty on legacy-path answers and on clips whose streams carry no
    # grouping evidence — a silent ladder is data, not a defect.
    grouping_levels: list[GroupingLevel] = field(default_factory=list)


@dataclass
class OnsetTempoResult:
    """
    Tempo estimated from word onset regularity, without word classification.

    Complementary to TempoResult (which requires Gemini classification).
    Works with step names, numbers, or any rhythmic speech.
    """
    bpm: float                # Estimated tempo from rhythmic sections
    confidence: float         # 0-1, based on consistency and coverage
    rhythmic_sections: list[RhythmicSection]  # Detected rhythmic windows
    total_duration: float     # Total audio duration (onset span) in seconds
    rhythmic_coverage: float  # Fraction of total duration covered by rhythmic sections
    ioi_histogram_peak_bpm: float | None = None  # Secondary estimate for cross-check


# === Trigger types (streaming mode) ===


class TriggerState(Enum):
    """State of the analysis trigger pipeline."""
    IDLE = "idle"            # Only wake word detector runs
    LISTENING = "listening"  # Wake word detected, Whisper running on buffered audio


@dataclass
class TriggerEvent:
    """Emitted when the trigger pipeline decides analysis is warranted."""
    audio_segment: bytes              # Raw audio to send to Gemini
    words: list[TimestampedWord]      # Already transcribed by Whisper
    onset_tempo: OnsetTempoResult     # Already computed from word onsets
    timestamp: float                  # Wall clock time of trigger


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
    onset_tempo: OnsetTempoResult | None = None
    normalized_tempo: NormalizedTempo | None = None  # Coherent BPM + meter + subdivision
    # Deprecated — use normalized_tempo instead:
    normalized_bpm: float | None = None
    tempo_multiplier: int | None = None
    subdivision: SubdivisionResult | None = None
    meter: Meter | None = None  # Always set when normalized_tempo is set (overrides Gemini)
    exercise: ExerciseDetectionResult | None = None
    quality: QualityProfile | None = None
    counting_signature: CountingSignature | None = None
    structure: PhraseStructure | None = None
    words: list[TimestampedWord] = field(default_factory=list)
    markers: list[TimedMarker] = field(default_factory=list)
    stress_labels: list[tuple[str, int]] | None = None  # (word, 0|1) from WhiStress
