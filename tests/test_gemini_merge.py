"""Tests for Gemini-Whisper merge logic. No API key needed."""

from musical_perception.analyze import _merge_gemini_with_timestamps
from musical_perception.types import (
    GeminiAnalysisResult,
    GeminiWord,
    MarkerType,
    Meter,
    PhraseStructure,
    QualityProfile,
    TimestampedWord,
)


def _make_result(words):
    """Helper to build a GeminiAnalysisResult from a word list."""
    return GeminiAnalysisResult(
        words=words,
        exercise=None,
        counting_structure=None,
        meter=None,
        quality=None,
        structure=None,
        model="test",
    )


def test_basic_merge():
    """Matching words get Gemini classification + Whisper timestamp."""
    gemini = _make_result([
        GeminiWord("one", MarkerType.BEAT, 1),
        GeminiWord("and", MarkerType.AND, 1),
        GeminiWord("two", MarkerType.BEAT, 2),
    ])
    whisper = [
        TimestampedWord("one", 0.0, 0.4),
        TimestampedWord("and", 0.4, 0.8),
        TimestampedWord("two", 0.8, 1.2),
    ]
    markers = _merge_gemini_with_timestamps(gemini, whisper)
    assert len(markers) == 3
    assert markers[0].marker_type == MarkerType.BEAT
    assert markers[0].timestamp == 0.0
    assert markers[0].beat_number == 1
    assert markers[1].marker_type == MarkerType.AND
    assert markers[1].timestamp == 0.4
    assert markers[2].marker_type == MarkerType.BEAT
    assert markers[2].beat_number == 2


def test_merge_skips_non_markers():
    """Words classified as None by Gemini don't produce markers."""
    gemini = _make_result([
        GeminiWord("okay", None, None),
        GeminiWord("one", MarkerType.BEAT, 1),
    ])
    whisper = [
        TimestampedWord("okay", 0.0, 0.3),
        TimestampedWord("one", 0.5, 0.8),
    ]
    markers = _merge_gemini_with_timestamps(gemini, whisper)
    assert len(markers) == 1
    assert markers[0].raw_word == "one"
    assert markers[0].timestamp == 0.5


def test_merge_handles_whisper_missing_word():
    """If Whisper didn't transcribe a word Gemini found, skip it."""
    gemini = _make_result([
        GeminiWord("one", MarkerType.BEAT, 1),
        GeminiWord("a", MarkerType.AH, 1),
        GeminiWord("two", MarkerType.BEAT, 2),
    ])
    whisper = [
        TimestampedWord("one", 0.0, 0.4),
        TimestampedWord("two", 0.8, 1.2),
    ]
    markers = _merge_gemini_with_timestamps(gemini, whisper)
    assert len(markers) == 2
    assert markers[0].beat_number == 1
    assert markers[1].beat_number == 2


def test_merge_handles_extra_whisper_words():
    """Whisper words not in Gemini output are ignored."""
    gemini = _make_result([
        GeminiWord("one", MarkerType.BEAT, 1),
    ])
    whisper = [
        TimestampedWord("tell", 0.0, 0.2),
        TimestampedWord("me", 0.2, 0.3),
        TimestampedWord("when", 0.3, 0.4),
        TimestampedWord("one", 0.5, 0.8),
    ]
    markers = _merge_gemini_with_timestamps(gemini, whisper)
    assert len(markers) == 1
    assert markers[0].timestamp == 0.5


def test_merge_normalizes_case_and_punctuation():
    """Words are matched after normalization."""
    gemini = _make_result([
        GeminiWord("One", MarkerType.BEAT, 1),
        GeminiWord("two.", MarkerType.BEAT, 2),
    ])
    whisper = [
        TimestampedWord("one", 0.0, 0.4),
        TimestampedWord("Two", 0.8, 1.2),
    ]
    markers = _merge_gemini_with_timestamps(gemini, whisper)
    assert len(markers) == 2


def test_merge_empty_inputs():
    """Empty inputs produce empty output."""
    gemini = _make_result([])
    markers = _merge_gemini_with_timestamps(gemini, [])
    assert markers == []


def test_merge_preserves_raw_word_from_whisper():
    """The raw_word in TimedMarker comes from Whisper, not Gemini."""
    gemini = _make_result([
        GeminiWord("Six", MarkerType.BEAT, 6),
    ])
    whisper = [
        TimestampedWord("six", 1.0, 1.3),
    ]
    markers = _merge_gemini_with_timestamps(gemini, whisper)
    assert len(markers) == 1
    assert markers[0].raw_word == "six"


# === Tests for new typed fields ===


def test_quality_profile():
    """QualityProfile stores three numeric dimensions."""
    q = QualityProfile(
        articulation=0.7, weight=0.5, energy=0.4,
    )
    assert q.articulation == 0.7
    assert q.weight == 0.5
    assert q.energy == 0.4


def test_meter():
    """Meter stores beats_per_measure and beat_unit."""
    m = Meter(beats_per_measure=3, beat_unit=4)
    assert m.beats_per_measure == 3
    assert m.beat_unit == 4


def test_phrase_structure():
    """PhraseStructure stores counts and sides."""
    s = PhraseStructure(counts=32, sides=2)
    assert s.counts == 32
    assert s.sides == 2


def test_gemini_result_carries_typed_fields():
    """GeminiAnalysisResult carries Meter, QualityProfile, PhraseStructure."""
    result = GeminiAnalysisResult(
        words=[],
        exercise=None,
        counting_structure=None,
        meter=Meter(beats_per_measure=4, beat_unit=4),
        quality=QualityProfile(
            articulation=0.5, weight=0.5, energy=0.5,
        ),
        structure=PhraseStructure(counts=16, sides=1),
        model="test",
    )
    assert result.meter.beats_per_measure == 4
    assert result.quality.articulation == 0.5
    assert result.structure.counts == 16


def test_merge_with_typed_fields():
    """Merge still works when GeminiAnalysisResult has typed fields populated."""
    result = GeminiAnalysisResult(
        words=[
            GeminiWord("one", MarkerType.BEAT, 1),
            GeminiWord("and", MarkerType.AND, 1),
        ],
        exercise=None,
        counting_structure=None,
        meter=Meter(beats_per_measure=4, beat_unit=4),
        quality=QualityProfile(
            articulation=0.7, weight=0.5, energy=0.4,
        ),
        structure=PhraseStructure(counts=16, sides=2),
        model="test",
    )
    whisper = [
        TimestampedWord("one", 0.0, 0.4),
        TimestampedWord("and", 0.4, 0.8),
    ]
    markers = _merge_gemini_with_timestamps(result, whisper)
    assert len(markers) == 2
    assert markers[0].marker_type == MarkerType.BEAT
