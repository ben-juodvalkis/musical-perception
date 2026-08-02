"""Tests for the PerceptionBundle seam. No models, no network, no files."""

import numpy as np

from musical_perception.analyze import analyze
from musical_perception.bundle import PerceptionBundle
from musical_perception.types import (
    GeminiAnalysisResult,
    GeminiCountingStructure,
    GeminiWord,
    LandmarkTimeSeries,
    MarkerType,
    Meter,
    PhraseStructure,
    QualityProfile,
    TimestampedWord,
)


def _words_at(bpm: float, count: int) -> list[TimestampedWord]:
    """Counted beats at a steady BPM starting at t=0."""
    period = 60.0 / bpm
    names = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    return [
        TimestampedWord(names[i % 8], round(i * period, 3), round(i * period + 0.2, 3))
        for i in range(count)
    ]


def _gemini_for(words: list[TimestampedWord]) -> GeminiAnalysisResult:
    """Index-keyed classification marking every word a beat."""
    return GeminiAnalysisResult(
        words=[
            GeminiWord(w.word, MarkerType.BEAT, i + 1, index=i)
            for i, w in enumerate(words)
        ],
        exercise=None,
        counting_structure=GeminiCountingStructure(
            total_counts=len(words), prep_counts=None,
            subdivision_type="none", estimated_bpm=None,
        ),
        meter=Meter(beats_per_measure=4, beat_unit=4),
        quality=QualityProfile(articulation=0.5, weight=0.5, energy=0.5),
        structure=PhraseStructure(counts=16, sides=2),
        model="stub",
    )


def _stub_bundle(words, gemini_result, calls=None):
    def transcribe(path):
        if calls is not None:
            calls.append(("transcribe", path))
        return words

    def analyze_media(path, *, onset_bpm=None, transcript_words=None):
        if calls is not None:
            calls.append(("analyze_media", onset_bpm, tuple(transcript_words or ())))
        return gemini_result

    return PerceptionBundle(transcribe=transcribe, analyze_media=analyze_media)


def test_analyze_runs_offline_with_stub_bundle():
    """analyze() with a bundle touches no model wrappers and no files."""
    words = _words_at(bpm=100, count=16)
    result = analyze("fake.wav", bundle=_stub_bundle(words, _gemini_for(words)))
    assert result.normalized_tempo is not None
    assert abs(result.normalized_tempo.bpm - 100.0) < 2.0
    assert result.meter.beats_per_measure == 4
    # Counted "one..eight" twice: the counts estimator (ADR-012) reads the
    # numeric cycle of 8, overriding the stub's structure.counts=16.
    assert result.structure.counts == 8
    assert len(result.markers) == 16


def test_bundle_receives_transcript_and_onset_hint():
    """The bundle's analyze_media gets Whisper's words and the onset BPM."""
    words = _words_at(bpm=120, count=16)
    calls = []
    analyze("fake.wav", bundle=_stub_bundle(words, _gemini_for(words), calls))
    kinds = [c[0] for c in calls]
    assert kinds == ["transcribe", "analyze_media"]
    _, onset_bpm, transcript = calls[1]
    assert onset_bpm is not None and abs(onset_bpm - 120.0) < 5.0
    assert transcript == tuple(w.word for w in words)


def test_bundle_wins_over_model_kwargs():
    """Legacy model kwargs are ignored when a bundle is provided."""
    words = _words_at(bpm=100, count=8)
    result = analyze(
        "fake.wav",
        model_name="nonexistent-model",
        gemini_model="nonexistent-gemini",
        bundle=_stub_bundle(words, _gemini_for(words)),
    )
    assert result.normalized_tempo is not None  # would have raised if loaded


def test_pose_skipped_when_bundle_has_no_landmarks():
    """use_pose with a pose-less bundle warns and keeps Gemini quality."""
    import warnings

    words = _words_at(bpm=100, count=8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analyze(
            "fake.mov", use_pose=True,
            bundle=_stub_bundle(words, _gemini_for(words)),
        )
    assert any("pose" in str(w.message).lower() for w in caught)
    assert result.quality.articulation == 0.5  # untouched Gemini value


def test_pose_runs_through_bundle():
    """A bundle-provided extract_landmarks feeds the quality synthesis."""
    words = _words_at(bpm=100, count=8)
    n = 60
    series = LandmarkTimeSeries(
        timestamps=np.linspace(0, 2, n),
        landmarks=np.tile(np.linspace(0.4, 0.6, 33 * 3).reshape(1, 33, 3), (n, 1, 1)),
        fps=30.0,
        detection_rate=1.0,
    )
    bundle = _stub_bundle(words, _gemini_for(words))
    bundle.extract_landmarks = lambda path: series
    result = analyze("fake.mov", use_pose=True, bundle=bundle)
    # Static pose → near-zero movement energy pulls the synthesized value
    # below the pure-Gemini 0.5 baseline (0.7 gemini + 0.3 pose).
    assert result.quality is not None
    assert result.quality.energy < 0.5
