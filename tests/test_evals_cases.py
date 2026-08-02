"""Tests for the case loader and the trace record/replay round-trip."""

import json

import pytest

try:
    import yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from musical_perception.bundle import PerceptionBundle
from musical_perception.evals.traces import (
    make_recording_bundle,
    replay_bundle,
    slugify,
    write_meta,
)
from musical_perception.types import (
    GeminiAnalysisResult,
    GeminiWord,
    MarkerType,
    Meter,
    TimestampedWord,
)

yaml_only = pytest.mark.skipif(not HAS_YAML, reason="pyyaml not installed")


_CASE_YAML = """\
id: sample-case
input:
  trace: traces/sample/
  media: "video/sample clip.mov"
tags: {source: youtube, slot: frappe, count_style: numbers}
expect:
  slot: frappe
  marking_bpm: 160
  meter: "2/4"
  counts: 64
  sides: both
notes: test fixture
"""


@yaml_only
def test_load_case_maps_annotation_spellings(tmp_path):
    from musical_perception.evals.cases import load_cases

    (tmp_path / "sample.yaml").write_text(_CASE_YAML)
    (case,) = load_cases(tmp_path)
    assert case.id == "sample-case"
    assert case.expect["meter"] == Meter(2, 4)
    assert case.expect["sides"] == 2
    assert case.expected_bpm == 160.0
    assert case.tags["slot"] == "frappe"


@yaml_only
def test_load_case_rejects_unknown_expect_field(tmp_path):
    from musical_perception.evals.cases import load_cases

    (tmp_path / "bad.yaml").write_text(
        "id: bad\ninput: {trace: traces/x/}\nexpect: {bpm: 100}\n"
    )
    with pytest.raises(ValueError, match="unknown expect fields"):
        load_cases(tmp_path)


@yaml_only
def test_performance_bpm_wins_over_marking(tmp_path):
    from musical_perception.evals.cases import load_cases

    (tmp_path / "c.yaml").write_text(
        "id: c\ninput: {trace: traces/c/}\n"
        "expect: {marking_bpm: 100, performance_bpm: 120}\n"
    )
    (case,) = load_cases(tmp_path)
    assert case.expected_bpm == 120.0


def test_slugify():
    assert slugify("grande battement") == "grande-battement"
    assert slugify("Exercise 1 Demo") == "exercise-1-demo"


def test_record_then_replay_round_trip(tmp_path):
    """A recorded trace replays to identical words and parsed Gemini result."""
    words = [
        TimestampedWord("one", 0.0, 0.2),
        TimestampedWord("two", 0.6, 0.8),
    ]
    raw = {
        "words": [
            {"index": 0, "word": "one", "marker_type": "beat", "beat_number": 1},
            {"index": 1, "word": "two", "marker_type": "beat", "beat_number": 2},
        ],
        "exercise": {"exercise_type": "frappe", "display_name": "Frappé",
                     "confidence": 0.9, "reasoning": "test"},
        "counting_structure": {"total_counts": 2, "prep_counts": None,
                               "subdivision_type": "none", "estimated_bpm": None},
        "meter": {"beats_per_measure": 4, "beat_unit": 4},
        "quality": [],
        "structure": {"counts": 8, "sides": 1},
    }

    def fake_analyze_media(path, *, onset_bpm=None, transcript_words=None):
        from musical_perception.perception.gemini import parse_raw_response
        return parse_raw_response(raw, "fake-model")

    inner = PerceptionBundle(
        transcribe=lambda path: words,
        analyze_media=fake_analyze_media,
    )
    trace_dir = tmp_path / "trace"
    recording = make_recording_bundle(inner, trace_dir)
    got_words = recording.transcribe("clip.mov")
    got_result = recording.analyze_media(
        "clip.mov", onset_bpm=100.0, transcript_words=[w.word for w in got_words]
    )
    write_meta(trace_dir, "clip.mov", use_pose=False,
               whisper_model_name="test-model", gemini_model="fake-model")

    replayed, meta = replay_bundle(trace_dir)
    assert replayed.transcribe("anything") == words
    replay_result = replayed.analyze_media("anything", onset_bpm=100.0)
    assert isinstance(replay_result, GeminiAnalysisResult)
    assert replay_result.words == got_result.words
    assert replay_result.structure == got_result.structure
    assert replay_result.words[0] == GeminiWord("one", MarkerType.BEAT, 1, index=0)
    assert meta["analyze_flags"] == {"use_pose": False}
    assert meta["gemini"]["temperature"] == 0.0
    assert json.loads((trace_dir / "gemini.json").read_text())["inputs"][
        "onset_bpm_sent"] == 100.0


def test_replay_warns_on_onset_drift(tmp_path):
    """Recomputed onset differing from the frozen one warns, never fails."""
    import warnings as warnings_mod

    from musical_perception.perception.gemini import parse_raw_response

    words = [TimestampedWord("one", 0.0, 0.2)]
    minimal_raw = {"words": [], "exercise": {}, "counting_structure": {},
                   "meter": {}, "quality": [], "structure": {}}
    inner = PerceptionBundle(
        transcribe=lambda p: words,
        analyze_media=lambda p, *, onset_bpm=None, transcript_words=None:
            parse_raw_response(minimal_raw, "g"),
    )
    trace_dir = tmp_path / "t"
    rec = make_recording_bundle(inner, trace_dir)
    rec.transcribe("x")
    rec.analyze_media("x", onset_bpm=100.0, transcript_words=["one"])
    write_meta(trace_dir, "x", use_pose=False,
               whisper_model_name="m", gemini_model="g")

    replayed, _ = replay_bundle(trace_dir)
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        replayed.analyze_media("x", onset_bpm=131.0)
    assert any("onset_bpm" in str(w.message) for w in caught)
