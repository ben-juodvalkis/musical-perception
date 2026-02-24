"""Tests for exercise detection. No audio or models needed."""

from musical_perception.scaffolding.exercise import detect_exercise, find_exercise_matches
from musical_perception.types import TimestampedWord


def _words(text_timestamps):
    """Helper: list of (word, start, end) -> list of TimestampedWord."""
    return [TimestampedWord(w, s, e) for w, s, e in text_timestamps]


def test_detect_plie():
    words = _words([
        ("okay", 0.0, 0.3),
        ("plie", 0.5, 0.8),
        ("in", 1.0, 1.2),
        ("first", 1.2, 1.5),
    ])
    result = detect_exercise(words)
    assert result.primary_exercise == "plie"
    assert result.display_name == "Plié"
    assert result.confidence > 0.0


def test_detect_tendu():
    words = _words([
        ("tendu", 0.5, 0.8),
        ("to", 0.9, 1.0),
        ("the", 1.0, 1.1),
        ("front", 1.1, 1.4),
    ])
    result = detect_exercise(words)
    assert result.primary_exercise == "tendu"


def test_detect_grand_battement_multiword():
    """Multi-word exercise detection."""
    words = _words([
        ("grand", 0.5, 0.8),
        ("battement", 0.8, 1.2),
    ])
    result = detect_exercise(words)
    assert result.primary_exercise == "grand_battement"


def test_no_exercise_detected():
    words = _words([
        ("hello", 0.0, 0.3),
        ("everyone", 0.3, 0.6),
    ])
    result = detect_exercise(words)
    assert result.primary_exercise is None
    assert result.confidence == 0.0


def test_empty_words():
    result = detect_exercise([])
    assert result.primary_exercise is None


def test_search_window():
    """Exercise outside the search window should be ignored initially."""
    words = _words([
        ("hello", 0.0, 0.3),
        ("tendu", 10.0, 10.3),  # Outside default 5s window
    ])
    # Should still find it via fallback to full search
    result = detect_exercise(words, search_window_seconds=5.0)
    assert result.primary_exercise == "tendu"


def test_confidence_boost_for_repeated_mentions():
    words = _words([
        ("plie", 0.5, 0.8),
        ("and", 1.0, 1.2),
        ("plie", 1.5, 1.8),
    ])
    result = detect_exercise(words, search_window_seconds=None)
    single_words = _words([("plie", 0.5, 0.8)])
    single_result = detect_exercise(single_words, search_window_seconds=None)
    assert result.confidence >= single_result.confidence
