"""Tests for word classification. No audio or models needed."""

from musical_perception.scaffolding.markers import classify_marker, extract_markers
from musical_perception.types import MarkerType, TimestampedWord


def test_beat_words():
    assert classify_marker("one") == MarkerType.BEAT
    assert classify_marker("two") == MarkerType.BEAT
    assert classify_marker("eight") == MarkerType.BEAT
    assert classify_marker("5") == MarkerType.BEAT


def test_and_words():
    assert classify_marker("and") == MarkerType.AND
    assert classify_marker("&") == MarkerType.AND
    assert classify_marker("an") == MarkerType.AND


def test_ah_words():
    assert classify_marker("ah") == MarkerType.AH
    assert classify_marker("the") == MarkerType.AH
    assert classify_marker("da") == MarkerType.AH
    assert classify_marker("ta") == MarkerType.AH


def test_unrecognized():
    assert classify_marker("hello") is None
    assert classify_marker("tendu") is None


def test_punctuation_stripped():
    assert classify_marker("one,") == MarkerType.BEAT
    assert classify_marker("and.") == MarkerType.AND


def test_extract_markers_beat_association():
    """Subdivisions should be associated with their preceding beat."""
    words = [
        TimestampedWord("one", 0.0, 0.4),
        TimestampedWord("and", 0.4, 0.8),
        TimestampedWord("two", 0.8, 1.2),
        TimestampedWord("and", 1.2, 1.6),
    ]
    markers = extract_markers(words)
    assert len(markers) == 4
    assert markers[0].beat_number == 1
    assert markers[1].beat_number == 1  # "and" belongs to beat 1
    assert markers[2].beat_number == 2
    assert markers[3].beat_number == 2  # "and" belongs to beat 2


def test_extract_markers_skips_non_markers():
    words = [
        TimestampedWord("okay", 0.0, 0.3),
        TimestampedWord("one", 0.5, 0.8),
        TimestampedWord("tendus", 1.0, 1.5),
        TimestampedWord("two", 1.5, 1.8),
    ]
    markers = extract_markers(words)
    assert len(markers) == 2
    assert markers[0].raw_word == "one"
    assert markers[1].raw_word == "two"
