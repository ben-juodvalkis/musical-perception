"""Tests for onset-based rhythmic section detection."""

from musical_perception.types import TimestampedWord
from musical_perception.precision.rhythm import detect_onset_tempo


def _word(text, start, end):
    """Helper to create a TimestampedWord."""
    return TimestampedWord(word=text, start=start, end=end)


# --- Test 1: Regular counting at 120 BPM ---

def test_regular_counting_120bpm():
    """8 evenly-spaced words at 0.5s intervals = 120 BPM."""
    words = [
        _word("one", 0.0, 0.2),
        _word("two", 0.5, 0.7),
        _word("three", 1.0, 1.2),
        _word("four", 1.5, 1.7),
        _word("five", 2.0, 2.2),
        _word("six", 2.5, 2.7),
        _word("seven", 3.0, 3.2),
        _word("eight", 3.5, 3.7),
    ]
    result = detect_onset_tempo(words)
    assert result is not None
    assert abs(result.bpm - 120.0) < 5.0
    assert result.confidence > 0.5
    assert len(result.rhythmic_sections) >= 1


# --- Test 2: Step names at ~80 BPM ---

def test_step_names_80bpm():
    """Ballet step names spoken rhythmically at ~80 BPM (0.75s intervals)."""
    interval = 60.0 / 80  # 0.75s
    step_names = ["tendu", "front", "brush", "through",
                  "tendu", "side", "close", "fifth"]
    words = [
        _word(name, i * interval, i * interval + 0.3)
        for i, name in enumerate(step_names)
    ]
    result = detect_onset_tempo(words)
    assert result is not None
    assert abs(result.bpm - 80.0) < 5.0
    assert result.confidence > 0.4


# --- Test 3: Explanation then counting ---

def test_explanation_then_counting():
    """Non-rhythmic speech followed by rhythmic counting."""
    # Irregular explanation (0.0-4.0s)
    explanation = [
        _word("we're", 0.0, 0.2),
        _word("going", 0.25, 0.5),
        _word("to", 0.55, 0.65),
        _word("do", 0.9, 1.1),
        _word("a", 1.8, 1.9),
        _word("tendu", 2.0, 2.5),
        _word("exercise", 3.0, 3.6),
    ]
    # Regular counting (5.0-8.5s, 120 BPM)
    counting = [
        _word("one", 5.0, 5.2),
        _word("two", 5.5, 5.7),
        _word("three", 6.0, 6.2),
        _word("four", 6.5, 6.7),
        _word("five", 7.0, 7.2),
        _word("six", 7.5, 7.7),
        _word("seven", 8.0, 8.2),
        _word("eight", 8.5, 8.7),
    ]
    words = explanation + counting
    result = detect_onset_tempo(words)
    assert result is not None
    assert abs(result.bpm - 120.0) < 10.0
    # Should detect the rhythmic section, not the explanation
    assert result.rhythmic_coverage < 1.0
    assert len(result.rhythmic_sections) >= 1


# --- Test 4: Two rhythmic phrases separated by a pause ---

def test_two_rhythmic_sections():
    """Two counting phrases separated by a long pause, same tempo."""
    phrase1 = [_word(str(i + 1), i * 0.5, i * 0.5 + 0.2) for i in range(8)]
    # 6-second gap (wider than the 3s window, so sections stay separate)
    phrase2 = [_word(str(i + 1), 10.0 + i * 0.5, 10.0 + i * 0.5 + 0.2) for i in range(8)]
    words = phrase1 + phrase2
    result = detect_onset_tempo(words)
    assert result is not None
    assert abs(result.bpm - 120.0) < 5.0
    assert len(result.rhythmic_sections) >= 2


# --- Test 5: Insufficient data ---

def test_insufficient_data():
    """Fewer than 3 words returns None."""
    assert detect_onset_tempo([]) is None
    assert detect_onset_tempo([_word("one", 0.0, 0.2)]) is None
    assert detect_onset_tempo([
        _word("one", 0.0, 0.2),
        _word("two", 0.5, 0.7),
    ]) is None


# --- Test 6: Completely irregular speech ---

def test_no_rhythmic_sections():
    """Irregular conversational speech produces None."""
    words = [
        _word("so", 0.0, 0.1),
        _word("today", 0.3, 0.6),
        _word("we", 1.5, 1.6),
        _word("are", 1.7, 1.9),
        _word("going", 3.0, 3.4),
        _word("to", 3.5, 3.6),
        _word("work", 5.0, 5.3),
        _word("on", 5.4, 5.5),
        _word("something", 7.0, 7.5),
    ]
    result = detect_onset_tempo(words)
    assert result is None


# --- Test 7: Outlier robustness ---

def test_outlier_word_in_rhythmic_section():
    """One slightly late word shouldn't break detection."""
    words = [
        _word("one", 0.0, 0.2),
        _word("two", 0.5, 0.7),
        _word("three", 1.0, 1.2),
        _word("four", 1.7, 1.9),  # Late (should be 1.5)
        _word("five", 2.0, 2.2),
        _word("six", 2.5, 2.7),
        _word("seven", 3.0, 3.2),
        _word("eight", 3.5, 3.7),
    ]
    result = detect_onset_tempo(words)
    assert result is not None
    assert abs(result.bpm - 120.0) < 15.0


# --- Test 8: Histogram cross-check populated ---

def test_histogram_populated():
    """IOI histogram peak should be present for regular counting."""
    words = [_word(str(i + 1), i * 0.5, i * 0.5 + 0.2) for i in range(16)]
    result = detect_onset_tempo(words)
    assert result is not None
    assert result.ioi_histogram_peak_bpm is not None
    assert abs(result.ioi_histogram_peak_bpm - result.bpm) < 20.0
