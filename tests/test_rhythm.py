"""Tests for onset-based rhythmic section detection."""

import numpy as np

from musical_perception.types import TimestampedWord
from musical_perception.precision.rhythm import _grid_period, detect_onset_tempo


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


# --- Test 9: Histogram handles identical IOIs ---

def test_histogram_identical_iois():
    """Perfectly regular input (all IOIs identical) should not crash."""
    # 10 words at exactly 0.5s intervals — all IOIs are 0.5
    words = [_word(str(i + 1), i * 0.5, i * 0.5 + 0.2) for i in range(10)]
    result = detect_onset_tempo(words)
    assert result is not None
    # Histogram should handle the degenerate case gracefully
    assert result.ioi_histogram_peak_bpm is not None
    assert abs(result.ioi_histogram_peak_bpm - 120.0) < 1.0


# --- Grid fitting (ADR-015) ---

def test_grid_period_ignores_an_agogic_gap():
    """A bar-boundary stretch must not drag the period — the clip-4 defect.

    Nine intervals on a 0.6s pulse plus four expressive gaps at ~0.9s: the
    mean reads 84.7 BPM, the pulse is 100.
    """
    iois = np.array([0.6, 0.6, 0.62, 0.9, 0.58, 0.6, 0.64, 0.9, 0.6, 0.62, 0.9, 0.6, 0.9])
    period, support, _ = _grid_period(iois)
    assert abs(60.0 / period - 100.0) < 5.0
    assert abs(60.0 / float(np.mean(iois)) - 100.0) > 10.0  # what the mean would say
    assert support < 1.0  # the gaps are unexplained, and the fit says so


def test_grid_period_folds_integer_multiples_of_a_sparse_pulse():
    """Words on some beats only: a 1x/2x mixture must recover the base."""
    iois = np.array([0.667, 0.667, 1.334, 0.667, 1.334, 0.667, 0.667, 1.334])
    period, support, _ = _grid_period(iois)
    assert abs(60.0 / period - 90.0) < 3.0
    assert support == 1.0  # every interval is a whole number of beats


def test_grid_period_leaves_a_clean_window_alone():
    """No gaps, no multiples: the grid must agree with the mean it replaces."""
    iois = np.array([0.58, 0.6, 0.62, 0.6, 0.59, 0.61])
    period, support, _ = _grid_period(iois)
    assert abs(period - float(np.mean(iois))) < 0.005
    assert support == 1.0


def test_grid_period_keeps_the_mean_below_the_identifiability_floor():
    """Two or three intervals cannot falsify a grid, so they keep the mean."""
    iois = np.array([0.5, 0.9, 0.55])
    period, support, _ = _grid_period(iois)
    assert period == float(np.mean(iois))
    assert support < 1.0


def test_agogic_gaps_do_not_drag_reported_tempo():
    """End to end: 100 BPM counting with a lengthened beat every fourth word."""
    onsets, t = [], 0.0
    for i in range(20):
        onsets.append(t)
        t += 0.9 if i % 4 == 3 else 0.6
    words = [_word(str(i), round(o, 3), round(o + 0.2, 3)) for i, o in enumerate(onsets)]
    result = detect_onset_tempo(words)
    assert result is not None
    assert abs(result.bpm - 100.0) < 6.0


def test_sparse_marking_recovers_the_beat_not_the_word_rate():
    """Step names on beats 1, 2 and 4 of each bar at 90 BPM."""
    period = 60.0 / 90
    onsets = [
        (bar * 4 + beat) * period
        for bar in range(8)
        for beat in (0, 1, 3)
    ]
    words = [_word("tendu", round(o, 3), round(o + 0.2, 3)) for o in onsets]
    result = detect_onset_tempo(words)
    assert result is not None
    assert abs(result.bpm - 90.0) < 5.0


def test_partly_explained_reading_is_less_confident_than_a_fully_explained_one():
    """Grid support, not regularity alone, separates these two (ADR-015).

    Both land on 100 BPM and both are internally tidy, but the second only
    gets there by discarding intervals that fit no whole number of beats.
    Confidence has to notice the difference; CV alone does not.
    """
    period = 60.0 / 100
    clean = [_word(str(i), round(i * period, 3), round(i * period + 0.2, 3))
             for i in range(24)]

    onsets, t = [], 0.0
    for i in range(24):
        onsets.append(t)
        t += 0.85 if i % 3 == 2 else period  # 0.85s fits neither 1 nor 2 beats
    ragged = [_word(str(i), round(o, 3), round(o + 0.2, 3)) for i, o in enumerate(onsets)]

    clean_result = detect_onset_tempo(clean)
    ragged_result = detect_onset_tempo(ragged)
    assert clean_result is not None and ragged_result is not None
    assert abs(ragged_result.bpm - 100.0) < 6.0
    assert clean_result.confidence > ragged_result.confidence


def test_perfectly_regular_iois_do_not_crash_histogram():
    """Float-hair IOI ranges (~1e-15) must not blow up np.histogram.

    Regression: synthetic perfectly-regular timelines produced an IOI range
    just above zero, slipping past the exact-equality guard and making
    np.histogram raise "Too many bins for data range".
    """
    period = 60.0 / 100.0  # 0.6s — accumulates float hair over multiples
    words = [
        _word(str(i), round(i * period, 3), round(i * period + 0.2, 3))
        for i in range(8)
    ]
    result = detect_onset_tempo(words)
    assert result is not None
    assert abs(result.bpm - 100.0) < 2.0
    assert result.ioi_histogram_peak_bpm is not None
    assert abs(result.ioi_histogram_peak_bpm - 100.0) < 2.0
