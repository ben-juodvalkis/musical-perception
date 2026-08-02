"""Tests for the counts estimator. Pure math, hardcoded data."""

from musical_perception.precision.structure import (
    CountsEstimate,
    _counting_cycle,
    _snap,
    estimate_counts,
)
from musical_perception.types import MarkerType, TimedMarker


def _beats(words, bpm=100.0, start=0.0):
    period = 60.0 / bpm
    return [
        TimedMarker(MarkerType.BEAT, i + 1, round(start + i * period, 3), w)
        for i, w in enumerate(words)
    ]


def _step_names(n, bpm=100.0):
    names = ["front", "side", "tombe", "coupé", "close", "brush", "back", "plié"]
    return _beats([names[i % 8] for i in range(n)], bpm=bpm)


# === snapping ===

def test_snap_prefers_smaller_on_tie():
    assert _snap(40) == 32   # equidistant to 32 and 48
    assert _snap(28) == 24   # equidistant to 24 and 32
    assert _snap(33) == 32
    assert _snap(60) == 64


# === counting cycles ===

def test_cycle_with_restarts():
    assert _counting_cycle([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]) == 8


def test_cycle_without_restart_uses_max():
    assert _counting_cycle([1, 2, 3, 4, 5, 6]) == 6


def test_cycle_inconsistent_maxima_abstains():
    assert _counting_cycle([1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2]) is None


def test_cycle_partial_last_cycle_ok():
    # 1..8, then a cut-off 1..3 — completed cycle still wins
    assert _counting_cycle([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3]) == 8


# === regime 1: numeric counting ===

def test_numeric_counting_two_rounds_of_eight():
    words = ["one", "two", "three", "four", "five", "six", "seven", "eight"] * 2
    est = estimate_counts(_beats(words), bpm=100.0)
    assert est.method == "counting"
    assert est.counts == 8


def test_quantity_numbers_do_not_trigger_counting_regime():
    """Sparse numbers among step names (quantities) stay in vote regime."""
    words = ["two", "battement", "front", "passé", "one", "more",
             "battement", "side", "brush", "close"]
    est = estimate_counts(_beats(words), bpm=100.0, gemini_counts=16,
                          gemini_total_counts=17)
    assert est.method != "counting"


# === regime 2: voting ===

def test_vote_commits_on_agreement():
    # 32 step-name markers at 104 → tally says 32; gemini fields say 32/34
    est = estimate_counts(
        _step_names(32, bpm=104), bpm=104.0,
        subdivision="none", gemini_counts=32, gemini_total_counts=34,
    )
    assert est.method == "vote"
    assert est.counts == 32


def test_vote_outvotes_a_bad_gemini_read():
    """The grande-battement 18-flip scenario: other evidence wins."""
    est = estimate_counts(
        _step_names(35, bpm=104), bpm=104.0,
        subdivision="none", gemini_counts=18, gemini_total_counts=34,
    )
    assert est.counts == 32  # tally(35)→32 + total(34)→32 beat gemini(18)→16


def test_vote_tie_abstains():
    # Construct a genuine 2-2 split: two signals at 16, two at 32
    est = estimate_counts(
        _step_names(16, bpm=100),           # tally 16
        bpm=100.0,                          # span ~9s → ~16
        gemini_bpm=200.0,                   # distinct hypothesis → span ~32
        subdivision="none",
        gemini_counts=32,                   # 32
        gemini_total_counts=None,
    )
    assert est.counts is None
    assert est.method == "abstain"


def test_single_signal_abstains():
    est = estimate_counts([], gemini_counts=32)
    assert est.counts is None


def test_no_evidence_abstains():
    est = estimate_counts([])
    assert est.counts is None
    assert isinstance(est, CountsEstimate)


def test_same_bpm_casts_one_span_vote():
    """Two agreeing tempo readings must not fake independent agreement."""
    markers = _step_names(16, bpm=100)
    est = estimate_counts(
        markers, bpm=100.0, gemini_bpm=101.0,  # within 5% — same hypothesis
        subdivision="none",
    )
    span_votes = [
        signals for signals in est.votes.values()
        if any(s.startswith("span") for s in signals)
    ]
    all_span = [s for sig in span_votes for s in sig if s.startswith("span")]
    assert all_span == ["span_x_bpm"]


def test_subdivision_scales_tally():
    # 20 beats marked duple → tally votes 40→32; agreeing gemini_counts=32 commits
    est = estimate_counts(
        _step_names(20, bpm=100), bpm=None,
        subdivision="duple", gemini_counts=32,
    )
    assert est.counts == 32
