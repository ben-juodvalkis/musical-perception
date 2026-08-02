"""Tier-0 gate: sweep-level recovery over the synthetic suite.

Gates assert sweep rates and pin the exact known-failure set — a new
failure trips them, and so does a silent fix (update the set consciously,
with the change that earned it).
"""

from musical_perception.evals.aggregate import aggregate
from musical_perception.evals.synthetic import SUITE, build_timeline, run_suite
from musical_perception.types import MarkerType, Meter

# KEEP-layer defects the suite exposed on day one (2026-08-01):
# - clean-triplet in 4/4: interpret_meter prefers /2 over /3 when both land
#   in the 70-140 band, even with gemini_subdivision="triplet" → 120 not 80.
# - half-tempo marking in 3/4: tempo recovers but the meter flips.
KNOWN_FAILING = {"t0-4-4-clean-triplet", "t0-3-4-half"}


def _failing_ids(results):
    bad = set()
    for r in results:
        for s in r.scores:
            if s.field == "tempo" and s.outcome != "correct":
                bad.add(r.case_id)
            if s.field == "meter_triple" and s.credit < 1.0:
                bad.add(r.case_id)
    return bad


def test_sweep_failure_set_is_exactly_the_known_one():
    assert _failing_ids(run_suite()) == KNOWN_FAILING


def test_sweep_rates_meet_floor():
    summary = aggregate(run_suite())
    assert summary["fields"]["tempo"]["accuracy"] >= 0.90
    assert summary["fields"]["meter_triple"]["mean_credit"] >= 0.85
    assert summary["fields"]["tempo"]["abstained"] == 0


def test_suite_is_deterministic():
    a, b = run_suite(), run_suite()
    for ra, rb in zip(a, b):
        assert [(s.outcome, s.predicted) for s in ra.scores] == \
               [(s.outcome, s.predicted) for s in rb.scores]


def test_suite_has_expected_shape():
    assert len(SUITE) == 24
    assert len({c.id for c in SUITE}) == 24
    corruptions = {c.tags["corruption"] for c in SUITE}
    assert {"clean", "jitter", "dropped", "interleaved",
            "prep", "half_tempo", "stress"} <= corruptions


# === build_timeline unit checks ===

def test_timeline_duple_inserts_and_between_beats():
    words, markers = build_timeline(Meter(4, 4), 120, "duple", 4)
    texts = [w.word for w in words]
    assert texts[:4] == ["one", "and", "two", "and"]
    assert sum(1 for m in markers if m.marker_type == MarkerType.AND) == 4


def test_timeline_words_are_lowercase():
    words, _ = build_timeline(Meter(4, 4), 104, "triplet", 8)
    assert all(w.word == w.word.lower() for w in words)


def test_timeline_explanation_words_are_not_markers():
    words, markers = build_timeline(
        Meter(4, 4), 104, "none", 8, interleaved_explanation=True
    )
    assert len(words) == 8 + 8  # 8 beats + 8 explanation words
    assert len(markers) == 8


def test_timeline_half_tempo_doubles_spacing():
    w_full, _ = build_timeline(Meter(4, 4), 104, "none", 8)
    w_half, _ = build_timeline(Meter(4, 4), 104, "none", 8, half_tempo_marking=True)
    gap_full = w_full[1].start - w_full[0].start
    gap_half = w_half[1].start - w_half[0].start
    assert abs(gap_half - 2 * gap_full) < 0.01


def test_timeline_drop_keeps_minimum_beats():
    _, markers = build_timeline(
        Meter(4, 4), 104, "none", 16, drop_rate=0.9, seed=3
    )
    beats = [m for m in markers if m.marker_type == MarkerType.BEAT]
    assert len(beats) >= 4


def test_timeline_prep_counts_precede_beat_one():
    words, markers = build_timeline(Meter(4, 4), 104, "none", 8, prep_counts=4)
    assert [w.word for w in words[:4]] == ["five", "six", "seven", "eight"]
    beat_one = next(m for m in markers if m.beat_number == 1)
    assert words[3].start < beat_one.timestamp
