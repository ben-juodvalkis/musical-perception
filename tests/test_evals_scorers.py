"""Tests for the eval scorer library. Pure — no models, no files."""

from musical_perception.evals.aggregate import (
    aggregate,
    expected_calibration_error,
    risk_coverage,
    spearman,
    summarize_field,
    wilson_interval,
)
from musical_perception.evals.scorers import (
    CaseResult,
    ScoreResult,
    canonical_slot,
    score_counts,
    score_meter_triple,
    score_quality,
    score_sides,
    score_slot,
    score_tempo,
)
from musical_perception.types import (
    ExerciseDetectionResult,
    Meter,
    NormalizedTempo,
    PhraseStructure,
    QualityProfile,
    TempoCandidate,
)


def _triple(bpm, beats=4, unit=4, subdivision="none", confidence=0.8):
    return NormalizedTempo(
        bpm=bpm, meter=Meter(beats, unit), subdivision=subdivision,
        confidence=confidence, raw_bpm=bpm, tempo_multiplier=1,
    )


def _exercise(name, confidence=0.9):
    return ExerciseDetectionResult(
        primary_exercise=name, display_name=name.title(),
        confidence=confidence, all_matches=[],
    )


# === tempo ===

def test_tempo_within_tolerance_is_correct():
    r = score_tempo(104.2, 104.0)
    assert r.outcome == "correct" and r.credit == 1.0


def test_tempo_abstains_on_none():
    r = score_tempo(None, 104.0)
    assert r.outcome == "abstained" and r.credit == 0.0


def test_tempo_double_is_metric_level_not_tempo_error():
    r = score_tempo(208.0, 104.0)
    assert r.outcome == "wrong"
    assert r.failure_mode == "metric_level_x2"


def test_tempo_half_and_third_metric_levels():
    assert score_tempo(52.0, 104.0).failure_mode == "metric_level_div2"
    assert score_tempo(312.0, 104.0).failure_mode == "metric_level_x3"
    assert score_tempo(40.0, 120.0).failure_mode == "metric_level_div3"


def test_tempo_generic_error():
    r = score_tempo(131.0, 104.0)
    assert r.outcome == "wrong" and r.failure_mode == "tempo_error"


# === truth in family (ADR-014, informational) ===

def _family(*bpms):
    return [
        TempoCandidate(bpm=b, meter=Meter(4, 4), subdivision="none",
                       multiplier=1, in_comfort_band=70 <= b <= 140)
        for b in bpms
    ]


def test_tempo_family_not_measured_without_alternates():
    assert score_tempo(208.0, 104.0).truth_in_family is None
    assert score_tempo(104.0, 104.0).truth_in_family is None


def test_tempo_truth_found_in_alternates():
    """Clip 12: primary 124.4 is wrong, but 62.2 is in the family."""
    r = score_tempo(124.4, 60.0, alternates=_family(62.2, 186.6, 31.1))
    assert r.outcome == "wrong" and r.credit == 0.0   # outcome unchanged
    assert r.truth_in_family is True


def test_tempo_truth_absent_from_family():
    r = score_tempo(124.4, 97.0, alternates=_family(62.2, 186.6, 31.1))
    assert r.outcome == "wrong"
    assert r.truth_in_family is False


def test_tempo_family_never_rescues_an_outcome():
    """A family containing the truth is reported, never scored as correct."""
    r = score_tempo(80.9, 160.0, alternates=_family(161.8, 323.6, 53.9))
    assert r.outcome == "wrong"
    assert r.failure_mode == "metric_level_div2"
    assert r.truth_in_family is True


def test_tempo_correct_answer_is_not_family_annotated():
    r = score_tempo(104.0, 104.0, alternates=_family(208.0, 52.0))
    assert r.outcome == "correct" and r.truth_in_family is None


# === meter triple ===

def test_triple_exact_match_correct():
    r = score_meter_triple(_triple(104), Meter(4, 4), 104, "none")
    assert r.outcome == "correct" and r.credit == 1.0
    assert r.confidence == 0.8


def test_triple_abstains_on_none():
    r = score_meter_triple(None, Meter(4, 4))
    assert r.outcome == "abstained"


def test_triple_equivalent_reading_partial_credit():
    """3/4 @120 none ≡ 4/4 @40 triplet — identical rhythmic surface."""
    predicted = _triple(40, beats=4, unit=4, subdivision="triplet")
    r = score_meter_triple(predicted, Meter(3, 4), 120, "none")
    assert r.outcome == "wrong" and r.credit == 0.5
    assert r.failure_mode == "equivalent_reading"


def test_triple_meter_wrong():
    r = score_meter_triple(_triple(104, beats=3), Meter(4, 4), 104, "none")
    assert r.failure_mode == "meter_wrong" and r.credit == 0.0


def test_triple_meter_only_expectation():
    """A case may pin meter only — bpm/subdivision unset are not scored."""
    r = score_meter_triple(_triple(999), Meter(4, 4))
    assert r.outcome == "correct"


# === counts / sides ===

def test_counts_exact():
    assert score_counts(PhraseStructure(32, 2), 32).outcome == "correct"
    assert score_counts(PhraseStructure(16, 2), 32).outcome == "wrong"
    assert score_counts(None, 32).outcome == "abstained"


def test_sides_exact():
    assert score_sides(PhraseStructure(32, 2), 2).outcome == "correct"
    assert score_sides(PhraseStructure(32, 1), 2).outcome == "wrong"


# === slot ===

def test_slot_alias_and_accent_folding():
    assert canonical_slot("Grande Battement") == "grand_battement"
    assert canonical_slot("plié") == "plie"
    assert score_slot(_exercise("grande_battement"), "grand_battement").outcome == "correct"


def test_slot_unknown_is_abstention():
    r = score_slot(_exercise("unknown"), "frappe")
    assert r.outcome == "abstained"


def test_slot_wrong():
    r = score_slot(_exercise("tendu"), "frappe")
    assert r.outcome == "wrong" and r.failure_mode == "slot_wrong"


# === quality ===

def test_quality_hit_and_miss():
    q = QualityProfile(articulation=0.25, weight=0.5, energy=0.9)
    rows = score_quality(q, {"articulation": 0.3, "energy": 0.5})
    by = {r.field: r for r in rows}
    assert by["quality_articulation"].outcome == "correct"
    assert by["quality_energy"].outcome == "wrong"
    assert "quality_weight" not in by  # unlabeled dimension not scored


def test_quality_abstains_on_none():
    rows = score_quality(None, {"articulation": 0.3})
    assert rows[0].outcome == "abstained"


# === aggregation ===

def test_wilson_interval_sane():
    lo, hi = wilson_interval(9, 10)
    assert 0.55 < lo < 0.9 < hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_summarize_field_abstention_not_wrong():
    rows = [
        ScoreResult("tempo", "correct", 1.0, 100, 100),
        ScoreResult("tempo", "abstained", 0.0, None, 100),
        ScoreResult("tempo", "wrong", 0.0, 200, 100, failure_mode="metric_level_x2"),
    ]
    s = summarize_field(rows)
    assert s["n"] == 3 and s["abstained"] == 1
    assert s["accuracy"] == 0.5  # of the two committed
    assert s["failure_modes"] == {"metric_level_x2": 1}
    assert s["truth_in_family"] is None and s["truth_in_family_n"] is None


def test_summarize_field_counts_family_recall_without_touching_accuracy():
    rows = [
        ScoreResult("tempo", "correct", 1.0, 100, 100),
        ScoreResult("tempo", "wrong", 0.0, 200, 100, truth_in_family=True),
        ScoreResult("tempo", "wrong", 0.0, 131, 100, truth_in_family=False),
    ]
    s = summarize_field(rows)
    assert s["truth_in_family"] == 1 and s["truth_in_family_n"] == 2
    assert s["accuracy"] == round(1 / 3, 3)  # unchanged by the family measure
    assert s["mean_credit"] == round(1 / 3, 3)


def test_ece_none_without_confidence():
    rows = [ScoreResult("counts", "correct", 1.0, 32, 32)]
    assert expected_calibration_error(rows) is None


def test_ece_perfect_calibration_low():
    rows = [
        ScoreResult("tempo", "correct", 1.0, 1, 1, confidence=0.9),
        ScoreResult("tempo", "correct", 1.0, 1, 1, confidence=0.9),
        ScoreResult("tempo", "wrong", 0.0, 2, 1, confidence=0.1),
    ]
    ece = expected_calibration_error(rows, bins=5)
    assert ece is not None and ece < 0.15


def test_risk_coverage_monotone_coverage():
    rows = [
        ScoreResult("tempo", "correct", 1.0, 1, 1, confidence=0.9),
        ScoreResult("tempo", "wrong", 0.0, 2, 1, confidence=0.5),
    ]
    curve = risk_coverage(rows)
    assert [p["coverage"] for p in curve] == [0.5, 1.0]
    assert curve[0]["risk"] == 0.0 and curve[1]["risk"] == 0.5


def test_spearman_ranks_with_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    assert spearman([1, 1, 1], [1, 2, 3]) is None  # zero variance
    assert spearman([1, 2], [1, 2]) is None  # too few


def test_aggregate_slices_by_tags():
    cases = [
        CaseResult("a", tags={"source": "youtube"},
                   scores=[ScoreResult("tempo", "correct", 1.0, 100, 100)]),
        CaseResult("b", tags={"source": "rig"},
                   scores=[ScoreResult("tempo", "wrong", 0.0, 50, 100,
                                       failure_mode="metric_level_div2")]),
    ]
    summary = aggregate(cases)
    assert summary["n_cases"] == 2
    assert summary["fields"]["tempo"]["correct"] == 1
    assert summary["slices"]["source"]["youtube"]["tempo"]["accuracy"] == 1.0
    assert summary["slices"]["source"]["rig"]["tempo"]["accuracy"] == 0.0
