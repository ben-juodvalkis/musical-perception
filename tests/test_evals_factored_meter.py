"""W12: the factored meter slice — derived truth, duple-family credit, and
the property the commission rests on: it gates NOTHING."""

from pathlib import Path

import pytest

from musical_perception.evals.aggregate import aggregate
from musical_perception.evals.runner import outcomes_map
from musical_perception.evals.scorers import (
    ABSTAINED,
    CORRECT,
    REPORTED_ONLY_FIELDS,
    WRONG,
    CaseResult,
    ScoreResult,
    factored_truth,
    score_meter_factored,
)
from musical_perception.types import GroupingLevel, Meter, NormalizedTempo

_REPO = Path(__file__).resolve().parent.parent


def _norm(bar, unit=4, subdivision="none", levels=None):
    return NormalizedTempo(
        bpm=100.0, meter=Meter(beats_per_measure=bar, beat_unit=unit),
        subdivision=subdivision, confidence=0.8, raw_bpm=100.0,
        tempo_multiplier=1, grouping_levels=levels or [],
    )


def _by_field(rows):
    return {r.field: r for r in rows}


# --- the pre-registered mapping table ------------------------------------

@pytest.mark.parametrize("bar,unit,sub,exp_div,exp_bar,exp_accept", [
    (2, 4, "none", "none", 2, (2, 4)),
    (4, 4, "duple", "duple", 4, (2, 4)),
    (3, 4, "triplet", "triplet", 3, (3,)),
    (6, 8, "none", "none", 6, (6,)),
])
def test_mapping_table(bar, unit, sub, exp_div, exp_bar, exp_accept):
    t = factored_truth(Meter(beats_per_measure=bar, beat_unit=unit), sub)
    assert (t["division"], t["bar"], t["accepted"]) == (exp_div, exp_bar, exp_accept)


def test_six_eight_division_truth_is_none_whatever_the_label_says():
    """Owner ruling R-6/8: the pulse IS the counted eighth, so nothing
    divides below it — even if the case file says `duple`."""
    t = factored_truth(Meter(beats_per_measure=6, beat_unit=8), "duple")
    assert t["division"] == "none"
    assert t["accent_rung"] == 3          # accent every 3, reported not scored
    # 6/4 is not 6/8 and must not inherit the override.
    assert factored_truth(Meter(beats_per_measure=6, beat_unit=4), "duple")["division"] == "duple"


# --- duple-family credit -------------------------------------------------

def test_duple_family_credit_accepts_either_duple_bar():
    for truth_bar in (2, 4):
        for predicted_bar in (2, 4):
            rows = _by_field(score_meter_factored(
                _norm(predicted_bar), Meter(beats_per_measure=truth_bar, beat_unit=4), "none"))
            assert rows["meter_grouping"].outcome == CORRECT


def test_triple_bar_gets_no_family_credit():
    rows = _by_field(score_meter_factored(
        _norm(4), Meter(beats_per_measure=3, beat_unit=4), "none"))
    assert rows["meter_grouping"].outcome == WRONG


def test_exact_bar_is_reported_even_when_family_credit_applies():
    rows = _by_field(score_meter_factored(
        _norm(4), Meter(beats_per_measure=2, beat_unit=4), "none"))
    assert rows["meter_grouping"].outcome == CORRECT
    assert "exact=n" in rows["meter_grouping"].detail


# --- absence, abstention, and the ladder ---------------------------------

def test_missing_subdivision_truth_produces_no_division_row():
    rows = _by_field(score_meter_factored(_norm(4), Meter(beats_per_measure=4, beat_unit=4), None))
    assert "meter_division" not in rows      # absence, not a zero
    assert "meter_grouping" in rows


def test_none_prediction_abstains_rather_than_failing():
    rows = _by_field(score_meter_factored(None, Meter(beats_per_measure=4, beat_unit=4), "none"))
    assert {r.outcome for r in rows.values()} == {ABSTAINED}


def test_ladder_is_reported_not_scored():
    """The ADR-017 ladder rides in `detail`; it never moves an outcome."""
    levels = [GroupingLevel(level=8, strength=1.0, source="counting")]
    with_ladder = _by_field(score_meter_factored(
        _norm(4, levels=levels), Meter(beats_per_measure=4, beat_unit=4), "none"))
    without = _by_field(score_meter_factored(
        _norm(4), Meter(beats_per_measure=4, beat_unit=4), "none"))
    assert with_ladder["meter_grouping"].outcome == without["meter_grouping"].outcome
    assert "ladder=8:1.00" in with_ladder["meter_grouping"].detail


# --- the load-bearing property: it gates nothing -------------------------

def _case(extra):
    return CaseResult(
        case_id="c1", tags={"count_style": "numbers"},
        scores=[ScoreResult("meter_triple", WRONG, 0.0, "x", "y", confidence=0.9)] + extra,
    )


def test_factored_rows_never_reach_outcomes():
    extra = score_meter_factored(_norm(4), Meter(beats_per_measure=4, beat_unit=4), "none")
    assert set(outcomes_map([_case(extra)])["c1"]) == {"meter_triple"}


def test_factored_rows_change_no_headline_number():
    """Adding the factored rows must leave `fields`, ECE, risk-coverage and
    the tag slices bit-for-bit alone — the whole basis for `gates nothing`."""
    extra = score_meter_factored(_norm(4), Meter(beats_per_measure=4, beat_unit=4), "none")
    without = aggregate([_case([])])
    with_ = aggregate([_case(extra)])
    for block in ("fields", "ece", "risk_coverage", "slices", "tempo_metrics"):
        assert with_[block] == without[block], block
    assert without["factored_meter"] is None
    assert set(with_["factored_meter"]) == set(REPORTED_ONLY_FIELDS)


def test_reported_only_fields_are_declared_in_one_place():
    assert REPORTED_ONLY_FIELDS == ("meter_division", "meter_grouping")
