"""W1.5: `maturity` on cases, and the promise that provisional truth
never gates and never pools into a headline number.

The charter's whole ingestion carve-out rests on these tests: agents may
write NEW case files for new material, but only if the harness can be
trusted to keep agent-proposed labels out of the tier-1 gate and out of
every aggregate the owner reads as a result. Hardcoded data throughout —
no media, no models.
"""

import json
from pathlib import Path

import pytest

from musical_perception.evals.aggregate import aggregate
from musical_perception.evals.report import (
    render_html,
    render_markdown_baseline,
)
from musical_perception.evals.runner import (
    compare_outcomes,
    outcomes_map,
    provisional_ids,
)
from musical_perception.evals.scorers import CORRECT, WRONG, CaseResult, ScoreResult

try:
    import yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

needs_yaml = pytest.mark.skipif(not HAS_YAML, reason="needs pyyaml")

_REPO = Path(__file__).resolve().parent.parent


def _case_yaml(cid, *, maturity=None, accompanied=None) -> str:
    body = {
        "id": cid,
        "input": {"trace": f"traces/{cid}/"},
        "tags": {"count_style": "numbers"},
        "expect": {"marking_bpm": 104},
    }
    if maturity is not None:
        body["maturity"] = maturity
    if accompanied is not None:
        body["tags"]["accompanied"] = accompanied
    return yaml.safe_dump(body)


def _result(case_id, *, provisional, outcome=CORRECT, bpm=104.0):
    return CaseResult(
        case_id=case_id,
        tags={"count_style": "numbers"},
        provisional=provisional,
        scores=[ScoreResult(
            field="tempo", outcome=outcome,
            credit=1.0 if outcome == CORRECT else 0.0,
            predicted=bpm, expected=104.0, confidence=0.8,
        )],
    )


# --- the case-file key --------------------------------------------------

@needs_yaml
def test_maturity_defaults_to_verified(tmp_path):
    """Every case written before W1.5 keeps exactly the meaning it had."""
    from musical_perception.evals.cases import load_cases

    (tmp_path / "c.yaml").write_text(_case_yaml("c"))
    (case,) = load_cases(tmp_path)
    assert case.maturity == "verified"
    assert case.provisional is False


@needs_yaml
def test_maturity_provisional_parses(tmp_path):
    from musical_perception.evals.cases import load_cases

    (tmp_path / "c.yaml").write_text(_case_yaml("c", maturity="provisional"))
    (case,) = load_cases(tmp_path)
    assert case.maturity == "provisional"
    assert case.provisional is True


@needs_yaml
def test_unknown_maturity_is_an_error(tmp_path):
    """A typo must not silently become a gating row."""
    from musical_perception.evals.cases import load_cases

    (tmp_path / "c.yaml").write_text(_case_yaml("c", maturity="probational"))
    with pytest.raises(ValueError, match="maturity must be one of"):
        load_cases(tmp_path)


@needs_yaml
def test_every_committed_case_is_verified():
    """The corpus this baseline was blessed on is owner-verified, all 30 of
    it. If a provisional case ever lands here, this test says so loudly —
    that is a review event, not a detail."""
    from musical_perception.evals.cases import load_cases

    cases = load_cases(_REPO / "evals" / "cases")
    assert len(cases) == 30
    assert [c.id for c in cases if c.provisional] == []


# --- the tag vocabulary (owner ruling B5) -------------------------------

@needs_yaml
def test_accompaniment_only_is_a_first_class_condition(tmp_path):
    from musical_perception.evals.cases import load_cases

    (tmp_path / "c.yaml").write_text(
        _case_yaml("c", accompanied="accompaniment_only")
    )
    (case,) = load_cases(tmp_path)
    assert case.tags["accompanied"] == "accompaniment_only"
    assert case.accompaniment_only is True


@needs_yaml
def test_accompanied_booleans_still_mean_what_they_meant(tmp_path):
    from musical_perception.evals.cases import load_cases

    (tmp_path / "f.yaml").write_text(_case_yaml("f", accompanied=False))
    (tmp_path / "t.yaml").write_text(_case_yaml("t", accompanied=True))
    f, t = load_cases(tmp_path)
    assert (f.tags["accompanied"], f.accompaniment_only) == (False, False)
    assert (t.tags["accompanied"], t.accompaniment_only) == (True, False)


@needs_yaml
def test_unknown_accompanied_value_is_an_error(tmp_path):
    from musical_perception.evals.cases import load_cases

    (tmp_path / "c.yaml").write_text(_case_yaml("c", accompanied="piano"))
    with pytest.raises(ValueError, match="accompanied must be one of"):
        load_cases(tmp_path)


# --- the gate exclusion (the point of the whole workstream) -------------

def test_provisional_ids_lists_only_provisional_cases():
    results = [
        _result("z-prov", provisional=True),
        _result("a-verified", provisional=False),
        _result("b-prov", provisional=True),
    ]
    assert provisional_ids(results) == ["b-prov", "z-prov"]


def test_changed_provisional_row_does_not_gate():
    """A provisional row whose outcome moved says nothing about the
    pipeline — it says the agent's guess at truth was different."""
    baseline = {"p": {"tempo": CORRECT}, "v": {"tempo": CORRECT}}
    current = {"p": {"tempo": WRONG}, "v": {"tempo": CORRECT}}
    assert compare_outcomes(current, baseline, provisional={"p"}) == []


def test_the_same_change_gates_when_the_row_is_verified():
    """The control: without the exclusion this is a gate failure."""
    baseline = {"p": {"tempo": CORRECT}, "v": {"tempo": CORRECT}}
    current = {"p": {"tempo": WRONG}, "v": {"tempo": CORRECT}}
    assert compare_outcomes(current, baseline) == ["p.tempo: correct -> wrong"]


def test_new_provisional_case_does_not_gate():
    """This is what unblocks W4: ingesting new material adds rows the
    baseline has never seen, and that must not fail the tier-1 gate."""
    baseline = {"v": {"tempo": CORRECT}}
    current = {"v": {"tempo": CORRECT}, "new": {"tempo": WRONG}}
    assert compare_outcomes(current, baseline, provisional={"new"}) == []
    assert compare_outcomes(current, baseline) == ["new: new case (not in baseline)"]


def test_provisional_row_dropped_from_a_run_does_not_gate():
    baseline = {"v": {"tempo": CORRECT}, "p": {"tempo": CORRECT}}
    current = {"v": {"tempo": CORRECT}}
    assert compare_outcomes(current, baseline, provisional={"p"}) == []


def test_verified_regressions_still_gate_alongside_provisional_rows():
    """The exclusion must be surgical: it removes provisional rows and
    nothing else."""
    baseline = {"v": {"tempo": CORRECT}, "p": {"tempo": CORRECT}}
    current = {"v": {"tempo": WRONG}, "p": {"tempo": WRONG}}
    assert compare_outcomes(current, baseline, provisional={"p"}) == [
        "v.tempo: correct -> wrong"
    ]


def test_suite_provisional_ids_reads_a_run_or_baseline_block():
    """The run artifact is self-describing, so a stale baseline (written
    before W1.5, with no provisional key at all) degrades to 'nothing
    excluded' rather than crashing."""
    from musical_perception.evals.__main__ import suite_provisional_ids

    results = [_result("p", provisional=True), _result("v", provisional=False)]
    block = {"summary": aggregate(results), "outcomes": outcomes_map(results)}
    assert suite_provisional_ids(block) == {"p"}
    assert suite_provisional_ids({"summary": {"provisional": None}}) == set()
    assert suite_provisional_ids({"summary": {}}) == set()
    assert suite_provisional_ids(None) == set()


# --- the headline aggregates -------------------------------------------

def test_no_provisional_rows_leaves_the_summary_shape_untouched():
    """The byte-identity promise, in one assertion: on a verified-only
    corpus the only new key is `provisional`, and it is None."""
    results = [_result("v1", provisional=False), _result("v2", provisional=False)]
    summary = aggregate(results)
    assert summary["provisional"] is None
    assert summary["n_cases"] == 2
    assert summary["fields"]["tempo"]["n"] == 2


def test_provisional_rows_leave_every_headline_number_alone():
    verified = [_result("v1", provisional=False), _result("v2", provisional=False)]
    headline = aggregate(verified)
    with_provisional = aggregate(verified + [
        _result("p1", provisional=True, outcome=WRONG, bpm=52.0),
        _result("p2", provisional=True, outcome=WRONG, bpm=52.0),
    ])
    for key in ("n_cases", "fields", "tempo_metrics", "ece", "slices",
                "risk_coverage", "quality_spearman", "errors"):
        assert with_provisional[key] == headline[key], f"{key} moved"


def test_provisional_slice_has_its_own_n_and_its_own_numbers():
    summary = aggregate([
        _result("v1", provisional=False),
        _result("p1", provisional=True, outcome=WRONG, bpm=52.0),
        _result("p2", provisional=True, outcome=WRONG, bpm=52.0),
    ])
    prov = summary["provisional"]
    assert prov["n_cases"] == 2
    assert prov["case_ids"] == ["p1", "p2"]
    assert prov["fields"]["tempo"]["n"] == 2
    assert prov["fields"]["tempo"]["accuracy"] == 0.0
    # ...and the headline still reports only the one verified case
    assert summary["n_cases"] == 1
    assert summary["fields"]["tempo"]["accuracy"] == 1.0


def test_a_broken_provisional_case_is_not_a_headline_error():
    summary = aggregate([
        _result("v1", provisional=False),
        CaseResult(case_id="p1", provisional=True, error="ValueError: boom"),
    ])
    assert summary["errors"] == []
    assert summary["provisional"]["errors"] == ["p1"]


# --- reporting ----------------------------------------------------------

def test_reports_name_the_provisional_slice_when_one_exists():
    report = {
        "created_at": "2026-08-25T00:00:00+00:00", "git_sha": "abc1234",
        "package_version": "0", "suites": {"tier1": {
            "summary": aggregate([
                _result("v1", provisional=False),
                _result("p1", provisional=True, outcome=WRONG),
            ]),
            "outcomes": {}, "cases": [],
        }},
    }
    for text in (render_html(report), render_markdown_baseline(report)):
        assert "provisional slice" in text
        assert "p1" in text


def test_reports_stay_silent_when_nothing_is_provisional():
    report = {
        "created_at": "2026-08-25T00:00:00+00:00", "git_sha": "abc1234",
        "package_version": "0", "suites": {"tier1": {
            "summary": aggregate([_result("v1", provisional=False)]),
            "outcomes": {}, "cases": [],
        }},
    }
    for text in (render_html(report), render_markdown_baseline(report)):
        assert "provisional slice" not in text


# --- stage1 -------------------------------------------------------------

@needs_yaml
def test_stage1_row_is_provisional_when_the_case_is(tmp_path):
    """A verified grid under a provisional case is still a provisional
    row: a row is only as verified as its weakest label."""
    from musical_perception.annotation.grids import BeatGrid, save_grid
    from musical_perception.evals.stage1 import run_stage1

    (tmp_path / "cases").mkdir()
    (tmp_path / "cases" / "c.yaml").write_text(
        _case_yaml("c", maturity="provisional")
    )
    trace = tmp_path / "traces" / "c"
    trace.mkdir(parents=True)
    words = [{"word": f"w{i}", "start": 1.0 + 0.5 * i, "end": 1.2 + 0.5 * i}
             for i in range(4)]
    (trace / "whisper.json").write_text(json.dumps({"words": words}))
    times = [round(1.0 + 0.5 * i, 4) for i in range(4)]
    save_grid(
        BeatGrid(clip="c", provisional=False, beats=times, onsets=times),
        tmp_path / "grids",
    )

    out = run_stage1(tmp_path)
    assert out["clips"][0]["provisional"] is True
    assert out["aggregate_verified"] is None
    assert out["aggregate_provisional"]["n_clips"] == 1
