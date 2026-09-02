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
def test_the_gating_corpus_is_exactly_the_blessed_set():
    """Every case that gates is owner-verified, and the baseline pins them all.

    W1.5 wrote this as a tripwire: "if a provisional case ever lands here,
    this test says so loudly — that is a review event, not a detail." W4's
    Barre-1 ingestion (2026-08-29) is that event, and the tripwire fired on
    the first run; the ledger entry of that date records it as such. What
    the tripwire was protecting is not the corpus *size* but the *gating
    set*, so the assertion moves to the thing itself and gets stricter: the
    verified ids must be exactly the ids the blessed baseline pins, so no
    session can grow the gating set by writing `maturity: verified` on
    agent-authored truth.

    2026-09-01: the owner read back and verified the 26 Ballet Barre 6 rows,
    taking the gating set 30 -> 56. Promoting cases fires this tripwire by
    design — it stays red until the owner re-blesses, which is the review
    event W1.5 built it to announce.

    Reset 2026-09-01 (owner-attended, evening): the demo is the case, so
    the 17 piano takes are demoted to a reference slice that never gates
    and is never pinned, and the ballonne demo is deferred from step one
    by owner ruling (fast triple meter: no honest level inside 70-140).
    The gating set is therefore *verified minus reference*: 26
    rig/counting clips + 8 barre-6 demos = 34.
    """
    from musical_perception.evals.cases import load_cases

    cases = load_cases(_REPO / "evals" / "cases")
    gating = {c.id for c in cases if not c.provisional and not c.reference}
    blessed = set(
        json.loads((_REPO / "evals" / "baseline.json").read_text())
        ["suites"]["tier1"]["outcomes"]
    )
    assert len(blessed) == 34
    assert gating == blessed


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


# --- W1.6: what a bless is allowed to pin ------------------------------

def _run_report(*results) -> dict:
    """A run artifact of the shape `evals run` writes."""
    return {
        "schema": 1, "created_at": "2026-09-01T00:00:00+00:00",
        "git_sha": "abc1234", "package_version": "0",
        "suites": {"tier1": {
            "summary": aggregate(list(results)),
            "outcomes": outcomes_map(list(results)),
            "cases": [{"case_id": r.case_id} for r in results],
        }},
    }


def test_bless_pins_only_the_verified_rows():
    """The defect of 2026-09-01, in one assertion: a run scoring 1 verified
    and 2 provisional cases must bless a gating set of exactly 1."""
    from musical_perception.evals.runner import blessed_report

    report = _run_report(
        _result("v1", provisional=False),
        _result("p1", provisional=True, outcome=WRONG),
        _result("p2", provisional=True, outcome=WRONG),
    )
    assert set(report["suites"]["tier1"]["outcomes"]) == {"v1", "p1", "p2"}
    pinned = blessed_report(report)
    assert set(pinned["suites"]["tier1"]["outcomes"]) == {"v1"}


def test_bless_still_reports_the_rows_it_refuses_to_pin():
    """Provisional stops being *pinned*; it does not stop being *reported*."""
    from musical_perception.evals.runner import blessed_report

    report = _run_report(
        _result("v1", provisional=False),
        _result("p1", provisional=True, outcome=WRONG),
    )
    block = blessed_report(report)["suites"]["tier1"]
    assert block["summary"]["provisional"]["case_ids"] == ["p1"]
    assert block["cases"] == report["suites"]["tier1"]["cases"]
    assert block["outcomes_withheld_provisional"] == ["p1"]
    assert "provisional slice" in render_markdown_baseline(report)


def test_bless_of_a_verified_only_run_is_unchanged():
    """Every baseline blessed before the barre-1 ingestion stays exactly
    what it was — no withheld key, no reshaped outcomes."""
    from musical_perception.evals.runner import blessed_report

    report = _run_report(
        _result("v1", provisional=False), _result("v2", provisional=False)
    )
    assert blessed_report(report) == report


def test_bless_passes_through_a_suite_that_pins_nothing():
    """stage1's summary has no `provisional` key and its outcomes map is
    empty — the filter must not invent structure there."""
    from musical_perception.evals.runner import blessed_report

    report = {"suites": {"stage1": {
        "summary": {"clips": [], "pulse_source": "whisper"},
        "outcomes": {}, "cases": [],
    }}}
    assert blessed_report(report) == report


def test_pinning_and_comparison_are_two_guarantees_not_one():
    """W1.6's stated ruling, tested: the pinned set is what bounds the
    gating corpus, and the comparison-time skip is a separate runtime
    filter. Gating *decisions* are identical either way — which is why the
    leak was invisible — but only the pinned set stops a provisional row
    from ever being in the gate at all."""
    from musical_perception.evals.runner import blessed_report

    blessed_run = _run_report(
        _result("v1", provisional=False), _result("p1", provisional=True)
    )
    leaky = blessed_run["suites"]["tier1"]["outcomes"]              # pre-W1.6
    correct = blessed_report(blessed_run)["suites"]["tier1"]["outcomes"]

    # a later run where BOTH a verified and a provisional row moved
    later = _run_report(
        _result("v1", provisional=False, outcome=WRONG),
        _result("p1", provisional=True, outcome=WRONG),
    )
    current = later["suites"]["tier1"]["outcomes"]
    excluded = {"p1"}
    assert (compare_outcomes(current, leaky, provisional=excluded)
            == compare_outcomes(current, correct, provisional=excluded)
            == ["v1.tempo: correct -> wrong"])
    # ...and the difference that matters:
    assert "p1" in leaky and "p1" not in correct


def test_verifying_a_case_is_a_review_event_not_a_silent_promotion():
    """After the owner flips a row to verified, the gate fails until a
    re-bless. That is the cost of growing the gating set, and it is the
    point of pinning the verified set rather than the whole run."""
    from musical_perception.evals.runner import blessed_report

    pinned = blessed_report(_run_report(
        _result("v1", provisional=False), _result("p1", provisional=True)
    ))["suites"]["tier1"]["outcomes"]
    # next run: the owner has verified p1, so nothing excludes it any more
    now_verified = _run_report(
        _result("v1", provisional=False), _result("p1", provisional=False)
    )["suites"]["tier1"]["outcomes"]
    assert compare_outcomes(now_verified, pinned, provisional=set()) == [
        "p1: new case (not in baseline)"
    ]


def test_the_bless_command_writes_a_verified_only_baseline(tmp_path, monkeypatch):
    """End to end through the CLI, since the defect lived in the command
    and not in any function the unit tests could reach."""
    import argparse

    from musical_perception.evals import __main__ as cli

    (tmp_path / "runs").mkdir()
    report = _run_report(
        _result("v1", provisional=False),
        _result("p1", provisional=True, outcome=WRONG),
    )
    (tmp_path / "runs" / "run-2026-09-01.json").write_text(json.dumps(report))
    monkeypatch.setattr(cli, "BASELINE_MD", tmp_path / "baseline.md")

    rc = cli._cmd_bless(argparse.Namespace(evals_root=str(tmp_path), run=None))
    assert rc == 0

    written = json.loads((tmp_path / "baseline.json").read_text())
    assert set(written["suites"]["tier1"]["outcomes"]) == {"v1"}
    assert written["suites"]["tier1"]["outcomes_withheld_provisional"] == ["p1"]
    assert "provisional slice" in (tmp_path / "baseline.md").read_text()


# --- the reference slice (owner reset 2026-09-01: takes out of the benchmark)


def _rref(case_id, *, reference, provisional=False, outcome=CORRECT, bpm=104.0):
    return CaseResult(
        case_id=case_id,
        tags={"count_style": "numbers"},
        provisional=provisional,
        reference=reference,
        scores=[ScoreResult(
            field="tempo", outcome=outcome,
            credit=1.0 if outcome == CORRECT else 0.0,
            predicted=bpm, expected=104.0, confidence=0.8,
        )],
    )


@needs_yaml
def test_reference_keys_on_the_clip_role_tag(tmp_path):
    """The demotion is tag-keyed: `clip_role: take` and nothing else."""
    from musical_perception.evals.cases import load_cases

    body = yaml.safe_load(_case_yaml("t1"))
    body["tags"]["clip_role"] = "take"
    (tmp_path / "t1.yaml").write_text(yaml.safe_dump(body))
    body2 = yaml.safe_load(_case_yaml("d1"))
    body2["tags"]["clip_role"] = "demo"
    (tmp_path / "d1.yaml").write_text(yaml.safe_dump(body2))
    (tmp_path / "v1.yaml").write_text(_case_yaml("v1"))
    by_id = {c.id: c for c in load_cases(tmp_path)}
    assert by_id["t1"].reference is True
    assert by_id["d1"].reference is False
    assert by_id["v1"].reference is False


def test_reference_rows_leave_every_headline_number_alone():
    verified = [_result("v1", provisional=False), _result("v2", provisional=False)]
    headline = aggregate(verified)
    with_reference = aggregate(verified + [
        _rref("t1", reference=True, outcome=WRONG, bpm=52.0),
        _rref("t2", reference=True, outcome=WRONG, bpm=52.0),
    ])
    for key in ("n_cases", "fields", "tempo_metrics", "ece", "slices",
                "risk_coverage", "quality_spearman", "errors"):
        assert with_reference[key] == headline[key], f"{key} moved"
    ref = with_reference["reference"]
    assert ref["case_ids"] == ["t1", "t2"]
    assert ref["n_cases"] == 2
    assert with_reference["provisional"] is None


def test_reference_beats_provisional_for_slotting():
    """A row both provisional and reference lands in the reference slice
    (demotion is the stronger exclusion), never in both."""
    summary = aggregate([
        _result("v1", provisional=False),
        _rref("t1", reference=True, provisional=True),
    ])
    assert summary["provisional"] is None
    assert summary["reference"]["case_ids"] == ["t1"]
    assert summary["n_cases"] == 1


def test_bless_withholds_reference_rows_under_their_own_key():
    from musical_perception.evals.runner import blessed_report

    results = [
        _result("v1", provisional=False),
        _result("p1", provisional=True),
        _rref("t1", reference=True),
    ]
    report = {"suites": {"tier1": {
        "summary": aggregate(results), "outcomes": outcomes_map(results),
    }}}
    pinned = blessed_report(report)["suites"]["tier1"]
    assert set(pinned["outcomes"]) == {"v1"}
    assert pinned["outcomes_withheld_provisional"] == ["p1"]
    assert pinned["outcomes_withheld_reference"] == ["t1"]


def test_suite_reference_ids_reads_a_run_or_baseline_block():
    """Self-describing artifact; a pre-reset baseline (no reference key)
    degrades to 'nothing excluded' rather than crashing."""
    from musical_perception.evals.runner import reference_ids, suite_reference_ids

    results = [_rref("t1", reference=True), _result("v1", provisional=False)]
    assert reference_ids(results) == ["t1"]
    block = {"summary": aggregate(results), "outcomes": outcomes_map(results)}
    assert suite_reference_ids(block) == {"t1"}
    assert suite_reference_ids({"summary": {"reference": None}}) == set()
    assert suite_reference_ids({"summary": {}}) == set()
    assert suite_reference_ids(None) == set()


def test_reports_name_the_reference_slice_when_one_exists():
    results = [_result("v1", provisional=False), _rref("t1", reference=True)]
    report = {
        "schema": 1, "created_at": "2026-09-01T00:00:00+00:00", "git_sha": "x",
        "package_version": None,
        "suites": {"tier1": {
            "summary": aggregate(results), "outcomes": outcomes_map(results),
            "cases": [],
        }},
    }
    html = render_html(report)
    md = render_markdown_baseline(report)
    assert "reference slice" in html and "t1" in html
    assert "reference slice" in md and "t1" in md


@needs_yaml
def test_step_one_deferred_tag_also_lands_in_the_reference_slice(tmp_path):
    """Owner ruling (reset 2026-09-01): fast triple meters are deferred
    from step one — `step_one: deferred` shelves the row like a take."""
    from musical_perception.evals.cases import load_cases

    body = yaml.safe_load(_case_yaml("d1"))
    body["tags"]["clip_role"] = "demo"
    body["tags"]["step_one"] = "deferred"
    (tmp_path / "d1.yaml").write_text(yaml.safe_dump(body))
    (tmp_path / "v1.yaml").write_text(_case_yaml("v1"))
    by_id = {c.id: c for c in load_cases(tmp_path)}
    assert by_id["d1"].reference is True
    assert by_id["v1"].reference is False
