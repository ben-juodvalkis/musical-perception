"""Stage-1 pulse scoring, the Acc/OE tempo metrics, and the
onset-vs-token trace guard — hardcoded data, no media or models."""

import json
import warnings

import pytest

from musical_perception.evals.aggregate import (
    acc1,
    acc2,
    octave_errors,
    tempo_metrics,
)
from musical_perception.evals.scorers import (
    ABSTAINED,
    CORRECT,
    WRONG,
    CaseResult,
    ScoreResult,
)
from musical_perception.evals.stage1 import match_events, run_stage1, score_pulse
from musical_perception.evals.traces import onset_token_mismatch, replay_bundle

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

needs_yaml = pytest.mark.skipif(not HAS_YAML, reason="needs pyyaml")


# --- matching -----------------------------------------------------------

def test_match_events_within_tolerance():
    matches = match_events([1.0, 2.0, 3.0], [1.05, 2.5, 2.94], tol=0.07)
    assert matches == [(0, 0), (2, 2)]  # 2.5 is no one's match


def test_match_events_is_one_to_one_and_prefers_nearest():
    assert match_events([1.0], [0.98, 1.01], tol=0.07) == [(0, 1)]
    assert match_events([1.0, 1.02], [1.01], tol=0.07) == [(0, 0)]


def test_score_pulse_numbers():
    # ref 4 events; pred hits two of them (early), misses two, one spurious
    scored = score_pulse([1.0, 2.0, 3.0, 4.0], [0.95, 2.98, 5.5], tol=0.07)
    assert scored["matched"] == 2
    assert scored["precision"] == pytest.approx(2 / 3)
    assert scored["recall"] == pytest.approx(1 / 2)
    assert scored["f_measure"] == pytest.approx(4 / 7)
    assert scored["asynchrony_ms"] == pytest.approx([-50.0, -20.0])


def test_score_pulse_degenerate_cases():
    empty_pred = score_pulse([1.0], [], tol=0.07)
    assert (empty_pred["precision"], empty_pred["recall"]) == (None, 0.0)
    assert empty_pred["f_measure"] == 0.0
    empty_both = score_pulse([], [], tol=0.07)
    assert empty_both["f_measure"] is None


# --- Acc1/Acc2 + OE1/OE2 (Review 2 §4.2) --------------------------------

def test_octave_errors():
    assert octave_errors(208, 104) == pytest.approx((1.0, 0.0))
    assert octave_errors(104, 104) == pytest.approx((0.0, 0.0))
    assert octave_errors(52, 104) == pytest.approx((-1.0, 0.0))
    oe1, oe2 = octave_errors(156, 104)  # between levels: 1.5× the truth
    assert oe1 == pytest.approx(0.585, abs=1e-3)
    assert oe2 == pytest.approx(-0.415, abs=1e-3)  # nearest family member is 2×


def test_acc1_acc2():
    assert acc1(100, 104, 0.04)            # within 4%
    assert not acc1(208.5, 104, 0.04)
    assert acc2(208.5, 104, 0.04)          # within 4% of 2×
    assert acc2(34.8, 104, 0.04)           # within 4% of ⅓×
    assert not acc2(70, 104, 0.04)         # 2/3× is not in the family
    assert not acc2(156, 104, 0.04)        # between levels is invisible to Acc2


def _tempo_case(cid, predicted, expected, outcome):
    return CaseResult(case_id=cid, scores=[
        ScoreResult("tempo", outcome, 0.0, predicted, expected),
    ])


def test_tempo_metrics_aggregation():
    results = [
        _tempo_case("a", 104.0, 104.0, CORRECT),   # exact
        _tempo_case("b", 208.0, 104.0, WRONG),     # octave: Acc2 hit, OE2 0
        _tempo_case("c", 120.0, 104.0, WRONG),     # between levels
        _tempo_case("d", None, 104.0, ABSTAINED),  # excluded
        CaseResult(case_id="e", scores=[
            ScoreResult("counts", WRONG, 0.0, 12, 8),  # non-tempo: ignored
        ]),
    ]
    tm = tempo_metrics(results)
    assert tm["n_committed"] == 3
    assert tm["acc1"]["tol_04"] == pytest.approx(1 / 3, abs=1e-3)
    assert tm["acc2"]["tol_04"] == pytest.approx(2 / 3, abs=1e-3)
    assert tm["between_levels"] == 1  # only c: |OE2| ≈ 0.206
    per_case = {r["case"]: r for r in tm["per_case"]}
    assert per_case["b"]["oe1"] == pytest.approx(1.0)
    assert per_case["b"]["oe2"] == pytest.approx(0.0)
    assert per_case["c"]["oe2"] == pytest.approx(0.2064, abs=1e-3)
    assert tempo_metrics([results[-1]]) is None  # no tempo rows at all


# --- onset-vs-token guard (ADR-016 clip-17) -----------------------------

def test_onset_token_mismatch_thresholds():
    assert onset_token_mismatch(0, 10) is not None   # tokens, no acoustics
    assert onset_token_mismatch(0, 0) is None
    assert onset_token_mismatch(10, 0) is None       # nothing transcribed
    assert onset_token_mismatch(10, 40) is not None  # 40 > 1.5·10 + 8
    assert onset_token_mismatch(30, 40) is None      # 40 ≤ 1.5·30 + 8
    assert onset_token_mismatch(20, 38) is None      # boundary is inclusive
    assert onset_token_mismatch(20, 39) is not None


def _write_trace(trace_dir, n_words):
    trace_dir.mkdir(parents=True)
    words = [
        {"word": f"w{i}", "start": 1.0 + 0.3 * i, "end": 1.2 + 0.3 * i}
        for i in range(n_words)
    ]
    (trace_dir / "whisper.json").write_text(json.dumps({"words": words}))
    (trace_dir / "meta.json").write_text(json.dumps({"analyze_flags": {}}))
    (trace_dir / "gemini.json").write_text(json.dumps(
        {"model": "m", "raw_response": "{}", "inputs": {}}
    ))


def _write_grid(root, clip, n_onsets):
    from musical_perception.annotation.grids import BeatGrid, save_grid

    times = [round(1.0 + 0.5 * i, 4) for i in range(n_onsets)]
    save_grid(
        BeatGrid(clip=clip, provisional=True, beats=times, onsets=times),
        root / "grids",
    )


@needs_yaml
def test_replay_warns_on_hallucination_signature(tmp_path):
    _write_trace(tmp_path / "traces" / "clip-h", n_words=40)
    _write_grid(tmp_path, "clip-h", n_onsets=4)
    with pytest.warns(UserWarning, match="hallucination"):
        replay_bundle(tmp_path / "traces" / "clip-h")


@needs_yaml
def test_replay_quiet_when_consistent_or_gridless(tmp_path):
    _write_trace(tmp_path / "traces" / "clip-ok", n_words=8)
    _write_grid(tmp_path, "clip-ok", n_onsets=9)
    _write_trace(tmp_path / "traces" / "clip-nogrid", n_words=8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        replay_bundle(tmp_path / "traces" / "clip-ok")
        replay_bundle(tmp_path / "traces" / "clip-nogrid")


# --- run_stage1 integration on a temp evals root ------------------------

@needs_yaml
def test_run_stage1_scores_and_reports_missing(tmp_path):
    (tmp_path / "cases").mkdir()
    for cid in ("clip-a", "clip-b"):
        (tmp_path / "cases" / f"{cid}.yaml").write_text(yaml.safe_dump({
            "id": cid,
            "input": {"trace": f"traces/{cid}/"},
            "tags": {"count_style": "numbers"},
            "expect": {"marking_bpm": 104},
        }))
        _write_trace(tmp_path / "traces" / cid, n_words=4)
    _write_grid(tmp_path, "clip-a", n_onsets=4)  # clip-b has no grid

    out = run_stage1(tmp_path)
    assert out["missing_grids"] == ["clip-b"]
    assert [c["case_id"] for c in out["clips"]] == ["clip-a"]
    clip = out["clips"][0]
    assert clip["provisional"] is True
    assert clip["n_ref"] == 4 and clip["n_pred"] == 4
    assert out["aggregate_verified"] is None
    agg = out["aggregate_provisional"]
    assert agg["n_clips"] == 1
    # Slices are verified-only (owner ruling 2026-08-26). This clip's grid
    # is provisional, so its count_style must NOT appear — the table would
    # otherwise report an unverified number under a bare style name. This
    # assertion encodes the ruling; it is the deliverable, not a workaround
    # for a failure.
    assert out["slices"] == {}
    assert not out["errors"]


@needs_yaml
def test_slices_are_verified_only_and_do_not_pool_maturities(tmp_path):
    """The paired test the ruling actually needs.

    A filter that excludes everything satisfies "provisional is absent"
    just as well as a correct one (W1.5's standing lesson), so this
    asserts BOTH halves at once: the verified row of a style is present
    and scored, the provisional row of the SAME style is absent from it,
    and a style carried only by a provisional row produces no slice.
    """
    from musical_perception.annotation.grids import BeatGrid, save_grid

    spec = [
        # (case id, count_style, grid provisional?)
        ("clip-v", "numbers", False),      # verified -> must appear
        ("clip-p", "numbers", True),       # provisional, same style -> excluded
        ("clip-only", "vocables", True),   # style with no verified row -> gone
    ]
    (tmp_path / "cases").mkdir()
    for cid, style, prov in spec:
        (tmp_path / "cases" / f"{cid}.yaml").write_text(yaml.safe_dump({
            "id": cid,
            "input": {"trace": f"traces/{cid}/"},
            "tags": {"count_style": style},
            "expect": {"marking_bpm": 104},
        }))
        _write_trace(tmp_path / "traces" / cid, n_words=4)
        times = [round(1.0 + 0.3 * i, 4) for i in range(4)]
        save_grid(
            BeatGrid(clip=cid, provisional=prov, beats=times, onsets=times),
            tmp_path / "grids",
        )

    out = run_stage1(tmp_path)

    # All three clips are scored and reported...
    assert {c["case_id"] for c in out["clips"]} == {"clip-v", "clip-p", "clip-only"}
    assert out["aggregate_provisional"]["n_clips"] == 2
    assert out["aggregate_verified"]["n_clips"] == 1

    # ...but only the verified one reaches a slice.
    assert set(out["slices"]) == {"numbers"}
    assert out["slices"]["numbers"]["n_clips"] == 1
    assert "vocables" not in out["slices"]
