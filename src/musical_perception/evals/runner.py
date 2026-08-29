"""
Suite runner: replay cases through analyze(), score against expectations,
and compare runs to the blessed baseline.
"""

from pathlib import Path

from musical_perception.evals.cases import Case, load_cases
from musical_perception.evals.scorers import (
    CaseResult,
    score_counts,
    score_meter_triple,
    score_quality,
    score_sides,
    score_slot,
    score_tempo,
)
from musical_perception.evals.traces import replay_bundle


def score_parameters(result, case: Case) -> list:
    """Score a MusicalParameters against a case's expect block."""
    expect = case.expect
    scores = []
    normalized = result.normalized_tempo

    if case.expected_bpm is not None:
        scores.append(score_tempo(
            normalized.bpm if normalized else None,
            case.expected_bpm,
            confidence=normalized.confidence if normalized else None,
            alternates=normalized.alternates if normalized else None,
        ))
    if "meter" in expect:
        scores.append(score_meter_triple(
            normalized,
            expect["meter"],
            case.expected_bpm,
            expect.get("subdivision"),
        ))
    if "counts" in expect:
        scores.append(score_counts(result.structure, expect["counts"]))
    if "sides" in expect:
        scores.append(score_sides(result.structure, expect["sides"]))
    if "slot" in expect:
        scores.append(score_slot(result.exercise, expect["slot"]))
    if "character" in expect:
        scores.extend(score_quality(result.quality, expect["character"]))
    return scores


def _replay_media_name(case: Case, use_pose: bool) -> str:
    """A name for analyze()'s suffix checks only — the file is never opened."""
    if case.media:
        return case.media
    return "replay.mov" if use_pose else "replay.wav"


def run_case(case: Case, evals_root: Path) -> CaseResult:
    """Replay one frozen-trace case offline and score it."""
    from musical_perception.analyze import analyze

    try:
        bundle, meta = replay_bundle(Path(evals_root) / case.trace)
        use_pose = bool(meta.get("analyze_flags", {}).get("use_pose"))
        result = analyze(
            _replay_media_name(case, use_pose),
            use_pose=use_pose,
            bundle=bundle,
        )
        return CaseResult(
            case_id=case.id, tags=dict(case.tags),
            scores=score_parameters(result, case),
            provisional=case.provisional,
        )
    except Exception as e:  # a broken trace is a reportable row, not a crash
        return CaseResult(
            case_id=case.id, tags=dict(case.tags),
            error=f"{type(e).__name__}: {e}",
            provisional=case.provisional,
        )


def run_tier1(evals_root: Path) -> list[CaseResult]:
    return [run_case(c, evals_root) for c in load_cases(Path(evals_root) / "cases")]


def run_suites(suites: list[str], evals_root: Path) -> dict:
    """Run the named suites. tier0 = synthetic sweep; tier1 = frozen traces;
    stage1 = pulse scoring against beat grids (returns its own summary dict
    — provisional grids gate nothing); stage1-peakrate = the same scoring
    over W11's frozen acoustic pulse sidecars instead of word starts, as a
    separate suite so `stage1` keeps meaning exactly what it meant."""
    from musical_perception.evals import stage1, synthetic

    results = {}
    for suite in suites:
        if suite == "tier0":
            results["tier0"] = synthetic.run_suite()
        elif suite == "tier1":
            results["tier1"] = run_tier1(evals_root)
        elif suite == "stage1":
            results["stage1"] = stage1.run_stage1(Path(evals_root))
        elif suite == "stage1-peakrate":
            results["stage1-peakrate"] = stage1.run_stage1(
                Path(evals_root), pulse_source=stage1.PULSE_SOURCE_PEAKRATE
            )
        else:
            raise ValueError(
                f"unknown suite {suite!r} (expected tier0, tier1, stage1, "
                f"stage1-peakrate)"
            )
    return results


def outcomes_map(case_results: list[CaseResult]) -> dict:
    """{case_id: {field: outcome}} — the exact thing the tier-1 gate pins."""
    return {
        c.case_id: (
            {"__error__": c.error} if c.error
            else {s.field: s.outcome for s in c.scores}
        )
        for c in case_results
    }


def provisional_ids(case_results: list[CaseResult]) -> list[str]:
    """Case ids whose truth labels are agent-proposed (charter W1.5).

    Sorted so the run artifact — and therefore the blessed baseline — is
    self-describing and stable: a later run learns which of the baseline's
    rows were provisional without re-reading the case files.
    """
    return sorted(c.case_id for c in case_results if c.provisional)


def compare_outcomes(
    current: dict, baseline: dict, provisional: set[str] | None = None
) -> list[str]:
    """Human-readable per-case, per-field outcome changes vs the baseline.

    Any difference — regression or improvement — is a change the PR must
    own by re-blessing the baseline; that is how PRs carry their delta.

    Rows named in `provisional` are skipped entirely (charter W1.5): their
    truth is agent-proposed, so a difference against them says nothing
    about the pipeline and must never fail the tier-1 gate. Callers pass
    the union of this run's provisional ids and the baseline's own, so a
    row flipping maturity in either direction still cannot gate until the
    owner has verified it.
    """
    skip = set(provisional or ())
    changes = []
    for case_id in sorted(set(baseline) | set(current)):
        if case_id in skip:
            continue
        base, cur = baseline.get(case_id), current.get(case_id)
        if base is None:
            changes.append(f"{case_id}: new case (not in baseline)")
            continue
        if cur is None:
            changes.append(f"{case_id}: missing from this run (in baseline)")
            continue
        for fname in sorted(set(base) | set(cur)):
            b, c = base.get(fname), cur.get(fname)
            if b != c:
                changes.append(f"{case_id}.{fname}: {b} -> {c}")
    return changes


REBLESS_RECIPE = (
    "If this change is intentional, re-bless the baseline:\n"
    "  python -m musical_perception.evals run --suite tier0,tier1\n"
    "  python -m musical_perception.evals bless\n"
    "and commit evals/baseline.json + docs/evals/baseline.md with your PR."
)
