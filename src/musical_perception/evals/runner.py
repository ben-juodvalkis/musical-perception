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
        )
    except Exception as e:  # a broken trace is a reportable row, not a crash
        return CaseResult(case_id=case.id, tags=dict(case.tags), error=f"{type(e).__name__}: {e}")


def run_tier1(evals_root: Path) -> list[CaseResult]:
    return [run_case(c, evals_root) for c in load_cases(Path(evals_root) / "cases")]


def run_suites(suites: list[str], evals_root: Path) -> dict[str, list[CaseResult]]:
    """Run the named suites. tier0 = synthetic sweep; tier1 = frozen traces."""
    from musical_perception.evals import synthetic

    results = {}
    for suite in suites:
        if suite == "tier0":
            results["tier0"] = synthetic.run_suite()
        elif suite == "tier1":
            results["tier1"] = run_tier1(evals_root)
        else:
            raise ValueError(f"unknown suite {suite!r} (expected tier0, tier1)")
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


def compare_outcomes(current: dict, baseline: dict) -> list[str]:
    """Human-readable per-case, per-field outcome changes vs the baseline.

    Any difference — regression or improvement — is a change the PR must
    own by re-blessing the baseline; that is how PRs carry their delta.
    """
    changes = []
    for case_id in sorted(set(baseline) | set(current)):
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
