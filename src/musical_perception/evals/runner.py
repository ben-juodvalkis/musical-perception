"""
Suite runner: replay cases through analyze(), score against expectations,
and compare runs to the blessed baseline.
"""

from pathlib import Path

from musical_perception.evals.cases import Case, load_cases
from musical_perception.evals.scorers import (
    REPORTED_ONLY_FIELDS,
    CaseResult,
    score_counts,
    score_meter_factored,
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
        # W12: the factored slice, reported beside meter_triple and
        # gating nothing (REPORTED_ONLY_FIELDS).
        scores.extend(score_meter_factored(
            normalized, expect["meter"], expect.get("subdivision")
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
            reference=case.reference,
        )
    except Exception as e:  # a broken trace is a reportable row, not a crash
        return CaseResult(
            case_id=case.id, tags=dict(case.tags),
            error=f"{type(e).__name__}: {e}",
            provisional=case.provisional,
            reference=case.reference,
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
            else {
                s.field: s.outcome for s in c.scores
                if s.field not in REPORTED_ONLY_FIELDS
            }
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


def suite_provisional_ids(suite_block: dict | None) -> set[str]:
    """Case ids a run (or the blessed baseline) recorded as provisional.

    The run artifact carries its own list, so the baseline is
    self-describing: comparing two runs never needs to re-read the case
    files, and a row that changed maturity in either direction is excluded
    by taking the union of both sides (charter W1.5).
    """
    prov = ((suite_block or {}).get("summary") or {}).get("provisional")
    return set((prov or {}).get("case_ids") or ())


def reference_ids(case_results: list[CaseResult]) -> list[str]:
    """Case ids the owner demoted to the reference slice (reset 2026-09-01).

    Sorted for the same reason as `provisional_ids`: the run artifact —
    and therefore the blessed baseline — stays self-describing."""
    return sorted(c.case_id for c in case_results if c.reference)


def suite_reference_ids(suite_block: dict | None) -> set[str]:
    """Case ids a run (or the baseline) recorded as reference-slice rows,
    read from the artifact so comparisons never re-read the case files."""
    ref = ((suite_block or {}).get("summary") or {}).get("reference")
    return set((ref or {}).get("case_ids") or ())


def blessed_report(report: dict) -> dict:
    """The run artifact as it must be written to `evals/baseline.json`.

    **W1.6 — where the guarantee lives.** The baseline's `outcomes` map IS
    the gating corpus: `tests/test_evals_replay.py` reads it, and W1.5's
    tripwire asserts it equals the owner-verified case ids exactly. So the
    exclusion has to happen *here*, at pinning time — a `bless` that copies
    the run verbatim writes agent-authored truth into the gate, which is
    what happened on 2026-09-01 (30 pinned -> 52, 22 of them provisional).

    `compare_outcomes`'s `provisional=` skip is NOT this guarantee and does
    not replace it. That one is a runtime filter for rows provisional in
    the *current run* — fresh ingestion the baseline has never seen, or a
    row whose maturity moved since the bless. Both exist, with distinct
    jobs; only this one bounds what the baseline is allowed to claim.

    Provisional rows stay fully **reported**: `summary` (with its own
    slice and its own n) and `cases` are untouched, so the published
    baseline still shows them. They stop being **pinned**, which is the
    only thing that ever gated. Suites that pin no outcomes at all
    (stage1) pass through unchanged.
    """
    out = dict(report)
    out["suites"] = {}
    for suite, block in report.get("suites", {}).items():
        present = set(block.get("outcomes") or {})
        prov = suite_provisional_ids(block) & present
        # Reset 2026-09-01: reference rows (owner-demoted piano takes) are
        # withheld from pinning for the same reason provisional rows are —
        # the pinned map IS the gating corpus — under their own key, so a
        # reader can tell the two withholdings apart.
        ref = suite_reference_ids(block) & present
        if not prov and not ref:
            out["suites"][suite] = block
            continue
        pinned = dict(block)
        pinned["outcomes"] = {
            cid: v for cid, v in block["outcomes"].items()
            if cid not in prov and cid not in ref
        }
        # Named in the artifact so a reader can tell a filtered map from a
        # truncated one without re-deriving it from the case files.
        if prov:
            pinned["outcomes_withheld_provisional"] = sorted(prov)
        if ref:
            pinned["outcomes_withheld_reference"] = sorted(ref)
        out["suites"][suite] = pinned
    return out


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
