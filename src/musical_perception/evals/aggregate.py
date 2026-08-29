"""
Aggregation over ScoreResults: accuracy with intervals, abstention
accounting, calibration, and tag slices.

Rules carried from ADR-009: abstention is a first-class outcome (never
counted as wrong); n is always stated with a Wilson interval; slices are
reported (the mean is a headline, the worst slice is the story).
Numpy-only — no scipy.
"""

import math
from collections import defaultdict

import numpy as np

from musical_perception.evals.scorers import (
    ABSTAINED,
    CORRECT,
    REPORTED_ONLY_FIELDS,
    CaseResult,
    ScoreResult,
)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def expected_calibration_error(
    rows: list[ScoreResult], bins: int = 5
) -> float | None:
    """ECE over confidence-bearing, committed (non-abstained) scores."""
    scored = [r for r in rows if r.confidence is not None and r.outcome != ABSTAINED]
    if not scored:
        return None
    conf = np.array([r.confidence for r in scored])
    hit = np.array([1.0 if r.outcome == CORRECT else 0.0 for r in scored])
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(hit[mask].mean() - conf[mask].mean())
    return round(float(ece), 4)


def risk_coverage(rows: list[ScoreResult]) -> list[dict]:
    """Risk–coverage curve over confidence-bearing committed scores.

    Each point: commit only above a confidence threshold; report the
    fraction committed (coverage) and the error rate among them (risk).
    """
    scored = sorted(
        (r for r in rows if r.confidence is not None and r.outcome != ABSTAINED),
        key=lambda r: r.confidence, reverse=True,
    )
    if not scored:
        return []
    curve = []
    wrong = 0
    for i, r in enumerate(scored, start=1):
        if r.outcome != CORRECT:
            wrong += 1
        curve.append({
            "threshold": round(r.confidence, 3),
            "coverage": round(i / len(scored), 3),
            "risk": round(wrong / i, 3),
        })
    return curve


def _average_ranks(values: list[float]) -> np.ndarray:
    """Ranks with ties averaged (scipy-free)."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rank correlation; None when undefined (n<3 or zero variance)."""
    if len(a) != len(b) or len(a) < 3:
        return None
    ra, rb = _average_ranks(a), _average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def summarize_field(rows: list[ScoreResult]) -> dict:
    """Counts, coverage, accuracy-on-committed, credit, interval, modes."""
    n = len(rows)
    correct = sum(1 for r in rows if r.outcome == CORRECT)
    abstained = sum(1 for r in rows if r.outcome == ABSTAINED)
    committed = n - abstained
    wrong = committed - correct
    modes = defaultdict(int)
    for r in rows:
        if r.failure_mode:
            modes[r.failure_mode] += 1
    lo, hi = wilson_interval(correct, committed)
    # ADR-014: of the wrong answers we could check, how many still carried
    # the truth somewhere in their reported family? Informational — it does
    # not enter accuracy, credit, or any gate.
    family_checked = sum(1 for r in rows if r.truth_in_family is not None)
    family_hits = sum(1 for r in rows if r.truth_in_family)
    return {
        "n": n,
        "correct": correct,
        "wrong": wrong,
        "abstained": abstained,
        "coverage": round(committed / n, 3) if n else 0.0,
        "accuracy": round(correct / committed, 3) if committed else None,
        "accuracy_wilson95": [round(lo, 3), round(hi, 3)] if committed else None,
        "mean_credit": round(sum(r.credit for r in rows) / n, 3) if n else None,
        "truth_in_family": family_hits if family_checked else None,
        "truth_in_family_n": family_checked or None,
        "failure_modes": dict(modes),
    }


# Tempo-literature metrics (Review 2 §4.2, adopted at rung 1). Acc1/Acc2
# at the field-standard ±4% (Gouyon et al. 2006) plus the house ±8%;
# OE1 = log₂(est/ref), OE2 = OE1 minus the best factor in the metric-level
# family — the only standard metric that sees the "landed between levels"
# failure (|OE2| mass in (0, log₂1.5]).
ACC_TOL_STANDARD = 0.04
ACC_TOL_HOUSE = 0.08
ACC2_FAMILY = (1 / 3, 1 / 2, 1.0, 2.0, 3.0)


def octave_errors(predicted_bpm: float, expected_bpm: float) -> tuple[float, float]:
    """(OE1, OE2) in octaves for one committed tempo estimate."""
    oe1 = math.log2(predicted_bpm / expected_bpm)
    best = min((abs(oe1 - math.log2(f)), math.log2(f)) for f in ACC2_FAMILY)
    return oe1, oe1 - best[1]


def _within(predicted: float, target: float, tol: float) -> bool:
    return abs(predicted - target) / target <= tol


def acc1(predicted_bpm: float, expected_bpm: float, tol: float) -> bool:
    return _within(predicted_bpm, expected_bpm, tol)


def acc2(predicted_bpm: float, expected_bpm: float, tol: float) -> bool:
    """Within tol of any {⅓,½,1,2,3}× the annotation — `truth_in_family`
    with the literature's fixed family and name (Review 2 §4.2)."""
    return any(_within(predicted_bpm, expected_bpm * f, tol) for f in ACC2_FAMILY)


def tempo_metrics(case_results: list[CaseResult]) -> dict | None:
    """Acc1/Acc2 + OE1/OE2 over committed tempo rows. Informational:
    never enters outcomes, credit, or any gate."""
    rows = [
        (c.case_id, s.predicted, s.expected)
        for c in case_results for s in c.scores
        if s.field == "tempo" and s.outcome != ABSTAINED
        and isinstance(s.predicted, (int, float))
        and isinstance(s.expected, (int, float))
    ]
    if not rows:
        return None
    per_case = []
    for case_id, pred, exp in rows:
        oe1, oe2 = octave_errors(pred, exp)
        per_case.append({
            "case": case_id,
            "predicted": round(float(pred), 2),
            "expected": round(float(exp), 2),
            "oe1": round(oe1, 4),
            "oe2": round(oe2, 4),
        })
    n = len(rows)
    oe1s = np.array([r["oe1"] for r in per_case])
    oe2s = np.array([r["oe2"] for r in per_case])

    def _dist(arr: np.ndarray) -> dict:
        return {
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "abs_median": round(float(np.median(np.abs(arr))), 4),
            "max_abs": round(float(np.abs(arr).max()), 4),
        }

    return {
        "n_committed": n,
        "acc1": {
            "tol_04": round(sum(acc1(p, e, ACC_TOL_STANDARD) for _, p, e in rows) / n, 3),
            "tol_08": round(sum(acc1(p, e, ACC_TOL_HOUSE) for _, p, e in rows) / n, 3),
        },
        "acc2": {
            "tol_04": round(sum(acc2(p, e, ACC_TOL_STANDARD) for _, p, e in rows) / n, 3),
            "tol_08": round(sum(acc2(p, e, ACC_TOL_HOUSE) for _, p, e in rows) / n, 3),
        },
        "oe1": _dist(oe1s),
        "oe2": _dist(oe2s),
        "between_levels": int(((np.abs(oe2s) > 0.08) & (np.abs(oe2s) <= 0.585)).sum()),
        "per_case": per_case,
    }


def _summarize_cases(
    case_results: list[CaseResult],
    slice_keys: tuple[str, ...],
) -> dict:
    """The metric block for one cohort of cases (verified or provisional)."""
    every_row = [s for c in case_results for s in c.scores]
    # W12: reported-only rows leave every headline number alone — `fields`,
    # ECE, risk-coverage and the tag slices are all computed from the
    # gating rows exactly as before, so a corpus with no factored rows
    # produces byte-identical output to the pre-W12 harness.
    all_rows = [r for r in every_row if r.field not in REPORTED_ONLY_FIELDS]
    factored_rows = [r for r in every_row if r.field in REPORTED_ONLY_FIELDS]
    by_field = defaultdict(list)
    for row in all_rows:
        by_field[row.field].append(row)

    fields = {name: summarize_field(rows) for name, rows in sorted(by_field.items())}

    # Corpus-level quality ordering (the number the engine consumes)
    quality_rank = {}
    for name, rows in by_field.items():
        if not name.startswith("quality_"):
            continue
        pairs = [(r.predicted, r.expected) for r in rows if r.outcome != ABSTAINED]
        if pairs:
            rho = spearman([p for p, _ in pairs], [e for _, e in pairs])
            if rho is not None:
                quality_rank[name] = rho

    slices = {}
    for key in slice_keys:
        groups = defaultdict(list)
        for c in case_results:
            if key in c.tags:
                groups[str(c.tags[key])].extend(
                    r for r in c.scores if r.field not in REPORTED_ONLY_FIELDS
                )
        if groups:
            slices[key] = {
                value: {name: summarize_field(rs) for name, rs in _by_field(rows).items()}
                for value, rows in sorted(groups.items())
            }

    return {
        "n_cases": len(case_results),
        "errors": [c.case_id for c in case_results if c.error],
        "fields": fields,
        "tempo_metrics": tempo_metrics(case_results),
        "quality_spearman": quality_rank or None,
        "ece": expected_calibration_error(all_rows),
        "risk_coverage": risk_coverage(all_rows),
        "slices": slices,
        # W12: the factored meter slice, side by side with meter_triple and
        # gating NOTHING until a separate owner ruling. None when absent,
        # so pre-W12 corpora are byte-identical.
        "factored_meter": (
            {name: summarize_field(rs)
             for name, rs in _by_field(factored_rows).items()}
            if factored_rows else None
        ),
    }


def aggregate(
    case_results: list[CaseResult],
    slice_keys: tuple[str, ...] = ("count_style", "source", "slot"),
) -> dict:
    """Suite summary: per-field metrics, calibration, quality rank corr, slices.

    Charter W1.5: **every headline number here is verified-only.**
    Agent-proposed (`maturity: provisional`) cases are summarized with the
    same machinery under a separate `provisional` block carrying its own n
    — never pooled into the headline, never averaged with owner-verified
    truth. The block is `None` when the corpus has no provisional rows, so
    a verified-only corpus produces byte-identical output to the pre-W1.5
    harness.
    """
    verified = [c for c in case_results if not c.provisional]
    provisional = [c for c in case_results if c.provisional]

    summary = _summarize_cases(verified, slice_keys)
    summary["provisional"] = (
        {
            "case_ids": sorted(c.case_id for c in provisional),
            **_summarize_cases(provisional, slice_keys),
        }
        if provisional else None
    )
    return summary


def _by_field(rows: list[ScoreResult]) -> dict[str, list[ScoreResult]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r.field].append(r)
    return dict(sorted(grouped.items()))
