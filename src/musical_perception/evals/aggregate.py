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

from musical_perception.evals.scorers import ABSTAINED, CORRECT, CaseResult, ScoreResult


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
    return {
        "n": n,
        "correct": correct,
        "wrong": wrong,
        "abstained": abstained,
        "coverage": round(committed / n, 3) if n else 0.0,
        "accuracy": round(correct / committed, 3) if committed else None,
        "accuracy_wilson95": [round(lo, 3), round(hi, 3)] if committed else None,
        "mean_credit": round(sum(r.credit for r in rows) / n, 3) if n else None,
        "failure_modes": dict(modes),
    }


def aggregate(
    case_results: list[CaseResult],
    slice_keys: tuple[str, ...] = ("count_style", "source", "slot"),
) -> dict:
    """Suite summary: per-field metrics, calibration, quality rank corr, slices."""
    all_rows = [s for c in case_results for s in c.scores]
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
                groups[str(c.tags[key])].extend(c.scores)
        if groups:
            slices[key] = {
                value: {name: summarize_field(rs) for name, rs in _by_field(rows).items()}
                for value, rows in sorted(groups.items())
            }

    return {
        "n_cases": len(case_results),
        "errors": [c.case_id for c in case_results if c.error],
        "fields": fields,
        "quality_spearman": quality_rank or None,
        "ece": expected_calibration_error(all_rows),
        "risk_coverage": risk_coverage(all_rows),
        "slices": slices,
    }


def _by_field(rows: list[ScoreResult]) -> dict[str, list[ScoreResult]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r.field].append(r)
    return dict(sorted(grouped.items()))
