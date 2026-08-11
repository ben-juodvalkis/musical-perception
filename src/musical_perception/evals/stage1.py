"""
Stage-1 suite: pulse-stream scoring against beat-grid annotations
(agent-charter rung 1, EVAL-CHANGE).

Scores a per-clip predicted pulse stream against `evals/grids/` beat
grids with mir_eval-style one-to-one matching at ±70 ms (Review 2 §4.3)
and reports precision/recall/F plus *signed* asynchrony
(predicted − reference; negative = prediction early — Standing Lesson 1
predicts Whisper word starts run early).

The rung-1 pulse source is deliberately the Whisper word-start stream:
it is the pipeline's only timing channel today and the baseline rung 2's
acoustic extractor must beat on these same grids.

Provisional grids never gate anything: every row carries the grid's
`provisional` flag, aggregates are split provisional vs verified, and no
pytest gate consumes this suite.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

F_MEASURE_TOLERANCE_S = 0.07   # mir_eval.beat f_measure window
PULSE_SOURCE = "whisper-word-starts"


@dataclass
class ClipPulseScore:
    """Stage-1 result for one clip (times in seconds, asynchrony in ms)."""
    case_id: str
    provisional: bool
    count_style: str | None
    n_ref: int
    n_pred: int
    matched: int
    precision: float | None
    recall: float | None
    f_measure: float | None
    asynchrony_ms: list[float]     # signed, one per matched pair

    def summary(self) -> dict:
        a = np.array(self.asynchrony_ms) if self.asynchrony_ms else None
        return {
            "case_id": self.case_id,
            "provisional": self.provisional,
            "count_style": self.count_style,
            "n_ref": self.n_ref,
            "n_pred": self.n_pred,
            "matched": self.matched,
            "precision": _r3(self.precision),
            "recall": _r3(self.recall),
            "f_measure": _r3(self.f_measure),
            "asynchrony_mean_ms": _r1(a.mean()) if a is not None else None,
            "asynchrony_median_ms": _r1(np.median(a)) if a is not None else None,
            "asynchrony_sd_ms": _r1(a.std()) if a is not None else None,
        }


def _r3(x):
    return None if x is None else round(float(x), 3)


def _r1(x):
    return None if x is None else round(float(x), 1)


def match_events(
    reference: list[float], predicted: list[float], tol: float
) -> list[tuple[int, int]]:
    """One-to-one event matching within ±tol seconds.

    mir_eval's algorithm: all in-window (ref, pred) pairs sorted by |dt|,
    accepted greedily while both sides are unmatched — a maximum matching
    for window-based scoring.
    """
    pairs = [
        (abs(p - r), i, j)
        for i, r in enumerate(reference)
        for j, p in enumerate(predicted)
        if abs(p - r) <= tol
    ]
    pairs.sort()
    used_ref, used_pred, matches = set(), set(), []
    for _, i, j in pairs:
        if i in used_ref or j in used_pred:
            continue
        used_ref.add(i)
        used_pred.add(j)
        matches.append((i, j))
    return sorted(matches)


def score_pulse(
    reference: list[float],
    predicted: list[float],
    tol: float = F_MEASURE_TOLERANCE_S,
) -> dict:
    """P/R/F at ±tol plus signed asynchrony (ms) over matched pairs."""
    matches = match_events(reference, predicted, tol)
    n_ref, n_pred, m = len(reference), len(predicted), len(matches)
    precision = m / n_pred if n_pred else None
    recall = m / n_ref if n_ref else None
    if precision and recall:
        f = 2 * precision * recall / (precision + recall)
    elif n_ref or n_pred:
        f = 0.0
    else:
        f = None
    asynchrony = [
        (predicted[j] - reference[i]) * 1000.0 for i, j in matches
    ]
    return {
        "matched": m,
        "precision": precision,
        "recall": recall,
        "f_measure": f,
        "asynchrony_ms": asynchrony,
    }


def predicted_pulse_from_trace(trace_dir: Path) -> list[float]:
    """The rung-1 pulse stream: every transcript token's start time."""
    import json

    payload = json.loads((Path(trace_dir) / "whisper.json").read_text())
    return [float(w["start"]) for w in payload["words"]]


def _pooled(rows: list[ClipPulseScore]) -> dict | None:
    if not rows:
        return None
    n_ref = sum(r.n_ref for r in rows)
    n_pred = sum(r.n_pred for r in rows)
    matched = sum(r.matched for r in rows)
    asynchrony = [a for r in rows for a in r.asynchrony_ms]
    precision = matched / n_pred if n_pred else None
    recall = matched / n_ref if n_ref else None
    f = (
        2 * precision * recall / (precision + recall)
        if precision and recall else 0.0
    )
    per_clip_f = [r.f_measure for r in rows if r.f_measure is not None]
    a = np.array(asynchrony) if asynchrony else None
    return {
        "n_clips": len(rows),
        "n_ref": n_ref,
        "n_pred": n_pred,
        "matched": matched,
        "precision": _r3(precision),
        "recall": _r3(recall),
        "f_measure": _r3(f),
        "f_measure_macro": _r3(float(np.mean(per_clip_f))) if per_clip_f else None,
        "asynchrony_mean_ms": _r1(a.mean()) if a is not None else None,
        "asynchrony_median_ms": _r1(np.median(a)) if a is not None else None,
        "asynchrony_sd_ms": _r1(a.std()) if a is not None else None,
    }


def run_stage1(
    evals_root: Path, tol: float = F_MEASURE_TOLERANCE_S
) -> dict:
    """Score every case that has a beat grid; report missing grids loudly."""
    from musical_perception.annotation.grids import load_grids
    from musical_perception.evals.cases import load_cases

    evals_root = Path(evals_root)
    cases = load_cases(evals_root / "cases")
    grids = load_grids(evals_root / "grids")

    rows: list[ClipPulseScore] = []
    missing, errors = [], []
    for case in cases:
        grid = grids.get(case.id)
        if grid is None:
            missing.append(case.id)
            continue
        try:
            predicted = predicted_pulse_from_trace(evals_root / case.trace)
            scored = score_pulse(grid.beats, predicted, tol)
        except Exception as e:  # a broken row is reportable, not fatal
            errors.append(f"{case.id}: {type(e).__name__}: {e}")
            continue
        rows.append(ClipPulseScore(
            case_id=case.id,
            provisional=grid.provisional,
            count_style=case.tags.get("count_style"),
            n_ref=len(grid.beats),
            n_pred=len(predicted),
            **scored,
        ))

    provisional_rows = [r for r in rows if r.provisional]
    verified_rows = [r for r in rows if not r.provisional]
    styles = sorted({r.count_style for r in rows if r.count_style})
    return {
        "pulse_source": PULSE_SOURCE,
        "tolerance_s": tol,
        "n_cases": len(cases),
        "clips": [r.summary() for r in rows],
        # Provisional rows never gate and are always a separate slice
        # (charter rule 2); verified is empty until the owner's rung 1.5.
        "aggregate_provisional": _pooled(provisional_rows),
        "aggregate_verified": _pooled(verified_rows),
        "slices": {
            style: _pooled([r for r in rows if r.count_style == style])
            for style in styles
        },
        "missing_grids": missing,
        "errors": errors,
    }
