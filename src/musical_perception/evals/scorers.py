"""
Scorer library — one set of comparators shared by every eval tier.

Scoring discipline (ADR-009, Vision 08 §8.3):
- Every score is correct / wrong / **abstained** — a pipeline that returns
  None declined to commit, and abstention is never counted as wrong.
- Score musically, not with ==: exact ×2/×3/÷2/÷3 answers are metric-level
  failures (a different defect than a wrong tempo); musically equivalent
  (meter, bpm, subdivision) triples earn partial credit.
"""

import unicodedata
from dataclasses import dataclass, field

from musical_perception.types import (
    ExerciseDetectionResult,
    Meter,
    NormalizedTempo,
    PhraseStructure,
    QualityProfile,
    TempoCandidate,
)

CORRECT = "correct"
WRONG = "wrong"
ABSTAINED = "abstained"


@dataclass
class ScoreResult:
    """Outcome of scoring one expected field of one case."""
    field: str
    outcome: str                 # correct | wrong | abstained
    credit: float                # 1.0, partial (0.5), or 0.0
    predicted: object | None
    expected: object
    confidence: float | None = None   # None → excluded from calibration metrics
    failure_mode: str | None = None
    detail: str = ""
    # ADR-014, informational only (never gates, never changes `outcome`):
    # on a wrong tempo, did the expected answer appear anywhere in the
    # reported metric-level family? None → not measured for this row.
    truth_in_family: bool | None = None


@dataclass
class CaseResult:
    """All scores for one case, plus its tags for slicing."""
    case_id: str
    tags: dict = field(default_factory=dict)
    scores: list[ScoreResult] = field(default_factory=list)
    error: str | None = None
    # W1.5: agent-proposed truth. Provisional rows are scored and reported,
    # but they leave every headline aggregate and every gate alone.
    provisional: bool = False


_METRIC_LEVELS = [(2.0, "metric_level_x2"), (3.0, "metric_level_x3"),
                  (0.5, "metric_level_div2"), (1 / 3, "metric_level_div3")]


def _truth_in_family(
    predicted_bpm: float,
    expected_bpm: float,
    alternates: list[TempoCandidate] | None,
    rel_tol: float,
) -> tuple[bool | None, str]:
    """Did the expected BPM survive anywhere in the reported family?

    ADR-014 measure, deliberately non-gating: a wrong primary whose family
    still contains the truth is a *selection* failure, a wrong primary
    whose family does not is a *measurement* failure. Returns (flag, note);
    flag is None when the caller reported no family to check.
    """
    if alternates is None:
        return None, ""
    family = [predicted_bpm] + [c.bpm for c in alternates]
    hit = next(
        (b for b in family if abs(b - expected_bpm) / expected_bpm <= rel_tol),
        None,
    )
    if hit is None:
        return False, f"truth absent from family {[round(b, 1) for b in family]}"
    return True, f"truth in family as {hit:g}"


def score_tempo(
    predicted_bpm: float | None,
    expected_bpm: float,
    *,
    rel_tol: float = 0.08,
    confidence: float | None = None,
    alternates: list[TempoCandidate] | None = None,
) -> ScoreResult:
    """Relative-error tempo score with explicit metric-level classification.

    An answer that is exactly ×2/×3/÷2/÷3 off (within tolerance) is a
    metric_level failure, not a generic tempo error — that distinction is
    the subject of ADR-006/007 and tells you which module to fix.

    `alternates` (the primary's metric-level family, ADR-014) only annotates
    wrong answers with `truth_in_family`. It never changes an outcome:
    scoring policy stays a Vision 08 §8.3 question, not a side effect of
    reporting the family.
    """
    if predicted_bpm is None:
        return ScoreResult("tempo", ABSTAINED, 0.0, None, expected_bpm)

    rel_err = abs(predicted_bpm - expected_bpm) / expected_bpm
    if rel_err <= rel_tol:
        return ScoreResult(
            "tempo", CORRECT, 1.0, predicted_bpm, expected_bpm,
            confidence=confidence, detail=f"rel_err={rel_err:.3f}",
        )

    in_family, note = _truth_in_family(
        predicted_bpm, expected_bpm, alternates, rel_tol
    )

    for ratio, mode in _METRIC_LEVELS:
        target = expected_bpm * ratio
        if abs(predicted_bpm - target) / target <= rel_tol:
            return ScoreResult(
                "tempo", WRONG, 0.0, predicted_bpm, expected_bpm,
                confidence=confidence, failure_mode=mode,
                detail=f"answer sits at {ratio:g}× the expected tempo"
                       + (f"; {note}" if note else ""),
                truth_in_family=in_family,
            )

    return ScoreResult(
        "tempo", WRONG, 0.0, predicted_bpm, expected_bpm,
        confidence=confidence, failure_mode="tempo_error",
        detail=f"rel_err={rel_err:.3f}" + (f"; {note}" if note else ""),
        truth_in_family=in_family,
    )


_SUBDIVISION_FACTOR = {"none": 1, "duple": 2, "triplet": 3}


def _surface(meter: Meter, bpm: float, subdivision: str) -> tuple[float, int]:
    """Canonical rhythmic surface: (onset rate per minute, grouping).

    3/4 @120 none and 4/4 @40 triplet both produce 120 onsets/min grouped in
    threes — identical sound, and the accompanist does not care (ADR-007).
    """
    factor = _SUBDIVISION_FACTOR.get(subdivision or "none", 1)
    rate = bpm * factor
    group = meter.beats_per_measure if factor == 1 else factor
    return rate, group


def score_meter_triple(
    predicted: NormalizedTempo | None,
    expected_meter: Meter,
    expected_bpm: float | None = None,
    expected_subdivision: str | None = None,
    *,
    rel_tol: float = 0.08,
) -> ScoreResult:
    """Score the coherent (meter, bpm, subdivision) triple as one item."""
    expected_repr = (
        f"{expected_meter.beats_per_measure}/{expected_meter.beat_unit}"
        + (f" @{expected_bpm:g}" if expected_bpm else "")
        + (f" {expected_subdivision}" if expected_subdivision else "")
    )
    if predicted is None:
        return ScoreResult("meter_triple", ABSTAINED, 0.0, None, expected_repr)

    predicted_repr = (
        f"{predicted.meter.beats_per_measure}/{predicted.meter.beat_unit} "
        f"@{predicted.bpm:g} {predicted.subdivision}"
    )
    confidence = predicted.confidence

    meter_ok = (
        predicted.meter.beats_per_measure == expected_meter.beats_per_measure
        and predicted.meter.beat_unit == expected_meter.beat_unit
    )
    bpm_ok = (
        expected_bpm is None
        or abs(predicted.bpm - expected_bpm) / expected_bpm <= rel_tol
    )
    sub_ok = (
        expected_subdivision is None
        or predicted.subdivision == expected_subdivision
    )
    if meter_ok and bpm_ok and sub_ok:
        return ScoreResult(
            "meter_triple", CORRECT, 1.0, predicted_repr, expected_repr,
            confidence=confidence,
        )

    # Musically equivalent reading → partial credit (needs a full expected
    # triple to canonicalize both sides).
    if expected_bpm is not None and expected_subdivision is not None:
        exp_rate, exp_group = _surface(expected_meter, expected_bpm, expected_subdivision)
        pred_rate, pred_group = _surface(predicted.meter, predicted.bpm, predicted.subdivision)
        if pred_group == exp_group and abs(pred_rate - exp_rate) / exp_rate <= rel_tol:
            return ScoreResult(
                "meter_triple", WRONG, 0.5, predicted_repr, expected_repr,
                confidence=confidence, failure_mode="equivalent_reading",
                detail="same rhythmic surface, different notation",
            )

    mode = "meter_wrong" if not meter_ok else ("tempo_wrong" if not bpm_ok else "subdivision_wrong")
    return ScoreResult(
        "meter_triple", WRONG, 0.0, predicted_repr, expected_repr,
        confidence=confidence, failure_mode=mode,
    )


def score_counts(predicted: PhraseStructure | None, expected_counts: int) -> ScoreResult:
    if predicted is None or predicted.counts is None:
        return ScoreResult("counts", ABSTAINED, 0.0, None, expected_counts)
    ok = predicted.counts == expected_counts
    return ScoreResult(
        "counts", CORRECT if ok else WRONG, 1.0 if ok else 0.0,
        predicted.counts, expected_counts,
        failure_mode=None if ok else "counts_wrong",
    )


def score_sides(predicted: PhraseStructure | None, expected_sides: int) -> ScoreResult:
    if predicted is None:
        return ScoreResult("sides", ABSTAINED, 0.0, None, expected_sides)
    ok = predicted.sides == expected_sides
    return ScoreResult(
        "sides", CORRECT if ok else WRONG, 1.0 if ok else 0.0,
        predicted.sides, expected_sides,
        failure_mode=None if ok else "sides_wrong",
    )


# Gemini's snake_case is prompt-defined, not guaranteed — normalize spellings.
_SLOT_ALIASES = {
    "grande_battement": "grand_battement",
    "grand_battements": "grand_battement",
    "battement_tendu": "tendu",
    "battement_frappe": "frappe",
    "battement_fondu": "fondu",
    "plies": "plie",
    "grand_plie": "plie",
    "demi_plie": "plie",
    "releves": "releve",
}


def canonical_slot(name: str) -> str:
    """Lowercase, ASCII-fold, snake_case, and alias-map an exercise name."""
    folded = unicodedata.normalize("NFD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    slug = folded.strip().lower().replace(" ", "_").replace("-", "_")
    return _SLOT_ALIASES.get(slug, slug)


def score_slot(
    predicted: ExerciseDetectionResult | None, expected_slot: str
) -> ScoreResult:
    """Top-1 exercise identification. 'unknown' counts as abstention."""
    expected_canon = canonical_slot(expected_slot)
    if predicted is None or not predicted.primary_exercise:
        return ScoreResult("slot", ABSTAINED, 0.0, None, expected_canon)
    predicted_canon = canonical_slot(predicted.primary_exercise)
    if predicted_canon == "unknown":
        return ScoreResult(
            "slot", ABSTAINED, 0.0, "unknown", expected_canon,
            confidence=predicted.confidence,
        )
    ok = predicted_canon == expected_canon
    return ScoreResult(
        "slot", CORRECT if ok else WRONG, 1.0 if ok else 0.0,
        predicted_canon, expected_canon,
        confidence=predicted.confidence,
        failure_mode=None if ok else "slot_wrong",
    )


_QUALITY_DIMS = ("articulation", "weight", "energy")


def score_quality(
    predicted: QualityProfile | None,
    expected: dict[str, float],
    *,
    tol: float = 0.2,
) -> list[ScoreResult]:
    """Per-dimension quality score: within ±tol counts as a hit.

    Absolute values from a model are weak evidence; corpus-level ordering
    (Spearman, in aggregate.py) is what the engine actually consumes.
    """
    results = []
    for dim in _QUALITY_DIMS:
        if dim not in expected:
            continue
        exp = expected[dim]
        if predicted is None:
            results.append(ScoreResult(f"quality_{dim}", ABSTAINED, 0.0, None, exp))
            continue
        pred = getattr(predicted, dim)
        err = abs(pred - exp)
        ok = err <= tol
        results.append(ScoreResult(
            f"quality_{dim}", CORRECT if ok else WRONG, 1.0 if ok else 0.0,
            pred, exp, failure_mode=None if ok else "quality_off",
            detail=f"abs_err={err:.2f}",
        ))
    return results
