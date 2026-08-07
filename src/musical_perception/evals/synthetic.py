"""
Tier 0 — synthetic timelines for the precision (KEEP) layer.

A fixture builder emits word/marker timelines from a known
(meter, bpm, subdivision) triple with controlled corruption — the ADR-007
failure modes — and the suite asserts recovery within tolerance across a
sweep, never on one corrupted example (ADR-009 tier 0).
"""

from dataclasses import dataclass, field

import numpy as np

from musical_perception.evals.scorers import (
    CaseResult,
    score_meter_triple,
    score_tempo,
)
from musical_perception.types import MarkerType, Meter, TimedMarker, TimestampedWord

_NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
_EXPLANATION = ["so", "the", "arm", "stays", "long", "through", "the", "side"]
_WORD_LEN = 0.15
_EXPLANATION_GAP = 0.22  # syllable-rate speech, not on the pulse


def build_timeline(
    meter: Meter,
    bpm: float,
    subdivision: str,
    counts: int,
    *,
    jitter_sd: float = 0.0,
    drop_rate: float = 0.0,
    interleaved_explanation: bool = False,
    prep_counts: int = 0,
    half_tempo_marking: bool = False,
    swing: float = 0.0,
    seed: int = 0,
) -> tuple[list[TimestampedWord], list[TimedMarker]]:
    """
    Synthesize the spoken timeline of a counted marking.

    Returns (words, markers): every spoken word with timestamps, and the
    ground-truth markers a perfect classifier would produce (explanation
    words are non-markers and appear only in `words`).
    """
    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    spoken_period = period * (2 if half_tempo_marking else 1)

    events = []  # (t, word, marker_type, beat_number)

    t = 0.0
    if prep_counts:
        prep_words = _NUMBERS[8 - prep_counts:]
        for i, w in enumerate(prep_words):
            events.append((t, w, MarkerType.BEAT, i + (8 - prep_counts) + 1))
            t += spoken_period

    explain_after = counts // 2 if interleaved_explanation else None
    for i in range(counts):
        if explain_after is not None and i == explain_after:
            for w in _EXPLANATION:
                events.append((t, w, None, None))
                t += _EXPLANATION_GAP
        beat_number = i + 1
        events.append((t, _NUMBERS[i % 8], MarkerType.BEAT, beat_number))
        # swing pushes subdivision syllables late (real "ONE-and-ah" counting
        # is never even) — beats stay on the grid, syllables drift, and the
        # onset detector's IOI histogram locks off-level. The rig-waltz shape.
        if subdivision == "duple":
            events.append((t + spoken_period * (0.5 + swing), "and",
                           MarkerType.AND, beat_number))
        elif subdivision == "triplet":
            events.append((t + spoken_period * (1 / 3 + swing), "and",
                           MarkerType.AND, beat_number))
            events.append((t + spoken_period * (2 / 3 + swing / 2), "ah",
                           MarkerType.AH, beat_number))
        t += spoken_period

    if jitter_sd:
        events = [
            (max(0.0, et + rng.normal(0.0, jitter_sd * period)), w, m, b)
            for et, w, m, b in events
        ]
    if drop_rate:
        kept, beats_kept = [], 0
        for ev in events:
            if ev[2] == MarkerType.BEAT and rng.random() < drop_rate and beats_kept >= 4:
                continue
            if ev[2] == MarkerType.BEAT:
                beats_kept += 1
            kept.append(ev)
        events = kept

    events.sort(key=lambda ev: ev[0])
    words = [
        TimestampedWord(w, round(et, 3), round(et + _WORD_LEN, 3))
        for et, w, _, _ in events
    ]
    markers = [
        TimedMarker(marker_type=m, beat_number=b, timestamp=round(et, 3), raw_word=w)
        for et, w, m, b in events
        if m is not None
    ]
    return words, markers


@dataclass
class SyntheticCase:
    """One tier-0 combo: a known triple plus corruption knobs."""
    id: str
    meter: Meter
    bpm: float           # true beat rate the pipeline should recover
    subdivision: str
    counts: int = 16
    knobs: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)

    def __post_init__(self):
        self.tags = {
            "source": "synthetic",
            "count_style": "numbers",
            "corruption": self.tags.get("corruption", "clean"),
            **self.tags,
        }


def _case(id_, beats, unit, bpm, sub, corruption="clean", **knobs):
    return SyntheticCase(
        id=id_, meter=Meter(beats, unit), bpm=bpm, subdivision=sub,
        knobs=knobs, tags={"corruption": corruption},
    )


SUITE: list[SyntheticCase] = [
    # Clean sweep — meters × plain/duple counting
    _case("t0-2-4-clean", 2, 4, 120, "none"),
    _case("t0-3-4-clean", 3, 4, 90, "none"),
    _case("t0-4-4-clean", 4, 4, 104, "none"),
    _case("t0-6-8-clean", 6, 8, 100, "none"),
    _case("t0-2-4-clean-duple", 2, 4, 120, "duple"),
    _case("t0-3-4-clean-duple", 3, 4, 90, "duple"),
    _case("t0-4-4-clean-duple", 4, 4, 104, "duple"),
    _case("t0-6-8-clean-duple", 6, 8, 100, "duple"),
    # Triplet subdivision
    _case("t0-4-4-clean-triplet", 4, 4, 80, "triplet"),
    _case("t0-2-4-clean-triplet", 2, 4, 96, "triplet"),
    # Timing jitter (sloppy but honest counting)
    _case("t0-3-4-jitter", 3, 4, 90, "none", "jitter", jitter_sd=0.05, seed=1),
    _case("t0-4-4-jitter", 4, 4, 104, "none", "jitter", jitter_sd=0.05, seed=2),
    _case("t0-4-4-jitter-duple", 4, 4, 104, "duple", "jitter", jitter_sd=0.05, seed=3),
    _case("t0-2-4-jitter", 2, 4, 120, "none", "jitter", jitter_sd=0.05, seed=4),
    # Dropped counts (ASR misses words)
    _case("t0-4-4-dropped", 4, 4, 104, "none", "dropped", drop_rate=0.15, seed=5),
    _case("t0-3-4-dropped-duple", 3, 4, 90, "duple", "dropped", drop_rate=0.15, seed=6),
    # Interleaved explanation (the grande-battement failure mode)
    _case("t0-4-4-explained", 4, 4, 104, "none", "interleaved",
          interleaved_explanation=True),
    _case("t0-3-4-explained", 3, 4, 90, "none", "interleaved",
          interleaved_explanation=True),
    _case("t0-4-4-explained-duple", 4, 4, 126, "duple", "interleaved",
          interleaved_explanation=True),
    # Prep counts before beat 1
    _case("t0-4-4-prep", 4, 4, 104, "none", "prep", prep_counts=4),
    _case("t0-2-4-prep", 2, 4, 120, "none", "prep", prep_counts=4),
    # Half-tempo marking (teacher speaks at half speed; ADR-006 territory)
    _case("t0-4-4-half", 4, 4, 104, "none", "half_tempo", half_tempo_marking=True),
    _case("t0-3-4-half", 3, 4, 96, "none", "half_tempo", half_tempo_marking=True),
    # Combined stress
    _case("t0-4-4-stress", 4, 4, 104, "duple", "stress",
          jitter_sd=0.04, drop_rate=0.1, seed=7),
    # Swung triplet counting (the rig-waltz shape, ADR-013): syllables
    # drift late so onsets read off-level; beat markers carry the truth.
    _case("t0-3-4-swing-triplet", 3, 4, 90, "triplet", "swing", swing=0.12),
]


def run_synthetic_case(case: SyntheticCase) -> CaseResult:
    """Run one combo through the precision chain and score recovery.

    Meter comes from a simulated perfect Gemini observation — tier 0
    measures the KEEP layer (tempo math, subdivision, interpret_meter),
    not the perception model.
    """
    from musical_perception.precision.rhythm import detect_onset_tempo
    from musical_perception.precision.subdivision import analyze_subdivisions
    from musical_perception.precision.tempo import calculate_tempo, interpret_meter

    words, markers = build_timeline(
        case.meter, case.bpm, case.subdivision, case.counts, **case.knobs
    )
    onset_tempo = detect_onset_tempo(words)
    beat_ts = [m.timestamp for m in markers if m.marker_type == MarkerType.BEAT]
    marker_tempo = calculate_tempo(beat_ts)
    _ = analyze_subdivisions(markers)  # exercised for crashes; triple carries the score

    normalized = interpret_meter(
        onset_tempo=onset_tempo,
        gemini_tempo=marker_tempo,
        gemini_meter=case.meter,
        gemini_subdivision=case.subdivision,
    )

    scores = [
        score_tempo(
            normalized.bpm if normalized else None, case.bpm,
            confidence=normalized.confidence if normalized else None,
        ),
        score_meter_triple(normalized, case.meter, case.bpm, case.subdivision),
    ]
    return CaseResult(case_id=case.id, tags=dict(case.tags), scores=scores)


def run_suite() -> list[CaseResult]:
    """Run every tier-0 combo. Deterministic: seeds are fixed per case."""
    return [run_synthetic_case(c) for c in SUITE]
