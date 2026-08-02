"""
Phrase-structure estimation from evidence fusion.

KEEP — pure math. Owns `structure.counts` the way interpret_meter owns
tempo (ADR-007's discipline): one committed answer fused from several
weak signals, or an explicit abstention when the evidence disagrees.

Two regimes, detected from the markers themselves:

1. Numeric counting ("one two three…"): phrase length is read from the
   spoken count cycle — the value the count reaches before restarting.
2. Step-name marking ("front… tombe… coupé…"): no numbers to read, so
   independent estimators (Gemini's two structural reads, the marker
   tally, the marked span crossed with each tempo hypothesis) are
   snapped to musical phrase lengths and must agree; two concurring
   votes commit an answer, a tie or a lone voice abstains.

Abstention is deliberate product policy: a wrong phrase length ends a
combination in the wrong place; "ask the teacher" beats guessing.
"""

from dataclasses import dataclass, field

from musical_perception.types import MarkerType, TimedMarker

# Musically plausible phrase lengths (multiples of the 8-count, plus the
# 6/12 shapes 3/4 phrases produce).
_SNAP_GRID = (6, 8, 12, 16, 24, 32, 48, 64, 96)

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
}

_SUBDIVISION_FACTOR = {"none": 1, "duple": 2, "triplet": 3}


@dataclass
class CountsEstimate:
    """One committed counts answer (or an abstention) plus its evidence."""
    counts: int | None            # None = abstained
    method: str                   # "counting" | "vote" | "abstain"
    votes: dict = field(default_factory=dict)   # snapped value -> [signal names]
    detail: str = ""


def _normalize(word: str) -> str:
    return word.strip().lower().strip(".,!?;:")


def _snap(value: float) -> int:
    """Nearest musical phrase length; ties break to the smaller (a shorter
    phrase is the safer commitment — under-playing is recoverable)."""
    return min(_SNAP_GRID, key=lambda s: (abs(s - value), s))


def _counting_cycle(numbers: list[int]) -> int | None:
    """Phrase length from spoken count restarts.

    "1..8, 1..8" → cycles of 8. Without a restart, the maximum reached.
    Inconsistent cycle maxima → None (mixed speech misread as counting).
    """
    if not numbers:
        return None
    maxima, current_max = [], numbers[0]
    for prev, cur in zip(numbers, numbers[1:]):
        if cur <= prev:  # restart ("…eight, one…")
            maxima.append(current_max)
            current_max = cur
        else:
            current_max = max(current_max, cur)
    if not maxima:                      # never restarted
        return current_max
    maxima.append(current_max)
    completed = maxima[:-1]             # last cycle may be cut off mid-way
    if len(set(completed)) == 1:
        return completed[0]
    return None


def estimate_counts(
    markers: list[TimedMarker],
    *,
    bpm: float | None = None,
    gemini_bpm: float | None = None,
    subdivision: str | None = None,
    gemini_counts: int | None = None,
    gemini_total_counts: int | None = None,
) -> CountsEstimate:
    """
    Estimate the phrase length in counts from all available evidence.

    Args:
        markers: Timed markers from the Gemini/Whisper merge.
        bpm: Normalized tempo (onset-backed) if available.
        gemini_bpm: Gemini's independent tempo estimate
            (counting_structure.estimated_bpm) — kept separate because on
            step-name clips it can sit at a different metric level than
            the onset reading, and that disagreement is information.
        subdivision: Gemini's observed counting subdivision.
        gemini_counts: Gemini's structure.counts read.
        gemini_total_counts: Gemini's counting_structure.total_counts.
    """
    beats = [m for m in markers if m.marker_type == MarkerType.BEAT]

    # --- Regime 1: numeric counting ---
    numbered = [
        (m.timestamp, _NUMBER_WORDS[_normalize(m.raw_word)])
        for m in beats
        if _normalize(m.raw_word) in _NUMBER_WORDS
    ]
    if len(numbered) >= 4 and len(numbered) >= 0.6 * len(beats):
        cycle = _counting_cycle([n for _, n in numbered])
        if cycle is not None and cycle >= 4:
            return CountsEstimate(
                counts=cycle, method="counting",
                detail=f"{len(numbered)} count words, cycle of {cycle}",
            )

    # --- Regime 2: step-name marking — snapped votes must agree ---
    votes: dict[int, list[str]] = {}

    def cast(signal: str, value: float | None):
        if value is not None and value > 0:
            votes.setdefault(_snap(value), []).append(signal)

    cast("gemini_structure", gemini_counts)
    cast("gemini_total_counts", gemini_total_counts)

    factor = _SUBDIVISION_FACTOR.get(subdivision or "none", 1)
    if beats:
        cast("marker_tally", len(beats) * factor)

    span = beats[-1].timestamp - beats[0].timestamp if len(beats) >= 2 else 0.0
    if span > 0:
        # One span vote per *distinct* tempo hypothesis: when the two BPM
        # readings agree, casting both would fake independent agreement.
        if bpm:
            cast("span_x_bpm", span * bpm / 60.0 + 1)
        if gemini_bpm and (not bpm or abs(gemini_bpm - bpm) / bpm > 0.05):
            cast("span_x_gemini_bpm", span * gemini_bpm / 60.0 + 1)

    if not votes:
        return CountsEstimate(counts=None, method="abstain", detail="no evidence")

    ranked = sorted(votes.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    top_value, top_signals = ranked[0]
    runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
    if len(top_signals) >= 2 and len(top_signals) > runner_up:
        return CountsEstimate(
            counts=top_value, method="vote", votes=votes,
            detail=f"{len(top_signals)} signals agree on {top_value}",
        )
    return CountsEstimate(
        counts=None, method="abstain", votes=votes,
        detail="no two independent signals agree",
    )
