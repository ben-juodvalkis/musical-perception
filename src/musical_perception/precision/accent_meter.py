"""
Accent-periodicity meter votes.

KEEP — pure math over a beat sequence and an acoustic event stream.

Rung 3 (W2) of the ADR-016 reset. The question this module answers is
*grouping*: given a tactus (beat times) and the acoustic events around it,
how many beats make a bar, and which beat is the downbeat?

The method is a Parncutt/Povel-Essens salience clock. Per-beat salience is
built from three amplitude-free channels — agogic (interval lengthening),
density (events per beat), and voicing (Standing Lesson 6: silence is
evidence) — then a metrical-weight template for each candidate meter is
tiled over the sequence at every phase and correlated against it.

The templates carry the hierarchy that separates the confusable pairs:
4/4 differs from 2/4 only in its medium third beat, and 6/8 from 3/4 only
in its medium fourth. Everything this module can get wrong, it gets wrong
there.

Deliberate limitation: period 8 — the count-phrase level that pervades
this corpus — is not a hypothesis. An 8-periodic accent puts energy on
every harmonic of 1/8, so periodicity alone cannot separate it from 4/4
or 2/4; resolving it needs evidence from outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Metrical-weight templates, indexed by position within the bar.
# S=1.0 downbeat, m=0.5 secondary accent, w=0.0 weak.
METER_TEMPLATES: dict[str, list[float]] = {
    "2/4": [1.0, 0.0],
    "3/4": [1.0, 0.0, 0.0],
    "4/4": [1.0, 0.0, 0.5, 0.0],
    "6/8": [1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
}

# Salience channel weights. Equal thirds by default: no channel has earned
# priority over another on evidence yet, and inventing weights before
# measuring them is how a model launders its author's expectations.
DEFAULT_CHANNEL_WEIGHTS: dict[str, float] = {
    "agogic": 1.0,
    "density": 1.0,
    "voicing": 1.0,
}


@dataclass
class BeatSalience:
    """Per-beat salience, kept decomposed so a vote can be explained."""

    times: list[float]
    voiced: list[bool]
    agogic: list[float]
    density: list[float]
    combined: list[float]
    segment_ids: list[int]

    def __len__(self) -> int:
        return len(self.times)


@dataclass
class MeterVote:
    """One (meter, phase) hypothesis with its clock score."""

    meter: str
    period: int
    phase: int
    score: float  # correlation of template with salience, -1..1

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"MeterVote({self.meter} phase={self.phase} score={self.score:.3f})"


@dataclass
class AccentMeterResult:
    """Ranked meter votes plus the salience they were computed from."""

    votes: list[MeterVote]
    salience: BeatSalience
    abstained: bool
    reason: str | None = None
    channel_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CHANNEL_WEIGHTS)
    )

    @property
    def best(self) -> MeterVote | None:
        return self.votes[0] if self.votes else None

    @property
    def meter(self) -> str | None:
        """Top-voted meter, or None when the module abstains."""
        if self.abstained or not self.votes:
            return None
        return self.votes[0].meter

    @property
    def margin(self) -> float:
        """Score gap between the best meter and the best rival meter."""
        if len(self.votes) < 2:
            return 0.0
        top = self.votes[0].meter
        for v in self.votes[1:]:
            if v.meter != top:
                return self.votes[0].score - v.score
        return 0.0

    @property
    def confidence(self) -> float:
        """Margin squashed to 0..1. Not calibrated; a ranking aid only."""
        return float(np.clip(self.margin / 0.35, 0.0, 1.0))


def _segment_beats(
    times: list[float], free_time: list[tuple[float, float]]
) -> list[int]:
    """Assign each beat a segment id, cut at free-time regions.

    Phase is only continuous inside a segment: a stretch where the pulse
    stopped tells us nothing about how many beats went by.
    """
    if not free_time:
        return [0] * len(times)
    cuts = sorted(start for start, _ in free_time)
    ids = []
    for t in times:
        ids.append(sum(1 for c in cuts if c < t))
    return ids


def _local_median(values: list[float], index: int, half_window: int = 4) -> float:
    lo = max(0, index - half_window)
    hi = min(len(values), index + half_window + 1)
    window = values[lo:hi]
    return float(np.median(window)) if window else float("nan")


def beat_salience(
    beats: list[float],
    events: list[float] | None = None,
    *,
    voiced_flags: list[bool] | None = None,
    free_time: list[tuple[float, float]] | None = None,
    voiced_tolerance_s: float = 0.12,
    channel_weights: dict[str, float] | None = None,
) -> BeatSalience:
    """Build the per-beat salience vector.

    Args:
        beats: beat times in seconds, ascending. Include silent beats —
            their absence of voicing is one of the three channels.
        events: acoustic event times (rung-2 pulse extractor output).
        voiced_flags: explicit voicing per beat; derived from ``events``
            when omitted.
        free_time: (start, end) spans where the pulse stopped.
        voiced_tolerance_s: how near an event must fall to count a beat
            as voiced.
        channel_weights: overrides for the three channel weights.
    """
    weights = dict(DEFAULT_CHANNEL_WEIGHTS)
    if channel_weights:
        weights.update(channel_weights)
    events = sorted(events or [])
    n = len(beats)
    segments = _segment_beats(beats, free_time or [])

    # --- agogic: following IOI vs the local median, in relative terms.
    iois: list[float] = []
    for i in range(n):
        if i + 1 < n and segments[i + 1] == segments[i]:
            iois.append(beats[i + 1] - beats[i])
        else:
            iois.append(float("nan"))
    agogic: list[float] = []
    finite = [v for v in iois if np.isfinite(v)]
    for i, ioi in enumerate(iois):
        if not np.isfinite(ioi) or not finite:
            agogic.append(0.0)
            continue
        # Local window in the ORIGINAL index space, then drop the holes —
        # filtering first and reusing i would silently slide the window.
        lo, hi = max(0, i - 4), min(n, i + 5)
        window = [v for v in iois[lo:hi] if np.isfinite(v)]
        med = float(np.median(window)) if window else float(np.median(finite))
        agogic.append((ioi - med) / med if med > 0 else 0.0)

    # --- density: events inside each beat's interval.
    density: list[float] = []
    for i in range(n):
        start = beats[i]
        if i + 1 < n and segments[i + 1] == segments[i]:
            end = beats[i + 1]
        else:
            end = start + (float(np.median(finite)) if finite else 0.5)
        density.append(float(sum(1 for e in events if start <= e < end)))

    # --- voicing: is there an event at the beat itself?
    if voiced_flags is not None:
        voiced = list(voiced_flags)
    else:
        voiced = []
        for t in beats:
            voiced.append(
                any(abs(e - t) <= voiced_tolerance_s for e in events) if events else True
            )

    combined = _combine(agogic, density, voiced, weights)
    return BeatSalience(
        times=list(beats),
        voiced=voiced,
        agogic=agogic,
        density=density,
        combined=combined,
        segment_ids=segments,
    )


def _z(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    sd = arr.std()
    if sd <= 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / sd


def _combine(
    agogic: list[float],
    density: list[float],
    voiced: list[bool],
    weights: dict[str, float],
) -> list[float]:
    parts = (
        weights["agogic"] * _z(agogic)
        + weights["density"] * _z(density)
        + weights["voicing"] * _z([1.0 if v else 0.0 for v in voiced])
    )
    total = sum(weights.values()) or 1.0
    return list(parts / total)


def _clock_score(salience: np.ndarray, segments: np.ndarray, template: list[float],
                 phase: int) -> float:
    """Correlate a phase-shifted metrical template against the salience.

    Phase is counted from the first beat of each segment independently, so
    a free-time break does not smear the grid.
    """
    period = len(template)
    weights = np.empty(len(salience))
    for seg in np.unique(segments):
        idx = np.flatnonzero(segments == seg)
        for k, i in enumerate(idx):
            weights[i] = template[(k - phase) % period]
    if weights.std() <= 1e-9 or salience.std() <= 1e-9:
        return 0.0
    return float(np.corrcoef(weights, salience)[0, 1])


def meter_votes(
    salience: BeatSalience,
    *,
    meters: list[str] | None = None,
    min_beats: int = 6,
    min_margin: float = 0.05,
) -> AccentMeterResult:
    """Score every (meter, phase) hypothesis against the salience.

    Returns votes sorted best-first. Abstains when the sequence is too
    short to carry periodic evidence, or when the top two *meters* are
    separated by less than ``min_margin`` — a tie between metres is not a
    weak answer, it is no answer (ADR-015: abstention is designed
    behaviour, never punished as wrong).
    """
    meters = meters or list(METER_TEMPLATES)
    sal = np.asarray(salience.combined, dtype=float)
    segs = np.asarray(salience.segment_ids, dtype=int)

    if len(sal) < min_beats:
        return AccentMeterResult(
            votes=[],
            salience=salience,
            abstained=True,
            reason=f"only {len(sal)} beats, need {min_beats}",
        )

    votes: list[MeterVote] = []
    for name in meters:
        template = METER_TEMPLATES[name]
        for phase in range(len(template)):
            votes.append(
                MeterVote(
                    meter=name,
                    period=len(template),
                    phase=phase,
                    score=_clock_score(sal, segs, template, phase),
                )
            )
    votes.sort(key=lambda v: v.score, reverse=True)

    result = AccentMeterResult(votes=votes, salience=salience, abstained=False)
    if result.margin < min_margin:
        runner = next(
            (v.meter for v in votes[1:] if v.meter != votes[0].meter), "?"
        )
        return AccentMeterResult(
            votes=votes,
            salience=salience,
            abstained=True,
            reason=(
                f"margin {result.margin:.3f} < {min_margin} "
                f"({votes[0].meter} vs {runner})"
            ),
        )
    return result


def analyze_accent_meter(
    beats: list[float],
    events: list[float] | None = None,
    *,
    voiced_flags: list[bool] | None = None,
    free_time: list[tuple[float, float]] | None = None,
    channel_weights: dict[str, float] | None = None,
    min_margin: float = 0.05,
) -> AccentMeterResult:
    """Beat times in, ranked meter votes out. The module's front door."""
    salience = beat_salience(
        beats,
        events,
        voiced_flags=voiced_flags,
        free_time=free_time,
        channel_weights=channel_weights,
    )
    return meter_votes(salience, min_margin=min_margin)
