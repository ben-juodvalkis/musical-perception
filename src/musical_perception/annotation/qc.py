"""
Beat-grid QC checks (rung 2.5) — `docs/evals/annotation-convention.md` §4
as amended by the owner 2026-08-14.

Three checks run on every grid before it is trusted:

- **BPM vs case label** (ratified at rung 1.5): grid-implied BPM against
  the case's `marking_bpm`, flagged past 4%. It is the weakest of the
  three — it *false-passed* at +3.51% on a grid carrying three spurious
  labels and a missing beat.
- **Minimum IOI** (amendment): an interval far shorter than the clip's
  own beat is the double-mark signature that the BPM check hides.
- **Within-phrase IOI spread** (amendment): IOI variance measured
  *inside* phrases, which per-clip medians hide — on
  `rig-names-4-4-96-allegro` the whole-clip CV is 44.1% and the
  within-phrase CV 9.1%.

Both amendment checks are **suppressed inside tagged regions** (rung
2.5's `GridRegion`): a gap the annotator explained as silent-beat,
free-time, or excluded-explanation material is not evidence of an error.
Before rung 2.5 there was no way to say so in the file, which is why the
checks had to be run ad hoc.

Thresholds are frozen constants, pre-registered before the checks were
ever run (ledger 2026-08-14, rung M / W1). A misfiring threshold is a
reportable finding proposed for owner ratification, never a quiet edit.
"""

from dataclasses import dataclass, field
from statistics import median, pstdev

from musical_perception.annotation.grids import BeatGrid

MIN_IOI_RATIO = 0.5        # flag IOI < this × the clip's median IOI
PHRASE_BREAK_RATIO = 1.75  # IOI above this × median is a break, not a stretch
MAX_PHRASE_CV = 0.15       # flag within-phrase IOI CV above this
MIN_PHRASE_IOIS = 3        # a CV below this many intervals means nothing
BPM_TOLERANCE = 0.04       # the ratified 4% window (Standing Lesson 7)

CHECKS = ("bpm_vs_label", "min_ioi", "ioi_spread")


@dataclass
class Interval:
    """One inter-beat interval and why it may not count."""
    start: float
    end: float
    seconds: float
    suppressed_by: str | None = None   # region kind, when a tag explains it

    @property
    def counts(self) -> bool:
        return self.suppressed_by is None


@dataclass
class QCFinding:
    """One flagged observation. Severity is the owner's to judge."""
    clip: str
    check: str
    message: str
    times: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        return f"[{self.check}] {self.message}"


@dataclass
class ClipQC:
    """Every number the three checks computed, plus what they flagged."""
    clip: str
    provisional: bool
    n_beats: int
    n_intervals: int
    n_suppressed: int
    n_phrases: int
    median_ioi: float | None
    bpm_whole: float | None
    bpm_within_phrase: float | None
    max_phrase_cv: float | None
    min_ioi_ratio: float | None
    marking_bpm: float | None
    bpm_delta_pct: float | None
    findings: list[QCFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings


def intervals(grid: BeatGrid) -> list[Interval]:
    """Consecutive beat intervals, tagged with any region that explains them."""
    out = []
    for t0, t1 in zip(grid.beats, grid.beats[1:]):
        hit = next((r for r in grid.regions if r.overlaps(t0, t1)), None)
        out.append(Interval(t0, t1, round(t1 - t0, 4), hit.kind if hit else None))
    return out


def phrases(counted: list[Interval], median_ioi: float) -> list[list[Interval]]:
    """Maximal runs of counted intervals, split at breaks.

    A phrase ends at a suppressed interval (the tag says the pulse was not
    continuous there) or at an interval longer than PHRASE_BREAK_RATIO ×
    the clip median — long enough to be a break rather than the agogic or
    phrase-final stretch that ruling (f) tells the annotator to preserve.
    """
    grouped: list[list[Interval]] = []
    current: list[Interval] = []
    for iv in counted:
        if not iv.counts or iv.seconds > PHRASE_BREAK_RATIO * median_ioi:
            if current:
                grouped.append(current)
            current = []
            continue
        current.append(iv)
    if current:
        grouped.append(current)
    return grouped


def cv(values: list[float]) -> float:
    """Population CV — the definition the rung-1.5 numbers were computed with."""
    mean = sum(values) / len(values)
    return pstdev(values) / mean if mean else 0.0


def run_qc(grid: BeatGrid, marking_bpm: float | None = None) -> ClipQC:
    """The three ratified checks over one grid. Pure function of the grid."""
    all_ivs = intervals(grid)
    counted = [iv for iv in all_ivs if iv.counts]
    result = ClipQC(
        clip=grid.clip,
        provisional=grid.provisional,
        n_beats=len(grid.beats),
        n_intervals=len(all_ivs),
        n_suppressed=len(all_ivs) - len(counted),
        n_phrases=0,
        median_ioi=None,
        bpm_whole=None,
        bpm_within_phrase=None,
        max_phrase_cv=None,
        min_ioi_ratio=None,
        marking_bpm=marking_bpm,
        bpm_delta_pct=None,
    )
    if not counted:
        return result

    med = median(iv.seconds for iv in counted)
    result.median_ioi = round(med, 4)
    result.bpm_whole = round(60.0 / med, 2)

    grouped = phrases(counted, med)
    result.n_phrases = len(grouped)
    in_phrase = [iv.seconds for run in grouped for iv in run]
    if in_phrase:
        result.bpm_within_phrase = round(60.0 / median(in_phrase), 2)

    # --- check 1: BPM vs the case label (ratified at rung 1.5) ---------
    if marking_bpm:
        delta = (result.bpm_whole - marking_bpm) / marking_bpm
        result.bpm_delta_pct = round(delta * 100, 2)
        if abs(delta) > BPM_TOLERANCE:
            result.findings.append(QCFinding(
                grid.clip, "bpm_vs_label",
                f"grid-implied {result.bpm_whole} BPM vs label {marking_bpm} "
                f"({result.bpm_delta_pct:+.2f}%, over {BPM_TOLERANCE:.0%})",
            ))

    # --- check 2: minimum IOI (the double-mark signature) --------------
    shortest = min(counted, key=lambda iv: iv.seconds)
    result.min_ioi_ratio = round(shortest.seconds / med, 3)
    for iv in counted:
        if iv.seconds < MIN_IOI_RATIO * med:
            result.findings.append(QCFinding(
                grid.clip, "min_ioi",
                f"{iv.seconds:.3f} s interval at {iv.start:.3f} s is "
                f"{iv.seconds / med:.2f}× the {med:.3f} s median "
                f"(under {MIN_IOI_RATIO}×) — spurious double mark?",
                [iv.start, iv.end],
            ))

    # --- check 3: within-phrase IOI spread -----------------------------
    scored = [run for run in grouped if len(run) >= MIN_PHRASE_IOIS]
    if scored:
        result.max_phrase_cv = round(
            max(cv([iv.seconds for iv in run]) for run in scored), 4
        )
    for run in scored:
        spread = cv([iv.seconds for iv in run])
        if spread > MAX_PHRASE_CV:
            result.findings.append(QCFinding(
                grid.clip, "ioi_spread",
                f"phrase {run[0].start:.3f}–{run[-1].end:.3f} s has IOI CV "
                f"{spread:.1%} over {len(run)} intervals (over "
                f"{MAX_PHRASE_CV:.0%}) — missing beat or stray label?",
                [run[0].start, run[-1].end],
            ))
    return result
