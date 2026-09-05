"""
Factored joint rhythm posterior (rung 4 / W5) — the bar-pointer lattice.

KEEP — precision math. One exact posterior over (beat period, beat
phase) computed by the forward algorithm on a Krebs-2015 bar-pointer
state space with a Whiteley-2006 Poisson observation model, from the
replayable evidence streams: classified markers and raw word onsets.

There is no meter variable in the state space, and no division axis
either: division is decided by sub-syllable counts per beat
(subdivision.py's logic — syllables carry their metrical identity in
what is said, not where it lands), grouping is read out afterwards as a
per-level ladder, and the time-signature label is derived late, only
for the contract surface (owner direction, 2026-08-26).

Why a lattice and not a global-tempo enumeration: a first implementation
of this module scored (period, phase) hypotheses globally and was
falsified five separate ways by the DEV corpus — fast-level bumps
blanketing the beat circle, width-dependent peak heights subsidizing
whichever level had sharper fractional bumps, doubled grids collecting
two phasings of a half-filled stream from the phase marginal, junk
intervals rewarding whichever wrong grid divided them, and finally
INCOMMENSURATE grids precessing through a slow stream and collecting
partial credit at every pass (113.6 BPM beating both 59.9 and 118.3 on
a clean 60 BPM count). Every one of those is a hypothesis extracting
credit from data it does not explain, and the bar-pointer structure
forbids the whole class at once: the pointer advances deterministically
one frame at a time, tempo changes only at beat crossings and pays
exp(−lambda·|log ratio|) for each, and the per-frame Poisson emission
charges every hypothesis the full expected mass of its own template —
per frame, so paths of different tempi stay comparable. The failure
ledger is in the 2026-08-28 W5 entries.

Constants are declared a priori and checked against synthetic streams
(tests/test_posterior.py mirrors the tier-0 corruption sweep); none is
fitted to DEV clips. A_BEAT is the load-bearing judgment call: it sets
how many silent beat slots outweigh an octave of prior — where "marking
every other beat of a fast exercise" turns into "genuinely slow
marking". The synthetic contract admits [0.06, 0.21]: an 8-count
half-tempo marking must fold up while a 16-count genuine adagio must
not; 0.12 sits in the middle with margin both ways.
"""

import math
from collections import Counter

import numpy as np

from musical_perception.precision.tempo import (
    _derive_metric_reading,
    interpret_meter,
)
from musical_perception.types import (
    GroupingLevel,
    MarkerBelief,
    Meter,
    NormalizedTempo,
    OnsetTempoResult,
    TempoCandidate,
    TempoResult,
    TimedMarker,
    TimestampedWord,
)

# --- lattice ---------------------------------------------------------------
FPS = 50                      # frames per second (dt = 20 ms)
DT = 1.0 / FPS
T_MIN_FRAMES, T_MAX_FRAMES = 15, 75    # 200 BPM .. 40 BPM at 50 fps

# --- tempo prior (identical shape to W9's: T0 = sqrt(70*140), sigma = half
# the band's octave width; nothing here folds — the lattice IS every level)
PRIOR_T0 = math.sqrt(70.0 * 140.0)
PRIOR_SIGMA_OCT = 0.5 * math.log2(140.0 / 70.0)
# Tempo may drift one tempo state per beat crossing, paying
# DRIFT_LAMBDA·|log(T'/T)| — Krebs-2015's transition model. (A ±2
# window was tried for the ±60-100 ms wobble of real marking and
# reverted: it lets wrong-level paths wander onto anything, and broke
# the synthetic half-tempo fold.)
DRIFT_LAMBDA = 50.0

# --- observation model: per-frame Poisson (Whiteley 2006) ------------------
# Each evidence class is a point process whose rate is a background plus
# a Gaussian bump at the beat position of the pointer. Amplitudes are
# expected events per beat slot; backgrounds are events/sec unrelated to
# the pulse. The per-frame emission log(rate)·events − rate·dt charges
# every path its template's full mass, frame by frame.
A_BEAT = 0.12
BG_BEAT = 0.10
A_WORD = 0.30
BG_WORD = 1.20
# One constant absolute timing sigma (ASR jitter; stage1 asynchrony
# spread is ~30-45 ms). Constant in TIME at every tempo: the enumeration
# era showed that any width-height coupling becomes a level subsidy.
# (0.065 — jitter plus per-beat wobble — was tried and reverted: the
# forgiveness it buys the slow rows re-opens level ambiguity at the
# fast end and folds a genuine 160 BPM frappe.)
SIGMA_T_S = 0.05

# --- commitment ------------------------------------------------------------
# Confidence is the posterior mass of the ±8% neighborhood around the
# committed tempo (the probability the answer is one the scorer
# accepts). Below the floor the honest answer is abstention.
TOLERANCE = 0.08
COMMIT_FLOOR = 0.20

# --- sparse / degraded fallback --------------------------------------------
MIN_BEAT_MARKERS = 4
MIN_EVENTS = 8
# Below this beat-stream support the classifier's markers are not a
# pulse claim at all (mixed levels, everything-is-a-beat numbering);
# the legacy arbitration, whose windowed section-finding was built for
# exactly that shape, answers instead. 0.6 is not a new constant: it is
# rhythm.py's own cv_threshold=0.4 (the boundary below which a window
# does not count as rhythmic) expressed as 1 - cv, applied to the beat
# stream that would otherwise anchor the lattice.
SUPPORT_FLOOR = 0.6

_SUB_VOCAB = {"and", "an", "n", "ah", "a", "uh", "e"}
_DIVISIONS = ("none", "duple", "triplet")


def _normalize_text(word: str) -> str:
    return word.strip().lower().strip(".,!?;:")


def _stream_support(times: np.ndarray) -> float:
    """How self-consistent a marker stream is: 1 - robust IOI dispersion.

    A regular stream speaks with its full weight; a stream whose own
    intervals disagree (mixed levels, misclassifications) is discounted
    before it can outvote anything.
    """
    if len(times) < 3:
        return 0.5
    iois = np.diff(np.sort(times))
    iois = iois[iois > 0.05]
    if len(iois) < 2:
        return 0.5
    med = float(np.median(iois))
    if med <= 0:
        return 0.5
    spread = float(np.median(np.abs(iois - med)) / med)   # robust CV
    return float(np.clip(1.0 - spread, 0.1, 1.0))


def beliefs_from_markers(
    words: list[TimestampedWord],
    markers: list[TimedMarker],
) -> list[MarkerBelief]:
    """The single-draw partition, expressed as one-hot beliefs.

    Exactly reproduces the three streams the hard-label code built: a
    marker becomes a one-hot token of its own class (E included, which
    is why E is a class — an E token belongs to no stream, and folding
    it into `none` would enter it into the word stream); a word becomes
    a `none` token unless a marker already stands at its timestamp or
    its text is a subdivision vocable. Markers come first so the
    grouping ladder reads beat numbers in the order the classifier
    emitted them.
    """
    marker_ts = {round(m.timestamp, 4) for m in markers}
    beliefs = [
        MarkerBelief(
            timestamp=m.timestamp,
            probs={m.marker_type.value: 1.0},
            beat_number=m.beat_number,
            raw_word=m.raw_word,
        )
        for m in markers
    ]
    beliefs += [
        MarkerBelief(timestamp=w.start, probs={"none": 1.0}, raw_word=w.word)
        for w in words
        if round(w.start, 4) not in marker_ts
        and _normalize_text(w.word) not in _SUB_VOCAB
    ]
    return beliefs


def _weighted_stream(
    beliefs: list[MarkerBelief],
    classes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """One evidence class as (times, expected counts), time-sorted.

    Zero-mass tokens are dropped rather than carried at weight 0: a
    point process with no rate contribution is not an event, and
    keeping them would make `use_words` true on a clip whose words all
    belong to some other class.
    """
    pairs = [
        (b.timestamp, sum(b.p(c) for c in classes)) for b in beliefs
    ]
    pairs = sorted((t, w) for t, w in pairs if w > 0.0)
    times = np.array([t for t, _ in pairs])
    weights = np.array([w for _, w in pairs])
    return times, weights


def _map_times(beliefs: list[MarkerBelief], classes: tuple[str, ...]) -> np.ndarray:
    """Times of the tokens the MAP decode assigns to `classes`."""
    return np.array(sorted(
        b.timestamp for b in beliefs if b.map_class in classes
    ))


def estimate_rhythm(
    words: list[TimestampedWord],
    markers: list[TimedMarker],
    *,
    marker_beliefs: list[MarkerBelief] | None = None,
    gemini_meter: Meter | None = None,
    gemini_subdivision: str | None = None,
    onset_tempo: OnsetTempoResult | None = None,
    gemini_tempo: TempoResult | None = None,
    pulse_events: list[float] | None = None,
) -> NormalizedTempo | None:
    """
    Bar-pointer posterior over the marking streams.

    Returns the contract's NormalizedTempo: BPM is the window-mass Bayes
    commitment over the lattice's final tempo marginal, confidence is
    the posterior mass of the ±8% tempo neighborhood, division comes
    from sub-syllable counts per beat, the meter label is derived late
    (Gemini's claim is one vote — the state space has no meter
    variable), alternates carry the posterior's other tempo maxima with
    their masses, and `grouping_levels` reports the grouping ladder.

    Evidence-poor clips (fewer than MIN_BEAT_MARKERS classified beats or
    MIN_EVENTS events) and beat streams below SUPPORT_FLOOR fall back to
    `interpret_meter` unchanged.

    `marker_beliefs` (W6-a) replaces the hard labels with a
    distribution per token: the emission then charges expected support
    instead of integer counts, so N disagreeing draws vote by mass
    rather than by whichever one was sampled (Standing Lesson 4). Left
    None, the beliefs are built one-hot from `markers` and the answer
    is bit-for-bit the single-draw answer.

    `pulse_events` (PP-1) is the rung-2 acoustic pulse stream. When
    given, its all-pairs period tilts the tempo marginal by a bounded
    multiplicative prior before the commitment — a prior at level
    selection, never a fold (Standing Lesson 2). Left None, every
    number below is bit-for-bit its pre-PP-1 value.
    """
    beliefs = (
        marker_beliefs if marker_beliefs is not None
        else beliefs_from_markers(words, markers)
    )
    # Two views of the same tokens: the MAP decode for the guard
    # statistics (support, the division vet, the ladder), the expected
    # mass for the emission. One draw makes them the same object.
    beat_times = _map_times(beliefs, ("beat",))
    sub_times = _map_times(beliefs, ("and", "ah"))
    beat_ev = _weighted_stream(beliefs, ("beat",))
    sub_ev = _weighted_stream(beliefs, ("and", "ah"))
    word_ev = _weighted_stream(beliefs, ("none",))

    def fallback() -> NormalizedTempo | None:
        """Legacy arbitration, with division still measured, not
        relayed: on the fallback rows interpret_meter passes Gemini's
        subdivision claim through at multiplier 1, which is the exact
        defect W9-b names. Where the answer sits at beat level and
        there are beat markers to measure against, the counted-and-
        vetted division replaces the claim."""
        result = interpret_meter(
            onset_tempo, gemini_tempo, gemini_meter, gemini_subdivision
        )
        if (result is not None and result.tempo_multiplier == 1
                and len(beat_times) >= 2):
            result.subdivision = _division(
                beat_times, sub_times, gemini_subdivision
            )
        return result

    n_beat = float(beat_ev[1].sum())
    n_events = n_beat + float(sub_ev[1].sum()) + float(word_ev[1].sum())
    if n_beat < MIN_BEAT_MARKERS or n_events < MIN_EVENTS:
        return fallback()

    beat_support = _stream_support(beat_times)
    if beat_support < SUPPORT_FLOOR:
        return fallback()

    bpm_axis, log_marginal = _lattice_forward(
        beat_ev, word_ev, A_BEAT * beat_support
    )
    mass = np.exp(log_marginal - np.logaddexp.reduce(log_marginal))
    mass, pulse_bpm = _apply_pulse_prior(bpm_axis, mass, pulse_events)

    # Commit to the tempo whose ±8% neighborhood holds the most mass —
    # the Bayes decision under the scorer's utility (an answer is right
    # iff it lands within tolerance). Confidence IS that window's mass.
    log_tol = math.log(1.0 + TOLERANCE)
    log_bpms = np.log(bpm_axis)
    window = np.abs(log_bpms[:, None] - log_bpms[None, :]) <= log_tol
    window_mass = window @ mass
    center = int(window_mass.argmax())
    in_tol = window[center]
    confidence = float(window_mass[center])
    w = mass[in_tol]
    map_bpm = float(np.exp((w @ log_bpms[in_tol]) / w.sum()))

    if confidence < COMMIT_FLOOR:
        return None

    division = _division(beat_times, sub_times, gemini_subdivision)
    meter = gemini_meter or Meter(beats_per_measure=4, beat_unit=4)
    alternates = _alternates(
        bpm_axis, mass, in_tol, map_bpm, gemini_meter, gemini_subdivision
    )
    ladder = _grouping_ladder(beliefs, beat_times, 60.0 / map_bpm)

    return NormalizedTempo(
        bpm=round(map_bpm, 1),
        meter=meter,
        subdivision=division,
        confidence=round(confidence, 2),
        raw_bpm=round(map_bpm, 1),
        tempo_multiplier=1,
        alternates=alternates,
        grouping_levels=ladder,
    )


PULSE_PRIOR_WEIGHT = 0.5      # mixture weight; the (1-W) floor is why
PULSE_PRIOR_SIGMA = 0.10      # this prior can tilt but never veto


def _apply_pulse_prior(
    bpm_axis: np.ndarray,
    mass: np.ndarray,
    pulse_events: list[float] | None,
) -> tuple[np.ndarray, float | None]:
    """Tilt the tempo marginal toward the acoustic pulse's all-pairs period.

    PP-1. The prior is a mixture — a uniform floor of (1 - W) plus a
    log-normal bump of weight W at the measured period — so it can at
    most double the relative weight of the favoured region and can
    NEVER zero a hypothesis. That bound is the whole design: Standing
    Lesson 2 is about a hard fold destroying correct out-of-band
    measurements, and a prior that cannot reach zero cannot do that.

    No metric-level relatives are added. The lattice already votes over
    levels (Lesson 3); this only says a real periodicity was measured
    there. Returns (possibly unchanged) mass and the pulse BPM, or None
    when no prior was applied.
    """
    if not pulse_events:
        return mass, None
    from musical_perception.precision.pulse import all_pairs_period

    period = all_pairs_period(pulse_events)
    if period is None:                    # sparse, or a boundary artifact
        return mass, None
    pulse_bpm = 60.0 / period
    z = (np.log(bpm_axis) - math.log(pulse_bpm)) / PULSE_PRIOR_SIGMA
    prior = (1.0 - PULSE_PRIOR_WEIGHT) + PULSE_PRIOR_WEIGHT * np.exp(-0.5 * z * z)
    tilted = mass * prior
    total = tilted.sum()
    if not np.isfinite(total) or total <= 0.0:
        return mass, None
    return tilted / total, pulse_bpm


def _lattice_forward(
    beat_ev: tuple[np.ndarray, np.ndarray],
    word_ev: tuple[np.ndarray, np.ndarray],
    a_beat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward algorithm on the bar-pointer lattice.

    State: (T frames-per-beat, phi position-in-beat); the pointer
    advances one frame per frame, tempo changes only at beat crossings
    (phi wrapping to 0) to an adjacent tempo state, paying
    DRIFT_LAMBDA·|log ratio|. Emissions per frame and class c:
    events_c(f)·log(rate_c(phi,T)) − rate_c(phi,T)·dt, with
    rate = bg_c + amp_c·N(0;sigma)·gauss(phi·dt; sigma). Returns
    (bpm per tempo state, final log tempo marginal).

    Each stream arrives as (times, weights) — the expected number of
    events that token contributes to that class (W6-a). `events_c(f)`
    is therefore a fractional count, which is what a Poisson likelihood
    wants from an uncertain observer: half a beat's worth of belief
    buys half a beat's worth of log-rate credit and no more. One draw
    makes every weight exactly 1.0 and the arithmetic is the old
    integer arithmetic, bit for bit.
    """
    beat_times, beat_w = beat_ev
    word_times, word_w = word_ev
    t0 = float(beat_times[0]) - 0.2
    t1 = float(beat_times[-1]) + 0.2
    n_frames = max(int(math.ceil((t1 - t0) / DT)), T_MAX_FRAMES + 1)

    def counts(times: np.ndarray, weights: np.ndarray) -> np.ndarray:
        e = np.zeros(n_frames)
        if len(times):
            idx = np.clip(((times - t0) / DT).astype(int), 0, n_frames - 1)
            np.add.at(e, idx, weights)
        return e

    e_beat = counts(beat_times, beat_w)
    e_word = counts(word_times, word_w)
    use_words = len(word_times) > 0

    tempos = np.arange(T_MIN_FRAMES, T_MAX_FRAMES + 1)      # (K,)
    bpm_axis = 60.0 / (tempos * DT)
    offsets = np.concatenate([[0], np.cumsum(tempos)])      # state layout
    n_states = int(offsets[-1])

    # Per-state emission ingredients.
    peak = 1.0 / (SIGMA_T_S * math.sqrt(2.0 * math.pi))
    log_rate_b = np.empty(n_states)
    log_rate_w = np.empty(n_states)
    rate_sum = np.empty(n_states)
    state_T = np.empty(n_states, dtype=int)
    for k, T in enumerate(tempos):
        sl = slice(offsets[k], offsets[k + 1])
        phi = np.arange(T)
        d = np.minimum(phi, T - phi) * DT                   # sec to beat
        g = np.exp(-0.5 * (d / SIGMA_T_S) ** 2)
        rb = BG_BEAT + a_beat * peak * g
        rw = BG_WORD + A_WORD * peak * g
        log_rate_b[sl] = np.log(rb)
        log_rate_w[sl] = np.log(rw)
        rate_sum[sl] = (rb + (rw if use_words else 0.0)) * DT
        state_T[sl] = T

    # Predecessor map for the deterministic advance (phi>0 <- phi-1);
    # beat states (phi=0) gather from the last position of adjacent
    # tempo states with the drift cost.
    prev_idx = np.empty(n_states, dtype=int)
    for k, T in enumerate(tempos):
        base = offsets[k]
        prev_idx[base] = base                                # placeholder
        prev_idx[base + 1: base + T] = np.arange(base, base + T - 1)
    beat_states = offsets[:-1]                               # phi = 0
    last_states = offsets[1:] - 1                            # phi = T-1
    drift_from = []                                          # per k: (idx, cost)
    for k, T in enumerate(tempos):
        srcs, costs = [], []
        for dk in (-1, 0, 1):
            j = k + dk
            if 0 <= j < len(tempos):
                srcs.append(last_states[j])
                costs.append(DRIFT_LAMBDA * abs(math.log(T / tempos[j])))
        drift_from.append((np.array(srcs), np.array(costs)))

    # Init: log-normal tempo prior, per-T mass uniform over phase (the
    # -log T term — without it, T phase-states of headroom hand faster
    # tempos a free log T of mass).
    prior = -0.5 * (np.log2(bpm_axis / PRIOR_T0) / PRIOR_SIGMA_OCT) ** 2
    alpha = np.empty(n_states)
    for k, T in enumerate(tempos):
        alpha[offsets[k]: offsets[k + 1]] = prior[k] - math.log(T)

    emit_base = -rate_sum
    for f in range(n_frames):
        emit = emit_base
        if e_beat[f]:
            emit = emit + e_beat[f] * log_rate_b
        if use_words and e_word[f]:
            emit = emit + e_word[f] * log_rate_w
        if f:
            shifted = alpha[prev_idx]
            for k in range(len(tempos)):
                srcs, costs = drift_from[k]
                shifted[beat_states[k]] = np.logaddexp.reduce(
                    alpha[srcs] - costs
                )
            alpha = shifted + emit
        else:
            alpha = alpha + emit

    log_marginal = np.array([
        np.logaddexp.reduce(alpha[offsets[k]: offsets[k + 1]])
        for k in range(len(tempos))
    ])
    return bpm_axis, log_marginal


def _division(
    beat_times: np.ndarray,
    sub_times: np.ndarray,
    gemini_subdivision: str | None,
) -> str:
    """Division from sub-syllable counts per beat, vetted by timing
    CONSISTENCY.

    The count decides the candidate category: spoken subdivision
    syllables carry their metrical identity in how many there are per
    beat, not in where they land relative to the ideal positions — on
    the real corpus a duple "and" sits at frac 0.61-0.77 (swing) and a
    triplet lands its pair near 0.55 and 0.9, nowhere near 1/3 and
    2/3, so nearest-ideal-position classification misreads both.

    Timing then vets the candidate (the owner's point, measured and
    confirmed 2026-08-28): a REAL subdivision is a rhythmic event that
    recurs at a stable phase of the beat, swing included — one tight
    cluster for duple, two for triplet — while incidental between-beat
    speech (step names, explanation syllables the classifier tagged as
    and/ah) scatters across the beat. Each within-beat rank of the sub
    positions must be circularly concentrated; a candidate whose subs
    have no stable phase is not a subdivision, and the answer is
    `none`. Positions are measured between surrounding beat MARKERS
    (local, drift-free). Sparse strays (fewer than one per two gaps)
    are `none` before any of this; Gemini's claim decides only when
    there are no beat markers to measure against.
    """
    if not len(sub_times):
        return "none"
    if len(beat_times) < 2:
        return gemini_subdivision if gemini_subdivision in _DIVISIONS else "none"

    gap_positions: dict[int, list[float]] = {}
    for t in sub_times:
        i = int(np.searchsorted(beat_times, t)) - 1
        if 0 <= i < len(beat_times) - 1:
            span = beat_times[i + 1] - beat_times[i]
            if 0.2 < span < 3.0:
                gap_positions.setdefault(i, []).append(
                    float((t - beat_times[i]) / span)
                )
    n_gaps = len(beat_times) - 1
    n_subs = sum(len(v) for v in gap_positions.values())
    # Recurrence needs at least three observations — two points always
    # "recur" (the same identifiability floor as rhythm.py's
    # GRID_MIN_IOIS). Below it, no subdivision claim is checkable, and
    # the unverifiable answer is none.
    if n_subs < 3:
        return "none"
    avg = n_subs / n_gaps if n_gaps else 0.0
    if avg < 0.5:
        return "none"
    candidate = "duple" if avg < 1.5 else ("triplet" if avg < 2.5 else None)
    if candidate is None:
        return gemini_subdivision if gemini_subdivision in _DIVISIONS else "none"

    # Consistency vet: per within-gap rank, the circular resultant
    # length of the positions. Uniform scatter gives R ~ 1/sqrt(n);
    # a recurring phase gives R near 1. The threshold was chosen with
    # DEV distributions visible (disclosed, W9-style): the measured
    # clusters sit at R >= 0.85 and the measured scatter at R <= 0.5,
    # so any value in (0.5, 0.85) separates them; 0.6 leaves the wider
    # margin on the cluster side, where a swung but genuine subdivision
    # must never be rejected.
    for rank in range(2 if candidate == "triplet" else 1):
        pos = [sorted(v)[rank] for v in gap_positions.values() if len(v) > rank]
        if len(pos) < 4:
            continue
        angles = 2.0 * math.pi * np.array(pos)
        resultant = float(np.hypot(np.cos(angles).mean(),
                                   np.sin(angles).mean()))
        if resultant < 0.6:
            return "none"
    return candidate


def _alternates(
    bpms: np.ndarray,
    mass: np.ndarray,
    map_in_tol: np.ndarray,
    map_bpm: float,
    gemini_meter: Meter | None,
    gemini_subdivision: str | None,
) -> list[TempoCandidate]:
    """ADR-014 family from the posterior's other tempo maxima.

    Each candidate carries the posterior mass of its own ±8%
    neighborhood — the family finally has real weights.
    """
    out = []
    remaining = mass.copy()
    remaining[map_in_tol] = 0.0
    log_tol = math.log(1.0 + TOLERANCE)
    for _ in range(4):
        idx = int(remaining.argmax())
        peak_bpm = float(bpms[idx])
        window = np.abs(np.log(bpms / peak_bpm)) <= log_tol
        peak_mass = float(remaining[window].sum())
        if peak_mass < 0.01:
            break
        remaining[window] = 0.0
        ratio = peak_bpm / map_bpm
        multiplier = 1
        for m in (2, 3, -2, -3):
            target = m if m > 0 else 1.0 / -m
            if abs(math.log(ratio / target)) <= log_tol:
                multiplier = m
                break
        meter, subdivision = _derive_metric_reading(
            multiplier, gemini_meter, gemini_subdivision
        )
        out.append(TempoCandidate(
            bpm=round(peak_bpm, 1),
            meter=meter,
            subdivision=subdivision,
            multiplier=multiplier,
            in_comfort_band=70.0 <= peak_bpm <= 140.0,
            weight=round(peak_mass, 3),
        ))
    return out


def _grouping_ladder(
    beliefs: list[MarkerBelief],
    beat_times: np.ndarray,
    period: float,
) -> list[GroupingLevel]:
    """Per-level grouping evidence above the beat (the factored ladder).

    Sources available in the replayable streams: the counted-number
    cycle (strong where the teacher counts), and boundary gaps between
    marked runs (Temperley's gap rule). Accent alternation is not
    measurable from these traces (no salience features are frozen), and
    W2 showed it is mostly absent in this corpus anyway; the ladder
    reports what the evidence supports and stays silent where it is
    silent — that silence is the honest output, not a failure.
    """
    levels: dict[int, GroupingLevel] = {}

    nums = [b.beat_number for b in beliefs
            if b.map_class == "beat" and b.beat_number is not None]
    if len(nums) >= 4:
        resets = [nums[i - 1] for i in range(1, len(nums)) if nums[i] < nums[i - 1]]
        if resets:
            top, hits = Counter(resets).most_common(1)[0]
            share = hits / len(resets)
            if top >= 2 and share >= 0.5:
                levels[top] = GroupingLevel(
                    level=int(top),
                    strength=round(min(1.0, share * min(1.0, len(resets) / 2)), 2),
                    source="counting",
                )

    if len(beat_times) >= 6 and period > 0:
        iois = np.diff(beat_times)
        med = float(np.median(iois))
        gaps = np.where(iois > 1.6 * med)[0]
        if len(gaps) >= 2:
            spans = np.diff(beat_times[gaps])
            beats_between = np.round(spans / period)
            good = beats_between[(beats_between >= 2) & (beats_between <= 16)]
            if len(good):
                top, hits = Counter(good.astype(int)).most_common(1)[0]
                share = hits / len(good)
                if share >= 0.5 and int(top) not in levels:
                    levels[int(top)] = GroupingLevel(
                        level=int(top),
                        strength=round(0.5 * share, 2),
                        source="gaps",
                    )

    return [levels[k] for k in sorted(levels)]
