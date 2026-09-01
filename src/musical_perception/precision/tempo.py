"""
Tempo calculation from beat timestamps.

KEEP — precision math that AI models won't replace.
Pure functions: timestamps in, BPM out. No I/O, no models.
"""

import math

import numpy as np

from musical_perception.types import (
    Meter,
    NormalizedTempo,
    OnsetTempoResult,
    TempoCandidate,
    TempoResult,
)

# Metric levels a raw pulse can plausibly sit at, in the order the family
# is reported: beat level first, then measure levels, then subdivisions.
_METRIC_LEVELS = (1, 2, 3, -2, -3)

# The absolute plausibility range for a beat rate, shared by the reported
# family (ADR-014) and by level selection: a genuinely slow (60 BPM
# marking) or genuinely fast (160 BPM frappé) tempo is a candidate, a
# 900 BPM one is not.
FAMILY_LOW = 20.0
FAMILY_HIGH = 400.0


# Level-selection prior (Standing Lesson 2: priors are priors, not
# post-processing). The 70-140 "comfort band" was a hard indicator
# function: a reading 2% outside it was folded by a whole metric level,
# which destroyed correct out-of-band measurements (rung-2 checklist
# clips frappe / rig-names-2-4-160-long / rig-numbers-4-4-60-halftempo).
#
# The same prior, softened: read [low, high] as the central interval of a
# log-normal over beat rate — geometric centre T0 = sqrt(low*high), and
# half its octave width as one standard deviation — and combine it with a
# prior over metric distance before choosing a level. Nothing is snapped;
# a level is chosen.
#
# LEVEL_PRIOR_EXPONENT 2 makes P(fold by factor k) proportional to k^-2:
# scale-free in the fold factor, with no separate constant for x2 and x3.
# It is the only integer exponent that both leaves a genuine 160 BPM
# frappe alone and still lifts a genuine half-tempo marking (~52 BPM) to
# the beat level; see the 2026-08-28 ledger entry for the admissible
# interval and the disclosure that the three candidates were checked
# against the corpus.
LEVEL_PRIOR_EXPONENT = 2.0

# Abstention: if even the best level sits further than this many sigma
# from T0, no reading is plausible and the caller gets multiplier 0 — the
# behaviour the old "no x2/x3 transform fits" branch provided.
ABSTAIN_SIGMA = 3.0


def _prior_shape(low: float, high: float) -> tuple[float, float]:
    """(centre, sigma) of the log-normal the band [low, high] stands for."""
    return math.sqrt(low * high), 0.5 * math.log2(high / low)


def _level_scores(
    bpm: float, low: float, high: float
) -> list[tuple[float, float, int]]:
    """(score, candidate_bpm, multiplier) for every plausible metric level.

    Score is an unnormalized log posterior: log-normal tempo prior plus
    log level prior. Candidates outside the absolute plausibility range
    shared with `tempo_family` are not levels at all and are dropped.
    """
    t0, sigma = _prior_shape(low, high)
    scored = []
    for multiplier in _METRIC_LEVELS:
        candidate = round(_apply_multiplier(bpm, multiplier), 1)
        if not FAMILY_LOW <= candidate <= FAMILY_HIGH:
            continue
        octaves = math.log2(candidate / t0)
        score = (
            -0.5 * (octaves / sigma) ** 2
            - LEVEL_PRIOR_EXPONENT * math.log(abs(multiplier))
        )
        scored.append((score, candidate, multiplier))
    return scored


def normalize_tempo(
    bpm: float,
    low: float = 70.0,
    high: float = 140.0,
) -> tuple[float, int]:
    """
    Choose the metric level of a raw pulse under a soft tempo prior.

    From onset regularity alone, N BPM is indistinguishable from N×2, N×3,
    N/2 or N/3 at the beat level (ADR-014). Something has to break the tie,
    and the only thing available is a prior over absolute beat rate. This
    function applies that prior *at level selection* — multiplicatively,
    against every candidate level — instead of folding the measurement
    into a fixed interval.

    `low`/`high` no longer bound the answer. They parameterize the prior:
    the returned BPM may sit outside them when the measurement is good
    enough that no fold is worth its metric-distance cost. With the 70-140
    defaults the resulting "keep it as measured" range is about 55-178 BPM.

    The multiplier tracks how the original pulse relates to the chosen beat:
    - multiplier=1: already at beat level
    - multiplier=2: original was at measure level (doubled to reach beat)
    - multiplier=3: original was at measure level in triple meter
    - multiplier=-2: original was at subdivision level (halved to reach beat)
    - multiplier=-3: original was at triplet subdivision level
    - multiplier=0: no level is plausible (best candidate further than
      ABSTAIN_SIGMA from the prior's centre, or nothing in family range)

    Args:
        bpm: Raw BPM value to normalize.
        low: Lower edge of the prior's central interval.
        high: Upper edge of the prior's central interval.

    Returns:
        (normalized_bpm, multiplier) tuple. When multiplier=0, the raw BPM is
        returned unchanged — the caller should treat it as unreliable.
    """
    if bpm <= 0 or low <= 0 or high <= low:
        return round(bpm, 1), 0

    scored = _level_scores(bpm, low, high)
    if not scored:
        return round(bpm, 1), 0

    # Ties break toward _METRIC_LEVELS order (beat level first), which
    # `max` gives for free: it keeps the first of equal scores.
    _, candidate, multiplier = max(scored, key=lambda s: s[0])

    t0, sigma = _prior_shape(low, high)
    if abs(math.log2(candidate / t0)) > ABSTAIN_SIGMA * sigma:
        return round(bpm, 1), 0

    return candidate, multiplier


def _apply_multiplier(bpm: float, multiplier: int) -> float:
    """The BPM a metric-level multiplier implies for a raw pulse."""
    return bpm * multiplier if multiplier > 0 else bpm / -multiplier


def _derive_metric_reading(
    multiplier: int,
    gemini_meter: Meter | None,
    gemini_subdivision: str | None,
) -> tuple[Meter, str]:
    """Meter + subdivision implied by a metric-level multiplier.

    The single derivation table (ADR-006/007) shared by the primary answer
    and every candidate in its family:

    - multiplier=1: pulse is already at beat level → trust Gemini
    - multiplier=2: pulse was at measure level, doubled → 4/4, no subdivision
    - multiplier=3: pulse was at measure level, tripled → 3/4, no subdivision
    - multiplier=-2: pulse was at subdivision level, halved → duple
    - multiplier=-3: pulse was at subdivision level, divided by 3 → triplet
    """
    if multiplier == 1:
        # BPM was already at beat level — trust Gemini's observations
        return (
            gemini_meter or Meter(beats_per_measure=4, beat_unit=4),
            gemini_subdivision or "none",
        )
    if multiplier == 2:
        # Raw was at measure level, doubled → duple meter, no subdivision
        return Meter(beats_per_measure=4, beat_unit=4), "none"
    if multiplier == 3:
        # Raw was at measure level, tripled → triple meter, no subdivision
        return Meter(beats_per_measure=3, beat_unit=4), "none"
    if multiplier == -2:
        # Raw was at subdivision level, halved → duple subdivision
        return gemini_meter or Meter(beats_per_measure=4, beat_unit=4), "duple"
    if multiplier == -3:
        # Raw was at subdivision level, divided by 3 → triplet subdivision
        return gemini_meter or Meter(beats_per_measure=4, beat_unit=4), "triplet"
    raise ValueError(f"unexpected multiplier {multiplier}")


def tempo_family(
    raw_bpm: float,
    gemini_meter: Meter | None = None,
    gemini_subdivision: str | None = None,
    low: float = 70.0,
    high: float = 140.0,
) -> list[TempoCandidate]:
    """
    All musically-sane readings of one raw pulse (ADR-014).

    From onset regularity alone, a raw reading of N BPM is indistinguishable
    from N×2, N×3, N/2 or N/3 at the beat level: a teacher marking every
    other beat of a 124 BPM exercise and a teacher marking a genuinely slow
    62 BPM exercise produce identical audio. `normalize_tempo()` resolves
    that by scoring the levels under a soft prior; this function reports
    the whole family instead of collapsing it, so the levels that prior
    did not pick stay visible.

    Members are generated over FAMILY_LOW-FAMILY_HIGH (a broad absolute
    plausibility range), ordered beat level → measure levels → subdivisions,
    each carrying the meter/subdivision its multiplier implies.

    Args:
        raw_bpm: The measured pulse, before any normalization.
        gemini_meter: Meter observation, used for beat-level candidates.
        gemini_subdivision: Subdivision observation, same.
        low: Lower bound of the comfort band (for `in_comfort_band`).
        high: Upper bound of the comfort band.

    Returns:
        List of TempoCandidate, possibly empty when no level is plausible.
    """
    if raw_bpm <= 0:
        return []

    candidates = []
    for multiplier in _METRIC_LEVELS:
        bpm = round(_apply_multiplier(raw_bpm, multiplier), 1)
        if not FAMILY_LOW <= bpm <= FAMILY_HIGH:
            continue
        meter, subdivision = _derive_metric_reading(
            multiplier, gemini_meter, gemini_subdivision
        )
        candidates.append(TempoCandidate(
            bpm=bpm,
            meter=meter,
            subdivision=subdivision,
            multiplier=multiplier,
            in_comfort_band=low <= bpm <= high,
        ))
    return candidates


# Tolerance the answer is judged against (ADR-009 scorer, and the
# posterior path's +-8% commitment window). `confidence` is the
# probability of landing inside it, so both paths report the same
# quantity on the same scale.
_TEMPO_TOLERANCE = 0.08
# Weak prior on spoken-count jitter, in units of the beat period, worth
# _PRIOR_N pseudo-intervals. It exists so that one interval — whose
# sample dispersion is exactly 0 — cannot claim certainty (W14-c).
_PRIOR_CV = 0.12
_PRIOR_N = 2.0
# Asymptotic efficiency factor for the median's standard error under a
# normal: sqrt(pi/2).
_MEDIAN_SE_FACTOR = 1.2533


def _tolerance_probability(intervals: list[float], median_interval: float) -> float:
    """P(true beat period is within +-8% of the median interval).

    The dispersion is pooled with a weak prior before the standard error
    is taken, so confidence RISES with evidence instead of starting at
    1.0 and falling. With a single interval the sample dispersion is 0
    and the prior carries the whole estimate, which is the correct
    answer to "how sure are you after one interval": not very.
    """
    n = len(intervals)
    if n == 0 or median_interval <= 0:
        return 0.0

    sample_var = float(np.var(intervals))
    prior_var = (_PRIOR_CV * median_interval) ** 2
    # Shrink toward the prior; n-1 sample degrees of freedom, so one
    # interval contributes no dispersion evidence at all.
    dof = max(n - 1, 0)
    pooled_var = (_PRIOR_N * prior_var + dof * sample_var) / (_PRIOR_N + dof)

    se = _MEDIAN_SE_FACTOR * math.sqrt(pooled_var) / math.sqrt(n)
    if se <= 0:
        return 1.0

    z = (_TEMPO_TOLERANCE * median_interval) / se
    return math.erf(z / math.sqrt(2.0))


def calculate_tempo(timestamps: list[float]) -> TempoResult | None:
    """
    Calculate tempo from a list of beat timestamps.

    Uses median interval for robustness to outliers.

    Two separate numbers come back (W14-c), because the old single number
    was doing both jobs and got one of them backwards:

    - `regularity` = 1 - CV, interval evenness. Maximal at one interval,
      which is fine for "are these even?" and wrong for "are we sure?".
    - `confidence` = P(true period within the scorer's +-8% tolerance),
      which shrinks when the evidence is thin.

    Args:
        timestamps: Times (in seconds) when beats occurred

    Returns:
        TempoResult with BPM and confidence, or None if insufficient data
    """
    if len(timestamps) < 2:
        return None

    intervals = []
    for i in range(1, len(timestamps)):
        intervals.append(timestamps[i] - timestamps[i - 1])

    if not intervals:
        return None

    median_interval = np.median(intervals)
    bpm = 60.0 / median_interval

    # Regularity: lower standard deviation = more even intervals. This is
    # the pre-W14-c `confidence`, preserved bit-for-bit because
    # interpret_meter's arbitration is tuned against it.
    std_interval = np.std(intervals)
    cv = std_interval / median_interval if median_interval > 0 else 1.0
    regularity = max(0.0, 1.0 - cv)

    confidence = _tolerance_probability(intervals, float(median_interval))

    return TempoResult(
        bpm=round(bpm, 1),
        confidence=round(confidence, 2),
        beat_count=len(timestamps),
        intervals=intervals,
        regularity=round(regularity, 2),
    )


def interpret_meter(
    onset_tempo: OnsetTempoResult | None,
    gemini_tempo: TempoResult | None,
    gemini_meter: Meter | None,
    gemini_subdivision: str | None,
) -> NormalizedTempo | None:
    """
    Produce a coherent metric interpretation from raw tempo signals.

    Picks the best raw BPM, chooses its metric level under the soft tempo
    prior (`normalize_tempo`), and derives meter and subdivision from how
    the BPM was scaled. The multiplier encodes the metric level of the raw
    pulse:

    - multiplier=1: raw was already at beat level → trust Gemini meter/subdivision
    - multiplier=2: raw was at measure level, doubled → 4/4, no subdivision
    - multiplier=3: raw was at measure level, tripled → 3/4, no subdivision
    - multiplier=-2: raw was at subdivision level, halved → duple subdivision
    - multiplier=-3: raw was at subdivision level, divided by 3 → triplet subdivision

    Args:
        onset_tempo: Classification-free tempo from word onsets.
        gemini_tempo: Tempo from Gemini-classified beat markers.
        gemini_meter: Gemini's meter guess (used as fallback).
        gemini_subdivision: Gemini's subdivision observation (used as fallback).

    Returns:
        NormalizedTempo with coherent BPM + meter + subdivision, or None
        if no usable tempo signal exists.
    """
    # Arbitration (ADR-013). ADR-006 made onsets primary because Gemini's
    # markers were sparse garbage then; ADR-010's index-keyed merge changed
    # that. Confidence alone cannot arbitrate across metric levels (a
    # regular measure-level marker stream is also "confident"), so the
    # discriminator is the beat band itself: when the onset reading sits
    # OUTSIDE 70-140 (syllable or measure level) and a dense, regular
    # marker tempo sits INSIDE it, the markers are the beat-level signal
    # and win. Whenever onsets are already at beat level, ADR-006/007
    # behavior is preserved unchanged (issue-10 cross-ratio included).
    #
    # The band survives HERE on purpose (W9, 2026-08-28) while it was
    # removed from normalize_tempo: here it is a level *discriminator*
    # between two arms, not a fold applied to a measurement, and all three
    # clips it hands to the markers are correct because of it. Softening
    # it is a separate named question with its own evidence.
    onset_at_beat_level = (
        onset_tempo is not None
        and onset_tempo.confidence >= 0.3
        and 70.0 <= onset_tempo.bpm <= 140.0
    )
    # Reads `regularity`, not `confidence` (W14-c): this gate asks "dense
    # and even" — a regularity question — and 0.6 was tuned against the
    # 1-CV number, which `regularity` still carries bit-for-bit. Every
    # arbitration branch is therefore unchanged by the confidence fix.
    marker_at_beat_level = (
        gemini_tempo is not None
        and gemini_tempo.regularity >= 0.6
        and gemini_tempo.beat_count >= 8
        and 70.0 <= gemini_tempo.bpm <= 140.0
    )
    marker_is_strong = marker_at_beat_level and not onset_at_beat_level

    raw_bpm = None
    confidence = 0.0
    if marker_is_strong:
        raw_bpm = gemini_tempo.bpm
        confidence = gemini_tempo.confidence
    elif onset_tempo is not None and onset_tempo.confidence >= 0.3:
        raw_bpm = onset_tempo.bpm
        confidence = onset_tempo.confidence
    elif gemini_tempo is not None:
        raw_bpm = gemini_tempo.bpm
        confidence = gemini_tempo.confidence
    elif onset_tempo is not None:
        raw_bpm = onset_tempo.bpm
        confidence = onset_tempo.confidence

    if raw_bpm is None:
        return None

    normalized_bpm, multiplier = normalize_tempo(raw_bpm)

    if multiplier == 0:
        return None

    # Cross-signal check: when onset is in range but Gemini BPM is much
    # lower, the ratio tells us about meter. onset/gemini ≈ 3 → triple meter.
    # Only meaningful when onset was the primary signal — when markers won
    # the arbitration, Gemini's meter/subdivision already describe their level.
    if multiplier == 1 and not marker_is_strong \
            and onset_tempo is not None and gemini_tempo is not None:
        ratio = raw_bpm / gemini_tempo.bpm if gemini_tempo.bpm > 0 else 1.0
        if 2.5 <= ratio <= 3.5:
            # Onset at beat level, Gemini at measure level of triple meter.
            # Note: this overloads multiplier=3 — here raw_bpm was NOT tripled
            # (it was already in range), but we use 3 to signal triple meter.
            # raw_bpm is stored separately so consumers don't need to reverse it.
            multiplier = 3

    # Derive meter and subdivision from the multiplier. Unreachable else:
    # normalize_tempo() returns only {1,2,3,-2,-3,0}, and multiplier==0
    # triggers the early return above.
    meter, subdivision = _derive_metric_reading(
        multiplier, gemini_meter, gemini_subdivision
    )

    # The rest of the metric-level family (ADR-014). Purely additive: the
    # primary answer above is unaffected. The member matching the primary
    # BPM is dropped — it IS the primary, under a different narrative.
    alternates = [
        c for c in tempo_family(raw_bpm, gemini_meter, gemini_subdivision)
        if c.bpm != normalized_bpm
    ]

    return NormalizedTempo(
        bpm=normalized_bpm,
        meter=meter,
        subdivision=subdivision,
        confidence=round(confidence, 2),
        raw_bpm=round(raw_bpm, 1),
        tempo_multiplier=multiplier,
        alternates=alternates,
    )
