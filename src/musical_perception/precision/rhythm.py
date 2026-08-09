"""
Rhythmic section detection from word onset timing.

KEEP — precision math, no AI dependencies. Identifies rhythmic sections
of speech from word onset timestamps alone, without requiring word
classification (beat/and/ah). Complementary to the Gemini-based tempo
pipeline: provides a classification-free tempo estimate that works with
step names, numbers, or any rhythmic speech.

Algorithm: sliding-window analysis of inter-onset interval regularity.
Windows with low coefficient of variation (CV < 0.4) are classified as
rhythmic. Each section's beat period is then recovered by grid-fitting all
of its inter-onset intervals (ADR-015) rather than averaging them.
"""

import numpy as np

from musical_perception.types import (
    OnsetTempoResult,
    RhythmicSection,
    TimestampedWord,
)

# Grid-fit parameters (ADR-015). Chosen on principle, not tuned:
#
# GRID_TOL 0.20 is the widest tolerance that keeps the 1x acceptance band
#   [0.8m, 1.2m] disjoint from the 2x band [1.6m, 2.4m] with margin, and is
#   about the size of real expressive timing deviation. An IOI landing in
#   the dead zone between bands is an agogic gap, and gets dropped rather
#   than averaged in.
# GRID_MAX_SPAN 3 is the same integer set ADR-014's metric-level family
#   uses; a teacher who speaks on every 4th beat is out of scope.
# GRID_MIN_IOIS 4 is the identifiability floor: the grid hypothesis spends
#   one free integer per interval, so on two or three intervals it can fit
#   anything and falsify nothing. Below the floor the window keeps its mean.
GRID_TOL = 0.20
GRID_MAX_SPAN = 3
GRID_MIN_IOIS = 4
_GRID_PASSES = 2


def detect_onset_tempo(
    words: list[TimestampedWord],
    *,
    window_sec: float = 3.0,
    step_sec: float = 0.5,
    cv_threshold: float = 0.4,
    min_words_per_window: int = 3,
    min_ioi: float = 0.15,
    max_ioi: float = 2.0,
    grid_tol: float = GRID_TOL,
) -> OnsetTempoResult | None:
    """
    Detect tempo from word onset timing without word classification.

    Slides overlapping windows over word onsets and identifies sections
    where words are regularly spaced (low CV). Within each section the beat
    period is recovered by grid-fitting the IOIs (see `_grid_period`), which
    is what keeps expressive bar-boundary gaps and sparse marking from
    dragging the estimate off the pulse (ADR-015).

    Args:
        words: Timestamped words from transcription.
        window_sec: Sliding window duration in seconds.
        step_sec: Step size between windows in seconds.
        cv_threshold: Maximum CV to consider a window rhythmic.
        min_words_per_window: Minimum word onsets per window.
        min_ioi: Minimum inter-onset interval (filters sub-word artifacts).
        max_ioi: Maximum inter-onset interval (filters long pauses).
        grid_tol: Relative tolerance for an IOI to count as k beats.

    Returns:
        OnsetTempoResult with BPM and rhythmic sections, or None if
        insufficient data or no rhythmic sections found.
    """
    if len(words) < 3:
        return None

    onsets = np.array([w.start for w in words])
    word_texts = [w.word for w in words]

    # Secondary estimate: IOI histogram peak
    all_iois = np.diff(onsets)
    musical_iois = all_iois[(all_iois >= min_ioi) & (all_iois <= max_ioi)]
    histogram_bpm = _ioi_histogram_peak(musical_iois)

    # Primary: sliding window analysis
    sections = _compute_window_sections(
        onsets, word_texts, window_sec, step_sec, cv_threshold, min_words_per_window,
        min_ioi,
    )

    if not sections:
        return None

    merged, supports = _refit_sections(
        _merge_overlapping_sections(sections), onsets, min_ioi, max_ioi, grid_tol
    )

    # Compute final BPM via duration-weighted median
    bpms = np.array([s.bpm for s in merged])
    durations = np.array([s.end - s.start for s in merged])
    sorted_idx = np.argsort(bpms)
    cumw = np.cumsum(durations[sorted_idx])
    median_pos = int(np.searchsorted(cumw, cumw[-1] / 2.0))
    median_pos = min(median_pos, len(bpms) - 1)
    bpm = round(float(bpms[sorted_idx[median_pos]]), 1)

    total_duration = float(onsets[-1] - onsets[0])
    rhythmic_duration = sum(s.end - s.start for s in merged)
    coverage = round(min(1.0, rhythmic_duration / total_duration), 3) if total_duration > 0 else 0.0

    confidence = _compute_confidence(
        merged, total_duration, histogram_bpm, float(np.mean(supports))
    )

    return OnsetTempoResult(
        bpm=bpm,
        confidence=confidence,
        rhythmic_sections=merged,
        total_duration=round(total_duration, 2),
        rhythmic_coverage=coverage,
        ioi_histogram_peak_bpm=histogram_bpm,
    )


def _grid_period(
    iois: np.ndarray,
    tol: float = GRID_TOL,
    max_span: int = GRID_MAX_SPAN,
) -> tuple[float, float, float]:
    """Beat period of a run of onsets, fitted as an integer grid over its IOIs.

    The mean IOI assumes every interval spans exactly one beat and that all
    of them are drawn from one distribution. Marking violates both: a
    bar-boundary gap is one long interval among steady ones, and step-name
    marking speaks on some beats only, mixing 1x, 2x and 3x the period.
    Averaging those lands between metric levels (ADR-015).

    So: anchor on the median (the mean is the quantity being corrected, so
    it cannot also be the reference), ask how many beats each interval
    spans, drop the ones that fit no whole number of beats, and divide
    elapsed time by beats spanned.

    Below `GRID_MIN_IOIS` intervals the fit is not identifiable and the
    caller keeps the plain mean — today's answer, unchanged.

    Returns:
        (period, support, cv) — the fitted period in seconds, the fraction
        of IOIs the grid explains, and the dispersion of the grid-folded
        IOIs around it.
    """
    if len(iois) < GRID_MIN_IOIS:
        mean_ioi = float(np.mean(iois))
        if mean_ioi <= 0:
            return 0.0, 0.0, 1.0
        # Discounted in proportion to how far below the floor it sits. Three
        # intervals that happen to agree are not better evidence than a fitted
        # window that explains four of five — which is what full support here
        # would claim, and it is the spurious confidence this measure exists
        # to prevent.
        return mean_ioi, len(iois) / GRID_MIN_IOIS, float(np.std(iois) / mean_ioi)

    period = float(np.median(iois))
    kept, spans = iois, np.ones(len(iois))

    for _ in range(_GRID_PASSES):
        if period <= 0:
            break
        candidate_spans = np.clip(np.round(iois / period), 1, max_span)
        fits = np.abs(iois - candidate_spans * period) <= tol * candidate_spans * period
        if not fits.any():
            break
        kept, spans = iois[fits], candidate_spans[fits]
        period = float(kept.sum() / spans.sum())

    if period <= 0:
        return 0.0, 0.0, 1.0

    folded = kept / spans
    return period, len(kept) / len(iois), float(np.std(folded) / period)


def _compute_window_sections(
    onsets: np.ndarray,
    word_texts: list[str],
    window_sec: float,
    step_sec: float,
    cv_threshold: float,
    min_words_per_window: int,
    min_ioi: float = 0.15,
) -> list[RhythmicSection]:
    """Slide windows over onsets and identify rhythmic sections.

    Unchanged by ADR-015: the sweep decides *where* speech is rhythmic, and
    that boundary is not what was wrong. The tempo each window reports here
    is provisional — `_refit_sections` measures it again over the whole
    merged section.
    """
    sections = []
    t = float(onsets[0])
    end_time = float(onsets[-1])

    while t + window_sec <= end_time + step_sec:
        mask = (onsets >= t) & (onsets < t + window_sec)
        indices = np.where(mask)[0]

        if len(indices) >= min_words_per_window:
            window_onsets = onsets[indices]
            window_iois = np.diff(window_onsets)

            # Filter sub-word artifacts within the window
            window_iois = window_iois[window_iois >= min_ioi]

            if len(window_iois) >= 2:
                mean_ioi = float(np.mean(window_iois))
                if mean_ioi > 0:
                    cv = float(np.std(window_iois) / mean_ioi)
                    if cv < cv_threshold:
                        sections.append(RhythmicSection(
                            start=round(float(t), 2),
                            end=round(float(t + window_sec), 2),
                            bpm=round(60.0 / mean_ioi, 1),
                            mean_ioi=round(mean_ioi, 4),
                            cv=round(cv, 3),
                            word_count=len(indices),
                            words=[word_texts[i] for i in indices],
                        ))

        t += step_sec

    return sections


def _refit_sections(
    sections: list[RhythmicSection],
    onsets: np.ndarray,
    min_ioi: float,
    max_ioi: float,
    grid_tol: float,
) -> tuple[list[RhythmicSection], list[float]]:
    """Re-measure each merged section's tempo from all of its own onsets.

    The window sweep is what decides *where* speech is rhythmic; it is a poor
    instrument for deciding *how fast*, because a 3-second window holds only
    4-6 intervals and the merge then elects one of them to speak for the
    whole section. Refitting over the section's full onset run puts every
    interval it contains behind one estimate (ADR-015).
    """
    refit, supports = [], []
    for section in sections:
        selected = onsets[(onsets >= section.start) & (onsets <= section.end)]
        iois = np.diff(selected)
        iois = iois[(iois >= min_ioi) & (iois <= max_ioi)]
        if len(iois) < 2:
            refit.append(section)
            supports.append(0.0)
            continue
        period, support, cv = _grid_period(iois, grid_tol)
        if period <= 0:
            refit.append(section)
            supports.append(0.0)
            continue
        refit.append(RhythmicSection(
            start=section.start,
            end=section.end,
            bpm=round(60.0 / period, 1),
            mean_ioi=round(period, 4),
            cv=round(cv, 3),
            word_count=section.word_count,
            words=section.words,
        ))
        supports.append(support)
    return refit, supports


def _merge_overlapping_sections(
    sections: list[RhythmicSection],
) -> list[RhythmicSection]:
    """Merge overlapping rhythmic windows into consolidated sections."""
    if not sections:
        return []

    sorted_sections = sorted(sections, key=lambda s: s.start)
    merged = [sorted_sections[0]]

    for section in sorted_sections[1:]:
        prev = merged[-1]
        if section.start <= prev.end:
            # Overlapping: keep BPM from the more regular window (lower CV)
            best = section if section.cv < prev.cv else prev
            # Deduplicate by text — overlapping windows will contain the same
            # word strings from the same onset indices. Repeated step names
            # (e.g. two "tendu"s) are collapsed, which slightly understates
            # word_count but keeps the display list readable.
            all_words = list(dict.fromkeys(prev.words + section.words))
            merged[-1] = RhythmicSection(
                start=prev.start,
                end=max(prev.end, section.end),
                bpm=best.bpm,
                mean_ioi=best.mean_ioi,
                cv=best.cv,
                word_count=len(all_words),
                words=all_words,
            )
        else:
            merged.append(section)

    return merged


def _ioi_histogram_peak(iois: np.ndarray) -> float | None:
    """Find the dominant IOI from histogram peak, return BPM or None."""
    if len(iois) < 3:
        return None

    ioi_min, ioi_max = float(iois.min()), float(iois.max())
    if ioi_max - ioi_min < 1e-6:
        # All IOIs (near-)identical — perfectly regular, no histogram needed.
        # Strict equality is not enough: float-hair ranges (~1e-15) make
        # np.histogram raise "too many bins for data range".
        median = float(np.median(iois))
        return round(60.0 / median, 1) if median > 0 else None

    n_bins = max(10, min(50, len(iois) // 2))
    hist, bin_edges = np.histogram(iois, bins=n_bins, range=(ioi_min, ioi_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_idx = int(np.argmax(hist))

    if hist[peak_idx] < 2:
        return None

    return round(60.0 / float(bin_centers[peak_idx]), 1)


def _compute_confidence(
    sections: list[RhythmicSection],
    total_duration: float,
    histogram_bpm: float | None,
    grid_support: float = 1.0,
) -> float:
    """Overall confidence: coverage, cross-section agreement, grid fit, support.

    `grid_support` (ADR-015) is the share of IOIs the fitted grid actually
    explains. Without it a sparse reading scores as confidently as a dense
    one — three surviving intervals fit any grid perfectly — which is
    exactly the reading the arbitration in `interpret_meter()` should
    trust least.
    """
    if not sections:
        return 0.0

    # Factor 1: Coverage (fraction of audio that is rhythmic)
    rhythmic_duration = sum(s.end - s.start for s in sections)
    coverage = min(1.0, rhythmic_duration / total_duration) if total_duration > 0 else 0.0

    # Factor 2: BPM consistency across sections
    bpms = np.array([s.bpm for s in sections])
    if len(bpms) > 1:
        bpm_cv = float(np.std(bpms) / np.mean(bpms)) if np.mean(bpms) > 0 else 1.0
        consistency = max(0.0, 1.0 - bpm_cv)
    else:
        consistency = 0.5

    # Factor 3: Mean grid fit (inverse of mean folded-IOI dispersion)
    mean_cv = float(np.mean([s.cv for s in sections]))
    regularity = max(0.0, 1.0 - mean_cv)

    # Factor 4: Histogram agreement
    if histogram_bpm is not None:
        median_bpm = float(np.median(bpms))
        ratio = min(median_bpm, histogram_bpm) / max(median_bpm, histogram_bpm)
        agreement = ratio
    else:
        agreement = 0.5

    # Factor 5: How much of the evidence the fitted grid explains
    support = max(0.0, min(1.0, grid_support))

    # Weighted combination
    confidence = (
        0.30 * coverage
        + 0.25 * consistency
        + 0.15 * regularity
        + 0.10 * agreement
        + 0.20 * support
    )

    return round(max(0.0, min(1.0, confidence)), 2)
