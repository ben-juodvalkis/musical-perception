"""
Rhythmic section detection from word onset timing.

KEEP — precision math, no AI dependencies. Identifies rhythmic sections
of speech from word onset timestamps alone, without requiring word
classification (beat/and/ah). Complementary to the Gemini-based tempo
pipeline: provides a classification-free tempo estimate that works with
step names, numbers, or any rhythmic speech.

Algorithm: sliding-window analysis of inter-onset interval regularity.
Windows with low coefficient of variation (CV < 0.4) are classified as
rhythmic. BPM is estimated from the mean IOI within rhythmic sections.
"""

import numpy as np

from musical_perception.types import (
    OnsetTempoResult,
    RhythmicSection,
    TimestampedWord,
)


def detect_onset_tempo(
    words: list[TimestampedWord],
    *,
    window_sec: float = 3.0,
    step_sec: float = 0.5,
    cv_threshold: float = 0.4,
    min_words_per_window: int = 3,
    min_ioi: float = 0.15,
    max_ioi: float = 2.0,
) -> OnsetTempoResult | None:
    """
    Detect tempo from word onset timing without word classification.

    Slides overlapping windows over word onsets and identifies sections
    where words are regularly spaced (low CV). Estimates BPM from the
    mean inter-onset interval in those sections.

    Args:
        words: Timestamped words from transcription.
        window_sec: Sliding window duration in seconds.
        step_sec: Step size between windows in seconds.
        cv_threshold: Maximum CV to consider a window rhythmic.
        min_words_per_window: Minimum word onsets per window.
        min_ioi: Minimum inter-onset interval (filters sub-word artifacts).
        max_ioi: Maximum inter-onset interval (filters long pauses).

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

    merged = _merge_overlapping_sections(sections)

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

    confidence = _compute_confidence(merged, total_duration, histogram_bpm)

    return OnsetTempoResult(
        bpm=bpm,
        confidence=confidence,
        rhythmic_sections=merged,
        total_duration=round(total_duration, 2),
        rhythmic_coverage=coverage,
        ioi_histogram_peak_bpm=histogram_bpm,
    )


def _compute_window_sections(
    onsets: np.ndarray,
    word_texts: list[str],
    window_sec: float,
    step_sec: float,
    cv_threshold: float,
    min_words_per_window: int,
    min_ioi: float = 0.15,
) -> list[RhythmicSection]:
    """Slide windows over onsets and identify rhythmic sections."""
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
    if ioi_min == ioi_max:
        # All IOIs identical — perfectly regular, no histogram needed
        return round(60.0 / ioi_min, 1) if ioi_min > 0 else None

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
) -> float:
    """Compute overall confidence from section consistency, coverage, and histogram agreement."""
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

    # Factor 3: Mean regularity (inverse of mean CV across sections)
    mean_cv = float(np.mean([s.cv for s in sections]))
    regularity = max(0.0, 1.0 - mean_cv)

    # Factor 4: Histogram agreement
    if histogram_bpm is not None:
        median_bpm = float(np.median(bpms))
        ratio = min(median_bpm, histogram_bpm) / max(median_bpm, histogram_bpm)
        agreement = ratio
    else:
        agreement = 0.5

    # Weighted combination
    confidence = (
        0.30 * coverage
        + 0.30 * consistency
        + 0.25 * regularity
        + 0.15 * agreement
    )

    return round(max(0.0, min(1.0, confidence)), 2)
