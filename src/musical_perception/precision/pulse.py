"""
Acoustic pulse extractor (rung 2) — the voice-as-drum event stream.

Review-1 "steal this first" #1 + #2, frozen in the rung-2 pre-registration
(RESEARCH-LOG 2026-08-14) before any scoring:

1. peakRate events (Oganian & Chang 2019) via the annotation layer's
   frozen detector — identical `PeakRateParams`, so rung-1.5's measured
   selection behaviour carries over unchanged.
2. de Jong & Wempe (2009) syllable-nuclei REGIONS via Praat intensity:
   peaks above a quantile-relative silence threshold with ≥ `min_dip_db`
   dips on both sides (4 dB — review-1 §1.2's marked-speech retuning,
   conservative end), voiced-gated with the same AC pitch settings.
   The FIRST peakRate event inside each nucleus region is the event
   time (first, not largest: the documented five/eight diphthong
   re-fire is a second rise inside one nucleus); events outside every
   region are dropped.

No tactus selection happens here: the output is a syllable-rate stream,
scored by the level-collapsed §2.1 metrics that were designed for it.
"""

from dataclasses import dataclass, field

import numpy as np

from musical_perception.annotation.peakrate import (
    PeakRateParams,
    _voiced_times,
    peak_rate_events,
)


@dataclass(frozen=True)
class AcousticPulseParams:
    """Frozen extractor constants (pre-registered; not tuned on results)."""
    peakrate: PeakRateParams = field(default_factory=PeakRateParams)
    intensity_min_pitch_hz: float = 50.0   # Praat To Intensity minimum pitch
    intensity_time_step_s: float = 0.01
    silence_db: float = 25.0               # threshold: reference − this
    min_dip_db: float = 4.0                # marked speech: review-1 §1.2
    # W2.5 (2026-08-26), pre-registered before measurement. Both are closed
    # vocabularies; an unknown value is a hard error, never a silent
    # fallback to the old behaviour.
    # Default flipped to "all" by W2.5's pre-registered adoption rule
    # (ALL F_lc 0.839 -> 0.876, step_names 0.742 -> 0.811, zero clips
    # losing R@tac). "first" is retained so rung 2's blessed stream stays
    # reproducible. `voiced_median` is the parked backlog-(i) speech-band
    # floor: measured, and it changes region bounds without changing a
    # single emitted event on all 28 verified clips -- kept only so that
    # falsification is re-runnable.
    silence_reference: str = "q99"         # q99 | voiced_median
    events_per_nucleus: str = "all"        # all | first

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["peakrate"] = self.peakrate.as_dict()
        return d


def _nucleus_regions(
    y: np.ndarray, sr: int, params: AcousticPulseParams
) -> list[tuple[float, float]]:
    """Syllable-nucleus regions (start, end) from the Praat intensity contour.

    A nucleus is an intensity peak above the quantile-relative silence
    threshold separated from its neighbours by dips of at least
    `min_dip_db` (the paper's dip criterion applied directly — scipy
    prominence is NOT a substitute: equal-height plateau samples make
    neighbouring maxima each inherit full prominence). Candidates whose
    shared valley is too shallow merge into the higher peak. Region
    bounds are the intensity minima between consecutive nuclei; the
    outer bounds fall at the nearest below-threshold frame (or the clip
    edge).
    """
    import parselmouth  # lazy: praat-parselmouth lives in the [prosody] extra
    from scipy.signal import find_peaks

    snd = parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=sr)
    intensity = snd.to_intensity(
        minimum_pitch=params.intensity_min_pitch_hz,
        time_step=params.intensity_time_step_s,
    )
    values = np.array(intensity.values[0], dtype=float)
    times = np.array(intensity.xs())
    finite = np.isfinite(values)
    if not finite.any():
        return []
    floor = float(values[finite].min()) - 1.0
    values = np.where(finite, values, floor)

    if params.silence_reference == "q99":
        reference = float(np.percentile(values[finite], 99))
    elif params.silence_reference == "voiced_median":
        # W2.5 V2: the clip's own speech-band level rather than its loudest
        # 1%. Voiced frames ARE the speech band; unvoiced frames and silence
        # are exactly what the threshold exists to exclude.
        voiced_ref = _voiced_times(y, sr, params.peakrate)
        if voiced_ref.size:
            near = np.min(
                np.abs(times[:, None] - voiced_ref[None, :]), axis=1
            ) <= params.peakrate.voiced_window_s
        else:
            near = np.zeros(times.shape, dtype=bool)
        pool = values[finite & near]
        reference = (
            float(np.median(pool)) if pool.size
            else float(np.percentile(values[finite], 99))
        )
    else:
        raise ValueError(
            f"unknown silence_reference: {params.silence_reference!r}"
        )
    threshold = reference - params.silence_db
    candidates, _ = find_peaks(values, height=threshold)
    kept: list[int] = []
    for c in candidates:
        if not kept:
            kept.append(int(c))
            continue
        valley = float(values[kept[-1]:c + 1].min())
        if valley <= min(values[kept[-1]], values[c]) - params.min_dip_db:
            kept.append(int(c))
        elif values[c] > values[kept[-1]]:
            kept[-1] = int(c)  # same nucleus, higher summit
    peaks = np.array(kept, dtype=int)
    if peaks.size == 0:
        return []

    voiced = _voiced_times(y, sr, params.peakrate)
    if voiced.size == 0:
        return []
    peaks = np.array([
        p for p in peaks
        if np.min(np.abs(voiced - times[p])) <= params.peakrate.voiced_window_s
    ])
    if peaks.size == 0:
        return []

    below = values < threshold

    def outer_bound(peak: int, direction: int) -> float:
        i = peak
        while 0 <= i < len(values) and not below[i]:
            i += direction
        i = min(max(i, 0), len(values) - 1)
        return float(times[i])

    bounds = [outer_bound(peaks[0], -1)]
    for a, b in zip(peaks[:-1], peaks[1:]):
        valley = a + int(np.argmin(values[a:b + 1]))
        bounds.append(float(times[valley]))
    bounds.append(outer_bound(peaks[-1], +1))
    return list(zip(bounds[:-1], bounds[1:]))


def acoustic_pulse_events(
    y: np.ndarray, sr: int, params: AcousticPulseParams = AcousticPulseParams()
) -> np.ndarray:
    """Acoustic pulse event times (seconds) for one mono clip.

    Deterministic pure function of (audio, sr): voiced-gated peakRate
    events, region-filtered to the first event per syllable nucleus.
    """
    events = peak_rate_events(y, sr, params.peakrate)
    if events.size == 0:
        return events

    if params.events_per_nucleus not in ("first", "all"):
        raise ValueError(
            f"unknown events_per_nucleus: {params.events_per_nucleus!r}"
        )
    kept: list[float] = []
    regions = _nucleus_regions(y, sr, params)
    for start, end in regions:
        inside = events[(events >= start) & (events <= end)]
        if not inside.size:
            continue
        if params.events_per_nucleus == "first":
            kept.append(float(inside[0]))
        else:
            kept.extend(float(t) for t in inside)
    return np.array(sorted(set(kept)))
