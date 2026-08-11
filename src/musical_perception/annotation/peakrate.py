"""
peakRate landmark detection (Oganian & Chang 2019) — the tap-assist core.

Recipe frozen from review-1 "steal this first" #1: bandpass the speech
band, rectify, low-pass the envelope at 10 Hz (zero-phase — MacIntyre et
al. 2022: the cutoff shifts timing, so it is a frozen constant, not a
knob), differentiate, half-wave rectify, and pick prominent peaks; keep
only peaks that Praat calls voiced within ±30 ms. peakRate events align
with vowel onsets (≈ P-centers) in running English — the annotation
anchor the grids require.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeakRateParams:
    """Frozen detector constants; recorded verbatim into every grid."""
    sr: int = 16000                 # analysis rate (media is resampled to this)
    band_lo_hz: float = 300.0       # speech-band edges: vowel formant energy,
    band_hi_hz: float = 3000.0      # rejects f0 rumble and fricative hiss
    lowpass_hz: float = 10.0        # envelope smoothing — FROZEN (timing-critical)
    butter_order: int = 4
    prominence_mad_k: float = 3.0   # peak prominence ≥ k · MAD(derivative)
    # Degenerate-input guard, conditional so real audio is never affected:
    # when MAD is orders of magnitude below the max slope the signal is
    # silence-dominated (MAD ~1e-7·max on synthetic silence vs ≥ 2e-3·max
    # measured across every DEV clip) and zero-phase Butterworth ringing
    # (≤ ~4% of max, measured) would pass any MAD multiple — only then
    # switch to a fraction-of-max prominence floor.
    mad_degenerate_frac: float = 0.0005
    prominence_rel_floor: float = 0.05
    min_distance_s: float = 0.12    # syllable-rate floor
    voiced_gate: bool = True        # require Praat voicing within ±window
    voiced_window_s: float = 0.03
    pitch_floor_hz: float = 75.0    # adult close-mic floor (skips creak octaves)
    pitch_ceiling_hz: float = 450.0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def envelope(y: np.ndarray, sr: int, params: PeakRateParams) -> np.ndarray:
    """Speech-band amplitude envelope, zero-phase throughout."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    band = butter(
        params.butter_order,
        [params.band_lo_hz / nyq, params.band_hi_hz / nyq],
        btype="band", output="sos",
    )
    rectified = np.abs(sosfiltfilt(band, np.asarray(y, dtype=float)))
    smooth = butter(params.butter_order, params.lowpass_hz / nyq, output="sos")
    return sosfiltfilt(smooth, rectified)


def _voiced_times(y: np.ndarray, sr: int, params: PeakRateParams) -> np.ndarray:
    """Praat AC pitch frame times that carry a voiced estimate."""
    import parselmouth  # lazy: praat-parselmouth lives in the [prosody] extra

    snd = parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(
        time_step=0.01,
        pitch_floor=params.pitch_floor_hz,
        pitch_ceiling=params.pitch_ceiling_hz,
    )
    freq = pitch.selected_array["frequency"]
    times = pitch.xs()
    return times[(freq > 0) & np.isfinite(freq)]


def peak_rate_events(
    y: np.ndarray, sr: int, params: PeakRateParams = PeakRateParams()
) -> np.ndarray:
    """peakRate event times (seconds) for one mono clip.

    Deterministic; returns a sorted float array. With `voiced_gate` the
    breath/click/plosive-burst peaks that carry no voicing are dropped —
    the pairing that makes the suggestions correctable rather than noisy.
    """
    from scipy.signal import find_peaks

    y = np.asarray(y, dtype=float)
    if y.size < sr // 10:  # under 100 ms of audio has no syllables to find
        return np.array([])

    env = envelope(y, sr, params)
    rate = np.diff(env) * sr          # envelope slope per second
    rate_hw = np.clip(rate, 0.0, None)

    mad = float(np.median(np.abs(rate - np.median(rate))))
    max_slope = float(rate_hw.max())
    if mad < params.mad_degenerate_frac * max_slope:
        prominence = params.prominence_rel_floor * max_slope
    else:
        prominence = params.prominence_mad_k * mad
    prominence = max(prominence, 1e-12)
    peaks, _ = find_peaks(
        rate_hw,
        prominence=prominence,
        distance=max(1, int(params.min_distance_s * sr)),
    )
    times = (peaks + 0.5) / sr        # diff[i] sits between samples i and i+1

    if params.voiced_gate and times.size:
        voiced = _voiced_times(y, sr, params)
        if voiced.size == 0:
            return np.array([])
        keep = [
            t for t in times
            if np.min(np.abs(voiced - t)) <= params.voiced_window_s
        ]
        times = np.array(keep)

    return times
