"""
Nod kinematics — head-gesture evidence from pose landmark time series.

KEEP — pure math, no model dependencies. Source-agnostic: consumes the same
33-point LandmarkTimeSeries as dynamics.py and gesture.py.

W10 prototype (rung M). Not wired into analyze.py. W7 asked whether *limb*
movement carries a recoverable pulse and answered no; its recommendation was
to change the event definition rather than the peak-picker, because "a dancer
places phrase arrivals on the beat, which is a segmentation problem, not a
periodicity problem". This module makes that change, and takes its event
definitions from the one piece of literature that studies our exact event:

    Bishop & Goebl (2018), "Beating time: How ensemble musicians' cueing
    gestures communicate beat position and tempo" (Psychology of Music) —
    gesture acceleration patterns indicate beat position, specifically peak
    acceleration and the deceleration period following it, in leaders'
    *head-nodding* gestures; and visual cues at re-entry points after long
    pauses are especially salient.

Two consequences shape the design. The landmark set is the **head**, not the
limbs. And the test is **alignment to a known beat grid**, not periodicity in
isolation — W7's periodicity dissolved under a scale change, and alignment to
owner-verified truth is a stronger question than self-consistency.

Method, and why each step:

1.  **Head centroid, torso-normalized.** Nose and both ears, averaged with
    `nanmean` so one missing ear does not discard a frame, divided by the
    median shoulder-to-hip distance. Units are torso-lengths.
2.  **Vertical is the nod axis.** A nod is up-and-down. MediaPipe's image y
    grows *downward*, so the bottom of a nod is a local *maximum* of y.
3.  **Three pre-declared event definitions, never one chosen post-hoc.**
    Peak acceleration (the literature's claim), nod bottom (the ictus), and
    head-speed minima (W7's falsified definition moved to the head, kept as
    the control that says whether the landmark set or the kinematic quantity
    was what mattered).
4.  **Circular-shift null.** Rotating the event train modulo clip duration
    preserves the event count and every inter-onset interval and destroys
    only phase, so it asks the one question that matters — are the events at
    *these* beat positions — and is immune to the density inflation that a
    wide match tolerance would otherwise buy. Its blind spot is stated where
    it bites: against a perfectly isochronous grid a shift of one beat period
    realigns, so the null is weak by construction on isochronous references
    and the per-clip mean null F is reported so a reader can see it.
5.  **NaN gaps are handled before anything else.** W7's secondary finding:
    a clip reporting `detection_rate = 1.00` still carried 0.43 % NaN, which
    was enough to poison a median threshold and erase every event in it.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from musical_perception.types import LandmarkTimeSeries

# MediaPipe landmark indices (shared convention with dynamics.py/gesture.py)
_NOSE = 0
_LEFT_EAR = 7
_RIGHT_EAR = 8
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24

# The head. Ears bracket the skull, so nose-plus-ears tracks the nod without
# tracking the jaw, which moves when the teacher talks and not when she nods.
_HEAD_LANDMARKS = [_NOSE, _LEFT_EAR, _RIGHT_EAR]

# Same 60 ms boxcar as gesture.py: removes landmark jitter (well above 15 Hz)
# without touching anything at beat rate (< 4 Hz). Applied once per
# differentiation, since differencing amplifies exactly what it removes.
_SMOOTH_SECONDS = 0.06

# Two nod events closer than this are one nod with a wobble in it. Matches the
# minimum-IOI QC floor ratified for beat grids (annotation convention §4) and
# the value gesture.py uses, so the arms stay comparable.
_MIN_IOI_SECONDS = 0.20

# A NaN run shorter than this is interpolated across; longer runs are holes,
# and events are not emitted inside them.
_MAX_INTERP_GAP_SECONDS = 0.50

# An extremum counts as an event only if the signal travelled at least this
# many robust noise scales (1.4826 x MAD) to reach it. Fixed at 3.0 because
# that is the value the owner-ratified peakRate detector already uses for the
# same job (`prominence_mad_k: 3.0` in every committed beat grid's params) —
# an anchor from this repository's own convention rather than a swept knob.
_MIN_PROMINENCE_MADS = 3.0

EVENT_KINDS = ("peak_acceleration", "nod_bottom", "speed_minima")


@dataclass
class HeadSeries:
    """Torso-normalized head kinematics for one clip."""
    times: np.ndarray        # (N,) seconds
    vertical: np.ndarray     # (N,) torso-lengths, +down (MediaPipe y sense)
    speed: np.ndarray        # (N,) torso-lengths/s, 3-D centroid speed
    accel: np.ndarray        # (N,) torso-lengths/s^2, vertical acceleration
    valid: np.ndarray        # (N,) bool — False inside an un-interpolated hole
    nan_fraction: float      # fraction of head landmarks missing before repair
    hole_seconds: float      # total duration of un-interpolated holes

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0


def _torso_scale(landmarks: np.ndarray) -> float:
    """Median shoulder-midpoint to hip-midpoint distance, in frame units."""
    shoulder = (landmarks[:, _LEFT_SHOULDER, :] + landmarks[:, _RIGHT_SHOULDER, :]) / 2
    hip = (landmarks[:, _LEFT_HIP, :] + landmarks[:, _RIGHT_HIP, :]) / 2
    dist = np.linalg.norm(shoulder - hip, axis=-1)
    with np.errstate(invalid="ignore"):
        scale = float(np.nanmedian(dist)) if np.isfinite(dist).any() else 0.0
    # A degenerate or missing torso must not become a divide-by-zero that
    # silently inflates every number in the clip.
    return scale if np.isfinite(scale) and scale > 1e-6 else 1.0


def _smooth(signal: np.ndarray, width: int) -> np.ndarray:
    """Boxcar smoothing with edge padding.

    `mode="same"` alone zero-pads, which drags the first and last samples
    toward zero and manufactures an extremum at each end of every clip.
    """
    if width <= 1 or len(signal) < width:
        return signal
    kernel = np.ones(width) / width
    pad = width // 2
    padded = np.pad(signal, pad, mode="edge")
    return np.convolve(padded, kernel, mode="same")[pad:pad + len(signal)]


def _interpolate_gaps(
    values: np.ndarray, times: np.ndarray, max_gap: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate short NaN runs; report which samples stay invalid.

    Returns (repaired, valid). Samples inside a run longer than `max_gap`,
    and any leading/trailing run, are repaired for continuity but flagged
    invalid so no event is emitted from them.
    """
    out = np.array(values, dtype=np.float64)
    finite = np.isfinite(out)
    valid = finite.copy()
    if not finite.any():
        return np.zeros_like(out), np.zeros(len(out), dtype=bool)

    idx = np.arange(len(out))
    out = np.interp(idx, idx[finite], out[finite])

    # Mark long runs (and the edges, which np.interp fills by extension)
    # invalid. `finite` is the pre-repair mask, so runs are the False blocks.
    edges = np.flatnonzero(np.diff(np.concatenate(([0], (~finite).view(np.int8), [0]))))
    for start, stop in zip(edges[::2], edges[1::2]):
        run_seconds = times[min(stop, len(times) - 1)] - times[start]
        if run_seconds > max_gap or start == 0 or stop >= len(out):
            valid[start:stop] = False
    return out, valid


def head_series(lts: LandmarkTimeSeries) -> HeadSeries:
    """Torso-normalized head position, speed and vertical acceleration."""
    lm = np.asarray(lts.landmarks, dtype=np.float64)
    ts = np.asarray(lts.timestamps, dtype=np.float64)
    empty = HeadSeries(*(np.array([]),) * 4, np.array([], dtype=bool), 0.0, 0.0)
    if lm.ndim != 3 or len(lm) < 5:
        return empty

    dt = 1.0 / lts.fps if lts.fps else float(np.median(np.diff(ts)))
    scale = _torso_scale(lm)
    width = max(1, int(round(_SMOOTH_SECONDS / dt)))

    head = lm[:, _HEAD_LANDMARKS, :]                      # (N, 3, 3)
    nan_fraction = float(np.isnan(head).mean())
    with np.errstate(invalid="ignore"):
        # An all-NaN frame is a legitimate input (the pose model dropped it);
        # nanmean's warning about it is noise, and the frame is repaired below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            centroid = np.nanmean(head, axis=1) / scale   # (N, 3)

    # Repair each axis on the same validity mask: a frame is usable only if
    # the whole centroid is.
    frame_ok = np.isfinite(centroid).all(axis=1)
    repaired = np.empty_like(centroid)
    for axis in range(3):
        col = np.where(frame_ok, centroid[:, axis], np.nan)
        repaired[:, axis], valid = _interpolate_gaps(col, ts, _MAX_INTERP_GAP_SECONDS)
    hole_seconds = float((~valid).sum() * dt)

    vertical = _smooth(repaired[:, 1], width)
    velocity_y = np.gradient(vertical, dt)
    accel = _smooth(np.gradient(_smooth(velocity_y, width), dt), width)

    step = np.linalg.norm(np.diff(repaired, axis=0), axis=-1) / dt
    speed = _smooth(np.concatenate(([step[0]], step)), width)

    return HeadSeries(ts, vertical, speed, accel, valid, nan_fraction, hole_seconds)


def _finite_median(signal: np.ndarray) -> float:
    """Median over finite samples only.

    A plain median across NaN returns NaN, after which nothing clears the
    floor and the clip silently reports zero events — the exact mechanism
    that zeroed 14 of 22 clips in W7.
    """
    finite = signal[np.isfinite(signal)]
    return float(np.median(finite)) if len(finite) else 0.0


def _local_extrema(signal: np.ndarray, maxima: bool) -> np.ndarray:
    """Indices of strict interior local extrema."""
    s = signal if maxima else -signal
    return np.flatnonzero((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])) + 1


def _prominence(signal: np.ndarray, idx: np.ndarray, maxima: bool) -> np.ndarray:
    """Height of each extremum above the higher of its bracketing troughs.

    Height alone cannot separate a nod from a wiggle: a signal that sits at
    baseline between nods puts half its noise above its own median, so a
    median floor keeps every one of those wiggles. Prominence asks how far
    the signal actually travelled to get there, which is the question.
    """
    s = signal if maxima else -signal
    troughs = _local_extrema(s, maxima=False)
    if len(troughs) == 0:
        return np.abs(s[idx] - np.min(s))
    out = np.empty(len(idx))
    for k, i in enumerate(idx):
        left = troughs[troughs < i]
        right = troughs[troughs > i]
        lo = s[left[-1]] if len(left) else s[0]
        hi = s[right[0]] if len(right) else s[-1]
        out[k] = s[i] - max(lo, hi)
    return out


def _robust_scale(signal: np.ndarray) -> float:
    """1.4826 x MAD over finite samples — a noise scale that outliers cannot move."""
    finite = signal[np.isfinite(signal)]
    if len(finite) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(finite - np.median(finite))))


def _thin(indices: np.ndarray, times: np.ndarray, salience: np.ndarray,
          min_ioi: float) -> np.ndarray:
    """Greedy strongest-first thinning to a minimum inter-onset interval."""
    kept: list[int] = []
    for i in indices[np.argsort(-salience)]:
        if all(abs(times[i] - times[j]) >= min_ioi for j in kept):
            kept.append(int(i))
    return np.array(sorted(kept), dtype=int)


def nod_events(
    series: HeadSeries, kind: str, min_ioi: float = _MIN_IOI_SECONDS
) -> np.ndarray:
    """Event times (seconds) under one of the three pre-declared definitions.

    The same rule shapes all three arms so the comparison between them is
    fair: take the local extrema of the driving signal, keep those on the
    eventful side of that signal's own median (a nod is an event, not every
    wobble), and thin strongest-first to `min_ioi`.

    The floor is deliberately a **signal-level** one, not a rank over the
    candidates. A rank floor ("keep the top half of the extrema") was
    written first and the positive control killed it: it caps recall at 0.5
    by construction, so a perfectly recovered synthetic nod scored F = 0.615
    and would have been read as a weak channel. Signal-level floors let a
    clip that really does nod on every beat keep every nod.
    """
    if kind not in EVENT_KINDS:
        raise ValueError(f"unknown event kind: {kind!r}")
    if len(series.times) < 5:
        return np.array([])

    if kind == "peak_acceleration":
        # "Peak acceleration" is taken sign-agnostically, as magnitude: the
        # nod's turn points carry it, and choosing a sign here would be a
        # post-hoc parameter the pre-registration did not declare.
        driver, maxima = np.abs(series.accel), True
    elif kind == "nod_bottom":
        # Image y grows downward, so the bottom of the nod is a maximum.
        driver, maxima = series.vertical, True
    else:  # speed_minima — W7's definition, moved to the head
        driver, maxima = series.speed, False

    idx = _local_extrema(driver, maxima=maxima)
    if len(idx) == 0:
        return np.array([])
    # One floor, one form, all three arms: the deflection that produced this
    # extremum must be at least half the driving signal's own robust noise
    # scale. Dimensionless, so it means the same thing in torso-lengths,
    # torso-lengths/s and torso-lengths/s^2.
    salience = _prominence(driver, idx, maxima) - _MIN_PROMINENCE_MADS * _robust_scale(driver)

    if len(idx) == 0:
        return np.array([])

    # Events never come out of an un-interpolated hole: an unknown is not a nod.
    ok = series.valid[idx]
    idx, salience = idx[ok], salience[ok]
    if len(idx) == 0:
        return np.array([])

    # Median over *finite* saliences only. A plain median across NaN returns
    # NaN and then nothing passes the floor — the W7 failure, exactly.
    finite = np.isfinite(salience)
    idx, salience = idx[finite], salience[finite]
    keep = salience > 0.0
    idx, salience = idx[keep], salience[keep]
    if len(idx) == 0:
        return np.array([])

    return series.times[_thin(idx, series.times, salience, min_ioi)]


def circular_shift_null(
    reference: list[float],
    predicted: np.ndarray,
    duration: float,
    score_fn,
    n_draws: int = 500,
    seed: int = 20260830,
) -> tuple[float, float, float]:
    """Monte-Carlo p, observed score, and mean null score under phase rotation.

    The event train is rotated modulo `duration`, which preserves the event
    count and every inter-onset interval (the wrap creates one new interval)
    and destroys only the phase relative to `reference`.
    """
    observed = score_fn(reference, list(predicted))
    if duration <= 0 or len(predicted) == 0:
        return 1.0, observed, observed

    rng = np.random.default_rng(seed)
    t0 = float(predicted[0]) - float(predicted[0])  # rotations are relative
    base = np.asarray(predicted, dtype=np.float64) - float(np.min(predicted)) + t0
    null = np.empty(n_draws)
    for k in range(n_draws):
        shifted = np.sort((base + rng.uniform(0.0, duration)) % duration)
        null[k] = score_fn(reference, list(shifted))
    p = (1.0 + float((null >= observed).sum())) / (1.0 + n_draws)
    return p, observed, float(null.mean())


def partition_reentry(
    beats: list[float], gap_seconds: float = 2.0
) -> tuple[list[int], list[int]]:
    """Split grid beats into re-entry and interior indices.

    A re-entry beat is the first beat after a silence of at least
    `gap_seconds` — including the clip's first beat, which is a re-entry from
    the pre-roll. Bishop & Goebl: visual cues at re-entry points after long
    pauses are the salient ones, and a ballet class is re-entry after talk.
    """
    if not beats:
        return [], []
    reentry, interior = [0], []
    for i in range(1, len(beats)):
        (reentry if beats[i] - beats[i - 1] >= gap_seconds else interior).append(i)
    return reentry, interior
