"""
Gesture pulse — periodic movement evidence from pose landmark time series.

KEEP — pure math, no model dependencies. Source-agnostic: consumes the same
33-point LandmarkTimeSeries as dynamics.py.

W7 prototype (rung M). Not wired into analyze.py: this module exists to
answer whether movement carries a usable pulse before anything downstream
is allowed to depend on it. The motivating case is the voice-less take —
a clip where the teacher demonstrates to piano and says nothing, so the
marker channel the rest of the pipeline rests on is empty.

Method, and why each step:

1.  **Speed, torso-normalized.** Per-frame displacement of the gesturing
    limbs, divided by the median shoulder-to-hip distance, so the signal is
    in torso-lengths per second and a camera that moves closer does not
    read as a faster dancer.
2.  **Events at arrivals, not at peaks.** Candidate gesture beats are local
    *minima* of speed — the moment a limb reaches a position and stops.
    Peak speed falls between beats; arrival is what a dancer places on one.
    This is the movement analogue of Standing Lesson 1: the energetic event
    and the perceptual beat are not the same instant.
3.  **Rayleigh phase test, not mean IOI.** For a candidate period the event
    phases are tested for circular concentration (Standing Lesson 3: levels
    vote, they do not average). The null is exact for a Poisson process,
    and the search over candidate periods is corrected with a Monte-Carlo
    max-statistic null so that "best of 400 periods" is not read as
    significance.
4.  **Windowed.** A ballet exercise runs 30-150 s with talking in it and no
    obligation to hold one tempo throughout. Periodicity is measured in
    short windows and reported as a fraction of windows that carry it,
    which is a coverage statement rather than a single fragile estimate.
"""

from dataclasses import dataclass, field

import numpy as np

from musical_perception.types import LandmarkTimeSeries

# MediaPipe landmark indices (shared convention with dynamics.py)
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28

# The limbs that carry gesture. Hips and shoulders are deliberately excluded:
# they define the body's scale and its slow drift, not its articulation.
_GESTURE_LANDMARKS = [_LEFT_WRIST, _RIGHT_WRIST, _LEFT_ANKLE, _RIGHT_ANKLE]

# Landmark jitter lives well above 15 Hz; a 60 ms boxcar removes it without
# touching anything at beat rate (< 4 Hz).
_SMOOTH_SECONDS = 0.06

# Two gesture arrivals closer than this are one arrival with a wobble in it.
# Matches the minimum-IOI QC floor ratified for beat grids (convention §4).
_MIN_IOI_SECONDS = 0.20

# Candidate beat periods: 20-240 BPM. Deliberately wider than the tempo
# prior (Standing Lesson 2 — priors belong at level selection, not here).
_MIN_PERIOD = 0.25
_MAX_PERIOD = 3.00
_N_PERIODS = 400

_WINDOW_SECONDS = 12.0
_HOP_SECONDS = 6.0
_MIN_EVENTS_PER_WINDOW = 6
_N_NULL_DRAWS = 200
_SEED = 20260823


@dataclass
class PeriodicityWindow:
    """One analysis window's periodicity verdict."""
    start: float
    end: float
    n_events: int
    period: float          # seconds, best candidate
    resultant: float       # Rayleigh R in [0, 1] — phase concentration
    p_value: float         # max-statistic Monte-Carlo p over the period grid

    @property
    def bpm(self) -> float:
        return 60.0 / self.period if self.period > 0 else 0.0

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


@dataclass
class GestureResult:
    """Movement-pulse evidence for one clip."""
    duration: float
    event_times: np.ndarray
    windows: list[PeriodicityWindow] = field(default_factory=list)

    @property
    def event_rate(self) -> float:
        return len(self.event_times) / self.duration if self.duration > 0 else 0.0

    @property
    def significant_windows(self) -> list[PeriodicityWindow]:
        return [w for w in self.windows if w.significant]

    @property
    def coverage(self) -> float:
        """Fraction of analysed windows carrying significant periodicity."""
        return len(self.significant_windows) / len(self.windows) if self.windows else 0.0

    @property
    def dominant_period(self) -> float | None:
        """
        Median period across significant windows.

        Median rather than mean, and over significant windows only: an
        insignificant window's `period` is the argmax of noise and carries
        no information worth averaging in.
        """
        sig = self.significant_windows
        if not sig:
            return None
        return float(np.median([w.period for w in sig]))

    @property
    def dominant_bpm(self) -> float | None:
        p = self.dominant_period
        return 60.0 / p if p else None


def _torso_scale(landmarks: np.ndarray) -> float:
    """Median shoulder-midpoint to hip-midpoint distance, in frame units."""
    shoulder = (landmarks[:, _LEFT_SHOULDER, :] + landmarks[:, _RIGHT_SHOULDER, :]) / 2
    hip = (landmarks[:, _LEFT_HIP, :] + landmarks[:, _RIGHT_HIP, :]) / 2
    dist = np.linalg.norm(shoulder - hip, axis=-1)
    scale = float(np.median(dist))
    # A degenerate or missing torso must not turn into a divide-by-zero that
    # silently inflates every speed in the clip.
    return scale if scale > 1e-6 else 1.0


def _smooth(signal: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(signal) < width:
        return signal
    kernel = np.ones(width) / width
    return np.convolve(signal, kernel, mode="same")


def movement_speed(lts: LandmarkTimeSeries) -> tuple[np.ndarray, np.ndarray]:
    """
    Torso-normalized gesture speed.

    Returns:
        (times, speed) — times are frame midpoints in seconds, speed is in
        torso-lengths per second. Both empty if the clip is too short.
    """
    lm = np.asarray(lts.landmarks, dtype=np.float64)
    ts = np.asarray(lts.timestamps, dtype=np.float64)
    if lm.ndim != 3 or len(lm) < 3:
        return np.array([]), np.array([])

    dt = 1.0 / lts.fps if lts.fps else float(np.median(np.diff(ts)))
    scale = _torso_scale(lm)

    limbs = lm[:, _GESTURE_LANDMARKS, :]                 # (N, L, 3)
    step = np.linalg.norm(np.diff(limbs, axis=0), axis=-1)  # (N-1, L)
    speed = step.mean(axis=1) / (dt * scale)

    width = max(1, int(round(_SMOOTH_SECONDS / dt)))
    speed = _smooth(speed, width)
    times = (ts[:-1] + ts[1:]) / 2
    return times, speed


def gesture_events(
    times: np.ndarray,
    speed: np.ndarray,
    min_ioi: float = _MIN_IOI_SECONDS,
) -> np.ndarray:
    """
    Arrival times — local minima of speed, thinned to a minimum IOI.

    A minimum only counts as an arrival if it sits below the clip's median
    speed: the dancer must actually have slowed, not merely decelerated
    slightly while still travelling.
    """
    if len(speed) < 3:
        return np.array([])

    # Undetected frames arrive as NaN. They must not become arrivals (an
    # unknown is not a stillness) and must not enter the floor: one NaN in
    # a plain median turns the whole clip's threshold into NaN and silently
    # yields zero events, which reads as "this dancer never stopped".
    valid = np.isfinite(speed)
    if valid.sum() < 3:
        return np.array([])
    floor = float(np.median(speed[valid]))

    probe = np.where(valid, speed, np.inf)
    interior = np.arange(1, len(probe) - 1)
    is_min = (probe[interior] <= probe[interior - 1]) & (probe[interior] < probe[interior + 1])
    candidates = interior[is_min & (probe[interior] < floor)]
    if len(candidates) == 0:
        return np.array([])

    # Thin greedily by depth: the deepest arrival in any min_ioi neighbourhood
    # wins, so a jittery approach does not outvote the moment of stillness.
    order = candidates[np.argsort(speed[candidates])]
    kept: list[int] = []
    for idx in order:
        t = times[idx]
        if all(abs(t - times[k]) >= min_ioi for k in kept):
            kept.append(int(idx))
    return np.sort(times[np.array(sorted(kept))])


def _resultant(events: np.ndarray, period: float) -> float:
    """Rayleigh resultant length R of event phases at a candidate period."""
    phases = 2 * np.pi * (events / period)
    return float(np.hypot(np.cos(phases).sum(), np.sin(phases).sum()) / len(events))


def _best_period(events: np.ndarray, periods: np.ndarray) -> tuple[float, float]:
    scores = np.array([_resultant(events, p) for p in periods])
    i = int(np.argmax(scores))
    return float(periods[i]), float(scores[i])


def _hard_core_draw(rng: np.random.Generator, n: int, window: float,
                    min_ioi: float) -> np.ndarray:
    """
    `n` uniform points in [0, window] with every gap >= min_ioi.

    Exact rather than rejection-sampled: draw n uniform points in the space
    left over once every mandatory gap is reserved, sort, then hand each
    point back its share of the reserved space.
    """
    free = window - (n - 1) * min_ioi
    if free <= 0:
        return np.arange(n) * min_ioi
    return np.sort(rng.uniform(0.0, free, size=n)) + np.arange(n) * min_ioi


def windowed_periodicity(
    events: np.ndarray,
    duration: float,
    window: float = _WINDOW_SECONDS,
    hop: float = _HOP_SECONDS,
    n_null: int = _N_NULL_DRAWS,
    seed: int = _SEED,
    min_ioi: float = _MIN_IOI_SECONDS,
) -> list[PeriodicityWindow]:
    """
    Rayleigh periodicity per window, against a hard-core uniform null.

    The null places the same number of events uniformly at random subject to
    the same minimum IOI the detector enforces. Two nulls were tried and
    rejected first, both disclosed because each failure says something:

    * **Plain uniform** — lacks the min-IOI constraint the observations
      carry, so the test reported the constraint rather than the
      periodicity, and scored every clip at the short edge of the period
      grid. A null must share the observations' constraints or it measures
      them.
    * **Shuffled IOIs** — permuting intervals is the *identity* on an
      isochronous train, so this null has exactly zero power against the
      one hypothesis the module exists to test. Caught by
      `test_periodic_events_beat_the_shuffled_ioi_null`, on a synthetic
      perfectly-periodic input that it scored p = 0.31.

    The null statistic is the *maximum* resultant over the whole period
    grid, which is what keeps "best of 400 candidate periods" from reading
    as evidence.
    """
    rng = np.random.default_rng(seed)
    periods = np.linspace(_MIN_PERIOD, _MAX_PERIOD, _N_PERIODS)
    out: list[PeriodicityWindow] = []

    start = 0.0
    while start + window <= duration + 1e-9:
        end = start + window
        local = events[(events >= start) & (events < end)] - start
        if len(local) >= _MIN_EVENTS_PER_WINDOW:
            period, r = _best_period(local, periods)
            null = np.array([
                _best_period(_hard_core_draw(rng, len(local), window, min_ioi), periods)[1]
                for _ in range(n_null)
            ])
            # +1 in numerator and denominator: an observed value never gets
            # p = 0 from a finite null.
            p = float((np.sum(null >= r) + 1) / (n_null + 1))
            out.append(PeriodicityWindow(start, end, len(local), period, r, p))
        start += hop

    return out


def analyze_gesture(lts: LandmarkTimeSeries, seed: int = _SEED) -> GestureResult:
    """Full gesture-pulse analysis for one clip."""
    times, speed = movement_speed(lts)
    if len(times) == 0:
        return GestureResult(duration=0.0, event_times=np.array([]))

    duration = float(times[-1] - times[0])
    events = gesture_events(times, speed)
    windows = windowed_periodicity(events - times[0], duration, seed=seed) if len(events) else []
    return GestureResult(duration=duration, event_times=events, windows=windows)
