"""Tests for the gesture pulse prototype (W7). Pure math — no media, no models."""

import numpy as np
import pytest

from musical_perception.precision.gesture import (
    analyze_gesture,
    gesture_events,
    movement_speed,
    windowed_periodicity,
)
from musical_perception.types import LandmarkTimeSeries

FPS = 50.0


def _series(duration: float, period: float, nan_frames: slice | None = None,
            scale: float = 1.0) -> LandmarkTimeSeries:
    """A body whose limbs oscillate at `period`, so arrivals land every period."""
    n = int(duration * FPS)
    ts = np.arange(n) / FPS
    lm = np.zeros((n, 33, 3), dtype=np.float64)
    # Torso: shoulders at y=0, hips at y=scale, so torso length == scale.
    lm[:, 11, 1] = 0.0
    lm[:, 12, 1] = 0.0
    lm[:, 23, 1] = scale
    lm[:, 24, 1] = scale
    swing = scale * 0.5 * np.sin(2 * np.pi * ts / period)
    for idx in (15, 16, 27, 28):
        lm[:, idx, 0] = swing
    if nan_frames is not None:
        lm[nan_frames, :, :] = np.nan
    return LandmarkTimeSeries(timestamps=ts, landmarks=lm, fps=FPS, detection_rate=1.0)


def test_speed_is_torso_normalized():
    """A dancer twice as close to the camera is not a dancer moving twice as fast."""
    _, near = movement_speed(_series(6.0, 0.6, scale=2.0))
    _, far = movement_speed(_series(6.0, 0.6, scale=1.0))
    assert np.allclose(np.median(near), np.median(far), rtol=1e-6)


def test_events_land_at_arrivals():
    """A sinusoidal swing reverses twice per cycle; both reversals are arrivals."""
    lts = _series(12.0, 1.0)
    times, speed = movement_speed(lts)
    events = gesture_events(times, speed)
    assert len(events) > 0
    iois = np.diff(events)
    assert np.median(iois) == pytest.approx(0.5, abs=0.1)


def test_min_ioi_is_enforced():
    lts = _series(12.0, 1.0)
    times, speed = movement_speed(lts)
    events = gesture_events(times, speed, min_ioi=0.75)
    assert len(events) >= 2
    assert np.diff(events).min() >= 0.75


def test_nan_gaps_do_not_erase_every_event():
    """
    Regression, 2026-08-23: undetected frames arrive as NaN, and a plain
    median over them made the threshold NaN, so 14 of 22 real clips silently
    reported zero events. An unknown frame must be skipped, not fatal.
    """
    lts = _series(12.0, 1.0, nan_frames=slice(100, 140))
    times, speed = movement_speed(lts)
    assert np.isnan(speed).any(), "the fixture must actually contain a gap"
    assert len(gesture_events(times, speed)) > 0


def test_periodic_events_beat_the_hard_core_null():
    events = np.arange(0, 24.0, 0.6)
    windows = windowed_periodicity(events, 24.0, n_null=200)
    assert windows and all(w.significant for w in windows)
    assert np.median([w.period for w in windows]) == pytest.approx(0.6, abs=0.05)


def test_jittered_events_do_not_beat_the_null():
    """The null shares the min-IOI constraint, so mere irregularity must fail."""
    rng = np.random.default_rng(7)
    iois = rng.uniform(0.25, 1.2, size=60)
    events = np.cumsum(iois)
    windows = windowed_periodicity(events[events < 24.0], 24.0, n_null=200)
    assert windows
    assert sum(w.significant for w in windows) == 0


def test_analyze_gesture_on_a_still_body():
    n = int(8.0 * FPS)
    lts = LandmarkTimeSeries(
        timestamps=np.arange(n) / FPS,
        landmarks=np.zeros((n, 33, 3)),
        fps=FPS,
        detection_rate=1.0,
    )
    result = analyze_gesture(lts)
    assert result.dominant_period is None
    assert result.coverage == 0.0
