"""Tests for precision/dynamics.py. No video files or MediaPipe needed."""

import numpy as np

from musical_perception.precision.dynamics import compute_quality, synthesize
from musical_perception.types import LandmarkTimeSeries, QualityProfile


def _make_landmarks(positions_fn, n_frames=60, fps=30.0):
    """
    Build a LandmarkTimeSeries from a function that generates key positions.

    positions_fn(t) should return (33, 3) array for time t in seconds.
    All frames are treated as detected (no NaN).
    """
    timestamps = np.arange(n_frames) / fps
    landmarks = np.array([positions_fn(t) for t in timestamps])
    return LandmarkTimeSeries(
        timestamps=timestamps,
        landmarks=landmarks,
        fps=fps,
        detection_rate=1.0,
    )


def _static_pose(hip_y=0.55):
    """A completely static pose — all landmarks fixed."""
    def fn(t):
        frame = np.full((33, 3), 0.5)
        frame[23, 1] = hip_y  # left hip
        frame[24, 1] = hip_y  # right hip
        return frame
    return fn


def _smooth_sine(amplitude=0.02, freq=1.0, hip_y=0.55):
    """Smooth sinusoidal movement on wrists — low jerk, gentle."""
    def fn(t):
        frame = np.full((33, 3), 0.5)
        frame[23, 1] = hip_y
        frame[24, 1] = hip_y
        # Wrists move in slow sine wave
        offset = amplitude * np.sin(2 * np.pi * freq * t)
        frame[15, 0] += offset  # left wrist x
        frame[16, 0] -= offset  # right wrist x
        return frame
    return fn


def _sharp_zigzag(amplitude=0.1, freq=5.0, hip_y=0.55):
    """Sharp zigzag movement on wrists — high jerk, staccato."""
    def fn(t):
        frame = np.full((33, 3), 0.5)
        frame[23, 1] = hip_y
        frame[24, 1] = hip_y
        # Wrists do sharp triangle wave
        period = 1.0 / freq
        phase = (t % period) / period
        offset = amplitude * (4 * abs(phase - 0.5) - 1)
        frame[15, 0] += offset
        frame[16, 0] -= offset
        return frame
    return fn


def _fast_movement(amplitude=0.15, freq=3.0, hip_y=0.55):
    """Fast large movement — high velocity, energetic."""
    def fn(t):
        frame = np.full((33, 3), 0.5)
        frame[23, 1] = hip_y
        frame[24, 1] = hip_y
        offset = amplitude * np.sin(2 * np.pi * freq * t)
        # All key landmarks move
        for idx in [11, 12, 15, 16, 27, 28]:
            frame[idx, 0] += offset
            frame[idx, 1] += offset * 0.5
        return frame
    return fn


# === compute_quality tests ===


def test_smooth_movement_high_articulation():
    """Smooth, slow movement → high articulation (legato)."""
    lm = _make_landmarks(_smooth_sine(amplitude=0.02, freq=0.5))
    q = compute_quality(lm)
    assert q.articulation > 0.6, f"Expected legato (>0.6), got {q.articulation}"


def test_sharp_movement_low_articulation():
    """Sharp, abrupt movement → low articulation (staccato)."""
    lm = _make_landmarks(_sharp_zigzag(amplitude=0.1, freq=5.0))
    q = compute_quality(lm)
    assert q.articulation < 0.4, f"Expected staccato (<0.4), got {q.articulation}"


def test_articulation_ordering():
    """Smooth movement should have higher articulation than sharp movement."""
    smooth = compute_quality(_make_landmarks(_smooth_sine(amplitude=0.02, freq=0.5)))
    sharp = compute_quality(_make_landmarks(_sharp_zigzag(amplitude=0.1, freq=5.0)))
    assert smooth.articulation > sharp.articulation


def test_low_hip_heavy_weight():
    """Low hip position → heavy weight."""
    lm = _make_landmarks(_static_pose(hip_y=0.65))
    q = compute_quality(lm)
    assert q.weight > 0.6, f"Expected heavy (>0.6), got {q.weight}"


def test_high_hip_light_weight():
    """High hip position → light weight."""
    lm = _make_landmarks(_static_pose(hip_y=0.40))
    q = compute_quality(lm)
    assert q.weight < 0.4, f"Expected light (<0.4), got {q.weight}"


def test_weight_ordering():
    """Lower hips should produce heavier weight than higher hips."""
    heavy = compute_quality(_make_landmarks(_static_pose(hip_y=0.65)))
    light = compute_quality(_make_landmarks(_static_pose(hip_y=0.40)))
    assert heavy.weight > light.weight


def test_fast_movement_high_energy():
    """Fast, large movement → high energy."""
    lm = _make_landmarks(_fast_movement(amplitude=0.15, freq=3.0))
    q = compute_quality(lm)
    assert q.energy > 0.5, f"Expected energetic (>0.5), got {q.energy}"


def test_static_pose_low_energy():
    """Static pose → low energy (calm)."""
    lm = _make_landmarks(_static_pose())
    q = compute_quality(lm)
    assert q.energy < 0.2, f"Expected calm (<0.2), got {q.energy}"


def test_energy_ordering():
    """Fast movement should have higher energy than static pose."""
    fast = compute_quality(_make_landmarks(_fast_movement()))
    static = compute_quality(_make_landmarks(_static_pose()))
    assert fast.energy > static.energy


def test_insufficient_frames():
    """Fewer than 4 frames → neutral quality (0.5, 0.5, 0.5)."""
    lm = LandmarkTimeSeries(
        timestamps=np.array([0.0, 0.033, 0.066]),
        landmarks=np.full((3, 33, 3), 0.5),
        fps=30.0,
        detection_rate=1.0,
    )
    q = compute_quality(lm)
    assert q.articulation == 0.5
    assert q.weight == 0.5
    assert q.energy == 0.5


def test_values_in_range():
    """All quality values should be clamped to [0, 1]."""
    for gen in [_smooth_sine(), _sharp_zigzag(), _fast_movement(), _static_pose()]:
        q = compute_quality(_make_landmarks(gen))
        assert 0.0 <= q.articulation <= 1.0
        assert 0.0 <= q.weight <= 1.0
        assert 0.0 <= q.energy <= 1.0


# === synthesize tests ===


def test_synthesize_gemini_only():
    """When only Gemini available, return it unchanged."""
    gemini = QualityProfile(articulation=0.8, weight=0.3, energy=0.5)
    result = synthesize(gemini=gemini, pose=None)
    assert result.articulation == 0.8
    assert result.weight == 0.3
    assert result.energy == 0.5


def test_synthesize_pose_only():
    """When only pose available, return it unchanged."""
    pose = QualityProfile(articulation=0.4, weight=0.7, energy=0.6)
    result = synthesize(gemini=None, pose=pose)
    assert result.articulation == 0.4
    assert result.weight == 0.7
    assert result.energy == 0.6


def test_synthesize_both():
    """When both available, weighted merge (Gemini 0.7, pose 0.3)."""
    gemini = QualityProfile(articulation=0.8, weight=0.4, energy=0.6)
    pose = QualityProfile(articulation=0.4, weight=0.8, energy=0.2)
    result = synthesize(gemini=gemini, pose=pose)
    # articulation: 0.7*0.8 + 0.3*0.4 = 0.56 + 0.12 = 0.68
    assert result.articulation == 0.68
    # weight: 0.7*0.4 + 0.3*0.8 = 0.28 + 0.24 = 0.52
    assert result.weight == 0.52
    # energy: 0.7*0.6 + 0.3*0.2 = 0.42 + 0.06 = 0.48
    assert result.energy == 0.48


def test_synthesize_both_none():
    """When neither available, return None."""
    result = synthesize(gemini=None, pose=None)
    assert result is None
