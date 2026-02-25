"""
Movement dynamics — derive QualityProfile from pose landmark time series.

KEEP — pure math, no model dependencies. Source-agnostic: works with any
pose model that produces 33-point landmarks in the LandmarkTimeSeries format.

Signal-to-dimension mapping:
    articulation ← jerk (rate of acceleration change). High jerk = staccato
        (sharp transitions), low jerk = legato (smooth, flowing movement).
    weight ← hip vertical position and velocity. Low/slow hips = heavy
        (grounded, pressing), high/fast hips = light (buoyant, airy).
    energy ← overall velocity magnitude. Low velocity = calm (controlled),
        high velocity = energetic (active, explosive).

These mappings are provisional and need validation across exercise types.
See ROADMAP-v2.md Open Questions #1 and #2.
"""

import numpy as np

from musical_perception.types import LandmarkTimeSeries, QualityProfile

# MediaPipe landmark indices
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28

# Key landmarks for dynamics computation: hips, shoulders, wrists, ankles
_KEY_LANDMARKS = [
    _LEFT_HIP, _RIGHT_HIP,
    _LEFT_SHOULDER, _RIGHT_SHOULDER,
    _LEFT_WRIST, _RIGHT_WRIST,
    _LEFT_ANKLE, _RIGHT_ANKLE,
]

# Synthesis weights (Gemini-privileged). Provisional — needs refinement
# with more video samples. See ADR-004.
_GEMINI_WEIGHT = 0.7
_POSE_WEIGHT = 1.0 - _GEMINI_WEIGHT


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def _compute_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute velocity from position time series using central differences.

    Args:
        positions: (N,) or (N, D) array of positions.
        dt: Time step between frames (1/fps).

    Returns:
        (N-2,) or (N-2, D) array of velocities (central difference,
        excludes first and last frame).
    """
    if len(positions) < 3:
        return np.array([])
    return (positions[2:] - positions[:-2]) / (2 * dt)


def _compute_jerk(positions: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute jerk (rate of acceleration change) from position time series.

    Uses finite differences: jerk = d³x/dt³, approximated via three
    successive forward differences.

    Args:
        positions: (N,) or (N, D) array of positions.
        dt: Time step between frames.

    Returns:
        (M,) or (M, D) array of jerk values. M < N due to differencing.
    """
    if len(positions) < 4:
        return np.array([])
    # First difference → velocity
    v = np.diff(positions, axis=0) / dt
    # Second difference → acceleration
    a = np.diff(v, axis=0) / dt
    # Third difference → jerk
    j = np.diff(a, axis=0) / dt
    return j


def compute_quality(landmarks: LandmarkTimeSeries) -> QualityProfile:
    """
    Derive a QualityProfile from pose landmark time series.

    Computes three dimensions from landmark motion:
        - articulation: from jerk magnitude (sharp vs smooth)
        - weight: from hip vertical position (low = heavy, high = light)
        - energy: from overall velocity magnitude (slow = calm, fast = energetic)

    All values are mapped to 0.0–1.0 using empirical scaling. The scaling
    constants are provisional — calibrated against a single 18-second
    fondu/développé video. Needs validation across exercise types.

    Args:
        landmarks: LandmarkTimeSeries from any pose estimation module.

    Returns:
        QualityProfile with three dimensions.
    """
    dt = 1.0 / landmarks.fps if landmarks.fps > 0 else 1.0 / 30.0
    lm = landmarks.landmarks  # (N, 33, 3)

    # Use only detected frames (non-NaN)
    valid_mask = ~np.isnan(lm[:, 0, 0])
    if valid_mask.sum() < 4:
        # Not enough data — return neutral
        return QualityProfile(articulation=0.5, weight=0.5, energy=0.5)

    valid_lm = lm[valid_mask]

    # --- Articulation from jerk ---
    # Average jerk across key landmarks (use x,y only — z is depth, noisy)
    jerk_magnitudes = []
    for idx in _KEY_LANDMARKS:
        xy = valid_lm[:, idx, :2]  # (N, 2)
        jerk = _compute_jerk(xy, dt)
        if len(jerk) > 0:
            jerk_magnitudes.append(np.mean(np.linalg.norm(jerk, axis=1)))

    if jerk_magnitudes:
        mean_jerk = np.mean(jerk_magnitudes)
        # Empirical scaling: jerk of ~50 = midpoint (0.5).
        # Higher jerk = more staccato (lower articulation).
        # Sigmoid-like mapping: articulation = 1 / (1 + jerk/50)
        articulation = 1.0 / (1.0 + mean_jerk / 50.0)
    else:
        articulation = 0.5

    # --- Weight from hip vertical position ---
    # MediaPipe y-axis: 0 = top of frame, 1 = bottom.
    # Lower hip position (higher y value) = heavier/more grounded.
    hip_y = (valid_lm[:, _LEFT_HIP, 1] + valid_lm[:, _RIGHT_HIP, 1]) / 2.0
    mean_hip_y = np.mean(hip_y)
    # Empirical: hip y around 0.35–0.65 in typical dance video.
    # Map so 0.35 → 0.0 (light), 0.50 → 0.5 (mid), 0.65 → 1.0 (heavy).
    weight = _clamp((mean_hip_y - 0.35) / 0.30)

    # --- Energy from velocity ---
    # Average velocity magnitude across key landmarks
    vel_magnitudes = []
    for idx in _KEY_LANDMARKS:
        xy = valid_lm[:, idx, :2]
        vel = _compute_velocity(xy, dt)
        if len(vel) > 0:
            vel_magnitudes.append(np.mean(np.linalg.norm(vel, axis=1)))

    if vel_magnitudes:
        mean_vel = np.mean(vel_magnitudes)
        # Empirical: velocity ~0.5 normalized units/s = midpoint.
        # Sigmoid mapping: energy = 1 - 1/(1 + vel/0.5)
        energy = 1.0 - 1.0 / (1.0 + mean_vel / 0.5)
    else:
        energy = 0.5

    return QualityProfile(
        articulation=round(_clamp(articulation), 2),
        weight=round(_clamp(weight), 2),
        energy=round(_clamp(energy), 2),
    )


def synthesize(
    gemini: QualityProfile | None,
    pose: QualityProfile | None,
) -> QualityProfile | None:
    """
    Merge Gemini and pose-derived QualityProfiles into one.

    Strategy: Gemini-privileged. When both sources are available, uses
    a weighted average (Gemini 0.7, pose 0.3). When only one source
    is available, returns it unchanged. When neither is available,
    returns None.

    This is a provisional strategy. With more video samples and
    cross-exercise validation, the weights and strategy should be
    refined — possibly per-dimension weights, Bayesian updating,
    or pose-override when pose confidence is high. See ADR-004.

    Args:
        gemini: QualityProfile from Gemini analysis, or None.
        pose: QualityProfile from pose landmark dynamics, or None.

    Returns:
        Merged QualityProfile, or None if both inputs are None.
    """
    if gemini is None and pose is None:
        return None
    if gemini is None:
        return pose
    if pose is None:
        return gemini

    return QualityProfile(
        articulation=round(
            _GEMINI_WEIGHT * gemini.articulation + _POSE_WEIGHT * pose.articulation, 2
        ),
        weight=round(
            _GEMINI_WEIGHT * gemini.weight + _POSE_WEIGHT * pose.weight, 2
        ),
        energy=round(
            _GEMINI_WEIGHT * gemini.energy + _POSE_WEIGHT * pose.energy, 2
        ),
    )
