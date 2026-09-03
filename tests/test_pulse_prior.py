"""PP-1 — the acoustic pulse as a bounded tempo prior.

Two properties matter and neither is about accuracy: the estimator is the
frozen EB-1 arithmetic, and the prior can tilt but never veto.
"""

import numpy as np

from musical_perception.precision.posterior import (
    PULSE_PRIOR_SIGMA,
    PULSE_PRIOR_WEIGHT,
    _apply_pulse_prior,
)
from musical_perception.precision.pulse import all_pairs_period


def test_all_pairs_recovers_a_clean_isochronous_period():
    events = [i * 0.5 for i in range(24)]          # 120 BPM
    period = all_pairs_period(events)
    assert period is not None
    assert abs(60.0 / period - 120.0) < 1.0


def test_all_pairs_survives_two_events_per_beat():
    # the corpus's actual disease: syllables at 2x the beat, jittered
    rng = np.random.default_rng(0)
    beats = np.arange(0, 12, 0.5)                  # 120 BPM
    events = sorted(np.concatenate([beats, beats + 0.25]) + rng.normal(0, 0.01, 48))
    period = all_pairs_period(events)
    assert period is not None
    # it must land on the beat or a clean metric relative of it, never between
    ratio = (60.0 / period) / 120.0
    assert min(abs(ratio - r) for r in (0.5, 1.0, 2.0)) < 0.1


def test_all_pairs_refuses_a_stream_too_sparse_to_read():
    assert all_pairs_period([0.0, 1.0, 2.0]) is None
    assert all_pairs_period([]) is None


def test_prior_is_inert_without_events():
    axis = np.linspace(60.0, 180.0, 64)
    mass = np.full(64, 1.0 / 64)
    out, bpm = _apply_pulse_prior(axis, mass, None)
    assert bpm is None
    assert out is mass


def test_prior_tilts_toward_the_measured_period():
    axis = np.linspace(60.0, 180.0, 121)           # 1 BPM steps
    mass = np.full(len(axis), 1.0 / len(axis))
    events = [i * 0.5 for i in range(24)]          # 120 BPM
    out, bpm = _apply_pulse_prior(axis, mass, events)
    assert bpm is not None and abs(bpm - 120.0) < 1.0
    assert out[np.argmin(np.abs(axis - 120.0))] > out[np.argmin(np.abs(axis - 70.0))]
    assert abs(out.sum() - 1.0) < 1e-12


def test_prior_can_never_zero_a_hypothesis():
    """The bound Standing Lesson 2 is about: a prior, not a fold."""
    axis = np.linspace(60.0, 180.0, 121)
    mass = np.full(len(axis), 1.0 / len(axis))
    events = [i * 0.5 for i in range(24)]
    out, _ = _apply_pulse_prior(axis, mass, events)
    assert (out > 0).all()
    # worst case the far tail is down-weighted by exactly the mixture floor
    floor_ratio = (1.0 - PULSE_PRIOR_WEIGHT)
    assert (out / out.sum()).min() >= floor_ratio * mass.min() / (1.0 + PULSE_PRIOR_WEIGHT)
    assert 0.0 < PULSE_PRIOR_WEIGHT < 1.0 and PULSE_PRIOR_SIGMA > 0.0
