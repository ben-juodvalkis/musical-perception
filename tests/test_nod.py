"""Tests for the W10 nod-kinematics module.

The positive controls are the point of this file. W7's lesson, earned the
hard way: a null with no power returns calm, plausible, entirely fictional
non-findings, and nothing in the output distinguishes it from an honest
negative. The more confidently a session expects a negative result, the more
it needs the control proving it could have detected a positive one.
"""

import numpy as np
import pytest

from musical_perception.precision.nod import (
    EVENT_KINDS,
    circular_shift_null,
    head_series,
    nod_events,
    partition_reentry,
)
from musical_perception.types import LandmarkTimeSeries

FPS = 50.0


def _synthetic_nod(beats, duration=30.0, fps=FPS, depth=0.15, seed=7):
    """A pose series whose head sits at the bottom of a nod on every beat.

    Deliberately *non*-isochronous (the caller supplies drifting beats with
    gaps), because a circular-shift null is degenerate against a perfectly
    periodic reference: a rotation by one beat period realigns it.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * fps)
    ts = np.arange(n) / fps
    y = np.zeros(n)
    # Each beat contributes one raised-cosine dip, so y (which grows downward)
    # peaks exactly at the beat time.
    for b in beats:
        half = 0.18
        m = np.abs(ts - b) < half
        y[m] += depth * 0.5 * (1 + np.cos(np.pi * (ts[m] - b) / half))
    y += 0.002 * rng.standard_normal(n)

    lm = np.zeros((n, 33, 3))
    lm[:, 0, 1] = 0.30 + y          # nose
    lm[:, 7, 1] = 0.30 + y
    lm[:, 8, 1] = 0.30 + y
    lm[:, 0, 0], lm[:, 7, 0], lm[:, 8, 0] = 0.50, 0.46, 0.54
    lm[:, 11, 1] = lm[:, 12, 1] = 0.50   # shoulders
    lm[:, 23, 1] = lm[:, 24, 1] = 0.80   # hips -> torso scale 0.30
    lm[:, 11, 0], lm[:, 23, 0] = 0.45, 0.45
    lm[:, 12, 0], lm[:, 24, 0] = 0.55, 0.55
    return LandmarkTimeSeries(timestamps=ts, landmarks=lm, fps=fps, detection_rate=1.0)


def _drifting_beats(start=1.0, n=40, period=0.60, drift=0.004, gap_after=(12, 25)):
    beats, t, p = [], start, period
    for i in range(n):
        beats.append(t)
        p += drift
        t += p + (2.5 if i in gap_after else 0.0)
    return beats


def _f_at(tol):
    from musical_perception.evals.stage1 import score_pulse

    return lambda ref, pred: score_pulse(ref, pred, tol=tol)["f_measure"] or 0.0


# --- positive controls -------------------------------------------------


def test_synthetic_nod_is_recovered_at_the_bottom():
    """N6's core: a known nod must come back, or every negative is worthless."""
    beats = _drifting_beats()
    lts = _synthetic_nod(beats, duration=beats[-1] + 3.0)
    series = head_series(lts)
    events = nod_events(series, "nod_bottom")
    f = _f_at(0.15)(beats, list(events))
    assert f >= 0.90, f"positive control failed to recover the nod: F={f:.3f}"


def test_circular_shift_null_rejects_a_real_alignment():
    """The null must have power: a true alignment has to come back p < 0.01."""
    beats = _drifting_beats()
    lts = _synthetic_nod(beats, duration=beats[-1] + 3.0)
    series = head_series(lts)
    events = nod_events(series, "nod_bottom")
    p, observed, mean_null = circular_shift_null(
        beats, events, series.duration, _f_at(0.15), n_draws=200
    )
    assert p < 0.01, f"null has no power: p={p:.3f} (F={observed:.3f}, null={mean_null:.3f})"
    assert observed > mean_null


def test_circular_shift_null_does_not_reject_phase_destroyed_events():
    """And it must not reject when the phase is gone: no false alarms."""
    beats = _drifting_beats()
    lts = _synthetic_nod(beats, duration=beats[-1] + 3.0)
    series = head_series(lts)
    events = np.asarray(nod_events(series, "nod_bottom"))
    rotated = np.sort((events + 7.3) % series.duration)
    p, _, _ = circular_shift_null(beats, rotated, series.duration, _f_at(0.15), n_draws=200)
    assert p > 0.05, f"null false-alarms on phase-destroyed events: p={p:.3f}"


# --- the W7 failure mode, pinned ---------------------------------------


def test_nan_gaps_do_not_erase_every_event():
    """A clip with scattered NaN still yields events under every definition."""
    beats = _drifting_beats()
    lts = _synthetic_nod(beats, duration=beats[-1] + 3.0)
    lm = np.array(lts.landmarks)
    rng = np.random.default_rng(3)
    holes = rng.choice(len(lm), size=int(0.05 * len(lm)), replace=False)
    lm[holes, :, :] = np.nan
    poisoned = LandmarkTimeSeries(lts.timestamps, lm, lts.fps, 1.0)

    series = head_series(poisoned)
    assert series.nan_fraction > 0.0
    for kind in EVENT_KINDS:
        assert len(nod_events(series, kind)) > 0, f"{kind} was zeroed by NaN"


def test_long_hole_yields_no_events_inside_it():
    beats = _drifting_beats()
    lts = _synthetic_nod(beats, duration=beats[-1] + 3.0)
    lm = np.array(lts.landmarks)
    t0, t1 = 5.0, 8.0
    inside = (lts.timestamps >= t0) & (lts.timestamps <= t1)
    lm[inside, :, :] = np.nan
    series = head_series(LandmarkTimeSeries(lts.timestamps, lm, lts.fps, 1.0))
    assert series.hole_seconds > 2.0
    events = nod_events(series, "nod_bottom")
    assert not ((events > t0 + 0.2) & (events < t1 - 0.2)).any()


# --- plumbing ----------------------------------------------------------


def test_degenerate_inputs_return_empty_rather_than_raising():
    lts = LandmarkTimeSeries(np.zeros(2), np.zeros((2, 33, 3)), FPS, 1.0)
    series = head_series(lts)
    assert len(series.times) == 0
    assert len(nod_events(series, "nod_bottom")) == 0


def test_unknown_event_kind_is_an_error():
    beats = _drifting_beats(n=5)
    series = head_series(_synthetic_nod(beats, duration=10.0))
    with pytest.raises(ValueError):
        nod_events(series, "whatever")


def test_min_ioi_is_respected():
    beats = _drifting_beats()
    series = head_series(_synthetic_nod(beats, duration=beats[-1] + 3.0))
    for kind in EVENT_KINDS:
        events = nod_events(series, kind)
        if len(events) > 1:
            assert np.diff(events).min() >= 0.20 - 1e-9


def test_partition_reentry_splits_on_gaps():
    beats = [1.0, 1.6, 2.2, 5.0, 5.6, 6.2]
    reentry, interior = partition_reentry(beats, gap_seconds=2.0)
    assert reentry == [0, 3]
    assert interior == [1, 2, 4, 5]
