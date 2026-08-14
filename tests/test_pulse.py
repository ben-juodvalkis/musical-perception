"""Acoustic pulse extractor (rung 2): synthetic-audio behaviour only —
the nuclei-region pairing on top of the peakRate core. No media."""

import numpy as np
import pytest

from musical_perception.precision.pulse import (
    AcousticPulseParams,
    acoustic_pulse_events,
)

try:
    import parselmouth  # noqa: F401
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False

pytestmark = pytest.mark.skipif(
    not HAS_PARSELMOUTH, reason="needs praat-parselmouth"
)

SR = 16000


def _harmonic_burst(t0: float, dur: float, total: float, f0: float = 150.0,
                    amp: float = 1.0):
    """A voiced-like syllable: harmonic complex, 10 ms attack, 150 ms release
    (speech-shaped decay — a hard cutoff rings the 10 Hz envelope filter)."""
    n = int(total * SR)
    y = np.zeros(n)
    ts = np.arange(int(dur * SR)) / SR
    tone = sum(np.sin(2 * np.pi * f0 * k * ts) for k in range(2, 11)) / 9.0
    attack = np.minimum(ts / 0.010, 1.0)
    release = np.minimum((dur - ts) / 0.150, 1.0)
    start = int(t0 * SR)
    y[start:start + len(ts)] = amp * tone * attack * np.clip(release, 0.0, 1.0)
    return y


def test_one_event_per_syllable():
    starts = [1.0, 1.6, 2.2, 2.8, 3.4]
    y = sum(_harmonic_burst(t0, 0.30, 5.0) for t0 in starts)
    events = acoustic_pulse_events(y, SR)
    assert len(events) == len(starts)
    for t0, t in zip(starts, events):
        assert abs(t - t0) < 0.06, f"event {t:.3f} vs syllable start {t0}"


def test_double_rise_in_one_nucleus_collapses_to_first():
    # A second envelope rise inside one syllable (the five/eight
    # diphthong re-fire): amplitude dips mid-syllable but never by the
    # 4 dB nucleus dip, so the region keeps only the FIRST peakRate event.
    ts = np.arange(int(0.5 * SR)) / SR
    tone = sum(np.sin(2 * np.pi * 150.0 * k * ts) for k in range(2, 11)) / 9.0
    shape = np.interp(
        ts, [0.0, 0.01, 0.20, 0.26, 0.27, 0.45, 0.50], [0, 1, 1, 0.75, 1, 1, 0]
    )
    y = np.zeros(3 * SR)
    y[SR:SR + len(ts)] = tone * shape
    events = acoustic_pulse_events(y, SR)
    assert len(events) == 1
    assert abs(events[0] - 1.0) < 0.06


def test_unvoiced_noise_burst_is_dropped():
    rng = np.random.default_rng(7)
    y = _harmonic_burst(1.0, 0.40, 4.0)
    shape = np.ones(int(0.4 * SR))
    shape[:int(0.01 * SR)] = np.linspace(0, 1, int(0.01 * SR))
    shape[-int(0.15 * SR):] = np.linspace(1, 0, int(0.15 * SR))
    y[int(2.5 * SR):int(2.9 * SR)] += 0.5 * rng.standard_normal(int(0.4 * SR)) * shape
    events = acoustic_pulse_events(y, SR)
    assert len(events) == 1
    assert abs(events[0] - 1.0) < 0.06


def test_silence_yields_nothing():
    assert len(acoustic_pulse_events(np.zeros(SR * 3), SR)) == 0
    assert len(acoustic_pulse_events(np.zeros(100), SR)) == 0


def test_params_serialize_for_provenance():
    d = AcousticPulseParams().as_dict()
    assert d["min_dip_db"] == 4.0
    assert d["peakrate"]["lowpass_hz"] == 10.0  # frozen, timing-critical
