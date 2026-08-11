"""Tap-assist annotator: peakRate detection on synthetic audio, grid I/O,
and the Audacity label round trip. Synthetic signals only — no media."""

import numpy as np
import pytest

from musical_perception.annotation.grids import (
    BeatGrid,
    beats_from_label_text,
    load_grid,
    load_grids,
    save_grid,
    to_label_text,
)
from musical_perception.annotation.peakrate import PeakRateParams, peak_rate_events

try:
    import parselmouth  # noqa: F401
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False

SR = 16000
NO_GATE = PeakRateParams(voiced_gate=False)


def _harmonic_burst(t0: float, dur: float, total: float, f0: float = 150.0):
    """A voiced-like syllable: harmonic complex, 10 ms attack, 150 ms release
    (speech-shaped decay — a hard cutoff rings the 10 Hz envelope filter)."""
    n = int(total * SR)
    y = np.zeros(n)
    ts = np.arange(int(dur * SR)) / SR
    tone = sum(np.sin(2 * np.pi * f0 * k * ts) for k in range(2, 11)) / 9.0
    attack = np.minimum(ts / 0.010, 1.0)
    release = np.minimum((dur - ts) / 0.150, 1.0)
    start = int(t0 * SR)
    y[start:start + len(ts)] = tone * attack * np.clip(release, 0.0, 1.0)
    return y


def test_peakrate_finds_each_syllable_onset():
    starts = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    y = sum(_harmonic_burst(t0, 0.25, 5.0) for t0 in starts)
    events = peak_rate_events(y, SR, NO_GATE)
    assert len(events) == len(starts)
    for t0, t in zip(starts, events):
        assert abs(t - t0) < 0.05, f"event {t:.3f} vs burst start {t0}"


def test_peakrate_min_distance_merges_close_bursts():
    y = _harmonic_burst(1.0, 0.25, 3.0) + _harmonic_burst(1.08, 0.25, 3.0)
    events = peak_rate_events(y, SR, NO_GATE)
    assert len(events) == 1
    assert abs(events[0] - 1.0) < 0.1


def test_peakrate_silence_yields_nothing():
    assert len(peak_rate_events(np.zeros(SR * 3), SR, NO_GATE)) == 0
    assert len(peak_rate_events(np.zeros(100), SR, NO_GATE)) == 0  # sub-100ms


@pytest.mark.skipif(not HAS_PARSELMOUTH, reason="needs praat-parselmouth")
def test_voiced_gate_drops_unvoiced_burst():
    rng = np.random.default_rng(7)
    y = _harmonic_burst(1.0, 0.40, 4.0)
    shape = np.ones(int(0.4 * SR))
    shape[:int(0.01 * SR)] = np.linspace(0, 1, int(0.01 * SR))
    shape[-int(0.15 * SR):] = np.linspace(1, 0, int(0.15 * SR))
    noise = np.zeros(4 * SR)
    noise[int(2.5 * SR):int(2.9 * SR)] = (
        0.5 * rng.standard_normal(int(0.4 * SR)) * shape
    )
    y = y + noise

    ungated = peak_rate_events(y, SR, NO_GATE)
    assert any(abs(t - 1.0) < 0.08 for t in ungated)
    assert any(2.45 < t < 2.95 for t in ungated)  # noise rises fire ungated

    gated = peak_rate_events(y, SR, PeakRateParams())
    assert len(gated) == 1  # the noise burst carries no voicing
    assert abs(gated[0] - 1.0) < 0.08


def test_grid_round_trip(tmp_path):
    grid = BeatGrid(
        clip="clip-a", provisional=True,
        beats=[0.5, 1.0], onsets=[0.5, 0.75, 1.0],
        media="audio/x.mp3", annotator="peakrate-tap-assist/1",
        params={"sr": 16000}, notes="test",
    )
    path = save_grid(grid, tmp_path)
    loaded = load_grid(path)
    assert loaded.clip == "clip-a"
    assert loaded.provisional is True
    assert loaded.beats == [0.5, 1.0]
    assert loaded.onsets == [0.5, 0.75, 1.0]
    assert loaded.params["sr"] == 16000
    assert load_grids(tmp_path) == {"clip-a": loaded}


def test_grid_validation_rejects_bad_data(tmp_path):
    with pytest.raises(ValueError, match="sorted"):
        save_grid(BeatGrid(clip="x", provisional=True, beats=[2.0, 1.0]), tmp_path)
    with pytest.raises(ValueError, match="negative"):
        save_grid(BeatGrid(clip="x", provisional=True, beats=[-0.1]), tmp_path)

    path = save_grid(BeatGrid(clip="x", provisional=True, beats=[1.0]), tmp_path)
    text = path.read_text().replace("provisional: true\n", "")
    path.write_text(text)
    with pytest.raises(ValueError, match="provisional"):
        load_grid(path)


def test_grid_filename_must_match_clip(tmp_path):
    path = save_grid(BeatGrid(clip="clip-a", provisional=True), tmp_path)
    path.rename(tmp_path / "clip-b.yaml")
    with pytest.raises(ValueError, match="filename"):
        load_grids(tmp_path)


def test_label_round_trip():
    grid = BeatGrid(clip="x", provisional=True, beats=[0.5, 1.25, 2.0])
    text = to_label_text(grid)
    assert text.splitlines()[0] == "0.5000\t0.5000\tbeat-1"
    assert beats_from_label_text(text) == [0.5, 1.25, 2.0]
    # corrected (deleted + nudged + reordered) labels still parse
    corrected = "1.3000\t1.3000\tbeat-2\n0.5000\t0.5000\tbeat-1\n"
    assert beats_from_label_text(corrected) == [0.5, 1.3]
