"""Tests for the accent-periodicity meter module (rung 3 / W2).

Hardcoded synthetic sequences only — no audio, no models, no grid files.
"""

import numpy as np

from musical_perception.precision.accent_meter import (
    METER_TEMPLATES,
    analyze_accent_meter,
    beat_salience,
    meter_votes,
)


def _isochronous(n, period=0.5, start=0.0):
    return [round(start + i * period, 6) for i in range(n)]


def test_flat_sequence_abstains():
    """Perfectly isochronous, uniformly voiced, no events: nothing to see."""
    beats = _isochronous(24)
    result = analyze_accent_meter(beats, events=[], voiced_flags=[True] * 24)
    assert result.abstained
    assert result.meter is None


def test_too_short_abstains():
    result = analyze_accent_meter(_isochronous(4), events=[])
    assert result.abstained
    assert "need" in result.reason


def test_voicing_channel_carries_period_three():
    """Two of every three beats voiced — the adr006-exercise-1-demo pattern."""
    beats = _isochronous(24)
    voiced = [(i % 3) != 2 for i in range(24)]
    sal = beat_salience(beats, events=[], voiced_flags=voiced)
    arr = np.asarray(sal.combined)
    on = arr[[i for i in range(24) if i % 3 == 0]].mean()
    off = arr[[i for i in range(24) if i % 3 == 2]].mean()
    assert on > off


def test_agogic_channel_finds_lengthened_downbeat():
    """The waltz finding: beat 1->2 held ~10% longer than the rest."""
    times, t = [], 0.0
    for bar in range(8):
        for pos, dur in enumerate((0.749, 0.656, 0.634)):
            times.append(round(t, 6))
            t += dur
    sal = beat_salience(times, events=[], voiced_flags=[True] * len(times))
    downbeats = [sal.agogic[i] for i in range(0, len(times), 3)]
    others = [sal.agogic[i] for i in range(len(times)) if i % 3]
    assert np.mean(downbeats) > np.mean(others)


def test_free_time_region_resets_phase():
    """A stretch where the pulse stopped must not carry phase across it."""
    beats = _isochronous(12) + _isochronous(12, start=20.0)
    sal = beat_salience(beats, events=[], free_time=[(6.0, 20.0)])
    assert sal.segment_ids[:12] == [0] * 12
    assert sal.segment_ids[12:] == [1] * 12


def test_density_channel_counts_events_per_beat():
    beats = _isochronous(8, period=1.0)
    events = [0.0, 0.5, 1.0, 3.0]  # beat 0 has two events, beat 1 one, beat 3 one
    sal = beat_salience(beats, events)
    assert sal.density[0] == 2.0
    assert sal.density[1] == 1.0
    assert sal.density[2] == 0.0


def test_votes_are_ranked_and_cover_every_phase():
    beats = _isochronous(24)
    voiced = [(i % 3) != 2 for i in range(24)]
    result = meter_votes(beat_salience(beats, events=[], voiced_flags=voiced),
                         min_margin=0.0)
    scores = [v.score for v in result.votes]
    assert scores == sorted(scores, reverse=True)
    expected = sum(len(t) for t in METER_TEMPLATES.values())
    assert len(result.votes) == expected


def test_confidence_is_zero_when_metres_tie():
    beats = _isochronous(24)
    result = meter_votes(beat_salience(beats, events=[], voiced_flags=[True] * 24))
    assert result.confidence == 0.0


def test_duple_and_triple_templates_are_separable_but_2_4_and_4_4_are_not():
    """The structural limit this module reports: it resolves family, not metre.

    Pinned as a test because it is a property of the template set, not of any
    corpus, and a future edit that changes it should have to say so.
    """
    n = 24

    def tiled(name, phase):
        t = METER_TEMPLATES[name]
        return np.array([t[(k - phase) % len(t)] for k in range(n)])

    def confus(a, b):
        return max(
            abs(np.corrcoef(tiled(a, 0), tiled(b, p))[0, 1])
            for p in range(len(METER_TEMPLATES[b]))
        )

    assert confus("2/4", "4/4") > 0.85
    assert confus("3/4", "6/8") > 0.85
    assert confus("2/4", "3/4") < 0.3
    assert confus("4/4", "6/8") < 0.3
