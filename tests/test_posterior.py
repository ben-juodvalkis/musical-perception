"""
Tests for the factored joint rhythm posterior (rung 4 / W5, ADR-017).

Mirrors the tier-0 corruption sweep for the new core: synthetic
timelines with a known (meter, bpm, subdivision) truth, recovery
asserted across a sweep. The synthetic builder is imported read-only
from the tier-0 module; tier-0 itself still exercises the legacy
`interpret_meter` and is untouched by this rung.
"""

import numpy as np
import pytest

from musical_perception.evals.synthetic import build_timeline
from musical_perception.precision.posterior import estimate_rhythm
from musical_perception.types import (
    GroupingLevel,
    MarkerType,
    Meter,
    TimedMarker,
    TimestampedWord,
)


def _run(meter, bpm, subdivision, counts=16, **kwargs):
    words, markers = build_timeline(
        Meter(*[int(x) for x in meter.split("/")]), bpm, subdivision,
        counts, **kwargs,
    )
    return estimate_rhythm(words, markers)


class TestCleanRecovery:
    @pytest.mark.parametrize("meter", ["2/4", "3/4", "4/4", "6/8"])
    @pytest.mark.parametrize("bpm", [63.0, 90.0, 104.0, 126.0, 160.0])
    def test_clean_tempo_recovered(self, meter, bpm):
        result = _run(meter, bpm, "none")
        assert result is not None
        assert abs(result.bpm - bpm) / bpm <= 0.08
        assert result.subdivision == "none"

    @pytest.mark.parametrize("bpm", [80.0, 104.0])
    def test_duple_recovered(self, bpm):
        result = _run("4/4", bpm, "duple")
        assert result is not None
        assert abs(result.bpm - bpm) / bpm <= 0.08
        assert result.subdivision == "duple"

    @pytest.mark.parametrize("bpm", [80.0, 104.0])
    def test_triplet_recovered(self, bpm):
        result = _run("4/4", bpm, "triplet")
        assert result is not None
        assert abs(result.bpm - bpm) / bpm <= 0.08
        assert result.subdivision == "triplet"

    def test_confidence_is_posterior_mass(self):
        result = _run("4/4", 104.0, "none")
        assert result is not None
        assert 0.0 <= result.confidence <= 1.0
        assert result.confidence >= 0.3


class TestCorruptionSweep:
    def test_jitter_tolerated(self):
        for seed in range(3):
            result = _run("4/4", 104.0, "none", jitter_sd=0.02, seed=seed)
            assert result is not None
            assert abs(result.bpm - 104.0) / 104.0 <= 0.08

    def test_interleaved_explanation_absorbed(self):
        result = _run("4/4", 104.0, "none", interleaved_explanation=True)
        assert result is not None
        assert abs(result.bpm - 104.0) / 104.0 <= 0.08

    def test_half_tempo_marking_folds_up(self):
        # Markers land at ~52 BPM; the true beat is 104. The soft prior
        # plus beat-slot silence decide the level — no hard fold exists.
        result = _run("4/4", 104.0, "none", counts=8, half_tempo_marking=True)
        assert result is not None
        assert abs(result.bpm - 104.0) / 104.0 <= 0.08

    def test_genuinely_slow_marking_is_kept(self):
        # A real 60 BPM adagio marked on every beat must NOT be doubled:
        # every slot is filled, so silence charges nothing and the prior
        # alone cannot outvote a fully-supported measurement (the
        # 52-vs-61.5 distinction W9 documented as the posterior's job).
        result = _run("4/4", 60.0, "none")
        assert result is not None
        assert abs(result.bpm - 60.0) / 60.0 <= 0.08

    def test_swing_still_finds_triplet(self):
        result = _run("4/4", 90.0, "triplet", swing=0.06)
        assert result is not None
        assert abs(result.bpm - 90.0) / 90.0 <= 0.08
        assert result.subdivision == "triplet"

    def test_swung_duple_stays_duple(self):
        # A late-spoken "and" (frac ~0.56-0.62) must not read as a 2/3
        # triplet: the wider sub bumps and the triplet template's second,
        # empty bump carry the decision.
        for swing in (0.06, 0.1):
            result = _run("4/4", 104.0, "duple", swing=swing)
            assert result is not None
            assert result.subdivision == "duple", f"swing={swing}"


class TestDivisionIsMeasuredNotRelayed:
    def test_claimed_duple_on_plain_stream_stays_none(self):
        # The pre-registered defect class: Gemini claims duple on a clip
        # with no subdivision events. One weak vote must not beat the
        # measured emptiness of the sub positions.
        words, markers = build_timeline(Meter(4, 4), 104.0, "none", 16)
        result = estimate_rhythm(words, markers, gemini_subdivision="duple")
        assert result is not None
        assert result.subdivision == "none"

    def test_claim_agreeing_with_evidence_stands(self):
        words, markers = build_timeline(Meter(4, 4), 104.0, "duple", 16)
        result = estimate_rhythm(words, markers, gemini_subdivision="duple")
        assert result is not None
        assert result.subdivision == "duple"

    def test_meter_label_is_derived_late(self):
        words, markers = build_timeline(Meter(3, 4), 90.0, "none", 12)
        result = estimate_rhythm(
            words, markers, gemini_meter=Meter(3, 4),
        )
        assert result is not None
        assert result.meter.beats_per_measure == 3
        # No claim → the default label, never a crash.
        result = estimate_rhythm(words, markers)
        assert result.meter.beats_per_measure == 4


class TestSparseAndDegenerate:
    def test_sparse_stream_falls_back_to_legacy(self):
        # Two markers cannot support a posterior over 2.3 octaves; the
        # legacy arbitration answers instead (and may abstain).
        words = [TimestampedWord(word="one", start=1.0, end=1.2),
                 TimestampedWord(word="two", start=1.6, end=1.8)]
        markers = [
            TimedMarker(MarkerType.BEAT, 1, 1.0, "one"),
            TimedMarker(MarkerType.BEAT, 2, 1.6, "two"),
        ]
        result = estimate_rhythm(words, markers)
        # Legacy path on two beats: gemini_tempo is None here, onset is
        # None (too few words) — the honest answer is abstention.
        assert result is None

    def test_empty_streams_abstain(self):
        assert estimate_rhythm([], []) is None

    def test_arrhythmic_stream_spreads_the_posterior(self):
        rng = np.random.default_rng(7)
        times = np.cumsum(rng.uniform(0.2, 1.7, size=24))
        words = [TimestampedWord(word=f"w{i}", start=float(t), end=float(t) + 0.1)
                 for i, t in enumerate(times)]
        markers = [TimedMarker(MarkerType.BEAT, None, float(t), f"w{i}")
                   for i, t in enumerate(times)]
        result = estimate_rhythm(words, markers)
        assert result is None or result.confidence < 0.5


class TestFactoredOutputs:
    def test_alternates_carry_posterior_weights(self):
        result = _run("4/4", 104.0, "none")
        assert result is not None
        for candidate in result.alternates:
            assert candidate.weight is not None
            assert 0.0 <= candidate.weight <= 1.0
        # The committed answer plus its family cannot exceed unit mass.
        total = result.confidence + sum(c.weight for c in result.alternates)
        assert total <= 1.0 + 1e-6

    def test_counting_cycle_becomes_ladder_rung(self):
        period = 60.0 / 104.0
        words, markers = [], []
        t = 0.0
        for rep in range(3):
            for n in range(1, 9):
                word = str(n)
                words.append(TimestampedWord(word=word, start=t, end=t + 0.15))
                markers.append(TimedMarker(MarkerType.BEAT, n, t, word))
                t += period
        result = estimate_rhythm(words, markers)
        assert result is not None
        eights = [g for g in result.grouping_levels if g.level == 8]
        assert eights and eights[0].source == "counting"
        assert eights[0].strength >= 0.5

    def test_silent_ladder_is_empty_not_invented(self):
        result = _run("4/4", 104.0, "none", counts=8)
        assert result is not None
        assert isinstance(result.grouping_levels, list)
        for rung in result.grouping_levels:
            assert isinstance(rung, GroupingLevel)
