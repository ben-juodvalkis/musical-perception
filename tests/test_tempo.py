"""Tests for precision tempo calculation. No audio or models needed."""

from musical_perception.precision.tempo import (
    calculate_tempo,
    interpret_meter,
    normalize_tempo,
    tempo_family,
)
from musical_perception.types import Meter, OnsetTempoResult, RhythmicSection, TempoResult


def test_steady_120bpm():
    """120 BPM = 0.5s intervals."""
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    result = calculate_tempo(timestamps)
    assert result is not None
    assert result.bpm == 120.0
    assert result.confidence > 0.95
    assert result.beat_count == 8


def test_steady_72bpm():
    """72 BPM = 0.833s intervals."""
    interval = 60.0 / 72
    timestamps = [i * interval for i in range(8)]
    result = calculate_tempo(timestamps)
    assert result is not None
    assert abs(result.bpm - 72.0) < 0.5
    assert result.confidence > 0.95


def test_insufficient_data():
    """Single timestamp can't determine tempo."""
    assert calculate_tempo([0.0]) is None
    assert calculate_tempo([]) is None


def test_two_beats():
    """Two beats should work but with lower confidence."""
    result = calculate_tempo([0.0, 0.5])
    assert result is not None
    assert result.bpm == 120.0
    assert result.beat_count == 2


def test_outlier_robustness():
    """Median-based calculation should handle one outlier."""
    # 120 BPM with one doubled interval
    timestamps = [0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5]
    result = calculate_tempo(timestamps)
    assert result is not None
    assert result.bpm == 120.0  # Median still picks 0.5s


def test_intervals_returned():
    """Raw intervals should be accessible."""
    timestamps = [0.0, 0.5, 1.0, 1.5]
    result = calculate_tempo(timestamps)
    assert result is not None
    assert len(result.intervals) == 3
    assert all(abs(i - 0.5) < 0.001 for i in result.intervals)


# --- normalize_tempo tests ---

def test_normalize_already_in_range():
    """BPM already in 70-140 stays unchanged."""
    bpm, mult = normalize_tempo(120.0)
    assert bpm == 120.0
    assert mult == 1

def test_normalize_already_in_range_boundaries():
    """Boundary values are in range."""
    assert normalize_tempo(70.0) == (70.0, 1)
    assert normalize_tempo(140.0) == (140.0, 1)

def test_normalize_double_up():
    """40 BPM (measure level) doubles to 80 BPM."""
    bpm, mult = normalize_tempo(40.0)
    assert bpm == 80.0
    assert mult == 2

def test_normalize_triple_up():
    """30 BPM triples to 90 BPM (triple meter measure level)."""
    bpm, mult = normalize_tempo(30.0)
    assert bpm == 90.0
    assert mult == 3

def test_normalize_halve_down():
    """240 BPM (subdivision level) halves to 120 BPM."""
    bpm, mult = normalize_tempo(240.0)
    assert bpm == 120.0
    assert mult == -2

def test_normalize_third_down():
    """360 BPM (triplet subdivision) divides by 3 to 120 BPM."""
    bpm, mult = normalize_tempo(360.0)
    assert bpm == 120.0
    assert mult == -3

def test_normalize_gemini_40bpm_case():
    """The actual Exercise 1 Demo case: Gemini said 40.5 BPM."""
    bpm, mult = normalize_tempo(40.5)
    assert bpm == 81.0
    assert mult == 2  # doubled from measure to beat level

def test_normalize_prefers_double_over_triple():
    """50 BPM is implausibly slow for a beat: x2=100 beats x3=150."""
    bpm, mult = normalize_tempo(50.0)
    assert bpm == 100.0
    assert mult == 2


def test_normalize_keeps_a_plausible_slow_reading():
    """W9: 60 BPM is a beat rate, not a measure rate.

    The old hard 70-140 band doubled it to 120 because 60 sat two BPM
    outside an interval edge. Under the soft prior a fold has to pay for
    itself, and moving a 60 BPM reading a whole metric level does not.
    """
    assert normalize_tempo(60.0) == (60.0, 1)


def test_normalize_soft_band_edges():
    """The prior's keep range is ~55-178 BPM, and it is a threshold, not a
    cliff: just inside it the measurement survives, just outside it folds."""
    assert normalize_tempo(56.0)[1] == 1
    assert normalize_tempo(54.0)[1] == 2
    assert normalize_tempo(175.0)[1] == 1
    assert normalize_tempo(181.0)[1] == -2

def test_normalize_extreme_bpm_returns_sentinel():
    """BPM too extreme for any ×2/×3 transform returns multiplier=0."""
    bpm, mult = normalize_tempo(5.0)
    assert bpm == 5.0
    assert mult == 0

    bpm, mult = normalize_tempo(1000.0)
    assert bpm == 1000.0
    assert mult == 0


# --- interpret_meter tests ---


def _onset(bpm, confidence=0.8):
    """Helper to create an OnsetTempoResult."""
    return OnsetTempoResult(
        bpm=bpm,
        confidence=confidence,
        rhythmic_sections=[RhythmicSection(
            start=0.0, end=5.0, bpm=bpm, mean_ioi=60.0 / bpm,
            cv=0.1, word_count=10,
        )],
        total_duration=10.0,
        rhythmic_coverage=0.5,
    )


def _gemini_tempo(bpm, confidence=0.8, beat_count=8):
    """Helper to create a TempoResult."""
    return TempoResult(bpm=bpm, confidence=confidence, beat_count=beat_count,
                       intervals=[60.0 / bpm] * max(1, beat_count - 1))


def test_interpret_issue10_waltz():
    """Issue #10: onset ~115, Gemini ~40 → 3/4, no subdivision."""
    result = interpret_meter(
        onset_tempo=_onset(115.0),
        gemini_tempo=_gemini_tempo(40.0),
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),  # Gemini says 4/4 (wrong)
        gemini_subdivision="triplet",  # Gemini says triplet (wrong)
    )
    assert result is not None
    assert abs(result.bpm - 115.0) < 5.0
    assert result.meter.beats_per_measure == 3  # Corrected to 3/4
    assert result.meter.beat_unit == 4
    assert result.subdivision == "none"  # No subdivision — each onset IS a beat
    assert result.tempo_multiplier == 3  # Cross-signal: onset/gemini ≈ 3 → triple meter


def test_interpret_straight_44():
    """Onset ~100, already in range → trust Gemini meter/subdivision."""
    result = interpret_meter(
        onset_tempo=_onset(100.0),
        gemini_tempo=_gemini_tempo(100.0),
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision="none",
    )
    assert result is not None
    assert result.bpm == 100.0
    assert result.meter.beats_per_measure == 4
    assert result.subdivision == "none"
    assert result.tempo_multiplier == 1


def test_interpret_duple_measure_level():
    """Gemini at measure level (40 BPM), doubled → 4/4, no subdivision."""
    result = interpret_meter(
        onset_tempo=None,
        gemini_tempo=_gemini_tempo(40.0),
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision="none",
    )
    assert result is not None
    assert result.bpm == 80.0
    assert result.meter.beats_per_measure == 4
    assert result.subdivision == "none"
    assert result.tempo_multiplier == 2


def test_interpret_triple_measure_level():
    """Raw ~30 BPM, tripled → 3/4, no subdivision."""
    result = interpret_meter(
        onset_tempo=None,
        gemini_tempo=_gemini_tempo(30.0),
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision=None,
    )
    assert result is not None
    assert result.bpm == 90.0
    assert result.meter.beats_per_measure == 3
    assert result.subdivision == "none"
    assert result.tempo_multiplier == 3


def test_interpret_duple_subdivision():
    """Raw ~240 BPM, halved → duple subdivision."""
    result = interpret_meter(
        onset_tempo=_onset(240.0),
        gemini_tempo=None,
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision=None,
    )
    assert result is not None
    assert result.bpm == 120.0
    assert result.meter.beats_per_measure == 4
    assert result.subdivision == "duple"
    assert result.tempo_multiplier == -2


def test_interpret_triplet_subdivision():
    """Raw ~360 BPM, divided by 3 → triplet subdivision."""
    result = interpret_meter(
        onset_tempo=_onset(360.0),
        gemini_tempo=None,
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision=None,
    )
    assert result is not None
    assert result.bpm == 120.0
    assert result.meter.beats_per_measure == 4
    assert result.subdivision == "triplet"
    assert result.tempo_multiplier == -3


def test_interpret_no_data():
    """No tempo signals → None."""
    result = interpret_meter(
        onset_tempo=None,
        gemini_tempo=None,
        gemini_meter=None,
        gemini_subdivision=None,
    )
    assert result is None


def test_interpret_extreme_bpm_returns_none():
    """BPM too extreme for normalization → None."""
    result = interpret_meter(
        onset_tempo=_onset(5.0),
        gemini_tempo=None,
        gemini_meter=None,
        gemini_subdivision=None,
    )
    assert result is None


def test_interpret_onset_preferred_over_gemini():
    """Onset tempo is used when confident, even if Gemini disagrees."""
    result = interpret_meter(
        onset_tempo=_onset(115.0, confidence=0.6),
        gemini_tempo=_gemini_tempo(40.0),
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision="triplet",
    )
    assert result is not None
    assert abs(result.bpm - 115.0) < 5.0
    assert result.raw_bpm == 115.0


def test_interpret_falls_back_to_gemini():
    """Low-confidence onset → falls back to Gemini tempo."""
    result = interpret_meter(
        onset_tempo=_onset(115.0, confidence=0.1),
        gemini_tempo=_gemini_tempo(100.0),
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision="duple",
    )
    assert result is not None
    assert result.bpm == 100.0
    assert result.raw_bpm == 100.0
    assert result.subdivision == "duple"  # Gemini subdivision passed through


def test_interpret_gemini_zero_bpm():
    """Gemini BPM=0 should not cause divide-by-zero in cross-signal check."""
    zero_tempo = TempoResult(bpm=0.0, confidence=0.5, beat_count=0, intervals=[])
    result = interpret_meter(
        onset_tempo=_onset(100.0),
        gemini_tempo=zero_tempo,
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision="none",
    )
    assert result is not None
    assert result.bpm == 100.0
    # ratio guard produces 1.0, so no cross-signal override
    assert result.tempo_multiplier == 1


# --- ADR-013: marker-vs-onset arbitration ---


def test_arbitration_strong_markers_beat_syllable_onset():
    """The rig-waltz shape: onsets lock onto swung triplet syllables (~216),
    markers carry the true beat (90.8 @ 0.92, 32 beats) — markers win."""
    result = interpret_meter(
        onset_tempo=_onset(215.9, confidence=0.79),
        gemini_tempo=_gemini_tempo(90.8, confidence=0.92, beat_count=32),
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision="triplet",
    )
    assert result.bpm == 90.8
    assert result.tempo_multiplier == 1
    assert result.subdivision == "triplet"


def test_arbitration_in_band_onset_stays_primary():
    """When onsets already read at beat level, ADR-006 behavior is untouched
    — even against denser, more confident markers."""
    result = interpret_meter(
        onset_tempo=_onset(115.0, confidence=0.6),
        gemini_tempo=_gemini_tempo(90.0, confidence=0.95, beat_count=32),
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision="none",
    )
    assert result.raw_bpm == 115.0


def test_arbitration_never_promotes_out_of_band_markers():
    """Measure-level markers (issue-10 shape) can't win the arbitration no
    matter how confident — both signals off-level falls back to onset."""
    result = interpret_meter(
        onset_tempo=_onset(216.0, confidence=0.79),
        gemini_tempo=_gemini_tempo(40.0, confidence=0.95, beat_count=32),
        gemini_meter=None,
        gemini_subdivision=None,
    )
    assert result.raw_bpm == 216.0
    assert result.bpm == 108.0  # onset primary, halved into band


def test_arbitration_needs_dense_markers():
    """A handful of confident markers is not enough to outvote onsets."""
    result = interpret_meter(
        onset_tempo=_onset(215.9, confidence=0.79),
        gemini_tempo=_gemini_tempo(90.8, confidence=0.92, beat_count=5),
        gemini_meter=None,
        gemini_subdivision=None,
    )
    assert result.raw_bpm == 215.9


def test_arbitration_frappe_shape_keeps_onset():
    """The frappé shape: the onset reading (74.3) is inside the beat band,
    so it keeps primacy even though weak in-band markers disagree —
    explicitly out of ADR-013's scope (needs accent evidence)."""
    result = interpret_meter(
        onset_tempo=_onset(74.3, confidence=0.73),
        gemini_tempo=_gemini_tempo(124.7, confidence=0.54, beat_count=40),
        gemini_meter=None,
        gemini_subdivision=None,
    )
    assert result.bpm == 74.3


# --- ADR-014: the metric-level family ---


def _by_multiplier(candidates):
    return {c.multiplier: c for c in candidates}


def test_family_in_band_reading_keeps_every_level():
    """A comfortable 104 BPM still has a family: only ×1 is in band, but
    the measure and subdivision levels stay plausible readings."""
    family = tempo_family(104.0)
    assert [c.multiplier for c in family] == [1, 2, 3, -2, -3]
    assert [c.bpm for c in family] == [104.0, 208.0, 312.0, 52.0, 34.7]
    assert [c.in_comfort_band for c in family] == [True, False, False, False, False]


def test_family_sub_70_reading_carries_the_slow_truth():
    """Clip 12's shape: a genuinely-slow 62.2 BPM marking. The band prior
    picks 124.4, but 62.2 itself remains in the family (ADR-014)."""
    family = _by_multiplier(tempo_family(62.2))
    assert family[1].bpm == 62.2
    assert family[1].in_comfort_band is False
    assert family[2].bpm == 124.4
    assert family[2].in_comfort_band is True


def test_family_over_140_reading_carries_the_fast_truth():
    """Clip 13's mirror shape: a genuinely-fast 161.8 BPM marking. ×3 (485.4)
    falls outside the plausibility range and is not offered."""
    family = tempo_family(161.8)
    assert [c.multiplier for c in family] == [1, 2, -2, -3]
    assert [c.bpm for c in family] == [161.8, 323.6, 80.9, 53.9]
    assert _by_multiplier(family)[1].in_comfort_band is False
    assert _by_multiplier(family)[-2].in_comfort_band is True


def test_family_members_carry_their_implied_reading():
    """Each candidate uses the same derivation table as the primary."""
    family = _by_multiplier(tempo_family(
        104.0, Meter(beats_per_measure=2, beat_unit=4), "none"
    ))
    assert family[1].meter.beats_per_measure == 2      # beat level → trust Gemini
    assert family[2].meter.beats_per_measure == 4 and family[2].subdivision == "none"
    assert family[3].meter.beats_per_measure == 3 and family[3].subdivision == "none"
    assert family[-2].subdivision == "duple"
    assert family[-3].subdivision == "triplet"


def test_family_drops_implausible_levels():
    """20-400 BPM bounds the family; nothing outside it is offered."""
    assert [c.bpm for c in tempo_family(30.0)] == [30.0, 60.0, 90.0]  # /2, /3 too slow
    assert [c.bpm for c in tempo_family(380.0)] == [380.0, 190.0, 126.7]  # ×2, ×3 too fast
    assert tempo_family(0.0) == []


# The primary answer, pinned. ADR-014 froze this sweep because the family
# was additive; W9 (2026-08-28) is the one change licensed to move it, and
# moved exactly the two rows ADR-014 had named as wrong. Everything else
# below is byte-for-byte the ADR-014 sweep.
PRIMARY_SWEEP = [
    # raw,    bpm,    multiplier, beats, subdivision
    # Clips 12 and 13 are the two rows ADR-014 documented and deliberately
    # did not fix: a genuinely slow 62.2 and a genuinely fast 161.8 that the
    # hard band folded away. W9 (2026-08-28) fixes them at the primary, so
    # they now stay where they were measured.
    (62.2,    62.2,    1,  4, "none"),   # clip 12: genuinely slow, kept
    (161.8,  161.8,    1,  4, "none"),   # clip 13: genuinely fast, kept
    (104.0,  104.0,   1,  4, "none"),
    (70.0,    70.0,   1,  4, "none"),
    (140.0,  140.0,   1,  4, "none"),
    (40.5,    81.0,   2,  4, "none"),
    (30.0,    90.0,   3,  3, "none"),
    (240.0,  120.0,  -2,  4, "duple"),
    (360.0,  120.0,  -3,  4, "triplet"),
]


def test_primary_selection_across_sweep():
    for raw, bpm, multiplier, beats, subdivision in PRIMARY_SWEEP:
        result = interpret_meter(
            onset_tempo=_onset(raw),
            gemini_tempo=None,
            gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
            gemini_subdivision="none",
        )
        assert result is not None, raw
        assert (result.bpm, result.tempo_multiplier) == (bpm, multiplier), raw
        assert result.meter.beats_per_measure == beats, raw
        assert result.subdivision == subdivision, raw


def test_interpret_reports_alternates_without_the_primary():
    """`alternates` holds the rest of the family — the primary is not
    repeated in it."""
    result = interpret_meter(
        onset_tempo=_onset(104.0),
        gemini_tempo=None,
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision="none",
    )
    assert result.bpm == 104.0
    assert [c.bpm for c in result.alternates] == [208.0, 312.0, 52.0, 34.7]


def test_interpret_keeps_the_slow_truth_as_primary():
    """Clip 12 end to end: the slow reading is now the primary (W9).

    ADR-014 could only surface 62.2 as an alternate because
    normalize_tempo folded the primary to 124.4. The soft prior selects
    the measured level, and the folded reading becomes the alternate.
    """
    result = interpret_meter(
        onset_tempo=_onset(62.2),
        gemini_tempo=None,
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision="none",
    )
    assert result.bpm == 62.2
    assert result.tempo_multiplier == 1
    assert 124.4 in [c.bpm for c in result.alternates]


def test_interpret_keeps_the_fast_truth_as_primary():
    """Clip 13 end to end: the fast reading is now the primary (W9)."""
    result = interpret_meter(
        onset_tempo=_onset(161.8),
        gemini_tempo=None,
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision="none",
    )
    assert result.bpm == 161.8
    assert result.tempo_multiplier == 1
    assert 80.9 in [c.bpm for c in result.alternates]


def test_interpret_cross_signal_triple_has_no_duplicate_primary():
    """The issue-10 waltz overloads multiplier=3 (BPM not actually tripled);
    the same-BPM family member must not reappear as an alternate."""
    result = interpret_meter(
        onset_tempo=_onset(115.0),
        gemini_tempo=_gemini_tempo(40.0),
        gemini_meter=Meter(beats_per_measure=4, beat_unit=4),
        gemini_subdivision="triplet",
    )
    assert result.tempo_multiplier == 3
    assert result.bpm not in [c.bpm for c in result.alternates]
