"""Tests for precision tempo calculation. No audio or models needed."""

from musical_perception.precision.tempo import calculate_tempo, interpret_meter, normalize_tempo
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
    """60 BPM: *2=120 (in range) and *3=180 (out). Should double."""
    bpm, mult = normalize_tempo(60.0)
    assert bpm == 120.0
    assert mult == 2

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


def _gemini_tempo(bpm, confidence=0.8):
    """Helper to create a TempoResult."""
    return TempoResult(bpm=bpm, confidence=confidence, beat_count=8, intervals=[60.0 / bpm] * 7)


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
