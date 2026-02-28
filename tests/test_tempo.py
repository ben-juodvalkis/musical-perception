"""Tests for precision tempo calculation. No audio or models needed."""

from musical_perception.precision.tempo import calculate_tempo, normalize_tempo


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
