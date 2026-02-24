"""Tests for precision tempo calculation. No audio or models needed."""

from musical_perception.precision.tempo import calculate_tempo


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
