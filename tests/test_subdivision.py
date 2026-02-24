"""Tests for subdivision analysis. No audio or models needed."""

from musical_perception.precision.subdivision import analyze_subdivisions
from musical_perception.types import TimedMarker, MarkerType


def _beat(num, ts):
    return TimedMarker(MarkerType.BEAT, num, ts, str(num))

def _and(num, ts):
    return TimedMarker(MarkerType.AND, num, ts, "and")

def _ah(num, ts):
    return TimedMarker(MarkerType.AH, num, ts, "ah")


def test_duple_subdivision():
    """1-and-2-and-3-and-4-and should detect duple."""
    markers = [
        _beat(1, 0.0), _and(1, 0.25),
        _beat(2, 0.5), _and(2, 0.75),
        _beat(3, 1.0), _and(3, 1.25),
        _beat(4, 1.5), _and(4, 1.75),
    ]
    result = analyze_subdivisions(markers)
    assert result.subdivision_type == "duple"
    assert result.subdivisions_per_beat == 2
    assert result.confidence > 0.9


def test_triplet_subdivision():
    """1-and-ah-2-and-ah should detect triplet."""
    markers = [
        _beat(1, 0.0), _and(1, 0.17), _ah(1, 0.33),
        _beat(2, 0.5), _and(2, 0.67), _ah(2, 0.83),
        _beat(3, 1.0), _and(3, 1.17), _ah(3, 1.33),
        _beat(4, 1.5), _and(4, 1.67), _ah(4, 1.83),
    ]
    result = analyze_subdivisions(markers)
    assert result.subdivision_type == "triplet"
    assert result.subdivisions_per_beat == 3
    assert result.confidence > 0.9


def test_no_subdivisions():
    """Beats only, no subdivisions."""
    markers = [
        _beat(1, 0.0),
        _beat(2, 0.5),
        _beat(3, 1.0),
        _beat(4, 1.5),
    ]
    result = analyze_subdivisions(markers)
    assert result.subdivision_type == "none"


def test_empty_markers():
    markers = []
    result = analyze_subdivisions(markers)
    assert result.subdivision_type == "none"
    assert result.confidence == 1.0
