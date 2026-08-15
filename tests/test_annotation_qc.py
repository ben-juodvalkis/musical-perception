"""Beat-grid QC checks (rung 2.5): the two owner-ratified checks plus the
older BPM-vs-label one, and the region suppression that makes them usable.

Synthetic grids only — the real-corpus numbers live in the ledger."""

from musical_perception.annotation.grids import BeatGrid, GridRegion
from musical_perception.annotation.qc import (
    MAX_PHRASE_CV,
    MIN_IOI_RATIO,
    cv,
    intervals,
    phrases,
    run_qc,
)


def _grid(beats, regions=None, clip="x"):
    return BeatGrid(clip=clip, provisional=False, beats=list(beats),
                    regions=list(regions or []))


def _isochronous(n, period=0.6, t0=1.0):
    return [round(t0 + i * period, 4) for i in range(n)]


def test_clean_isochronous_grid_raises_nothing():
    result = run_qc(_grid(_isochronous(16)), marking_bpm=100)
    assert result.ok
    assert result.bpm_whole == 100.0
    assert result.bpm_within_phrase == 100.0
    assert result.n_phrases == 1


def test_min_ioi_catches_a_double_mark():
    """The rung-1.5 signature: a stray label 0.13 s after a real beat."""
    beats = _isochronous(16)
    beats.insert(5, round(beats[4] + 0.13, 4))
    result = run_qc(_grid(beats))
    flags = [f for f in result.findings if f.check == "min_ioi"]
    assert len(flags) == 1
    assert result.min_ioi_ratio < MIN_IOI_RATIO
    assert "double mark" in flags[0].message


def test_free_time_region_suppresses_min_ioi():
    """The coda case: fast marking out of time is not a stray label."""
    beats = _isochronous(12) + [8.0, 8.15, 8.34, 8.5]
    tagged = _grid(beats, [GridRegion(7.9, 8.6, "free_time")])
    assert run_qc(tagged).ok
    untagged = [f.check for f in run_qc(_grid(beats)).findings]
    assert untagged.count("min_ioi") == 3


def test_silent_beat_region_suppresses_a_gap_the_break_ratio_misses():
    """An unvoiced beat can compress to 1.7x under rubato — under the 1.75x
    break ratio, so only the tag keeps it out of the phrase (the adagio case)."""
    beats = _isochronous(6)
    beats += [round(beats[-1] + 1.02, 4)]          # 1.7x the 0.6 s period
    beats += [round(beats[-1] + 0.6 * i, 4) for i in range(1, 6)]
    untagged = run_qc(_grid(beats))
    assert any(f.check == "ioi_spread" for f in untagged.findings)

    gap = [iv for iv in intervals(_grid(beats)) if iv.seconds > 1.0][0]
    tagged = _grid(beats, [GridRegion(gap.start, gap.end, "silent_beat")])
    result = run_qc(tagged)
    assert result.ok
    assert result.n_suppressed == 1


def test_phrases_split_at_breaks_not_at_agogic_stretch():
    beats = _isochronous(5) + [round(1.0 + 4 * 0.6 + 3.0, 4)]
    beats += [round(beats[-1] + 0.6 * i, 4) for i in range(1, 5)]
    counted = [iv for iv in intervals(_grid(beats)) if iv.counts]
    assert len(phrases(counted, 0.6)) == 2         # the 3.0 s gap is a break

    stretched = _isochronous(4) + [round(1.0 + 3 * 0.6 + 0.75, 4)]
    stretched += [round(stretched[-1] + 0.6 * i, 4) for i in range(1, 4)]
    counted = [iv for iv in intervals(_grid(stretched)) if iv.counts]
    assert len(phrases(counted, 0.6)) == 1         # 1.25x is ruling (f), kept


def test_ioi_spread_flags_variance_a_clip_median_hides():
    """Whole-clip median can look right while a phrase inside is ragged."""
    ragged = [1.0, 1.45, 2.3, 2.72, 3.62, 4.2]
    tail = [round(4.2 + 0.6 * i, 4) for i in range(1, 7)]
    result = run_qc(_grid(ragged + tail))
    assert any(f.check == "ioi_spread" for f in result.findings)
    assert result.max_phrase_cv > MAX_PHRASE_CV


def test_bpm_vs_label_flags_only_past_four_percent():
    grid = _grid(_isochronous(16, period=0.6))     # exactly 100 BPM
    assert run_qc(grid, marking_bpm=104).ok        # -3.85%, inside
    flagged = run_qc(grid, marking_bpm=110)        # -9.1%, outside
    assert [f.check for f in flagged.findings] == ["bpm_vs_label"]
    assert flagged.bpm_delta_pct == -9.09


def test_cv_is_population_sd_over_mean():
    assert cv([1.0, 1.0, 1.0]) == 0.0
    assert round(cv([0.9, 1.1]), 6) == 0.1


def test_untagged_grid_is_unaffected_by_the_new_machinery():
    """Format-1 behaviour: no regions means nothing is suppressed."""
    result = run_qc(_grid(_isochronous(10)))
    assert result.n_suppressed == 0
    assert result.n_intervals == 9
