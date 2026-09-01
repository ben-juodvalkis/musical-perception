"""W13(b) prefix-replay instrument: the pure logic, plus one truncation invariant.

The instrument lives in scripts/ (research driver, not pipeline code), so it
is loaded by path. Its end-to-end correctness proof is the identity check in
the report (full prefix == untruncated replay, 104/104); these tests pin the
convergence arithmetic that identity check cannot see.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "w13b-prefix-replay.py"


def _load():
    spec = importlib.util.spec_from_file_location("w13b_prefix_replay", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w13b = _load()


def test_numeric_match_uses_the_4pct_noise_floor():
    assert w13b.matches("tempo_bpm", 104.5, 100.0) is False
    assert w13b.matches("tempo_bpm", 103.9, 100.0) is True
    assert w13b.matches("tempo_bpm", 104.0, 100.0) is True  # the 4% edge is inclusive
    assert w13b.matches("counts", 8, 8) is True
    assert w13b.matches("counts", 9, 8) is False


def test_none_matches_only_none():
    assert w13b.matches("meter", None, None) is True
    assert w13b.matches("meter", None, "4/4") is False
    assert w13b.matches("meter", "4/4", None) is False


def test_convergence_is_the_last_departure_not_the_first_arrival():
    """A field that hits the final value early, leaves, and returns has NOT
    converged at the first arrival — the whole point of the metric."""
    times = [0.0, 1.0, 2.0, 3.0]
    series = [
        {"meter": "4/4"}, {"meter": "3/4"}, {"meter": "4/4"}, {"meter": "4/4"},
    ]
    series = [{**s, **{f: None for f in w13b.FIELDS if f != "meter"}} for s in series]
    final = series[-1]
    conv = w13b.convergence(times, series, final)
    assert conv["meter"] == 2.0


def test_convergence_excludes_fields_whose_final_is_none():
    times = [0.0, 1.0]
    series = [{f: None for f in w13b.FIELDS} for _ in times]
    assert w13b.convergence(times, series, series[-1])["tempo_bpm"] is None


def test_change_log_records_every_move_and_counts_them():
    times = [0.0, 1.0, 2.0]
    base = {f: None for f in w13b.FIELDS}
    series = [dict(base), {**base, "counts": 8}, {**base, "counts": 16}]
    log = w13b.change_log(times, series)
    assert [(c["t"], c["to"]) for c in log if c["field"] == "counts"] == [(1.0, 8), (2.0, 16)]


@pytest.mark.parametrize("trace", ["rig-names-4-4-104-clean"])
def test_prefix_bundle_truncates_words_and_markers_monotonically(trace):
    from musical_perception.evals.traces import replay_bundle

    inner, _ = replay_bundle(ROOT / "evals" / "traces" / trace)
    full = inner.transcribe("replay")
    span = max(w.end for w in full)

    seen = []
    for frac in (0.25, 0.5, 1.0):
        b = w13b.prefix_bundle(inner, span * frac, withhold_semantics=False)
        words = b.transcribe("replay")
        gem = b.analyze_media("replay", transcript_words=[w.word for w in words])
        assert all(w.end <= span * frac for w in words)
        assert all(gw.index is None or gw.index < len(words) for gw in gem.words)
        seen.append((len(words), len(gem.words)))
    assert seen[0][0] < seen[-1][0]
    assert [n for n, _ in seen] == sorted(n for n, _ in seen)
    assert [n for _, n in seen] == sorted(n for _, n in seen)


def test_withholding_suppresses_only_the_clip_level_semantic_fields():
    from musical_perception.evals.traces import replay_bundle

    inner, _ = replay_bundle(ROOT / "evals" / "traces" / "rig-names-4-4-104-clean")
    b = w13b.prefix_bundle(inner, None, withhold_semantics=True)
    gem = b.analyze_media("replay", transcript_words=None)
    assert (gem.exercise, gem.meter, gem.quality, gem.structure,
            gem.counting_structure) == (None,) * 5
    assert gem.words, "per-word classifications survive: only clip-level fields go"
