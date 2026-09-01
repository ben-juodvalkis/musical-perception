"""W14 — tests for the stopping-rule scorer.

The scorer is REPORTED-ONLY research code over frozen artifacts, so these
tests pin its semantics (what "committed" and "premature" mean), not any
pipeline outcome.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "w14", ROOT / "scripts" / "w14-stopping-rule.py")
w14 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w14)


def test_commit_k_needs_k_consecutive_equal_points():
    values = ["a", "b", "b", "b", "c"]
    assert w14.commit_k(values, 1) == 0        # k=1 commits immediately
    assert w14.commit_k(values, 2) == 2        # b,b
    assert w14.commit_k(values, 3) == 3        # b,b,b
    assert w14.commit_k(values, 4) is None     # never four in a row


def test_commit_k_refuses_to_commit_to_no_answer():
    """A rule that "commits" to None is not a stopping rule."""
    assert w14.commit_k([None, None, None, "a", "a"], 2) == 4


def test_commit_k_uses_the_4pct_tolerance_on_numbers():
    # 100 and 103 are within 4%: a stable pair by Standing Lesson 7.
    assert w14.commit_k([100.0, 103.0, 130.0], 2) == 1
    assert w14.commit_k([100.0, 130.0, 131.0], 2) == 2


def test_commit_theta_fires_at_first_prefix_reaching_theta():
    values = ["a", "b", "c"]
    confs = [0.2, 0.6, 0.9]
    assert w14.commit_theta(values, confs, 0.5) == 1
    assert w14.commit_theta(values, confs, 0.95) is None
    # a missing confidence is not a commit
    assert w14.commit_theta(values, [None, None, 0.9], 0.5) == 2


def test_best_point_honours_the_ceiling_and_prefers_the_earliest():
    block = {
        "1": {"premature_rate": 0.5, "median_commit_norm": 0.1},
        "2": {"premature_rate": 0.05, "median_commit_norm": 0.6},
        "3": {"premature_rate": 0.00, "median_commit_norm": 0.4},
    }
    assert w14.best_point(block)[0] == "3"
    too_loose = {"1": {"premature_rate": 0.5, "median_commit_norm": 0.1}}
    assert w14.best_point(too_loose) is None    # no relaxing the ceiling


def test_numeric_series_comes_from_the_recorded_stream_not_the_change_log():
    """The published change log is a >4%-move log, hence lossy for numbers."""
    row = {
        "grid": [0.0, 1.0, 2.0],
        "changes": [{"t": 0.0, "field": "tempo_bpm", "to": 100.0}],
        "series_num": {"tempo_bpm": [100.0, 102.0, 104.0]},
    }
    assert w14.series_of(row, "tempo_bpm") == [100.0, 102.0, 104.0]


def test_non_numeric_series_reconstructs_from_the_change_log():
    row = {
        "grid": [0.0, 1.0, 2.0, 3.0],
        "changes": [{"t": 1.0, "field": "meter", "to": "4/4"},
                    {"t": 3.0, "field": "meter", "to": "3/4"}],
        "series_num": {},
    }
    assert w14.series_of(row, "meter") == [None, "4/4", "4/4", "3/4"]


@pytest.mark.skipif(not (ROOT / "docs/research/w14-stopping-rule.json").is_file(),
                    reason="results artifact not generated")
def test_committed_results_are_internally_consistent():
    r = json.loads((ROOT / "docs/research/w14-stopping-rule.json").read_text())
    assert r["reconstruction_mismatches"] == []
    assert r["premature_ceiling"] == 0.10
    # `counts` carries no pipeline confidence, so F2 must abstain on it.
    for cond in ("granted", "withheld"):
        for mat in ("verified", "provisional"):
            assert r["results"]["f2"][cond][mat]["counts"] is None
    # The four metric fields share one confidence stream (ADR-017), so at any
    # theta they commit at the same prefix — identical median commit times.
    blk = r["results"]["f2"]["granted"]["verified"]
    for th in blk["meter"]:
        times = {blk[f][th]["median_commit_norm"]
                 for f in ("meter", "grouping", "division", "tempo_bpm")}
        assert len(times) == 1, f"theta={th}: metric block split into {times}"
