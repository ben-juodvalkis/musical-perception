"""Tier-1 gate: committed traces replay to exactly the blessed outcomes.

Any per-case, per-field outcome change vs evals/baseline.json fails —
including improvements. A PR that legitimately moves a number re-blesses
the baseline and carries the delta in its diff (ADR-009 rule 9).

The gate is **verified-only** (charter W1.5): cases carrying
`maturity: provisional` are agent-proposed truth, so a difference against
them measures the label, not the pipeline. They are excluded here and
reported in their own slice instead. The exclusion set is the union of
this run's provisional ids and the baseline's own, so a row flipping
maturity in either direction still cannot gate until the owner verifies
it.

Skips cleanly until the first baseline exists.
"""

import json
from pathlib import Path

import pytest

try:
    import yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

_REPO = Path(__file__).resolve().parent.parent
_EVALS = _REPO / "evals"
_BASELINE = _EVALS / "baseline.json"

pytestmark = pytest.mark.skipif(
    not (HAS_YAML and _BASELINE.is_file() and (_EVALS / "cases").is_dir()),
    reason="no blessed baseline / cases yet",
)


def _baseline_provisional(baseline: dict, suite: str) -> set[str]:
    from musical_perception.evals.__main__ import suite_provisional_ids

    return suite_provisional_ids(baseline.get("suites", {}).get(suite))


def _baseline_reference(baseline: dict, suite: str) -> set[str]:
    from musical_perception.evals.__main__ import suite_reference_ids

    return suite_reference_ids(baseline.get("suites", {}).get(suite))


def test_tier1_outcomes_match_baseline_exactly():
    from musical_perception.evals.runner import (
        REBLESS_RECIPE,
        compare_outcomes,
        outcomes_map,
        provisional_ids,
        reference_ids,
        run_tier1,
    )

    results = run_tier1(_EVALS)
    current = outcomes_map(results)
    baseline = json.loads(_BASELINE.read_text())
    blessed = baseline["suites"]["tier1"]["outcomes"]
    # Reference rows (owner-demoted takes, reset 2026-09-01) are skipped by
    # the union of both sides, exactly like provisional rows: a row moving
    # into or out of the reference slice cannot gate on the run it moved.
    excluded = (
        set(provisional_ids(results)) | _baseline_provisional(baseline, "tier1")
        | set(reference_ids(results)) | _baseline_reference(baseline, "tier1")
    )
    changes = compare_outcomes(current, blessed, provisional=excluded)
    assert not changes, "tier-1 outcomes changed vs baseline:\n  " + \
        "\n  ".join(changes) + "\n\n" + REBLESS_RECIPE


def test_tier0_outcomes_match_baseline_exactly():
    from musical_perception.evals.runner import REBLESS_RECIPE, compare_outcomes
    from musical_perception.evals.runner import outcomes_map
    from musical_perception.evals.synthetic import run_suite

    baseline = json.loads(_BASELINE.read_text())
    if "tier0" not in baseline.get("suites", {}):
        pytest.skip("baseline has no tier0 suite")
    blessed = baseline["suites"]["tier0"]["outcomes"]
    current = outcomes_map(run_suite())
    changes = compare_outcomes(current, blessed)
    assert not changes, "tier-0 outcomes changed vs baseline:\n  " + \
        "\n  ".join(changes) + "\n\n" + REBLESS_RECIPE
