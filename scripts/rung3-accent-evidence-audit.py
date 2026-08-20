#!/usr/bin/env python
"""Rung 3 (W2) evidence audit: is there bar-level periodic accent to find at all?

The grouping diagnostic asks whether a model recovers the meter. This asks the
prior question — whether the accent signal on the verified grids carries ANY
periodicity at the bar level, and if so at which lag. Answering it separates
"the model is wrong" from "the evidence is elsewhere", which are different
findings with different consequences for rung 4.

Per clip and per channel, the circular autocorrelation of the salience vector
at lags 2..8, reported against a phase-shuffle null (the same values in random
order, 400 draws) so a lag only counts as present if it beats chance.

Replayable from committed files only. Reads evals/; writes nothing.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from musical_perception.precision.accent_meter import beat_salience  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from importlib.machinery import SourceFileLoader  # noqa: E402

_report = SourceFileLoader(
    "rung3_report",
    os.path.join(os.path.dirname(__file__), "rung3-accent-meter-report.py"),
).load_module()

LAGS = [2, 3, 4, 6, 8]
N_SHUFFLE = 400
SEED = 20260820  # fixed: the audit must replay identically


def periodicity(sal: np.ndarray, lag: int) -> float:
    """Mean salience at every lag-th position, best over phase, z-free."""
    if len(sal) < 2 * lag:
        return float("nan")
    best = -np.inf
    for phase in range(lag):
        idx = np.arange(phase, len(sal), lag)
        if len(idx) < 2:
            continue
        on = sal[idx].mean()
        off = np.delete(sal, idx).mean()
        best = max(best, on - off)
    return float(best)


def main() -> int:
    events = json.load(open("docs/research/rung2-extractor-events.json"))["events"]
    cases = {}
    for path in sorted(glob.glob("evals/cases/*.yaml")):
        c = yaml.safe_load(open(path))
        cases[c["id"]] = c

    rng = np.random.default_rng(SEED)
    hdr = f"{'clip':38s} {'truth':5s} " + " ".join(f"lag{l:<5d}" for l in LAGS) + "  winner"
    print(hdr)
    print("-" * len(hdr))
    winners: dict[int, int] = {}
    for path in sorted(glob.glob("evals/grids/*.yaml")):
        grid = yaml.safe_load(open(path))
        cid = grid["clip"]
        if grid.get("provisional") or cid not in cases:
            continue
        truth = cases[cid].get("expect", {}).get("meter")
        times, voiced, free_time = _report.load_grid_beats(grid)
        sal = np.asarray(
            beat_salience(
                times, events.get(cid), voiced_flags=voiced, free_time=free_time
            ).combined
        )
        cells, sig = [], {}
        for lag in LAGS:
            obs = periodicity(sal, lag)
            if not np.isfinite(obs):
                cells.append("  n/a ")
                continue
            null = np.array(
                [periodicity(rng.permutation(sal), lag) for _ in range(N_SHUFFLE)]
            )
            p = float((null >= obs).mean())
            sig[lag] = (obs, p)
            star = "*" if p < 0.05 else " "
            cells.append(f"{obs:5.2f}{star}")
        strong = {l: v for l, v in sig.items() if v[1] < 0.05}
        win = max(strong, key=lambda l: strong[l][0]) if strong else None
        if win:
            winners[win] = winners.get(win, 0) + 1
        print(f"{cid:38s} {str(truth):5s} " + " ".join(f"{c:8s}" for c in cells)
              + f"  {win if win else '-'}")

    print("\n* = beats a 400-draw phase-shuffle null at p<0.05")
    print("winning lag (strongest significant), counted over verified grids:")
    for lag in sorted(winners):
        print(f"  lag {lag}: {winners[lag]} clips")
    print(f"  no significant lag: "
          f"{sum(1 for _ in glob.glob('evals/grids/*.yaml')) - sum(winners.values()) - 2} clips")
    template_confusability()
    return 0


def template_confusability() -> None:
    """How separable are the templates from each other, on any data at all?

    Tiled over a 24-beat sequence, best over relative phase. A high number
    here means the two metres cannot be told apart by correlation no matter
    how clean the accent signal is — a property of the model, not the corpus.
    """
    from musical_perception.precision.accent_meter import METER_TEMPLATES

    names = list(METER_TEMPLATES)
    n = 24
    def tiled(name, phase):
        t = METER_TEMPLATES[name]
        return np.array([t[(k - phase) % len(t)] for k in range(n)])
    print("\ntemplate confusability (max |corr| over relative phase, 24 beats):")
    print("       " + " ".join(f"{m:>6s}" for m in names))
    for a in names:
        row = []
        for b in names:
            best = max(
                abs(np.corrcoef(tiled(a, 0), tiled(b, p))[0, 1])
                for p in range(len(METER_TEMPLATES[b]))
            )
            row.append(f"{best:6.2f}")
        print(f"{a:>6s} " + " ".join(row))


if __name__ == "__main__":
    raise SystemExit(main())
