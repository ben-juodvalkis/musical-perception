#!/usr/bin/env python
"""Rung 3 (W2) grouping diagnostic: accent-periodicity meter votes vs verified grids.

Replayable from committed files only — no audio, no models, no API key.
Inputs: evals/grids/*.yaml, evals/cases/*.yaml, docs/research/rung2-extractor-events.json.
Reads them; writes nothing under evals/.

Usage: python scripts/rung3-accent-meter-report.py [--markdown]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from musical_perception.precision.accent_meter import analyze_accent_meter  # noqa: E402

# Bar length in grid-beat units, read from the owner-verified grid notes.
METER_TO_PERIOD = {"2/4": 2, "3/4": 3, "4/4": 4, "6/8": 6}

# Declared before measurement (pre-registration P6): the grid is at the number
# level and the 3/4 label lives in the 'and-ah' subdivision BELOW the tactus
# (ADR-006's equivalence case), so bar length in grid-beat units is 1.
DEGENERATE = {
    "rig-numbers-3-4-90-clean": "grid at the number level; 3/4 lives below the tactus",
}


def load_grid_beats(grid: dict) -> tuple[list[float], list[bool], list[tuple[float, float]]]:
    """Full beat sequence with silent beats reinstated, plus free-time spans."""
    beats = [(t, True) for t in grid["beats"]]
    free_time: list[tuple[float, float]] = []
    for region in grid.get("regions") or []:
        kind = region.get("kind")
        if kind == "silent_beat":
            beats.append(((region["start"] + region["end"]) / 2.0, False))
        elif kind == "free_time":
            free_time.append((region["start"], region["end"]))
    beats.sort(key=lambda p: p[0])
    times = [t for t, _ in beats]
    voiced = [v for _, v in beats]
    return times, voiced, free_time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    events_doc = json.load(open("docs/research/rung2-extractor-events.json"))
    events = events_doc["events"]
    cases = {}
    for path in sorted(glob.glob("evals/cases/*.yaml")):
        c = yaml.safe_load(open(path))
        cases[c["id"]] = c

    rows = []
    for path in sorted(glob.glob("evals/grids/*.yaml")):
        grid = yaml.safe_load(open(path))
        cid = grid["clip"]
        case = cases.get(cid)
        if case is None:
            continue
        truth_meter = case.get("expect", {}).get("meter")
        times, voiced, free_time = load_grid_beats(grid)
        result = analyze_accent_meter(
            times,
            events.get(cid),
            voiced_flags=voiced,
            free_time=free_time,
        )
        best = result.best
        rows.append(
            {
                "clip": cid,
                "provisional": bool(grid.get("provisional")),
                "truth": truth_meter,
                "truth_period": METER_TO_PERIOD.get(truth_meter or ""),
                "degenerate": cid in DEGENERATE,
                "pred": result.meter,
                "phase": None if result.abstained or not best else best.phase,
                "score": None if not best else round(best.score, 3),
                "margin": round(result.margin, 3),
                "conf": round(result.confidence, 2),
                "abstained": result.abstained,
                "reason": result.reason,
                "n_beats": len(times),
                "n_silent": sum(1 for v in voiced if not v),
            }
        )

    def scoreable(r):
        return (
            not r["provisional"]
            and not r["degenerate"]
            and r["truth_period"] is not None
        )

    fmt = "| {clip:38s} | {truth:5s} | {pred:6s} | {ph:>3s} | {sc:>6s} | {mg:>6s} | {mark:4s} |"
    head = fmt.format(clip="clip", truth="truth", pred="pred", ph="ph", sc="score",
                      mg="margin", mark="")
    print(head)
    if args.markdown:
        print("|" + "|".join(["-" * 40, "-" * 7, "-" * 8, "-" * 5, "-" * 8, "-" * 8, "-" * 6]) + "|")
    else:
        print("-" * len(head))

    def emit(r):
        ok = "" if not scoreable(r) else ("PASS" if r["pred"] == r["truth"] else "FAIL")
        if r["abstained"]:
            ok = "ABST" if scoreable(r) else ""
        print(
            fmt.format(
                clip=r["clip"],
                truth=str(r["truth"]),
                pred=str(r["pred"]),
                ph="-" if r["phase"] is None else str(r["phase"]),
                sc="-" if r["score"] is None else f"{r['score']:.3f}",
                mg=f"{r['margin']:.3f}",
                mark=ok,
            )
        )

    non44 = [r for r in rows if scoreable(r) and r["truth"] != "4/4"]
    four4 = [r for r in rows if scoreable(r) and r["truth"] == "4/4"]
    other = [r for r in rows if not scoreable(r)]

    print("\n--- non-4/4 slice (the diagnostic's primary set) ---")
    for r in non44:
        emit(r)
    print("\n--- 4/4 slice ---")
    for r in four4:
        emit(r)
    print("\n--- excluded (provisional / degenerate / no truth meter) ---")
    for r in other:
        note = []
        if r["provisional"]:
            note.append("provisional grid")
        if r["degenerate"]:
            note.append(DEGENERATE[r["clip"]])
        if r["truth_period"] is None and not r["degenerate"]:
            note.append("no truth meter")
        print(f"  {r['clip']:38s} truth={str(r['truth']):5s} pred={str(r['pred']):6s}"
              f"  [{'; '.join(note)}]")

    def tally(group, label):
        n = len(group)
        hit = sum(1 for r in group if r["pred"] == r["truth"] and not r["abstained"])
        abst = sum(1 for r in group if r["abstained"])
        print(f"{label}: {hit}/{n} correct, {abst} abstained")
        return hit, n

    print("\n=== summary ===")
    tally(non44, "non-4/4 grouping")
    tally(four4, "4/4 grouping")
    tally(non44 + four4, "all scoreable")
    for m in ("2/4", "3/4", "6/8"):
        sub = [r for r in non44 if r["truth"] == m]
        if sub:
            tally(sub, f"  {m}")

    # Family level. The template confusability check (see the evidence audit)
    # puts 2/4 vs 4/4 at r=0.90 and 3/4 vs 6/8 at r=0.93, so the finest
    # distinction this model can carry is duple vs triple/compound. Reporting
    # the coarse tally is not moving the goalposts — it is naming the
    # resolution the method actually has.
    FAMILY = {"2/4": "duple", "4/4": "duple", "3/4": "triple", "6/8": "triple"}
    fam = [r for r in non44 + four4 if not r["abstained"]]
    fam_hit = sum(1 for r in fam if FAMILY.get(r["pred"]) == FAMILY.get(r["truth"]))
    print(f"\nfamily (duple vs triple/compound), committed rows only: "
          f"{fam_hit}/{len(fam)} correct")
    fam_non44 = [r for r in non44 if not r["abstained"]]
    fam_non44_hit = sum(
        1 for r in fam_non44 if FAMILY.get(r["pred"]) == FAMILY.get(r["truth"])
    )
    print(f"family, non-4/4 slice: {fam_non44_hit}/{len(fam_non44)} correct")

    print("\nconfusions (scoreable, wrong, not abstained):")
    for r in non44 + four4:
        if r["pred"] != r["truth"] and not r["abstained"]:
            print(f"  {r['clip']:38s} {r['truth']} -> {r['pred']}  (margin {r['margin']:.3f})")
    print("\nabstentions:")
    for r in non44 + four4:
        if r["abstained"]:
            print(f"  {r['clip']:38s} truth={r['truth']}  {r['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
