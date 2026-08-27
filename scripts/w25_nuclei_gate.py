"""
W2.5 (2026-08-26) — the nuclei gate: silence floor vs one-event-per-nucleus.

Scores the rung-2 acoustic pulse extractor and three pre-registered
variants on the 28 owner-VERIFIED beat grids with the blessed §2.1
metrics, by importing the committed rung-2 kill-test harness. Nothing
under src/musical_perception/evals/ is modified or re-implemented.

Variants (pre-registered in RESEARCH-LOG 2026-08-26 before any result):
  V0 baseline        events_per_nucleus=first
  V1 all-in-nucleus  events_per_nucleus=all

REDUCED 2026-08-26 by owner ruling. As first run, this script also scored
V2/V3, the `silence_reference=voiced_median` speech-band floor. That
hypothesis was falsified — it moves nucleus bounds and changes not one
emitted event on any of the 28 verified clips — and the owner ruled the
code path out of `pulse.py` rather than keep a setting nobody may use, so
V2/V3 are no longer expressible. Their measured rows survive in the
committed artifact and in git at commit ca6ed2a; re-running this script
now regenerates the V0/V1 table only.

Both events_per_nucleus values are derived from ONE extraction pass, so
the pair is guaranteed to share identical regions.

Usage:  python scripts/w25_nuclei_gate.py
Writes: docs/research/w25-nuclei-gate.json / .md
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from rung2_kill_test import blessed_row, slice_table                # noqa: E402
from musical_perception.annotation.grids import load_grids          # noqa: E402
from musical_perception.annotation.__main__ import _load_audio      # noqa: E402
from musical_perception.annotation.peakrate import peak_rate_events # noqa: E402
from musical_perception.evals.cases import load_cases               # noqa: E402
from musical_perception.precision.pulse import (                    # noqa: E402
    AcousticPulseParams,
    _nucleus_regions,
)

OUT_JSON = REPO / "docs/research/w25-nuclei-gate.json"
OUT_MD = REPO / "docs/research/w25-nuclei-gate.md"
SLICES = ("ALL", "numbers", "step_names", "vocables")
TARGET = "rig-names-4-4-100-quiet"


def events_for_reference(cases, ids):
    """One extraction pass -> both 'first' and 'all' event streams."""
    params = AcousticPulseParams()
    first, allev, widths = {}, {}, {}
    for cid in sorted(ids):
        y = _load_audio(Path(cases[cid].media), params.peakrate.sr)
        ev = peak_rate_events(y, params.peakrate.sr, params.peakrate)
        regions = _nucleus_regions(y, params.peakrate.sr, params)
        f, a = [], []
        for s, e in regions:
            inside = ev[(ev >= s) & (ev <= e)]
            if not inside.size:
                continue
            f.append(round(float(inside[0]), 4))
            a.extend(round(float(t), 4) for t in inside)
        first[cid] = sorted(set(f))
        allev[cid] = sorted(set(a))
        widths[cid] = [round(e - s, 4) for s, e in regions]
        print(f"  [{reference}] {cid}: {len(regions)} nuclei, "
              f"first={len(first[cid])} all={len(allev[cid])}", file=sys.stderr)
    return first, allev, widths


def main() -> int:
    cases = {c.id: c for c in load_cases(REPO / "evals/cases")}
    grids = {k: g for k, g in load_grids(REPO / "evals/grids").items()
             if not g.provisional}
    assert len(grids) == 28, f"expected 28 verified grids, got {len(grids)}"
    styles = {cid: cases[cid].tags.get("count_style") for cid in grids}

    preds, widths = {}, {}
    f, al, w = events_for_reference(cases, grids)
    preds["V0"], preds["V1"] = f, al
    widths["q99"] = w

    # V0 must reproduce the blessed rung-2 extractor cache exactly.
    cached = json.loads(
        (REPO / "docs/research/rung2-extractor-events.json").read_text())["events"]
    repro = sum(1 for cid in grids if preds["V0"][cid] == cached[cid])
    print(f"\nV0 REPRODUCTION vs blessed cache: {repro}/28 byte-identical")

    per_clip, slices = {}, {}
    for v in ("V0", "V1"):
        per_clip[v] = {cid: blessed_row(grids[cid].beats, preds[v][cid])
                       for cid in sorted(grids)}
        slices[v] = slice_table(per_clip[v], styles)

    lost = {v: sorted(cid for cid in grids
                      if per_clip[v][cid]["r_tac"] < per_clip["V0"][cid]["r_tac"] - 1e-9)
            for v in ("V1",)}
    npred = {v: sum(len(preds[v][cid]) for cid in grids) for v in preds}

    lines = ["# W2.5 — the nuclei gate", "",
             "Blessed §2.1 metrics on the 28 owner-verified grids. "
             "V0 is the rung-2 blessed extractor.", "",
             "| variant | ref | per-nucleus | n_pred | " +
             " | ".join(f"{s} R/P/F" for s in SLICES) + " |",
             "|---|---|---|---|" + "---|" * len(SLICES)]
    meta = {"V0": ("q99", "first"), "V1": ("q99", "all")}
    for v in ("V0", "V1"):
        cells = " | ".join(
            f"{slices[v][s]['r_tac']:.3f}/{slices[v][s]['p_lc']:.3f}/"
            f"{slices[v][s]['f_lc']:.3f}" for s in SLICES)
        lines.append(f"| {v} | {meta[v][0]} | {meta[v][1]} | {npred[v]} | {cells} |")
    lines += ["", "## Per-clip R@tac (V0 -> V1)", "",
              "| clip | beats | V0 | V1 | delta |", "|---|---|---|---|---|"]
    for cid in sorted(grids):
        d = per_clip["V1"][cid]["r_tac"] - per_clip["V0"][cid]["r_tac"]
        lines.append(f"| {cid} | {len(grids[cid].beats)} | "
                     f"{per_clip['V0'][cid]['r_tac']:.3f} | "
                     f"{per_clip['V1'][cid]['r_tac']:.3f} | {d:+.3f} |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    OUT_JSON.write_text(json.dumps({
        "v0_cache_reproduction": f"{repro}/28",
        "variants": meta, "n_pred": npred, "slices": slices,
        "per_clip": per_clip, "clips_losing_r_tac": lost,
        "nucleus_widths": widths,
    }, indent=1))

    print("\n" + "\n".join(lines[:11]))
    for v in ("V1", "V2", "V3"):
        print(f"\n{v}: clips losing R@tac vs V0: {len(lost[v])} {lost[v]}")
        print(f"{v}: target {TARGET} R@tac "
              f"{per_clip['V0'][TARGET]['r_tac']:.4f} -> "
              f"{per_clip[v][TARGET]['r_tac']:.4f}, F_lc "
              f"{per_clip['V0'][TARGET]['f_lc']:.4f} -> "
              f"{per_clip[v][TARGET]['f_lc']:.4f}")
    print(f"\nwrote {OUT_JSON}, {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
