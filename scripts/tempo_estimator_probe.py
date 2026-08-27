"""
Owner probe (2026-08-26) — is the pulse->BPM step the weak link?

Commissioned nothing and blesses nothing. This is the evidence behind
workstream W9 (RESEARCH-LOG 2026-08-26, owner entry): a same-stream
comparison of two ways to turn a tick stream into a BPM.

  A. median-of-consecutive-gaps  — `precision.tempo.calculate_tempo`,
     which is what the shipping pipeline calls on Gemini-classified
     beat markers (analyze.py:202).
  B. pairwise-IOI histogram      — every pair of ticks within 4 s votes
     for the period that divides their gap by a small integer; the
     modal period wins.  Written here, once, and NOT tuned against the
     results; it is a probe, not a candidate implementation.

Both are run on the same two tick streams from the rung-2 acoustic
extractor -- V0 (first-event-per-nucleus, the rung-2 blessed stream) and
V1 (all-in-nucleus, W2.5's adopted stream) -- so the comparison isolates
the estimator, holding the stream fixed.

Ground truth is the metronome BPM in each rig clip's filename, NOT
tier-1's `marking_bpm` and NOT the blessed Acc1/Acc2/OE metrics.  See
the ledger entry for what this therefore does and does not establish.

Usage:  python scripts/tempo_estimator_probe.py
Writes: docs/research/tempo-estimator-probe.json / .md
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from musical_perception.annotation.grids import load_grids          # noqa: E402
from musical_perception.annotation.__main__ import _load_audio      # noqa: E402
from musical_perception.annotation.peakrate import peak_rate_events # noqa: E402
from musical_perception.precision.pulse import (                    # noqa: E402
    AcousticPulseParams,
    _nucleus_regions,
)
from musical_perception.precision.tempo import (                    # noqa: E402
    calculate_tempo,
    normalize_tempo,
)

OUT_JSON = REPO / "docs/research/tempo-estimator-probe.json"
OUT_MD = REPO / "docs/research/tempo-estimator-probe.md"

TOL_FRAC = 0.04          # "correct" = within 4% of the metronome
MATCH_TOL = 0.070        # the blessed +/-70 ms beat-matching tolerance
SUFFIXES = ("clean|long|waltz|quiet|coda|explained|adagio|allegro|"
            "bothsides|duple|fourx8|prep|halftempo|triplet|quantities")


def ioi_histogram_bpm(ts, lo=0.30, hi=1.30, sigma=0.020, max_gap=4.0):
    """Periodicity estimator: every tick pair within `max_gap` votes."""
    ts = np.asarray(sorted(ts), float)
    if ts.size < 3:
        return None
    periods = np.arange(lo, hi, 0.002)
    score = np.zeros_like(periods)
    for i in range(len(ts)):
        d = ts[i + 1:] - ts[i]
        d = d[d <= max_gap]
        if not d.size:
            continue
        for k in (1, 2, 3, 4):
            cand = d / k
            m = (cand >= lo) & (cand <= hi)
            if m.any():
                score += np.exp(
                    -0.5 * ((periods[:, None] - cand[m][None, :]) / sigma) ** 2
                ).sum(1)
    return float(60.0 / periods[int(np.argmax(score))])


def streams_for(media, params):
    """One extraction pass -> the V0 and V1 tick streams."""
    y = _load_audio(media, params.peakrate.sr)
    ev = peak_rate_events(y, params.peakrate.sr, params.peakrate)
    v0, v1 = [], []
    for s, e in _nucleus_regions(y, params.peakrate.sr, params):
        inside = ev[(ev >= s) & (ev <= e)]
        if not inside.size:
            continue
        v0.append(round(float(inside[0]), 4))
        v1.extend(round(float(t), 4) for t in inside)
    return sorted(set(v0)), sorted(set(v1))


def main():
    # Plain defaults on purpose: this script derives BOTH streams from the
    # regions itself, so it is independent of the events_per_nucleus default
    # and runs identically before and after W2.5 merges.
    params = AcousticPulseParams()
    grids = load_grids(REPO / "evals/grids")
    rows, phases = [], []
    kinds = {"on_found_beat": 0, "on_missed_beat": 0, "between": 0}

    for cid, g in sorted(grids.items()):
        if not cid.startswith("rig-") or g.provisional:
            continue
        m = re.search(rf"-(\d+)-(?:{SUFFIXES})$", cid)
        media = REPO / g.media
        if not m or not media.exists():
            print(f"  skip {cid}", file=sys.stderr)
            continue
        true_bpm = float(m.group(1))
        v0, v1 = streams_for(media, params)
        beats = np.asarray(g.beats, float)

        for t in sorted(set(v1) - set(v0)):
            d = np.abs(beats - t)
            j = int(np.argmin(d))
            if d[j] <= MATCH_TOL:
                hit0 = bool(v0) and bool(
                    np.any(np.abs(np.asarray(v0) - beats[j]) <= MATCH_TOL)
                )
                kinds["on_found_beat" if hit0 else "on_missed_beat"] += 1
            else:
                kinds["between"] += 1
                k = int(np.searchsorted(beats, t))
                if 0 < k < len(beats) and beats[k] > beats[k - 1]:
                    phases.append((t - beats[k - 1]) / (beats[k] - beats[k - 1]))

        def med(s):
            r = calculate_tempo(list(s))
            return (r.bpm, normalize_tempo(r.bpm)[0]) if r else (None, None)

        def hist(s):
            b = ioi_histogram_bpm(s)
            return (b, normalize_tempo(b)[0]) if b else (None, None)

        m0r, m0n = med(v0)
        m1r, m1n = med(v1)
        h0r, h0n = hist(v0)
        h1r, h1n = hist(v1)
        rows.append(dict(clip=cid, true_bpm=true_bpm, n_v0=len(v0), n_v1=len(v1),
                         med_v0_raw=m0r, med_v0=m0n, med_v1_raw=m1r, med_v1=m1n,
                         hist_v0_raw=h0r, hist_v0=h0n, hist_v1_raw=h1r, hist_v1=h1n))
        print(f"  {cid}: v0={len(v0)} v1={len(v1)}", file=sys.stderr)

    def hits(key):
        return sum(1 for r in rows
                   if r[key] is not None
                   and abs(r[key] - r["true_bpm"]) / r["true_bpm"] <= TOL_FRAC)

    n = len(rows)
    counts = {k: hits(k) for k in ("med_v0", "med_v1", "hist_v0", "hist_v1")}
    hist_bins, edges = np.histogram(phases, bins=8, range=(0, 1)) if phases else ([], [])

    result = dict(
        n_clips=n, tolerance_frac=TOL_FRAC, truth="metronome BPM in filename",
        correct=counts, extra_ticks=dict(total=sum(kinds.values()), **kinds),
        extra_phase_hist=[int(c) for c in hist_bins],
        extra_phase_edges=[round(float(e), 3) for e in edges],
        per_clip=rows,
    )
    OUT_JSON.write_text(json.dumps(result, indent=1) + "\n")

    L = ["# Tempo estimator probe (owner, 2026-08-26)", "",
         "Same tick streams, two estimators. Truth = the metronome BPM in each",
         f"rig clip's filename; \"correct\" = within {TOL_FRAC:.0%}. Not a blessed",
         "metric, not tier-1 tempo accuracy — see RESEARCH-LOG 2026-08-26.", "",
         f"| estimator | V0 (first-per-nucleus) | V1 (all-in-nucleus) |",
         "|---|---|---|",
         f"| median-of-consecutive-gaps (ships today) | {counts['med_v0']}/{n} | {counts['med_v1']}/{n} |",
         f"| pairwise-IOI histogram (probe) | {counts['hist_v0']}/{n} | {counts['hist_v1']}/{n} |",
         "", "## Where V1's extra ticks land", "",
         f"- on a beat V0 already found: **{kinds['on_found_beat']}**",
         f"- on a beat V0 missed (the recoveries): **{kinds['on_missed_beat']}**",
         f"- between beats (the clutter): **{kinds['between']}**", ""]
    if phases:
        L += ["Phase of the between-beat extras within their beat interval:", "", "```"]
        for c, lo in zip(hist_bins, edges[:-1]):
            L.append(f"  {lo:.3f}-{lo + 0.125:.3f}  {'#' * int(c)} {int(c)}")
        L += ["```", ""]
    L += ["## Per clip", "",
          "| clip | true | med V0 | med V1 | hist V0 | hist V1 |", "|---|---|---|---|---|---|"]
    f = lambda v: "—" if v is None else f"{v:.1f}"
    for r in rows:
        L.append(f"| {r['clip']} | {r['true_bpm']:.0f} | {f(r['med_v0'])} | "
                 f"{f(r['med_v1'])} | {f(r['hist_v0'])} | {f(r['hist_v1'])} |")
    OUT_MD.write_text("\n".join(L) + "\n")

    print(f"\ncorrect within {TOL_FRAC:.0%} (n={n}):")
    print(f"  median-of-consecutive-gaps:  V0 {counts['med_v0']}/{n}   V1 {counts['med_v1']}/{n}")
    print(f"  pairwise-IOI histogram:      V0 {counts['hist_v0']}/{n}   V1 {counts['hist_v1']}/{n}")
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
