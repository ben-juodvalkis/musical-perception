"""
Rung-2 kill-test analysis (annotation-convention §2) — committed analysis
code. Scores the whisper-word-start baseline and the precision-layer
acoustic pulse extractor on the 28 owner-VERIFIED beat grids with the
blessed metrics, and evaluates the four §2.3 gate conditions.

Imports the frozen scorer's matcher and report shapes; nothing under
src/musical_perception/evals/ is modified.

§2.1 edge semantics, pinned by exact reproduction of the committed §2.2
baseline table (the P0 validity gate below; every variant tried and the
search are in the rung-2 ledger entry):

- clusters: beat-centered slots, boundaries at midpoints between
  consecutive verified beats;
- annotated span: half the MEDIAN inter-beat interval beyond the first
  and last beat; predictions outside it are individual false positives;
- a cluster is TP iff it contains a prediction one-to-one matched to a
  beat at ±70 ms (the frozen mir_eval-style matcher);
- per-clip F_lc = harmonic mean of R@tac and P_lc; slice rows are
  per-clip macros.

Usage:  python scripts/rung2_kill_test.py
Writes: docs/research/rung2-extractor-events.json  (event cache, audited)
        docs/research/rung2-kill-test.json         (full results)
        docs/research/rung2-kill-test.md           (report)
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from musical_perception.annotation.grids import load_grids          # noqa: E402
from musical_perception.evals.cases import load_cases               # noqa: E402
from musical_perception.evals.stage1 import (                       # noqa: E402
    F_MEASURE_TOLERANCE_S,
    ClipPulseScore,
    _pooled,
    match_events,
    predicted_pulse_from_trace,
    score_pulse,
)

DECLINED = {"adr007-plies-demo", "rig-mixed-4-4-104-quantities"}
FROM_SCRATCH = {"adr006-exercise-1-demo", "adr010-grande-battement", "frappe"}
TOL = F_MEASURE_TOLERANCE_S
EVENTS_CACHE = REPO / "docs/research/rung2-extractor-events.json"
OUT_JSON = REPO / "docs/research/rung2-kill-test.json"
OUT_MD = REPO / "docs/research/rung2-kill-test.md"

# §2.2 blessed baseline (whisper-word-starts): {slice: (n, R@tac, P_lc, F_lc)}
TABLE_2_2 = {
    "ALL": (28, 0.449, 0.506, 0.452),
    "numbers": (14, 0.568, 0.604, 0.577),
    "step_names": (13, 0.349, 0.363, 0.343),
    "vocables": (1, 0.062, 1.000, 0.118),
}


def level_collapsed(beats: list[float], preds: list[float], tol: float) -> dict:
    """P_lc per the pinned §2.1 semantics. Returns tp/total/p_lc."""
    n = len(beats)
    if not preds:
        return {"tp": 0, "total": 0, "p_lc": None}
    matched_preds = {j for _, j in match_events(beats, preds, tol)}
    half_median = float(np.median(np.diff(beats))) / 2 if n > 1 else tol
    lo, hi = beats[0] - half_median, beats[-1] + half_median
    mids = [(beats[i] + beats[i + 1]) / 2 for i in range(n - 1)]
    slots: dict[int, list[int]] = {}
    outside = 0
    for j, p in enumerate(preds):
        if p < lo or p > hi:
            outside += 1
        else:
            slots.setdefault(int(np.searchsorted(mids, p)), []).append(j)
    tp = sum(1 for mem in slots.values() if any(j in matched_preds for j in mem))
    total = len(slots) + outside
    return {"tp": tp, "total": total, "p_lc": tp / total if total else None}


def blessed_row(beats: list[float], preds: list[float]) -> dict:
    """Per-clip blessed metrics: R@tac, P_lc, F_lc."""
    m = len(match_events(beats, preds, TOL))
    r = m / len(beats) if beats else None
    p = level_collapsed(beats, preds, TOL)["p_lc"]
    f = 2 * r * p / (r + p) if (r and p) else 0.0
    return {"r_tac": r, "p_lc": p, "f_lc": f, "matched": m}


def macro(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] if r[key] is not None else 0.0 for r in rows]))


def slice_table(per_clip: dict[str, dict], styles: dict[str, str]) -> dict:
    out = {}
    for s in ("ALL", "numbers", "step_names", "vocables"):
        rows = [v for cid, v in per_clip.items() if s == "ALL" or styles[cid] == s]
        out[s] = {
            "n": len(rows),
            "r_tac": round(macro(rows, "r_tac"), 3),
            "p_lc": round(macro(rows, "p_lc"), 3),
            "f_lc": round(macro(rows, "f_lc"), 3),
        }
    return out


def stage1_style(system: str, preds_by_clip: dict, grids, styles) -> dict:
    """A run_stage1-shaped report for one system over the verified grids,
    built from the frozen score_pulse/_pooled — the scorer is untouched."""
    rows = []
    for cid in sorted(preds_by_clip):
        scored = score_pulse(grids[cid].beats, preds_by_clip[cid], TOL)
        rows.append(ClipPulseScore(
            case_id=cid, provisional=False, count_style=styles[cid],
            n_ref=len(grids[cid].beats), n_pred=len(preds_by_clip[cid]), **scored,
        ))
    slices = sorted({r.count_style for r in rows if r.count_style})
    return {
        "pulse_source": system,
        "tolerance_s": TOL,
        "clips": [r.summary() for r in rows],
        "aggregate_verified": _pooled(rows),
        "slices": {s: _pooled([r for r in rows if r.count_style == s]) for s in slices},
    }


def asynchrony_by_cohort(preds_by_clip: dict, grids) -> dict:
    out = {}
    for name, members in (("seed-anchored", set(preds_by_clip) - FROM_SCRATCH),
                          ("from-scratch", FROM_SCRATCH & set(preds_by_clip))):
        deltas = []
        for cid in sorted(members):
            beats, preds = grids[cid].beats, preds_by_clip[cid]
            deltas += [(preds[j] - beats[i]) * 1000.0
                       for i, j in match_events(beats, preds, TOL)]
        a = np.array(deltas)
        out[name] = {
            "n_clips": len(members), "n_matched": len(deltas),
            "median_ms": round(float(np.median(a)), 1) if len(deltas) else None,
            "mean_ms": round(float(a.mean()), 1) if len(deltas) else None,
            "sd_ms": round(float(a.std()), 1) if len(deltas) else None,
        }
    return out


def extractor_events(cases_by_id: dict) -> dict[str, list[float]]:
    if EVENTS_CACHE.is_file():
        cached = json.loads(EVENTS_CACHE.read_text())
        return {cid: cached["events"][cid] for cid in cases_by_id}
    from musical_perception.annotation.__main__ import _load_audio
    from musical_perception.precision.pulse import (
        AcousticPulseParams,
        acoustic_pulse_events,
    )
    params = AcousticPulseParams()
    events = {}
    for cid, case in sorted(cases_by_id.items()):
        y = _load_audio(Path(case.media), params.peakrate.sr)
        events[cid] = [round(float(t), 4)
                       for t in acoustic_pulse_events(y, params.peakrate.sr, params)]
        print(f"  extracted {cid}: {len(events[cid])} events", file=sys.stderr)
    EVENTS_CACHE.write_text(json.dumps(
        {"extractor": "acoustic-pulse/1", "params": params.as_dict(),
         "events": events}, indent=1))
    return events


def fmt_slice_table(title: str, tab: dict) -> str:
    lines = [f"### {title}", "", "| slice | n | R@tac | P_lc | F_lc |",
             "|---|---|---|---|---|"]
    for s, row in tab.items():
        lines.append(f"| {s} | {row['n']} | {row['r_tac']:.3f} | "
                     f"{row['p_lc']:.3f} | {row['f_lc']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    cases = {c.id: c for c in load_cases(REPO / "evals/cases")}
    grids = load_grids(REPO / "evals/grids")
    verified = {cid: g for cid, g in grids.items() if not g.provisional}
    excluded = set(grids) - set(verified)
    assert excluded == DECLINED, f"declined-set mismatch: {excluded}"
    assert len(verified) == 28, f"expected 28 verified grids, got {len(verified)}"
    styles = {cid: cases[cid].tags.get("count_style") for cid in verified}

    base_preds = {cid: predicted_pulse_from_trace(REPO / "evals" / cases[cid].trace)
                  for cid in verified}
    base_clip = {cid: blessed_row(verified[cid].beats, base_preds[cid])
                 for cid in sorted(verified)}
    base_slices = slice_table(base_clip, styles)

    # P0 — metric validity gate: reproduce §2.2 exactly before reading
    # any candidate number.
    for s, (n, r, p, f) in TABLE_2_2.items():
        got = base_slices[s]
        assert (got["n"], got["r_tac"], got["p_lc"], got["f_lc"]) == (n, r, p, f), \
            f"§2.2 reproduction FAILED on {s}: {got} != {(n, r, p, f)}"
    print("P0 PASS — §2.2 baseline table reproduced exactly (all 12 numbers).")

    ext_preds = extractor_events({cid: cases[cid] for cid in verified})
    ext_clip = {cid: blessed_row(verified[cid].beats, ext_preds[cid])
                for cid in sorted(verified)}
    ext_slices = slice_table(ext_clip, styles)

    step_ids = sorted(cid for cid in verified if styles[cid] == "step_names")
    step_rows = [{
        "clip": cid,
        "n_ref": len(verified[cid].beats),
        "baseline_r": round(base_clip[cid]["r_tac"], 3),
        "extractor_r": round(ext_clip[cid]["r_tac"], 3),
        "improved": ext_clip[cid]["r_tac"] > base_clip[cid]["r_tac"],
    } for cid in step_ids]
    n_improved = sum(r["improved"] for r in step_rows)

    voc = ext_clip["rig-vocables-4-4-100-clean"]
    gate = {
        "1_step_names_r_tac": {
            "value": ext_slices["step_names"]["r_tac"], "threshold": 0.499,
            "pass": ext_slices["step_names"]["r_tac"] >= 0.499},
        "2_step_names_improved": {
            "value": n_improved, "threshold": 9, "of": len(step_ids),
            "pass": n_improved >= 9},
        "3_vocables_n1": {
            "r_tac": round(voc["r_tac"], 3), "p_lc": round(voc["p_lc"], 3),
            "thresholds": [0.60, 0.50],
            "pass": voc["r_tac"] >= 0.60 and voc["p_lc"] >= 0.50},
        "4_numbers_f_lc": {
            "value": ext_slices["numbers"]["f_lc"], "threshold": 0.527,
            "pass": ext_slices["numbers"]["f_lc"] >= 0.527},
    }
    verdict = "PASS" if all(g["pass"] for g in gate.values()) else "NEGATIVE"

    results = {
        "tolerance_s": TOL,
        "n_verified": len(verified),
        "declined_excluded": sorted(DECLINED),
        "p0_baseline_reproduction": "exact",
        "blessed_metrics": {
            "baseline": {"per_clip": {c: {k: (round(v, 4) if isinstance(v, float) else v)
                                          for k, v in r.items()}
                                      for c, r in base_clip.items()},
                         "slices": base_slices},
            "extractor": {"per_clip": {c: {k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in r.items()}
                                       for c, r in ext_clip.items()},
                          "slices": ext_slices},
        },
        "stage1_baseline": stage1_style("whisper-word-starts", base_preds,
                                        verified, styles),
        "stage1_extractor": stage1_style("acoustic-pulse/1", ext_preds,
                                         verified, styles),
        "step_names_per_clip": step_rows,
        "asynchrony_cohorts": {
            "note": ("cohorts per the rung-1.5 finding: seed-anchored grids sit ON "
                     "peakRate onsets by construction; from-scratch (the 3 video "
                     "clips) carry a ~20 ms method offset. Differences under "
                     "~25 ms are annotator noise and are not claimable."),
            "baseline": asynchrony_by_cohort(base_preds, verified),
            "extractor": asynchrony_by_cohort(ext_preds, verified),
        },
        "gate": gate,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(results, indent=1))

    md = ["# Rung-2 kill-test — blessed-gate results", "",
          f"Tolerance ±{TOL * 1000:.0f} ms; 28 owner-verified grids; "
          f"declined and excluded by name: {', '.join(sorted(DECLINED))}.",
          "", "P0: §2.2 baseline table reproduced exactly (all 12 numbers).", "",
          fmt_slice_table("Baseline (whisper-word-starts)", base_slices),
          fmt_slice_table("Extractor (acoustic-pulse/1)", ext_slices),
          "### step_names per clip (gate condition 2)", "",
          "| clip | n_ref | baseline R@tac | extractor R@tac | improved |",
          "|---|---|---|---|---|"]
    md += [f"| {r['clip']} | {r['n_ref']} | {r['baseline_r']:.3f} | "
           f"{r['extractor_r']:.3f} | {'YES' if r['improved'] else 'no'} |"
           for r in step_rows]
    md += ["", f"Improved on {n_improved} of {len(step_ids)} step_names clips.",
           "", "### Gate", "", "```json",
           json.dumps(gate, indent=1), "```", "",
           f"## VERDICT: {verdict}", ""]
    OUT_MD.write_text("\n".join(md))

    print(json.dumps(results["blessed_metrics"]["baseline"]["slices"], indent=1))
    print(json.dumps(results["blessed_metrics"]["extractor"]["slices"], indent=1))
    print(json.dumps(gate, indent=1))
    print("VERDICT:", verdict)
    print(f"wrote {OUT_JSON.relative_to(REPO)}, {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
