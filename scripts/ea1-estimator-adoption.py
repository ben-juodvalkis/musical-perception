#!/usr/bin/env python3
"""EA-1 — does EB-1's all-pairs estimator buy anything on the SHIPPING path?

Self-contained so it replays after the pipeline change it measures was
reverted (Standing Lesson 9: the replay path outlives the experiment).

Measures, on the stream `calculate_tempo` actually sees (Gemini-classified
beat markers, NOT the peakRate events EB-1 and AP-1 used):
  1. the median of consecutive gaps  (what ships)
  2. the all-pairs harmonic sum      (EB-1's winning arm)
and reports where they disagree, whether the marker arm is even reachable
under `interpret_meter`'s band gate, and what the committed answer does.

    python scripts/ea1-estimator-adoption.py [--json OUT]
"""
import argparse, json, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---- EB-1's all-pairs arm, parameters unchanged -------------------------
PERIOD_LO, PERIOD_HI, N_PERIODS = 0.20, 2.50, 600
MAX_SPAN, MAX_MULTIPLE, TOL, MIN_EVENTS = 3.0, 8, 0.15, 6


def all_pairs_period(timestamps) -> float | None:
    if len(timestamps) < MIN_EVENTS:
        return None
    ev = np.asarray(sorted(timestamps), dtype=float)
    diffs = ev[None, :] - ev[:, None]
    d = diffs[np.triu_indices_from(diffs, k=1)]
    d = d[(d > PERIOD_LO * 0.5) & (d <= MAX_SPAN)]
    if d.size < 5:
        return None
    best, best_s, best_mask = None, -1.0, None
    for p in np.geomspace(PERIOD_LO, PERIOD_HI, N_PERIODS):
        k = d / p
        m = np.round(k)
        resid = np.abs(k - m)
        ok = (m >= 1) & (m <= MAX_MULTIPLE) & (resid < TOL)
        if not ok.any():
            continue
        s = float(np.sum((1.0 - resid[ok] / TOL) / np.sqrt(m[ok])))
        if s > best_s:
            best, best_s, best_mask = float(p), s, ok
    if best is None:
        return None
    acc = d[best_mask]
    mult = np.round(acc / best)
    denom = float(np.sum(mult ** 2))
    if denom > 0:
        refined = float(np.sum(acc * mult) / denom)
        if PERIOD_LO <= refined <= PERIOD_HI:
            best = refined
    return best or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="docs/research/ea1-estimator-adoption.json")
    ap.add_argument("--evals-root", default="evals")
    args = ap.parse_args()

    from musical_perception.analyze import _markers_from_gemini
    from musical_perception.evals.cases import load_cases
    from musical_perception.evals.traces import replay_bundle
    from musical_perception.precision.rhythm import detect_onset_tempo
    from musical_perception.types import MarkerType

    root = Path(args.evals_root)
    rows = []
    for case in load_cases(root / "cases"):
        row = {"case": case.id, "reference": bool(case.reference)}
        try:
            bundle, _ = replay_bundle(root / case.trace)
            words = bundle.transcribe("replay.wav")
            onset = detect_onset_tempo(words)
            g = bundle.analyze_media(
                "replay.wav",
                onset_bpm=onset.bpm if onset else None,
                transcript_words=[w.word for w in words],
            )
            ts = [m.timestamp for m in _markers_from_gemini(g, words)
                  if m.marker_type == MarkerType.BEAT]
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            continue

        row["n_beat_markers"] = len(ts)
        if len(ts) >= 2:
            med = float(np.median(np.diff(sorted(ts))))
            row["median_bpm"] = round(60.0 / med, 1) if med > 0 else None
            p = all_pairs_period(ts)
            row["all_pairs_bpm"] = round(60.0 / p, 1) if p else None
            if row["median_bpm"] and row["all_pairs_bpm"]:
                row["ratio"] = round(row["all_pairs_bpm"] / row["median_bpm"], 3)
        # is the marker arm even reachable? interpret_meter hands the answer
        # to the markers only when the onset arm is NOT already at beat level
        row["onset_at_beat_level"] = bool(
            onset is not None and onset.confidence >= 0.3
            and 70.0 <= onset.bpm <= 140.0
        )
        rows.append(row)

    scored = [r for r in rows if not r.get("reference")]
    differs = [r for r in scored if r.get("ratio") and abs(r["ratio"] - 1.0) > 0.005]
    out = {
        "n_rows": len(scored),
        "n_estimators_disagree": len(differs),
        "n_marker_arm_reachable": sum(
            1 for r in scored if not r["onset_at_beat_level"]),
        "rows": rows,
    }
    Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(scored)} gating rows · estimators disagree on "
          f"{len(differs)} · marker arm reachable on "
          f"{out['n_marker_arm_reachable']} → {args.json}")


if __name__ == "__main__":
    main()
