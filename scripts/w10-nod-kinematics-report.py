#!/usr/bin/env python3
"""W10 — nod kinematics and phrase-arrival segmentation.

Read-only over committed traces and grids: no media, no models, no API key.
Reproduce with `python scripts/w10-nod-kinematics-report.py`.

Scores the three pre-declared head-nod event definitions against the
owner-verified beat grids, with a circular-shift null, and measures the
re-entry-vs-interior contrast that Bishop & Goebl (2018) predicts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from musical_perception.evals.stage1 import score_pulse
from musical_perception.precision.nod import (
    EVENT_KINDS,
    circular_shift_null,
    head_series,
    nod_events,
    partition_reentry,
)
from musical_perception.types import LandmarkTimeSeries

ROOT = Path(__file__).resolve().parent.parent
TRACES, GRIDS = ROOT / "evals" / "traces", ROOT / "evals" / "grids"

PRIMARY_TOL = 0.15     # pre-registered: a visual channel does not hold 70 ms
SECONDARY_TOL = 0.07   # the blessed mir_eval window, reported beside it
N_DRAWS = 500
N_CALIBRATION = 200
ALPHA = 0.05 / 3       # Bonferroni over the three pre-declared definitions
REENTRY_GAP = 2.0
SEED = 20260830


def load_pose(name: str) -> LandmarkTimeSeries:
    z = np.load(TRACES / name / "pose.npz")
    return LandmarkTimeSeries(z["timestamps"], z["landmarks"], float(z["fps"]),
                              float(z["detection_rate"]))


def grid_for(trace: str) -> tuple[str, dict] | None:
    for path in sorted(GRIDS.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if doc.get("clip", "").endswith(trace) or path.stem.endswith(trace):
            return path.stem, doc
    return None


def f_at(tol):
    return lambda ref, pred: score_pulse(ref, pred, tol=tol)["f_measure"] or 0.0


def recall_split(beats, events, tol):
    """Recall over re-entry beats vs interior beats."""
    from musical_perception.evals.stage1 import match_events

    matched = {i for i, _ in match_events(beats, list(events), tol)}
    reentry, interior = partition_reentry(beats, REENTRY_GAP)
    def r(group):
        return (sum(i in matched for i in group) / len(group)) if group else None
    return {
        "n_reentry": len(reentry), "n_interior": len(interior),
        "recall_reentry": r(reentry), "recall_interior": r(interior),
    }


def main() -> None:
    pose_traces = sorted(d.name for d in TRACES.iterdir() if (d / "pose.npz").exists())
    results = {"clips": {}, "coverage": {}, "calibration": {}, "params": {
        "primary_tol_s": PRIMARY_TOL, "secondary_tol_s": SECONDARY_TOL,
        "n_draws": N_DRAWS, "alpha_bonferroni": ALPHA,
        "reentry_gap_s": REENTRY_GAP, "seed": SEED,
    }}

    for name in pose_traces:
        series = head_series(load_pose(name))
        events = {k: nod_events(series, k) for k in EVENT_KINDS}
        results["coverage"][name] = {
            "duration_s": round(series.duration, 2),
            "nan_fraction": round(series.nan_fraction, 5),
            "hole_seconds": round(series.hole_seconds, 2),
            "n_events": {k: int(len(v)) for k, v in events.items()},
            "event_rate_hz": {k: round(len(v) / series.duration, 3) if series.duration else None
                              for k, v in events.items()},
        }

        found = grid_for(name)
        if not found:
            continue
        grid_name, grid = found
        beats = [float(b) for b in grid["beats"]]
        row = {"grid": grid_name, "provisional": bool(grid.get("provisional", True)),
               "n_beats": len(beats), "definitions": {}}
        for kind, ev in events.items():
            cell = {"n_events": int(len(ev))}
            for label, tol in (("primary", PRIMARY_TOL), ("secondary", SECONDARY_TOL)):
                sc = score_pulse(beats, list(ev), tol=tol)
                p, obs, null = circular_shift_null(
                    beats, ev, series.duration, f_at(tol), n_draws=N_DRAWS, seed=SEED)
                a = np.array(sc["asynchrony_ms"]) if sc["asynchrony_ms"] else None
                cell[label] = {
                    "tol_s": tol,
                    "precision": None if sc["precision"] is None else round(sc["precision"], 3),
                    "recall": None if sc["recall"] is None else round(sc["recall"], 3),
                    "f_measure": round(obs, 3),
                    "null_mean_f": round(null, 3),
                    "p_value": round(p, 4),
                    "significant": bool(p < ALPHA),
                    "asynchrony_mean_ms": None if a is None else round(float(a.mean()), 1),
                    "asynchrony_sd_ms": None if a is None else round(float(a.std()), 1),
                    **recall_split(beats, ev, tol),
                }
            row["definitions"][kind] = cell
        results["clips"][name] = row

    # Null calibration (N7): rotate a real event train and re-test. A test that
    # rejects phase-destroyed data at more than alpha is reporting its own
    # machinery, which is how W7's middle null produced friendly fiction.
    rng = np.random.default_rng(SEED)
    for name, row in results["clips"].items():
        if row["provisional"]:
            continue
        series = head_series(load_pose(name))
        beats = [float(b) for b in yaml.safe_load(
            (GRIDS / f"{row['grid']}.yaml").read_text())["beats"]]
        ev = np.asarray(nod_events(series, "peak_acceleration"))
        rejects = 0
        for _ in range(N_CALIBRATION):
            rotated = np.sort((ev + rng.uniform(0, series.duration)) % series.duration)
            p, _, _ = circular_shift_null(beats, rotated, series.duration,
                                          f_at(PRIMARY_TOL), n_draws=99,
                                          seed=int(rng.integers(1 << 30)))
            rejects += p < 0.05
        results["calibration"][name] = {
            "n_replicates": N_CALIBRATION, "false_positive_rate": round(rejects / N_CALIBRATION, 3)}

    out = ROOT / "docs" / "research" / "w10-nod-results.json"
    out.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")

    # --- summary to stdout ------------------------------------------------
    print(f"pose traces: {len(pose_traces)}   scored against grids: {len(results['clips'])}")
    zeroed = [n for n, c in results["coverage"].items()
              if any(v == 0 for v in c["n_events"].values())]
    print(f"clips with a zero-event definition: {len(zeroed)} {zeroed}")
    print()
    hdr = f"{'clip':22s} {'slice':11s} {'definition':18s} {'n_ev':>5s} {'P':>6s} {'R':>6s} {'F':>6s} {'nullF':>6s} {'p':>7s} {'sig':>4s} {'async_ms':>9s}"
    for tol_label, tol in (("primary", PRIMARY_TOL), ("secondary", SECONDARY_TOL)):
        print(f"=== tolerance +/-{tol:.2f} s ({tol_label}) ===")
        print(hdr)
        for name, row in sorted(results["clips"].items()):
            sl = "provisional" if row["provisional"] else "verified"
            for kind in EVENT_KINDS:
                c = row["definitions"][kind][tol_label]
                print(f"{name:22s} {sl:11s} {kind:18s} {c['n_events'] if False else row['definitions'][kind]['n_events']:5d} "
                      f"{c['precision'] if c['precision'] is not None else 0:6.3f} "
                      f"{c['recall'] if c['recall'] is not None else 0:6.3f} "
                      f"{c['f_measure']:6.3f} {c['null_mean_f']:6.3f} {c['p_value']:7.4f} "
                      f"{'YES' if c['significant'] else 'no':>4s} "
                      f"{c['asynchrony_mean_ms'] if c['asynchrony_mean_ms'] is not None else 0:9.1f}")
        print()

    print("=== re-entry vs interior recall (primary tolerance) ===")
    for name, row in sorted(results["clips"].items()):
        sl = "provisional" if row["provisional"] else "verified"
        for kind in EVENT_KINDS:
            c = row["definitions"][kind]["primary"]
            rr = c["recall_reentry"]
            ri = c["recall_interior"]
            delta = None if (rr is None or ri is None) else rr - ri
            print(f"{name:22s} {sl:11s} {kind:18s} n_re={c['n_reentry']:3d} n_int={c['n_interior']:3d} "
                  f"R_re={rr if rr is not None else float('nan'):.3f} R_int={ri if ri is not None else float('nan'):.3f} "
                  f"delta={delta if delta is not None else float('nan'):+.3f}")
    print()
    # POST-HOC, labelled as such: the pre-registered 2.0 s re-entry gap yields
    # too few re-entry beats to test N4. This sweep asks whether *any* gap
    # threshold reaches the pre-declared n >= 8 floor. It cannot rescue N4 --
    # a threshold chosen after seeing the deltas is not a test -- but a reader
    # is entitled to know whether the contrast was unmeasurable or merely
    # unmeasured.
    print("=== POST-HOC: re-entry gap sweep (E1, primary tolerance) — not a test ===")
    sweep = {}
    for gap in (0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        n_re = n_int = hit_re = hit_int = 0
        for name, row in results["clips"].items():
            if row["provisional"]:
                continue
            series = head_series(load_pose(name))
            beats = [float(b) for b in yaml.safe_load(
                (GRIDS / f"{row['grid']}.yaml").read_text())["beats"]]
            ev = nod_events(series, "peak_acceleration")
            from musical_perception.evals.stage1 import match_events
            matched = {i for i, _ in match_events(beats, list(ev), PRIMARY_TOL)}
            re_idx, in_idx = partition_reentry(beats, gap)
            n_re += len(re_idx); n_int += len(in_idx)
            hit_re += sum(i in matched for i in re_idx)
            hit_int += sum(i in matched for i in in_idx)
        r_re = hit_re / n_re if n_re else float("nan")
        r_int = hit_int / n_int if n_int else float("nan")
        sweep[gap] = {"n_reentry": n_re, "n_interior": n_int,
                      "recall_reentry": round(r_re, 3), "recall_interior": round(r_int, 3),
                      "delta": round(r_re - r_int, 3), "powered": n_re >= 8}
        print(f"gap>={gap:4.2f}s  n_re={n_re:3d} n_int={n_int:3d}  R_re={r_re:.3f} "
              f"R_int={r_int:.3f}  delta={r_re - r_int:+.3f}  "
              f"{'powered' if n_re >= 8 else 'UNDERPOWERED'}")
    results["reentry_sweep_post_hoc"] = {str(k): v for k, v in sweep.items()}
    out.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")

    print()
    print("=== null calibration (N7): FPR at alpha=0.05 on phase-destroyed events ===")
    for name, c in sorted(results["calibration"].items()):
        print(f"{name:22s} replicates={c['n_replicates']} FPR={c['false_positive_rate']:.3f}")
    print()
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
