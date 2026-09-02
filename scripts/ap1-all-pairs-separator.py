#!/usr/bin/env python
"""AP-1: does looking at ALL pairwise distances separate beats from syllables?

REPORTED-ONLY. Search space frozen in the 2026-09-02 pre-registration ledger
entry, which was committed before this file existed.

  events    peakRate (rung-2 extractor) on the clip's media, checksum-verified
            against the trace's media_sha256 — identical stream for every arm
  arm A     whole-clip median of CONSECUTIVE IOIs        (SW-1's control)
  arm B1    all-pairs kernel histogram, harmonic-summed  (PRIMARY)
  arm B2    comb / latent-grid score, null-subtracted    (SECONDARY)
  band      every arm projects by x/{1,2,1/2,3,1/3} into [70,140], in that
            order — the estimator is the only thing that varies
  pass      |bpm - truth| / truth <= 0.08 on the 34-row gating set

Reads evals/cases, evals/traces and the media the cases point at. Writes
nothing under evals/ or src/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from musical_perception.evals.aggregate import acc2, octave_errors  # noqa: E402
from musical_perception.evals.cases import load_cases  # noqa: E402

BAND_LO, BAND_HI = 70.0, 140.0
PASS_TOL = 0.08
MIN_EVENTS = 6
FACTORS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)
VIDEO_SUFFIXES = {".mov", ".m4v", ".mp4", ".avi", ".mkv", ".webm"}

# --- frozen AP-1 search space (pre-registration, 2026-09-02) ---------------
TAU_LO, TAU_HI = 0.20, 1.20          # 300 .. 50 BPM
MAX_LAG = 3.0                        # all-pairs difference cutoff, seconds
B1_SIGMA = 0.040                     # kernel width, seconds
B1_HARMONICS = 4                     # S(tau) = sum_k H(k*tau)/k
B2_SIGMA = 0.070                     # comb proximity width, seconds
B2_N_TAU = 200                       # log-spaced candidate periods
B2_N_PHASE = 20                      # phases per period
B2_N_NULL = 20                       # uniform-random trains for the null
B2_SEED = 0
GRID_MS = 0.001                      # tau resolution for B1


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_audio(path: str):
    import librosa

    if os.path.splitext(path)[1].lower() in VIDEO_SUFFIXES:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                 "-vn", "-ac", "1", "-ar", "16000", tmp.name],
                check=True,
            )
            return librosa.load(tmp.name, sr=16000, mono=True)
    return librosa.load(path, sr=16000, mono=True)


def peakrate_events(path: str) -> np.ndarray:
    from musical_perception.annotation.peakrate import PeakRateParams, peak_rate_events

    y, sr = load_audio(path)
    return np.asarray(peak_rate_events(y, sr, PeakRateParams()), dtype=float)


# ------------------------------------------------------------- the estimators


def project(bpm: float):
    """(in-band bpm, factor) or (None, None) — identical to SW-1's rule."""
    for f in FACTORS:
        cand = bpm * f
        if BAND_LO <= cand <= BAND_HI:
            return cand, f
    return None, None


def arm_a(ev: np.ndarray):
    """Median of CONSECUTIVE gaps. The control."""
    if ev.size < MIN_EVENTS:
        return None
    iois = np.diff(ev)
    iois = iois[iois > 0]
    if iois.size < MIN_EVENTS - 1:
        return None
    med = float(np.median(iois))
    return 60.0 / med if med > 0 else None


def _all_pairs(ev: np.ndarray) -> np.ndarray:
    d = ev[None, :] - ev[:, None]
    d = d[d > 0]
    return d[d <= MAX_LAG]


def arm_b1(ev: np.ndarray):
    """All-pairs kernel histogram, harmonic-summed. Returns (raw_bpm, detail)."""
    if ev.size < MIN_EVENTS:
        return None, {}
    d = _all_pairs(ev)
    if d.size < MIN_EVENTS:
        return None, {}
    # kernel density of pairwise distances on a 1 ms grid out to MAX_LAG
    grid = np.arange(GRID_MS, MAX_LAG + GRID_MS, GRID_MS)
    hist = np.zeros(grid.size)
    idx = np.clip((d / GRID_MS).astype(int) - 1, 0, grid.size - 1)
    np.add.at(hist, idx, 1.0)
    half = int(np.ceil(4 * B1_SIGMA / GRID_MS))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * GRID_MS / B1_SIGMA) ** 2)
    k /= k.sum()
    dens = np.convolve(hist, k, mode="same")

    taus = grid[(grid >= TAU_LO) & (grid <= TAU_HI)]
    score = np.zeros(taus.size)
    for h in range(1, B1_HARMONICS + 1):
        pos = np.clip((taus * h / GRID_MS).astype(int) - 1, 0, dens.size - 1)
        inside = (taus * h) <= MAX_LAG
        score += np.where(inside, dens[pos] / h, 0.0)
    tau = float(taus[int(np.argmax(score))])
    return 60.0 / tau, {"tau": round(tau, 4), "n_pairs": int(d.size)}


def arm_b2(ev: np.ndarray):
    """Comb / latent-grid score with a uniform-random null subtracted."""
    if ev.size < MIN_EVENTS:
        return None, {}
    t0, t1 = float(ev[0]), float(ev[-1])
    span = t1 - t0
    if span <= TAU_LO:
        return None, {}
    taus = np.geomspace(TAU_LO, TAU_HI, B2_N_TAU)
    rng = np.random.default_rng(B2_SEED)
    nulls = [np.sort(rng.uniform(t0, t1, ev.size)) for _ in range(B2_N_NULL)]

    def comb(train: np.ndarray, tau: float) -> float:
        best = 0.0
        for phi in np.linspace(0.0, tau, B2_N_PHASE, endpoint=False):
            g = np.arange(t0 + phi, t1 + 1e-9, tau)
            if g.size < 3:
                continue
            j = np.searchsorted(train, g)
            j = np.clip(j, 1, train.size - 1)
            delta = np.minimum(np.abs(train[j] - g), np.abs(train[j - 1] - g))
            s = float(np.mean(np.exp(-0.5 * (delta / B2_SIGMA) ** 2)))
            best = max(best, s)
        return best

    scores = np.array([comb(ev, t) - np.mean([comb(n, t) for n in nulls]) for t in taus])
    tau = float(taus[int(np.argmax(scores))])
    return 60.0 / tau, {"tau": round(tau, 4),
                        "at_ceiling": bool(tau >= TAU_HI * 0.95)}


# ---------------------------------------------------------------------- main


def score_row(raw: float | None, truth: float, detail: dict) -> dict:
    r = {"raw_bpm": None if raw is None else round(raw, 2), "truth": truth, **detail}
    if raw is None:
        r.update(bpm=None, factor=None, abstained=True, **{
            "pass": False, "acc2": False, "oe1": None, "oe2": None,
            "between_levels": False})
        return r
    bpm, factor = project(raw)
    r.update(bpm=None if bpm is None else round(bpm, 2), factor=factor,
             abstained=bpm is None)
    if bpm is None:
        r.update(**{"pass": False, "acc2": False, "oe1": None, "oe2": None,
                    "between_levels": False})
        return r
    oe1, oe2 = octave_errors(bpm, truth)
    r.update(**{"pass": abs(bpm - truth) / truth <= PASS_TOL,
                "acc2": bool(acc2(bpm, truth, PASS_TOL)),
                "oe1": round(oe1, 4), "oe2": round(oe2, 4),
                "between_levels": bool(0.08 < abs(oe2) <= 0.585)})
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from pathlib import Path

    cases = sorted(
        (c for c in load_cases(Path(os.path.join(ROOT, "evals/cases")))
         if not c.reference and c.maturity == "verified"),
        key=lambda c: c.id,
    )
    print(f"gating set: {len(cases)} rows")

    events: dict[str, np.ndarray] = {}
    truth: dict[str, float] = {}
    is_demo: dict[str, bool] = {}
    skipped: list[str] = []

    for c in cases:
        truth[c.id] = float(c.expected_bpm)
        is_demo[c.id] = c.tags.get("clip_role") == "demo"
        media = os.path.join(ROOT, c.media)
        meta = json.load(open(os.path.join(ROOT, "evals", c.trace, "meta.json")))
        want = meta.get("media_sha256")
        if not os.path.exists(media):
            skipped.append(f"{c.id} (media absent)")
            continue
        if want and sha256_of(media) != want:
            skipped.append(f"{c.id} (checksum mismatch)")
            continue
        events[c.id] = peakrate_events(media)

    print(f"  coverage {len(events)}/{len(cases)}"
          + (f"  SKIPPED BY NAME: {skipped}" if skipped else "  (no skips)"))

    ids = [c.id for c in cases]
    odd = {cid for i, cid in enumerate(ids) if i % 2 == 0}
    arms = {"A  median-consecutive (CONTROL)": "A",
            "B1 all-pairs harmonic (PRIMARY)": "B1",
            "B2 comb null-subtracted": "B2"}

    results: dict[str, dict] = {}
    for name, kind in arms.items():
        rows = {}
        for cid in ids:
            ev = events.get(cid)
            if ev is None:
                rows[cid] = {"skipped_source": True}
                continue
            if kind == "A":
                raw, detail = arm_a(ev), {}
            elif kind == "B1":
                raw, detail = arm_b1(ev)
            else:
                raw, detail = arm_b2(ev)
            detail = dict(detail, n_events=int(ev.size))
            rows[cid] = score_row(raw, truth[cid], detail)
        scored = [r for r in rows.values() if not r.get("skipped_source")]
        demo = [rows[c] for c in ids if is_demo[c] and not rows[c].get("skipped_source")]
        rig = [rows[c] for c in ids if not is_demo[c] and not rows[c].get("skipped_source")]
        o = [rows[c] for c in ids if c in odd and not rows[c].get("skipped_source")]
        e = [rows[c] for c in ids if c not in odd and not rows[c].get("skipped_source")]
        rate = lambda rs: (sum(bool(r["pass"]) for r in rs) / len(rs)) if rs else 0.0  # noqa: E731
        results[name] = {
            "kind": kind, "rows": rows, "n_scored": len(scored),
            "pass_total": sum(bool(r["pass"]) for r in scored),
            "pass_demo": sum(bool(r["pass"]) for r in demo), "n_demo": len(demo),
            "pass_rig": sum(bool(r["pass"]) for r in rig), "n_rig": len(rig),
            "acc2_total": sum(bool(r["acc2"]) for r in scored),
            "between_levels": sum(bool(r["between_levels"]) for r in scored),
            "between_levels_demo": sum(bool(r["between_levels"]) for r in demo),
            "abstained": sum(bool(r.get("abstained")) for r in scored),
            "pass_odd": sum(bool(r["pass"]) for r in o), "n_odd": len(o),
            "pass_even": sum(bool(r["pass"]) for r in e), "n_even": len(e),
            "half_gap": round(abs(rate(o) - rate(e)), 4),
            "at_ceiling": sum(1 for r in scored if r.get("at_ceiling")),
        }

    hdr = (f"{'arm':<34}{'pass':>9}{'demo':>7}{'rig':>8}{'Acc2':>7}{'btwn':>6}"
           f"{'btwnD':>7}{'abst':>6}{'odd':>7}{'even':>7}{'gap':>7}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for name, r in results.items():
        print(f"{name:<34}{r['pass_total']:>4}/{r['n_scored']:<4}"
              f"{r['pass_demo']:>3}/{r['n_demo']:<3}{r['pass_rig']:>4}/{r['n_rig']:<3}"
              f"{r['acc2_total']:>7}{r['between_levels']:>6}{r['between_levels_demo']:>7}"
              f"{r['abstained']:>6}{r['pass_odd']:>3}/{r['n_odd']:<3}"
              f"{r['pass_even']:>4}/{r['n_even']:<3}{r['half_gap']:>7.3f}")

    print("\nper-demo detail (truth · arm raw -> in-band · pass)")
    dhdr = f"{'demo':<28}{'truth':>7}" + "".join(f"{k.split()[0]:>22}" for k in arms)
    print(dhdr + "\n" + "-" * len(dhdr))
    for cid in ids:
        if not is_demo[cid]:
            continue
        line = f"{cid:<28}{truth[cid]:>7.0f}"
        for name in arms:
            r = results[name]["rows"][cid]
            mark = "PASS" if r.get("pass") else "    "
            line += f"{str(r.get('raw_bpm')):>8}->{str(r.get('bpm')):>7} {mark:<5}"
        print(line)

    print("\nrig-half flips vs control (id: A -> B1)")
    ctrl = results["A  median-consecutive (CONTROL)"]["rows"]
    prim = results["B1 all-pairs harmonic (PRIMARY)"]["rows"]
    for cid in ids:
        if ctrl[cid].get("skipped_source"):
            continue
        if bool(ctrl[cid]["pass"]) != bool(prim[cid]["pass"]):
            d = "WON " if prim[cid]["pass"] else "LOST"
            print(f"  {d} {cid:<32} truth {ctrl[cid]['truth']:>6.0f}  "
                  f"A {ctrl[cid]['bpm']}  B1 {prim[cid]['bpm']}")

    if args.json:
        payload = {
            "generated": "2026-09-02", "band": [BAND_LO, BAND_HI],
            "pass_tol": PASS_TOL, "tau_range_s": [TAU_LO, TAU_HI],
            "max_lag_s": MAX_LAG, "b1_sigma_s": B1_SIGMA,
            "b1_harmonics": B1_HARMONICS, "b2_sigma_s": B2_SIGMA,
            "b2_n_tau": B2_N_TAU, "b2_n_phase": B2_N_PHASE,
            "b2_n_null": B2_N_NULL, "b2_seed": B2_SEED,
            "factors": list(FACTORS), "skipped": skipped,
            "odd_half": sorted(odd), "even_half": sorted(set(ids) - odd),
            "arms": results,
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
