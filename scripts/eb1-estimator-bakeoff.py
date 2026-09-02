#!/usr/bin/env python
"""EB-1: five tempo estimators on one fixed event stream, plus the
harmonic/subharmonic resonance profile.

REPORTED-ONLY. Search space frozen in docs/research/eb1-estimator-bakeoff.md,
committed before this file existed.

  Arm A  median-consec (control) | all-pairs | comb | povel-essens | hopf
         -- identical peakRate events for all five; only the arithmetic varies
  Arm C  energy at f, 2f, 3f, f/2, f/3 relative to the true beat, computed
         from the linear comb AND the Hopf bank; plus the dominant peak's
         ratio to the true beat (the regime diagnostic)

Arm B (off-the-shelf trackers on the 8 demos) is a separate entry point.
Writes nothing under evals/ or src/.
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
FACTORS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)
VIDEO = {".mov", ".m4v", ".mp4", ".avi", ".mkv", ".webm"}
# candidate beat periods searched by comb / povel-essens / hopf readout
PERIOD_LO, PERIOD_HI, N_PERIODS = 0.20, 2.50, 400
PERIODS = np.geomspace(PERIOD_LO, PERIOD_HI, N_PERIODS)


def sha256_of(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_audio(path: str):
    import librosa

    if os.path.splitext(path)[1].lower() in VIDEO:
        with tempfile.NamedTemporaryFile(suffix=".wav") as t:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                            "-vn", "-ac", "1", "-ar", "16000", t.name], check=True)
            return librosa.load(t.name, sr=16000, mono=True)
    return librosa.load(path, sr=16000, mono=True)


def peakrate_events(path: str) -> np.ndarray:
    from musical_perception.annotation.peakrate import PeakRateParams, peak_rate_events

    y, sr = load_audio(path)
    return np.asarray(peak_rate_events(y, sr, PeakRateParams()), dtype=float)


# ------------------------------------------------------------------ estimators


def est_median_consec(ev: np.ndarray) -> float | None:
    if ev.size < 6:
        return None
    d = np.diff(ev)
    d = d[d > 0]
    m = float(np.median(d)) if d.size else 0.0
    return 60.0 / m if m > 0 else None


def _all_pairs(ev: np.ndarray, max_span: float = 3.0) -> np.ndarray:
    d = ev[None, :] - ev[:, None]
    d = d[np.triu_indices_from(d, k=1)]
    return d[(d > PERIOD_LO * 0.5) & (d <= max_span)]


def est_all_pairs(ev: np.ndarray) -> float | None:
    """Inner-Metric-Analysis family: the period supported by ALL pairwise
    distances, not only adjacent ones. Each candidate period scores the
    pairwise distances that are near an integer multiple of it."""
    if ev.size < 6:
        return None
    d = _all_pairs(ev)
    if d.size < 5:
        return None
    best, best_s = None, -1.0
    for p in PERIODS:
        k = d / p
        resid = np.abs(k - np.round(k))
        # only multiples up to 8x count; weight by 1/sqrt(multiple) so long
        # spans do not dominate simply by being numerous
        m = np.round(k)
        ok = (m >= 1) & (m <= 8) & (resid < 0.15)
        if not ok.any():
            continue
        s = float(np.sum((1.0 - resid[ok] / 0.15) / np.sqrt(m[ok])))
        if s > best_s:
            best, best_s = p, s
    return 60.0 / best if best else None


def _comb_profile(ev: np.ndarray, sigma_frac: float = 0.10) -> np.ndarray:
    """Event mass landing near grid points of each candidate period, best
    over phase. The linear analysis."""
    out = np.zeros(len(PERIODS))
    if ev.size < 3:
        return out
    t0 = float(ev[0])
    for i, p in enumerate(PERIODS):
        ph = ((ev - t0) / p) % 1.0
        # circular concentration = best-over-phase alignment, closed form
        out[i] = float(np.abs(np.mean(np.exp(2j * np.pi * ph))))
    return out


def est_comb(ev: np.ndarray) -> float | None:
    if ev.size < 6:
        return None
    prof = _comb_profile(ev)
    return 60.0 / PERIODS[int(np.argmax(prof))] if prof.max() > 0 else None


def est_povel_essens(ev: np.ndarray) -> float | None:
    """Clock induction: a candidate clock pays for events off its ticks AND
    for ticks with no event (Povel & Essens 1985 counterevidence, W = 4 for
    a silent tick, 1 for an unsupported event)."""
    if ev.size < 6:
        return None
    t0, t1 = float(ev[0]), float(ev[-1])
    best, best_c = None, np.inf
    for p in PERIODS:
        n = int((t1 - t0) / p)
        if n < 4:
            continue
        for phase in np.linspace(0, p, 8, endpoint=False):
            ticks = t0 + phase + p * np.arange(n + 1)
            near = np.abs(ticks[:, None] - ev[None, :]).min(axis=1)
            silent = int(np.sum(near > 0.10 * p * 2))
            nearest = np.abs(ev[:, None] - ticks[None, :]).min(axis=1)
            unsupported = int(np.sum(nearest > 0.10 * p * 2))
            c = (4.0 * silent + 1.0 * unsupported) / max(n, 1)
            if c < best_c:
                best, best_c = p, c
    return 60.0 / best if best else None


def _hopf_profile(ev: np.ndarray, dur: float) -> np.ndarray:
    """Bank of canonical Hopf oscillators driven by the event train.

    Velasco & Large (2011) ISMIR parameters: 289 oscillators log-spaced
    0.25-16 Hz, critical regime alpha=0, beta1=-1, beta2=-0.25, eps=1.
    zdot = z(alpha + i*omega + beta1|z|^2 + eps*beta2|z|^4/(1-eps|z|^2))
           + x(t)/(1-sqrt(eps)z) * 1/(1-sqrt(eps)conj(z))
    Returns steady-state mean amplitude per oscillator.
    """
    freqs = np.geomspace(0.25, 16.0, 289)
    omega = 2 * np.pi * freqs
    alpha, b1, b2, eps = 0.0, -1.0, -0.25, 1.0
    # fs raised from 200 to 2000 Hz and |z| clamped below the 1/sqrt(eps)
    # singularity: forward Euler at 200 Hz overshot the saturating nonlinearity
    # for the 16 Hz oscillators and the bank overflowed. Numerical fix only --
    # parameters are Velasco & Large's, unchanged. (Disclosed 2026-09-02.)
    fs = 2000.0
    zmax = 0.95 / np.sqrt(eps)
    n = int(dur * fs) + 1
    x = np.zeros(n)
    idx = np.clip((ev * fs).astype(int), 0, n - 1)
    np.add.at(x, idx, 1.0)
    w = np.hanning(int(0.02 * fs) * 2 + 1)
    x = np.convolve(x, w / w.sum(), mode="same")
    x *= 0.05                       # keep the drive in the small-signal regime

    z = np.zeros(len(freqs), dtype=complex)
    amp = np.zeros(len(freqs))
    dt = 1.0 / fs
    se = np.sqrt(eps)
    burn = int(n * 0.25)
    for k in range(n):
        r2 = np.abs(z) ** 2
        denom = np.maximum(1.0 - eps * r2, 1e-3)
        nonlin = alpha + 1j * omega + b1 * r2 + eps * b2 * (r2 ** 2) / denom
        drive = x[k] / np.maximum(np.abs(1.0 - se * z) ** 2, 1e-3)
        z = z + dt * (z * nonlin + drive)
        r = np.abs(z)
        over = r > zmax
        if over.any():
            z[over] = z[over] / r[over] * zmax
        if k >= burn:
            amp += np.abs(z)
    return amp / max(n - burn, 1), freqs


def est_hopf(ev: np.ndarray) -> float | None:
    if ev.size < 6:
        return None
    dur = float(ev[-1] - ev[0]) + 1.0
    amp, freqs = _hopf_profile(ev - ev[0], dur)
    band = (freqs >= 1.0 / PERIOD_HI) & (freqs <= 1.0 / PERIOD_LO)
    if not band.any() or amp[band].max() <= 0:
        return None
    f = freqs[band][int(np.argmax(amp[band]))]
    return 60.0 * f


ESTIMATORS = {
    "median-consec": est_median_consec,
    "all-pairs": est_all_pairs,
    "comb": est_comb,
    "povel-essens": est_povel_essens,
    "hopf": est_hopf,
}


def project(bpm: float):
    for f in FACTORS:
        c = bpm * f
        if BAND_LO <= c <= BAND_HI:
            return c, f
    return None, None


# ------------------------------------------------------------------- Arm C


def resonance_profile(ev: np.ndarray, truth_bpm: float) -> dict:
    """Energy at f, 2f, 3f, f/2, f/3 of the TRUE beat, linear and nonlinear,
    plus where the dominant peak actually sits."""
    out = {}
    beat_p = 60.0 / truth_bpm
    lin = _comb_profile(ev)
    dur = float(ev[-1] - ev[0]) + 1.0
    amp, freqs = _hopf_profile(ev - ev[0], dur)

    def lin_at(period):
        if period < PERIODS[0] or period > PERIODS[-1]:
            return None
        return float(lin[int(np.argmin(np.abs(PERIODS - period)))])

    def hopf_at(period):
        f = 1.0 / period
        if f < freqs[0] or f > freqs[-1]:
            return None
        return float(amp[int(np.argmin(np.abs(freqs - f)))])

    for name, mult in (("f", 1.0), ("2f", 2.0), ("3f", 3.0),
                       ("f/2", 0.5), ("f/3", 1.0 / 3.0)):
        p = beat_p / mult
        out[f"lin_{name}"] = lin_at(p)
        out[f"hopf_{name}"] = hopf_at(p)
    lmax = float(lin.max()) if lin.size else 0.0
    out["lin_peak_period"] = float(PERIODS[int(np.argmax(lin))]) if lmax > 0 else None
    out["lin_peak_ratio"] = (beat_p / out["lin_peak_period"]) if out["lin_peak_period"] else None
    out["lin_f_db_below_peak"] = (
        20 * np.log10(max(out["lin_f"], 1e-9) / max(lmax, 1e-9)) if out["lin_f"] else None
    )
    band = (freqs >= 1.0 / PERIOD_HI) & (freqs <= 1.0 / PERIOD_LO)
    hpk = freqs[band][int(np.argmax(amp[band]))] if band.any() else None
    out["hopf_peak_ratio"] = float((1.0 / hpk) and (beat_p * hpk)) if hpk else None
    hmax = float(amp[band].max()) if band.any() else 0.0
    out["hopf_f_db_below_peak"] = (
        20 * np.log10(max(out["hopf_f"], 1e-12) / max(hmax, 1e-12)) if out["hopf_f"] else None
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    from pathlib import Path

    cases = sorted((c for c in load_cases(Path(os.path.join(ROOT, "evals/cases")))
                    if not c.reference and c.maturity == "verified"), key=lambda c: c.id)
    ids = [c.id for c in cases]
    print(f"gating set: {len(cases)} rows")

    events, truth, is_demo, skipped = {}, {}, {}, []
    for c in cases:
        truth[c.id] = float(c.expected_bpm)
        is_demo[c.id] = c.tags.get("clip_role") == "demo"
        media = os.path.join(ROOT, c.media)
        meta = json.load(open(os.path.join(ROOT, "evals", c.trace, "meta.json")))
        if not os.path.exists(media):
            skipped.append(f"{c.id} (media absent)")
            continue
        if meta.get("media_sha256") and sha256_of(media) != meta["media_sha256"]:
            skipped.append(f"{c.id} (checksum mismatch)")
            continue
        events[c.id] = peakrate_events(media)
    print(f"  coverage peakrate: {len(events)}/{len(cases)}"
          + (f"  SKIPPED BY NAME: {skipped}" if skipped else "  (no skips)"))

    odd = {cid for i, cid in enumerate(ids) if i % 2 == 0}
    results = {}
    for name, fn in ESTIMATORS.items():
        rows = {}
        for cid in ids:
            ev = events.get(cid)
            if ev is None:
                rows[cid] = {"skipped_source": True}
                continue
            raw = fn(ev)
            r = {"raw_bpm": None if raw is None else round(raw, 2),
                 "truth": truth[cid], "n_events": int(ev.size)}
            bpm, fac = (None, None) if raw is None else project(raw)
            r["bpm"], r["factor"] = (None if bpm is None else round(bpm, 2)), fac
            if bpm is not None:
                r["pass"] = abs(bpm - truth[cid]) / truth[cid] <= PASS_TOL
                r["acc2"] = bool(acc2(bpm, truth[cid], PASS_TOL))
                _, oe2 = octave_errors(bpm, truth[cid])
                r["oe2"] = round(oe2, 4)
                r["between_levels"] = bool(0.08 < abs(oe2) <= 0.585)
            else:
                r.update({"pass": False, "acc2": False, "oe2": None,
                          "between_levels": False, "abstained": True})
            rows[cid] = r
        sc = [r for r in rows.values() if not r.get("skipped_source")]
        dm = [rows[c] for c in ids if is_demo[c] and not rows[c].get("skipped_source")]
        rg = [rows[c] for c in ids if not is_demo[c] and not rows[c].get("skipped_source")]
        o = [rows[c] for c in ids if c in odd and not rows[c].get("skipped_source")]
        e = [rows[c] for c in ids if c not in odd and not rows[c].get("skipped_source")]
        rate = lambda rs: (sum(bool(r["pass"]) for r in rs) / len(rs)) if rs else 0.0  # noqa: E731
        results[name] = {
            "rows": rows, "n": len(sc),
            "pass_total": sum(bool(r["pass"]) for r in sc),
            "pass_demo": sum(bool(r["pass"]) for r in dm), "n_demo": len(dm),
            "pass_rig": sum(bool(r["pass"]) for r in rg), "n_rig": len(rg),
            "acc2": sum(bool(r["acc2"]) for r in sc),
            "between_levels": sum(bool(r["between_levels"]) for r in sc),
            "abstained": sum(bool(r.get("abstained")) for r in sc),
            "pass_odd": sum(bool(r["pass"]) for r in o),
            "pass_even": sum(bool(r["pass"]) for r in e),
            "half_gap": round(abs(rate(o) - rate(e)), 4),
        }

    hdr = (f"{'estimator':<16}{'pass':>9}{'demo':>7}{'rig':>8}{'Acc2':>6}"
           f"{'btwn':>6}{'abst':>6}{'odd':>6}{'even':>6}{'gap':>7}")
    print("\nARM A — five estimators, identical peakRate events\n" + hdr + "\n" + "-" * len(hdr))
    for n, r in results.items():
        print(f"{n:<16}{r['pass_total']:>4}/{r['n']:<4}{r['pass_demo']:>3}/{r['n_demo']:<3}"
              f"{r['pass_rig']:>4}/{r['n_rig']:<3}{r['acc2']:>6}{r['between_levels']:>6}"
              f"{r['abstained']:>6}{r['pass_odd']:>6}{r['pass_even']:>6}{r['half_gap']:>7.3f}")

    print("\nARM C — resonance profile on the 8 demos (dB relative to the dominant peak)")
    print(f"{'demo':<26}{'truth':>7}{'lin peak/beat':>15}{'lin f':>9}{'hopf f':>9}   regime")
    profiles = {}
    for cid in ids:
        if not is_demo[cid] or cid not in events:
            continue
        p = resonance_profile(events[cid], truth[cid])
        profiles[cid] = p
        ratio = p["lin_peak_ratio"]
        lf, hf = p["lin_f_db_below_peak"], p["hopf_f_db_below_peak"]
        near_int = ratio is not None and min(abs(ratio - m) / m for m in (1/3, .5, 1, 2, 3)) <= 0.05
        if lf is not None and lf > -6:
            reg = "beat present linearly"
        elif hf is not None and hf > -6:
            reg = "MISSING PULSE (nonlinear only)"
        else:
            reg = "beat weak in both"
        reg += "" if near_int else "  [dominant peak at NON-INTEGER ratio -> clutter]"
        print(f"{cid.replace('barre6-','').replace('-demo',''):<26}{truth[cid]:>7.0f}"
              f"{(ratio if ratio else float('nan')):>15.2f}"
              f"{(lf if lf is not None else float('nan')):>9.1f}"
              f"{(hf if hf is not None else float('nan')):>9.1f}   {reg}")

    if args.json:
        def _j(o):
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            raise TypeError(type(o))

        json.dump({"arm_a": results, "arm_c": profiles, "skipped": skipped,
                   "odd_half": sorted(odd), "periods": [PERIOD_LO, PERIOD_HI, N_PERIODS]},
                  open(args.json, "w"), indent=1, sort_keys=True, default=_j)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
