#!/usr/bin/env python
"""SW-1: does one steady window beat reading the whole clip?

REPORTED-ONLY. Search space frozen in the 2026-09-01 commissioning ledger
entry; pre-registration in docs/research/sw1-steady-window-sweep.md, which
was committed before this file existed.

  sources       peakrate-media (rung-2 extractor on the clip's audio,
                checksum-verified) | whisper-trace (word onsets)
  windows       L in {3,5,8} s, slide step 0.5 s
  pick          minimum within-window IOI CV, >= 6 events; else whole-clip
                fallback, reported by name
  tempo         60 / median IOI, projected into [70,140] by x/{2,3};
                the factor is reported per clip
  controls      whole-clip estimate per source
  ceiling       oracle windows from the demo cases' intended-tempo spans
                (rig oracle = whole clip)

Reads evals/cases, evals/traces and the media the cases point at. Writes
nothing under evals/ or src/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
STEP_S = 0.5
LENGTHS = (3.0, 5.0, 8.0)
# Tried in this order; the band is exactly one octave wide, so two factors can
# only both land in band at the exact boundary. Order stated for determinism.
FACTORS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)
VIDEO_SUFFIXES = {".mov", ".m4v", ".mp4", ".avi", ".mkv", ".webm"}
SPAN_RE = re.compile(r"Intended-tempo span within this demo:\s*([0-9.]+)\s*-\s*([0-9.]+)\s*s")


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


def whisper_events(trace_dir: str) -> np.ndarray:
    words = json.load(open(os.path.join(trace_dir, "whisper.json")))["words"]
    return np.asarray(sorted(float(w["start"]) for w in words), dtype=float)


# ---------------------------------------------------------------- estimation


def project(bpm: float) -> tuple[float | None, float | None]:
    """(in-band bpm, factor) or (None, None) if no factor lands in band."""
    for f in FACTORS:
        cand = bpm * f
        if BAND_LO <= cand <= BAND_HI:
            return cand, f
    return None, None


def raw_bpm(ev: np.ndarray) -> float | None:
    if ev.size < MIN_EVENTS:
        return None
    iois = np.diff(ev)
    iois = iois[iois > 0]
    if iois.size < MIN_EVENTS - 1:
        return None
    med = float(np.median(iois))
    return 60.0 / med if med > 0 else None


def ioi_cv(ev: np.ndarray) -> float | None:
    if ev.size < MIN_EVENTS:
        return None
    iois = np.diff(ev)
    iois = iois[iois > 0]
    if iois.size < MIN_EVENTS - 1:
        return None
    m = float(iois.mean())
    return float(iois.std(ddof=0) / m) if m > 0 else None


def best_window(ev: np.ndarray, length: float):
    """Minimum-IOI-CV window; None if no window holds >= MIN_EVENTS events."""
    if ev.size < MIN_EVENTS:
        return None
    t0, t1 = float(ev[0]), float(ev[-1])
    if t1 - t0 < length:
        return None
    best = None
    t = t0
    while t <= t1 - length + 1e-9:
        inside = ev[(ev >= t) & (ev <= t + length)]
        cv = ioi_cv(inside)
        if cv is not None and (best is None or cv < best[0]):
            best = (cv, t, inside)
        t += STEP_S
    return best


def estimate(ev: np.ndarray, length: float | None, span: tuple[float, float] | None = None):
    """One (source, variant) estimate for one clip."""
    out = {"window": None, "fallback": False, "cv": None, "n_events": None}
    if span is not None:
        inside = ev[(ev >= span[0]) & (ev <= span[1])]
        out["window"] = [round(span[0], 2), round(span[1], 2)]
        chosen = inside
    elif length is None:
        chosen = ev
    else:
        b = best_window(ev, length)
        if b is None:
            out["fallback"] = True
            chosen = ev
        else:
            out["cv"] = round(b[0], 4)
            out["window"] = [round(b[1], 2), round(b[1] + length, 2)]
            chosen = b[2]
    out["n_events"] = int(chosen.size)
    r = raw_bpm(chosen)
    if r is None:
        out.update(raw_bpm=None, bpm=None, factor=None, abstained=True)
        return out
    bpm, factor = project(r)
    out.update(raw_bpm=round(r, 2), bpm=None if bpm is None else round(bpm, 2),
               factor=factor, abstained=bpm is None)
    if out["cv"] is None and chosen.size >= MIN_EVENTS:
        out["cv"] = None if ioi_cv(chosen) is None else round(ioi_cv(chosen), 4)
    return out


# ---------------------------------------------------------------------- main


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

    skipped: dict[str, list[str]] = {"peakrate-media": [], "whisper-trace": []}
    events: dict[str, dict[str, np.ndarray]] = {"peakrate-media": {}, "whisper-trace": {}}
    spans: dict[str, tuple[float, float]] = {}
    truth: dict[str, float] = {}
    is_demo: dict[str, bool] = {}

    for c in cases:
        truth[c.id] = float(c.expected_bpm)
        is_demo[c.id] = c.tags.get("clip_role") == "demo"
        m = SPAN_RE.search(c.notes or "")
        if m:
            spans[c.id] = (float(m.group(1)), float(m.group(2)))
        trace_dir = os.path.join(ROOT, "evals", c.trace)
        try:
            events["whisper-trace"][c.id] = whisper_events(trace_dir)
        except Exception as e:  # noqa: BLE001
            skipped["whisper-trace"].append(f"{c.id} ({type(e).__name__})")
        media = os.path.join(ROOT, c.media)
        meta = json.load(open(os.path.join(trace_dir, "meta.json")))
        want = meta.get("media_sha256")
        if not os.path.exists(media):
            skipped["peakrate-media"].append(f"{c.id} (media absent)")
            continue
        if want and sha256_of(media) != want:
            skipped["peakrate-media"].append(f"{c.id} (checksum mismatch)")
            continue
        events["peakrate-media"][c.id] = peakrate_events(media)

    for src in ("peakrate-media", "whisper-trace"):
        n = len(events[src])
        print(f"  coverage {src:<16} {n}/{len(cases)}"
              + (f"  SKIPPED BY NAME: {skipped[src]}" if skipped[src] else "  (no skips)"))

    variants: list[tuple[str, str, float | None, bool]] = []
    for src in ("peakrate-media", "whisper-trace"):
        for L in LENGTHS:
            variants.append((f"{src} · window {L:g}s", src, L, False))
        variants.append((f"{src} · whole-clip CONTROL", src, None, False))
        variants.append((f"{src} · ORACLE span CEILING", src, None, True))

    ids = [c.id for c in cases]
    odd = {cid for i, cid in enumerate(ids) if i % 2 == 0}   # rows 1,3,5...
    results = {}
    for name, src, L, oracle in variants:
        rows = {}
        for cid in ids:
            ev = events[src].get(cid)
            if ev is None:
                rows[cid] = {"skipped_source": True}
                continue
            span = spans.get(cid) if oracle else None
            r = estimate(ev, None if oracle else L, span)
            r["truth"] = truth[cid]
            if r["bpm"] is not None:
                r["pass"] = abs(r["bpm"] - truth[cid]) / truth[cid] <= PASS_TOL
                r["acc2"] = bool(acc2(r["bpm"], truth[cid], PASS_TOL))
                oe1, oe2 = octave_errors(r["bpm"], truth[cid])
                r["oe1"], r["oe2"] = round(oe1, 4), round(oe2, 4)
                r["between_levels"] = bool(0.08 < abs(oe2) <= 0.585)
            else:
                r.update(**{"pass": False, "acc2": False, "oe1": None,
                            "oe2": None, "between_levels": False})
            rows[cid] = r
        scored = [r for r in rows.values() if not r.get("skipped_source")]
        demo = [rows[c] for c in ids if is_demo[c] and not rows[c].get("skipped_source")]
        rig = [rows[c] for c in ids if not is_demo[c] and not rows[c].get("skipped_source")]
        o = [rows[c] for c in ids if c in odd and not rows[c].get("skipped_source")]
        e = [rows[c] for c in ids if c not in odd and not rows[c].get("skipped_source")]
        rate = lambda rs: (sum(bool(r["pass"]) for r in rs) / len(rs)) if rs else 0.0  # noqa: E731
        results[name] = {
            "source": src, "length_s": L, "oracle": oracle, "rows": rows,
            "n_scored": len(scored),
            "pass_total": sum(bool(r["pass"]) for r in scored),
            "pass_demo": sum(bool(r["pass"]) for r in demo), "n_demo": len(demo),
            "pass_rig": sum(bool(r["pass"]) for r in rig), "n_rig": len(rig),
            "acc2_total": sum(bool(r["acc2"]) for r in scored),
            "between_levels": sum(bool(r["between_levels"]) for r in scored),
            "abstained": sum(bool(r.get("abstained")) for r in scored),
            "fallbacks": [c for c in ids if rows[c].get("fallback")],
            "pass_odd": sum(bool(r["pass"]) for r in o), "n_odd": len(o),
            "pass_even": sum(bool(r["pass"]) for r in e), "n_even": len(e),
            "half_gap": round(abs(rate(o) - rate(e)), 4),
            "factor_not_1": sum(1 for r in scored if r.get("factor") not in (None, 1.0)),
        }

    hdr = (f"{'variant':<40}{'pass':>8}{'demo':>7}{'rig':>8}{'Acc2':>7}"
           f"{'btwn':>6}{'abst':>6}{'fallb':>7}{'odd':>7}{'even':>7}{'gap':>7}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for name, r in results.items():
        print(f"{name:<40}{r['pass_total']:>3}/{r['n_scored']:<4}"
              f"{r['pass_demo']:>3}/{r['n_demo']:<3}{r['pass_rig']:>4}/{r['n_rig']:<3}"
              f"{r['acc2_total']:>7}{r['between_levels']:>6}{r['abstained']:>6}"
              f"{len(r['fallbacks']):>7}{r['pass_odd']:>3}/{r['n_odd']:<3}"
              f"{r['pass_even']:>4}/{r['n_even']:<3}{r['half_gap']:>7.3f}")

    sweep = {k: v for k, v in results.items() if not v["oracle"] and v["length_s"] is not None}
    ranked = sorted(sweep.items(), key=lambda kv: (kv[1]["half_gap"],
                                                   -kv[1]["pass_demo"], -kv[1]["pass_total"]))
    print("\nselection (stability, then demo passes, then total passes):")
    for i, (name, r) in enumerate(ranked, 1):
        print(f"  {i}. {name:<38} gap {r['half_gap']:.3f}  demo {r['pass_demo']}/{r['n_demo']}"
              f"  total {r['pass_total']}/{r['n_scored']}")
    print("\nWINNER (NOT ADOPTED — reported only): " + ranked[0][0])

    for name, r in results.items():
        if r["fallbacks"]:
            print(f"  whole-clip fallback in {name}: {r['fallbacks']}")

    if args.json:
        payload = {
            "generated": "2026-09-02", "band": [BAND_LO, BAND_HI],
            "pass_tol": PASS_TOL, "min_events": MIN_EVENTS, "step_s": STEP_S,
            "lengths_s": list(LENGTHS), "factors": list(FACTORS),
            "skipped": skipped, "odd_half": sorted(odd),
            "even_half": sorted(set(ids) - odd),
            "variants": results,
            "ranking": [n for n, _ in ranked],
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
