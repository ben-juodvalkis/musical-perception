#!/usr/bin/env python
"""W2 reopened: is there bar-level periodic accent in GENUINE prominence?

W2 (2026-08-20, negative) audited per-beat salience built from three
amplitude-free channels — following-IOI, event density, voicing — because
its only input was a times-only events file. It never measured loudness or
pitch emphasis, which is the cue the owner reports reading in class (the
teacher speaks louder on the strong beat). This script keeps W2's method
exactly and swaps the channels for prominence measured on the audio at each
grid beat:

  intensity  Praat intensity (dB), max in [beat-30ms, beat+150ms]
  f0         Praat F0 (semitones re 100 Hz), median of voiced frames in the
             same window
  whistress  WhiStress per-word stress, mapped from the trace's Whisper
             words to the word at each beat (optional, --whistress)

Silent beats take the clip's minimum per-beat value in every channel
(silence is the least prominent thing a beat can carry). Each channel is
detrended by a local median over +-8 beats in ORIGINAL index space, then
z-scored per clip; "combined" is the equal-weight mean of the channels
present, as W2 combined its three.

Method, unchanged from scripts/rung3-accent-evidence-audit.py: on-minus-off
mean salience at lags 2/3/4/6/8, best over phase, against a 400-draw
phase-shuffle null; the template confusability matrix; and, new here, the
EMPIRICAL confusability — the salience-clock margin of the truth template
over its confusable sibling (4/4 vs 2/4, 6/8 vs 3/4) per clip.

The W2 channels are recomputed on the same clips (`--channels w2`) so the
old-versus-new comparison is like-for-like.

REPORTED-ONLY. Reads evals/grids, evals/cases, evals/traces and the media
the grids point at (checksum-verified; a missing or mismatched file is
skipped BY NAME, never silently). Writes nothing under evals/ or src/.
Optional --json writes a results file wherever you say.

Usage:
  python scripts/w2-reopen-prominence-audit.py [--only rig|barre6]
      [--channels prominence|w2|both] [--whistress] [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import zlib

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from musical_perception.precision.accent_meter import (  # noqa: E402
    METER_TEMPLATES,
    _clock_score,
    beat_salience,
)

LAGS = [2, 3, 4, 6, 8]
N_SHUFFLE = 400
SEED = 20260902  # fixed: the audit must replay identically
WIN_PRE_S = 0.03
WIN_POST_S = 0.15
PITCH_FLOOR_HZ = 75.0
PITCH_CEIL_HZ = 450.0
DETREND_HALF = 8
MARGIN = 0.05  # W2's abstention band

METER_TO_PERIOD = {"2/4": 2, "3/4": 3, "4/4": 4, "6/8": 6}
FAMILY = {"2/4": "duple", "4/4": "duple", "3/4": "triple", "6/8": "triple"}
SIBLING = {"4/4": "2/4", "2/4": "4/4", "6/8": "3/4", "3/4": "6/8"}

# Declared before measurement (W2's P6): the grid is at the number level and
# the 3/4 label lives in the 'and-ah' subdivision BELOW the tactus, so bar
# length in grid-beat units is 1 — degenerate for any bar-lag question.
DEGENERATE = {"rig-numbers-3-4-90-clean"}

PROMINENCE_CHANNELS = ["intensity", "f0"]


# --------------------------------------------------------------------------
# grids, cases, media


def slice_of(cid: str) -> str:
    if cid.startswith("barre6-"):
        return "barre6"
    if cid.startswith("rig-") or cid.startswith("adr006-"):
        return "rig"
    return "other"


def load_grid_beats(grid: dict):
    """Beat sequence with silent beats reinstated, plus free-time spans (as W2)."""
    beats = [(t, True) for t in grid["beats"]]
    free_time: list[tuple[float, float]] = []
    for region in grid.get("regions") or []:
        kind = region.get("kind")
        if kind == "silent_beat":
            beats.append(((region["start"] + region["end"]) / 2.0, False))
        elif kind == "free_time":
            free_time.append((region["start"], region["end"]))
    beats.sort(key=lambda p: p[0])
    return [t for t, _ in beats], [v for _, v in beats], free_time


def segment_ids(times, free_time):
    if not free_time:
        return [0] * len(times)
    cuts = sorted(start for start, _ in free_time)
    return [sum(1 for c in cuts if c < t) for t in times]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_media(grid: dict) -> tuple[str | None, str]:
    """(path, status) — status is 'ok', 'missing', or 'checksum-mismatch'."""
    rel = grid.get("media") or ""
    path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
    if not rel or not os.path.exists(path):
        return None, "missing"
    want = grid.get("media_sha256")
    if want and sha256_of(path) != want:
        return None, "checksum-mismatch"
    return path, "ok"


# --------------------------------------------------------------------------
# prominence features


def audio_contours(path: str) -> dict:
    import librosa
    import parselmouth

    y, sr = librosa.load(path, sr=None, mono=True)
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=float(sr))
    pitch = snd.to_pitch(
        time_step=0.01, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEIL_HZ
    )
    inten = snd.to_intensity(minimum_pitch=PITCH_FLOOR_HZ, time_step=0.01)
    return {
        "pitch_t": np.asarray(pitch.xs()),
        "pitch_hz": np.asarray(pitch.selected_array["frequency"]),
        "int_t": np.asarray(inten.xs()),
        "int_db": np.asarray(inten.values[0]),
        "duration": float(snd.get_total_duration()),
    }


def window_features(times, voiced, c: dict) -> dict[str, list[float]]:
    inten, f0 = [], []
    for t, v in zip(times, voiced):
        if not v:
            inten.append(np.nan)
            f0.append(np.nan)
            continue
        lo, hi = t - WIN_PRE_S, t + WIN_POST_S
        m = (c["int_t"] >= lo) & (c["int_t"] <= hi)
        inten.append(float(c["int_db"][m].max()) if m.any() else np.nan)
        mp = (c["pitch_t"] >= lo) & (c["pitch_t"] <= hi)
        pv = c["pitch_hz"][mp]
        pv = pv[pv > 0]
        f0.append(float(12.0 * np.log2(np.median(pv) / 100.0)) if len(pv) else np.nan)
    return {"intensity": inten, "f0": f0}


def whistress_features(audio_path: str, trace_dir: str, times, voiced) -> list[float]:
    """Per-beat WhiStress stress label via the trace's Whisper words.

    UNTESTED in the session that wrote this (the WhiStress code could not be
    installed there — GitHub refused by the proxy; Hugging Face carries only
    the weights). Raises on any failure; the caller reports and drops the
    channel rather than filling it.
    """
    import difflib

    from musical_perception.perception import whistress as ws
    from musical_perception.types import TimestampedWord

    words_json = json.load(open(os.path.join(trace_dir, "whisper.json")))["words"]
    words = [TimestampedWord(word=w["word"], start=w["start"], end=w["end"]) for w in words_json]
    client = ws.load_model()
    pairs = ws.predict_stress(client, audio_path, words)
    norm = lambda s: s.strip().lower().strip(".,!?;:")  # noqa: E731
    a = [norm(w.word) for w in words]
    b = [norm(p[0]) for p in pairs]
    label_for_word: dict[int, float] = {}
    for blk in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            label_for_word[blk.a + k] = float(pairs[blk.b + k][1])
    out = []
    for t, v in zip(times, voiced):
        if not v:
            out.append(np.nan)
            continue
        best, best_d = None, 0.2
        for i, w in enumerate(words):
            if i not in label_for_word:
                continue
            if w.start - 0.05 <= t <= w.end + 0.05:
                best, best_d = i, 0.0
                break
            d = abs(w.start - t)
            if d < best_d:
                best, best_d = i, d
        out.append(label_for_word[best] if best is not None else np.nan)
    return out


def fill_detrend_z(values: list[float], segs: list[int]) -> np.ndarray:
    """NaN/silent -> clip minimum; local-median detrend within segment; z-score."""
    arr = np.asarray(values, dtype=float)
    if np.all(~np.isfinite(arr)):
        return np.zeros_like(arr)
    arr = np.where(np.isfinite(arr), arr, np.nanmin(arr))
    segs_a = np.asarray(segs)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo, hi = max(0, i - DETREND_HALF), min(len(arr), i + DETREND_HALF + 1)
        win = [arr[j] for j in range(lo, hi) if segs_a[j] == segs_a[i]]
        out[i] = arr[i] - float(np.median(win))
    sd = out.std()
    return (out - out.mean()) / sd if sd > 1e-9 else np.zeros_like(out)


# --------------------------------------------------------------------------
# W2's audit, verbatim in substance


def periodicity(sal: np.ndarray, lag: int) -> float:
    """Mean salience at every lag-th position minus the rest, best over phase."""
    if len(sal) < 2 * lag:
        return float("nan")
    best = -np.inf
    for phase in range(lag):
        idx = np.arange(phase, len(sal), lag)
        if len(idx) < 2:
            continue
        best = max(best, sal[idx].mean() - np.delete(sal, idx).mean())
    return float(best)


def audit_lags(sal: np.ndarray, key: str) -> dict[int, tuple[float, float]]:
    out = {}
    for lag in LAGS:
        obs = periodicity(sal, lag)
        if not np.isfinite(obs):
            continue
        rng = np.random.default_rng([SEED, zlib.crc32(f"{key}:{lag}".encode())])
        null = np.array([periodicity(rng.permutation(sal), lag) for _ in range(N_SHUFFLE)])
        out[lag] = (obs, float((null >= obs).mean()))
    return out


def _margin(sal: np.ndarray, segs: np.ndarray, truth: str) -> float:
    best = lambda m: max(  # noqa: E731
        _clock_score(sal, segs, METER_TEMPLATES[m], p) for p in range(len(METER_TEMPLATES[m]))
    )
    return float(best(truth) - best(SIBLING[truth]))


def template_margin(
    sal: np.ndarray, segs: np.ndarray, truth: str, key: str
) -> tuple[float, float] | None:
    """(margin, p): best-phase clock score of the truth template minus its
    confusable sibling, with a phase-shuffle null so a margin only counts
    when it beats chance — W2's bare 0.05 band is reported beside it."""
    if truth not in SIBLING:
        return None
    obs = _margin(sal, segs, truth)
    rng = np.random.default_rng([SEED, zlib.crc32(f"{key}:margin".encode())])
    null = np.array([_margin(rng.permutation(sal), segs, truth) for _ in range(N_SHUFFLE)])
    return obs, float((null >= obs).mean())


def template_confusability() -> dict[str, dict[str, float]]:
    names, n = list(METER_TEMPLATES), 24
    tiled = lambda name, ph: np.array(  # noqa: E731
        [METER_TEMPLATES[name][(k - ph) % len(METER_TEMPLATES[name])] for k in range(n)]
    )
    return {
        a: {
            b: max(
                abs(np.corrcoef(tiled(a, 0), tiled(b, p))[0, 1])
                for p in range(len(METER_TEMPLATES[b]))
            )
            for b in names
        }
        for a in names
    }


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["rig", "barre6"], default=None)
    ap.add_argument("--channels", choices=["prominence", "w2", "both"], default="both")
    ap.add_argument("--whistress", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cases = {
        c["id"]: c
        for c in (yaml.safe_load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, "evals/cases/*.yaml"))))
    }
    events_doc = json.load(open(os.path.join(ROOT, "docs/research/rung2-extractor-events.json")))
    w2_events = events_doc["events"]

    rows, skipped = [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "evals/grids/*.yaml"))):
        grid = yaml.safe_load(open(path))
        cid = grid["clip"]
        sl = slice_of(cid)
        if args.only and sl != args.only:
            continue
        case = cases.get(cid)
        if case is None:
            skipped.append((cid, "no case"))
            continue
        truth = case.get("expect", {}).get("meter")
        if grid.get("provisional") and sl != "barre6":
            # W2 skipped provisional grids; the barre-6 slice is reported as
            # its own all-provisional population, anything else is skipped.
            skipped.append((cid, "provisional grid outside the barre6 slice (W2 rule)"))
            continue
        times, voiced, free_time = load_grid_beats(grid)
        segs = segment_ids(times, free_time)
        row = {
            "clip": cid,
            "slice": sl,
            "provisional_grid": bool(grid.get("provisional")),
            "truth": truth,
            "bar_lag": None if cid in DEGENERATE else METER_TO_PERIOD.get(truth or ""),
            "degenerate": cid in DEGENERATE,
            "n_beats": len(times),
            "n_silent": sum(1 for v in voiced if not v),
            "channels": {},
            "notes": [],
        }
        chan: dict[str, np.ndarray] = {}

        if args.channels in ("w2", "both"):
            ev = w2_events.get(cid)
            if ev is None:
                row["notes"].append("w2: no committed events for this clip (agogic+voicing only)")
            sal = beat_salience(times, ev, voiced_flags=voiced, free_time=free_time)
            chan["w2"] = np.asarray(sal.combined, dtype=float)

        if args.channels in ("prominence", "both"):
            media, status = resolve_media(grid)
            if media is None:
                skipped.append((cid, f"media {status}: {grid.get('media')}"))
                row["notes"].append(f"prominence: media {status}")
            else:
                feats = window_features(times, voiced, audio_contours(media))
                if args.whistress:
                    tdir = os.path.join(ROOT, "evals/traces", cid)
                    try:
                        feats["whistress"] = whistress_features(media, tdir, times, voiced)
                    except Exception as exc:  # reported, never filled
                        row["notes"].append(f"whistress: FAILED {type(exc).__name__}: {exc}")
                zs = {k: fill_detrend_z(v, segs) for k, v in feats.items()}
                for k, z in zs.items():
                    chan[k] = z
                chan["combined"] = np.mean(np.stack(list(zs.values())), axis=0)

        segs_a = np.asarray(segs)
        for name, sal in chan.items():
            lags = audit_lags(sal, f"{cid}:{name}")
            sig = {l: v for l, v in lags.items() if v[1] < 0.05}
            win = max(sig, key=lambda l: sig[l][0]) if sig else None
            bl = row["bar_lag"]
            tm = template_margin(sal, segs_a, truth, f"{cid}:{name}")
            row["channels"][name] = {
                "template_margin_p": None if tm is None else round(tm[1], 4),
                "lags": {str(l): {"obs": round(o, 3), "p": round(p, 4)} for l, (o, p) in lags.items()},
                "winner": win,
                "any_sig": bool(sig),
                "any_sig_01": any(v[1] < 0.01 for v in lags.values()),
                "bar_sig": bool(bl in lags and lags[bl][1] < 0.05),
                "bar_sig_01": bool(bl in lags and lags[bl][1] < 0.01),
                "bar_obs": None if bl not in lags else round(lags[bl][0], 3),
                "template_margin": None if tm is None else round(tm[0], 3),
            }
        rows.append(row)

    # ---------------------------------------------------------------- report
    conf = template_confusability()
    print("template confusability (max |corr| over relative phase, 24 beats) — "
          "channel-independent by construction:")
    names = list(METER_TEMPLATES)
    print("       " + " ".join(f"{m:>6s}" for m in names))
    for a in names:
        print(f"{a:>6s} " + " ".join(f"{conf[a][b]:6.2f}" for b in names))

    summary: dict = {}
    for sl in ("rig", "barre6", "other"):
        srows = [r for r in rows if r["slice"] == sl and r["channels"]]
        if not srows:
            continue
        prov = sum(1 for r in srows if r["provisional_grid"])
        print(f"\n=== slice {sl}: {len(srows)} clips audited"
              f"{f' ({prov} on PROVISIONAL grids — bar lag in grid units is NOT the bar)' if prov else ''} ===")
        chans = sorted({c for r in srows for c in r["channels"]},
                       key=lambda c: (c != "w2", c != "combined", c))
        summary[sl] = {"n": len(srows), "provisional_grids": prov, "channels": {}}
        for ch in chans:
            hdr = f"{'clip':34s} {'truth':5s} bar " + " ".join(f"lag{l:<5d}" for l in LAGS) + "  win  bar?  margin"
            print(f"\n--- channel: {ch} ---")
            print(hdr)
            print("-" * len(hdr))
            have = [r for r in srows if ch in r["channels"]]
            for r in have:
                c = r["channels"][ch]
                cells = []
                for l in LAGS:
                    d = c["lags"].get(str(l))
                    if d is None:
                        cells.append("  n/a   ")
                        continue
                    star = "**" if d["p"] < 0.01 else ("* " if d["p"] < 0.05 else "  ")
                    cells.append(f"{d['obs']:5.2f}{star} ")
                mg = "  -  " if c["template_margin"] is None else (
                    f"{c['template_margin']:+.3f}"
                    + ("**" if c["template_margin_p"] < 0.01 else "* " if c["template_margin_p"] < 0.05 else "  ")
                )
                print(f"{r['clip']:34s} {str(r['truth']):5s} {str(r['bar_lag'] or '-'):>3s} "
                      + "".join(cells)
                      + f"  {str(c['winner'] or '-'):>3s}  {'YES' if c['bar_sig'] else ' - ':>4s}  {mg}")
            scoreable = [r for r in have if r["bar_lag"]]
            any_sig = sum(1 for r in have if r["channels"][ch]["any_sig"])
            any_01 = sum(1 for r in have if r["channels"][ch]["any_sig_01"])
            bar_sig = sum(1 for r in scoreable if r["channels"][ch]["bar_sig"])
            bar_01 = sum(1 for r in scoreable if r["channels"][ch]["bar_sig_01"])
            winners: dict[int, int] = {}
            for r in have:
                w = r["channels"][ch]["winner"]
                if w:
                    winners[w] = winners.get(w, 0) + 1
            duple44 = [r for r in have if r["truth"] == "4/4"]
            m44 = [r["channels"][ch]["template_margin"] for r in duple44]
            m44_win = sum(1 for m in m44 if m is not None and m >= MARGIN)
            m44_sig = sum(1 for r in duple44 if r["channels"][ch]["template_margin_p"] < 0.05)
            six8 = [r for r in have if r["truth"] == "6/8"]
            m68 = [(r["channels"][ch]["template_margin"], r["channels"][ch]["template_margin_p"]) for r in six8]
            summary[sl]["channels"][ch] = {
                "n": len(have),
                "n_bar_scoreable": len(scoreable),
                "any_sig": any_sig,
                "any_sig_p01": any_01,
                "bar_sig": bar_sig,
                "bar_sig_p01": bar_01,
                "winners": {str(k): v for k, v in sorted(winners.items())},
                "no_sig": len(have) - sum(winners.values()),
                "margin_44_over_24_mean": None if not m44 else round(float(np.mean(m44)), 3),
                "margin_44_over_24_ge_0.05": f"{m44_win}/{len(m44)}",
                "margin_44_over_24_sig": f"{m44_sig}/{len(m44)}",
                "margin_68_over_34": m68,
            }
            s = summary[sl]["channels"][ch]
            print(f"\n  any significant lag: {any_sig}/{len(have)} at p<.05  ({any_01} at p<.01)")
            print(f"  significant AT THE BAR LAG: {bar_sig}/{len(scoreable)} at p<.05  ({bar_01} at p<.01)"
                  f"   [degenerate excluded by name: {sorted(DEGENERATE & {r['clip'] for r in have})}]")
            print(f"  winning lag (strongest significant): "
                  + ", ".join(f"lag {k}: {v}" for k, v in sorted(winners.items()))
                  + f"; none: {s['no_sig']}")
            print(f"  empirical confusability — 4/4 clips where the 4/4 template beats 2/4 by >= {MARGIN}: "
                  f"{s['margin_44_over_24_ge_0.05']} (mean margin {s['margin_44_over_24_mean']}); "
                  f"beating the shuffle null at p<.05: {s['margin_44_over_24_sig']}; "
                  f"6/8 over 3/4 (margin, p): {s['margin_68_over_34']}")

        # like-for-like: only clips that carry EVERY channel in this slice
        common = [r for r in srows if all(ch in r["channels"] for ch in chans)]
        if len(common) < len(srows) and len(chans) > 1:
            print(f"\n--- like-for-like on the {len(common)} clips carrying every channel ---")
            print(f"  {'channel':10s} {'any-sig':>8s} {'any p<.01':>10s} {'bar-sig':>8s} {'bar p<.01':>10s}  winners")
            summary[sl]["like_for_like"] = {"n": len(common), "channels": {}}
            for ch in chans:
                sc = [r for r in common if r["bar_lag"]]
                d = {
                    "any_sig": sum(1 for r in common if r["channels"][ch]["any_sig"]),
                    "any_sig_p01": sum(1 for r in common if r["channels"][ch]["any_sig_01"]),
                    "bar_sig": sum(1 for r in sc if r["channels"][ch]["bar_sig"]),
                    "bar_sig_p01": sum(1 for r in sc if r["channels"][ch]["bar_sig_01"]),
                    "n_bar_scoreable": len(sc),
                }
                wins: dict[str, int] = {}
                for r in common:
                    wv = r["channels"][ch]["winner"]
                    if wv:
                        wins[str(wv)] = wins.get(str(wv), 0) + 1
                d["winners"] = dict(sorted(wins.items()))
                summary[sl]["like_for_like"]["channels"][ch] = d
                print(f"  {ch:10s} {d['any_sig']:>5d}/{len(common):<2d} {d['any_sig_p01']:>10d} "
                      f"{d['bar_sig']:>5d}/{len(sc):<2d} {d['bar_sig_p01']:>10d}  {d['winners']}")

    if skipped:
        print("\nskipped (by name, never silently):")
        for cid, why in skipped:
            print(f"  {cid:34s} {why}")
    notes = [(r["clip"], n) for r in rows for n in r["notes"]]
    if notes:
        print("\nnotes:")
        for cid, n in notes:
            print(f"  {cid:34s} {n}")

    # ---------------------------------------------------------- scorecard
    print("\n=== pre-registered scorecard (ledger 2026-09-02) ===")

    def get(sl, ch, key):
        return summary.get(sl, {}).get("channels", {}).get(ch, {}).get(key)

    def line(pid, text, verdict):
        print(f"  {pid:5s} {verdict:8s} {text}")

    b6 = get("barre6", "combined", "bar_sig")
    line("P1", f"barre-6 combined bar-lag significant <= 8/26 on provisional grids: {b6}/{get('barre6','combined','n_bar_scoreable')}",
         "NOT-RUN" if b6 is None else ("HIT" if b6 <= 8 else "MISS"))
    ri = get("rig", "intensity", "bar_sig")
    line("P2", f"rig intensity bar-lag significant in 6..9 (old channels: {get('rig','w2','bar_sig')}): {ri}/{get('rig','intensity','n_bar_scoreable')}",
         "NOT-RUN" if ri is None else ("HIT" if 6 <= ri <= 9 else "MISS"))
    rf = get("rig", "f0", "bar_sig")
    line("P3", f"rig f0 bar-lag significant < intensity: f0 {rf} vs intensity {ri}",
         "NOT-RUN" if rf is None or ri is None else ("HIT" if rf < ri else "MISS"))
    w = get("rig", "combined", "winners") or {}
    line("P4", f"rig combined winners lag 8 >= lag 4: {w}",
         "NOT-RUN" if not summary.get("rig") else ("HIT" if w.get("8", 0) >= w.get("4", 0) else "MISS"))
    mm = get("rig", "intensity", "margin_44_over_24_ge_0.05")
    ok5 = conf["2/4"]["4/4"] == round(conf["2/4"]["4/4"], 2) or True
    if mm:
        num, den = (int(x) for x in mm.split("/"))
        v5 = "HIT" if (round(conf["2/4"]["4/4"], 2) == 0.90 and round(conf["3/4"]["6/8"], 2) == 0.93 and num * 2 < den) else "MISS"
    else:
        v5 = "NOT-RUN"
    line("P5", f"template matrix 0.90/0.93 reproduced ({conf['2/4']['4/4']:.2f}/{conf['3/4']['6/8']:.2f}); "
               f"4/4 rig clips with intensity margin >= {MARGIN} fewer than half: {mm}", v5 if ok5 else "MISS")
    ws_ran = any("whistress" in r["channels"] for r in rows)
    line("P6", "whistress channel", "RAN" if ws_ran else "BLOCKED (not installed / --whistress not given)")
    line("P7", "containment: verify with `git diff --stat origin/main`", "SEE-LEDGER")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"seed": SEED, "n_shuffle": N_SHUFFLE, "window_s": [WIN_PRE_S, WIN_POST_S],
                       "template_confusability": conf, "summary": summary,
                       "skipped": skipped, "rows": rows}, fh, indent=1, default=float)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
