"""W17 — the time-resolved tempo timeline for a single demonstration.

Every technique the project has, asked the same question every STEP seconds:
"given what you have heard so far, what is the tempo?" — and separately,
"what is the tempo right now?" Produces a per-technique curve over the clip
so the machine's commit moment can be compared against the owner's.

Two window modes per technique:
  causal    — everything from clip start to t   (answers "when does it commit")
  trailing  — only the last --trailing seconds  (answers "what is it now",
              which is what catches a tempo that drifts inside one demo)

The owner's hand-tapped beat grid is carried as a REFERENCE curve, never as
an input to any estimator.

    python scripts/w17-tempo-timeline.py --clip barre6-frappe-demo

Writes <out>/<clip>-timeline.{json,csv} and a self-contained SVG chart.
Deliberately prints no tempo numbers to stdout: the owner annotates the clip
BEFORE seeing any machine output, and a summary line here would spoil that.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eb1_import import all_pairs_bpm  # noqa: E402  (thin shim, see below)

SR = 22050


# ---------------------------------------------------------------- event sources

def load_pulse_events(clip: str) -> np.ndarray:
    p = ROOT / "evals" / "traces" / clip / "pulse.json"
    return np.asarray(json.loads(p.read_text())["events"], dtype=float)


def load_word_starts(clip: str) -> np.ndarray:
    """EVERY Whisper token, classified or not - the unfiltered stream."""
    p = ROOT / "evals" / "traces" / clip / "whisper.json"
    d = json.loads(p.read_text())
    words = d.get("words") or d.get("word_segments") or []
    out = [w["start"] for w in words if isinstance(w, dict) and w.get("start") is not None]
    return np.asarray(sorted(out), dtype=float)


def load_beat_markers(clip: str) -> np.ndarray:
    """Only the tokens Gemini classified as BEATS, paired to Whisper
    timestamps by transcript index - the same merge the shipping path uses
    (`analyze._pair_markers_by_index`, ADR-010). This is the honest word
    condition: feeding an estimator all 127 tokens when 53 are beats is a
    strawman, and the first W17 pass did exactly that.
    """
    tp = ROOT / "evals" / "traces" / clip
    words = json.loads((tp / "whisper.json").read_text())
    words = words.get("words") or words.get("word_segments") or []
    gem = json.loads((tp / "gemini.json").read_text())["raw_response"]["words"]
    out = []
    for gw in gem:
        if gw.get("marker_type") != "beat":
            continue
        i = gw.get("index")
        if i is None or not 0 <= i < len(words):
            continue                      # hallucinated index - nothing to anchor
        s = words[i].get("start")
        if s is not None:
            out.append(float(s))
    return np.asarray(sorted(out), dtype=float)


def load_grid_beats(clip: str) -> np.ndarray:
    import yaml
    p = ROOT / "evals" / "grids" / f"{clip}.yaml"
    if not p.is_file():
        return np.asarray([], dtype=float)
    return np.asarray(yaml.safe_load(p.read_text()).get("beats") or [], dtype=float)


def load_audio(clip: str, media: str) -> tuple[np.ndarray, int]:
    import librosa
    src = ROOT / media
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src),
             "-ac", "1", "-ar", str(SR), wav],
            check=True,
        )
        y, sr = librosa.load(wav, sr=SR, mono=True)
    return y, sr


# ---------------------------------------------------------------- estimators

def est_median_ioi(ev: np.ndarray) -> float | None:
    """The shipping path's historic estimator: median gap between neighbours."""
    if ev.size < 4:
        return None
    d = np.diff(ev)
    d = d[(d > 0.1) & (d < 2.5)]
    return float(60.0 / np.median(d)) if d.size >= 3 else None


def est_librosa_dp(y: np.ndarray, sr: int) -> float | None:
    import librosa
    if y.size < sr:
        return None
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, units="time")
        t = float(np.atleast_1d(tempo)[0])
        return t if t > 0 else None
    except Exception:
        return None


def est_librosa_plp(y: np.ndarray, sr: int) -> float | None:
    """Predominant local pulse — designed for tempo that moves."""
    import librosa
    if y.size < sr:
        return None
    try:
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        pulse = librosa.beat.plp(onset_envelope=oenv, sr=sr)
        beats = librosa.util.peak_pick(pulse, pre_max=5, post_max=5, pre_avg=5,
                                       post_avg=5, delta=0.05, wait=5)
        if beats.size < 4:
            return None
        times = librosa.frames_to_time(beats, sr=sr)
        d = np.diff(times)
        d = d[(d > 0.1) & (d < 2.5)]
        return float(60.0 / np.median(d)) if d.size >= 3 else None
    except Exception:
        return None


def est_librosa_acf(y: np.ndarray, sr: int) -> float | None:
    """Tempogram / autocorrelation global estimate."""
    import librosa
    if y.size < sr:
        return None
    try:
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        t = librosa.feature.tempo(onset_envelope=oenv, sr=sr, aggregate=np.median)
        v = float(np.atleast_1d(t)[0])
        return v if v > 0 else None
    except Exception:
        return None


def grid_local_bpm(beats: np.ndarray, lo: float, hi: float) -> float | None:
    """REFERENCE ONLY — local tempo from the owner's own taps in [lo, hi]."""
    b = beats[(beats >= lo) & (beats <= hi)]
    if b.size < 3:
        return None
    d = np.diff(b)
    d = d[(d > 0.05) & (d < 3.0)]
    return float(60.0 / np.median(d)) if d.size >= 2 else None


# ---------------------------------------------------------------- the sweep

EVENT_TECHNIQUES = {
    "pulse_allpairs": ("peakRate events, all-pairs period (EB-1/PP-1)", all_pairs_bpm),
    "pulse_median":   ("peakRate events, median gap (historic shipping path)", est_median_ioi),
}
WORD_TECHNIQUES = {
    "words_allpairs": ("ALL Whisper tokens, all-pairs (unclassified - strawman)", all_pairs_bpm),
    "words_median":   ("ALL Whisper tokens, median gap (unclassified - strawman)", est_median_ioi),
}
MARKER_TECHNIQUES = {
    "markers_allpairs": ("Gemini beat markers, all-pairs (the shipping condition)", all_pairs_bpm),
    "markers_median":   ("Gemini beat markers, median gap", est_median_ioi),
}
AUDIO_TECHNIQUES = {
    "librosa_dp":  ("librosa beat_track (dynamic programming)", est_librosa_dp),
    "librosa_plp": ("librosa predominant local pulse", est_librosa_plp),
    "librosa_acf": ("librosa tempogram autocorrelation", est_librosa_acf),
}


def run(clip: str, step: float, trailing: float, out_dir: Path,
        skip_audio: bool) -> dict:
    import yaml
    case = yaml.safe_load((ROOT / "evals" / "cases" / f"{clip}.yaml").read_text())
    media = case["input"]["media"]
    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(ROOT / media)],
        capture_output=True, text=True, check=True).stdout.strip())

    pulse = load_pulse_events(clip)
    words = load_word_starts(clip)
    markers = load_beat_markers(clip)
    grid = load_grid_beats(clip)
    y = sr = None
    if not skip_audio:
        y, sr = load_audio(clip, media)

    grid_times = np.arange(step, duration + 1e-9, step)
    rows: list[dict] = []

    for t in grid_times:
        lo_tr = max(0.0, t - trailing)
        for mode, lo in (("causal", 0.0), ("trailing", lo_tr)):
            for name, (_, fn) in EVENT_TECHNIQUES.items():
                ev = pulse[(pulse >= lo) & (pulse <= t)]
                rows.append(dict(t=round(float(t), 3), mode=mode, technique=name,
                                 bpm=fn(ev), n=int(ev.size)))
            for name, (_, fn) in WORD_TECHNIQUES.items():
                ev = words[(words >= lo) & (words <= t)]
                rows.append(dict(t=round(float(t), 3), mode=mode, technique=name,
                                 bpm=fn(ev), n=int(ev.size)))
            for name, (_, fn) in MARKER_TECHNIQUES.items():
                ev = markers[(markers >= lo) & (markers <= t)]
                rows.append(dict(t=round(float(t), 3), mode=mode, technique=name,
                                 bpm=fn(ev), n=int(ev.size)))
            if y is not None:
                seg = y[int(lo * sr):int(t * sr)]
                for name, (_, fn) in AUDIO_TECHNIQUES.items():
                    rows.append(dict(t=round(float(t), 3), mode=mode, technique=name,
                                     bpm=fn(seg, sr), n=int(seg.size)))
            rows.append(dict(t=round(float(t), 3), mode=mode, technique="grid_reference",
                             bpm=grid_local_bpm(grid, lo if mode == "trailing" else max(0.0, t - trailing), t),
                             n=int(((grid >= lo) & (grid <= t)).sum())))

    meta = dict(clip=clip, media=media, duration_s=duration, step_s=step,
                trailing_s=trailing, n_pulse_events=int(pulse.size),
                n_word_starts=int(words.size), n_beat_markers=int(markers.size), n_grid_beats=int(grid.size),
                audio_techniques=not skip_audio,
                techniques={k: v[0] for d in (EVENT_TECHNIQUES, WORD_TECHNIQUES,
                                              MARKER_TECHNIQUES, AUDIO_TECHNIQUES) for k, v in d.items()})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{clip}-timeline.json").write_text(
        json.dumps(dict(meta=meta, rows=rows), indent=1))
    with (out_dir / f"{clip}-timeline.csv").open("w") as fh:
        fh.write("t,mode,technique,bpm,n\n")
        for r in rows:
            fh.write(f"{r['t']},{r['mode']},{r['technique']},"
                     f"{'' if r['bpm'] is None else round(r['bpm'], 2)},{r['n']}\n")
    return dict(meta=meta, rows=rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="barre6-frappe-demo")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--trailing", type=float, default=8.0)
    ap.add_argument("--out", default="docs/research/w17")
    ap.add_argument("--skip-audio", action="store_true",
                    help="event-stream techniques only (fast; no ffmpeg/librosa)")
    a = ap.parse_args()
    res = run(a.clip, a.step, a.trailing, ROOT / a.out, a.skip_audio)
    m = res["meta"]
    # No tempo values printed on purpose - see module docstring.
    print(f"clip {m['clip']}  {m['duration_s']:.1f}s  step {m['step_s']}s  "
          f"trailing {m['trailing_s']}s")
    print(f"  pulse events {m['n_pulse_events']}  word starts {m['n_word_starts']}  "
          f"beat markers {m['n_beat_markers']}  owner taps {m['n_grid_beats']}")
    print(f"  {len(res['rows'])} rows -> {a.out}/{m['clip']}-timeline.{{json,csv}}")


if __name__ == "__main__":
    main()
