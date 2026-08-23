#!/usr/bin/env python3
"""
W7 diagnostic — does movement carry a pulse, and does it agree with the voice?

Read-only over the frozen Ballet Barre 1 traces. Needs no media, no models
and no API key: pose comes from the committed `pose.npz`, and the voice
channel is replayed from the committed whisper/gemini traces through the
ordinary pipeline, so the tempo compared against is the real one the
pipeline would have produced and not a reimplementation of it.

No ground truth exists for these clips (their case files are BLOCKED on
W1.5), so nothing here is an accuracy claim. The two answerable questions
are whether the gesture channel produces a periodic signal at all, and
whether it independently lands where the voice channel lands.

Usage:  python scripts/w7-pose-gesture-report.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from musical_perception.analyze import analyze  # noqa: E402
from musical_perception.evals.traces import replay_bundle  # noqa: E402
from musical_perception.precision.gesture import analyze_gesture  # noqa: E402
from musical_perception.types import LandmarkTimeSeries  # noqa: E402

TRACES = Path("evals/traces")
# Metric-level family: the ratios a beat tracker is allowed to be "right at
# the wrong level" by (ADR-014's alternates, duple and triple).
FAMILY = [1 / 3, 1 / 2, 2 / 3, 1.0, 3 / 2, 2.0, 3.0]
TOLERANCE = 0.08
VOICELESS_MAX_WORDS = 3


def load_pose(trace_dir: Path) -> LandmarkTimeSeries:
    z = np.load(trace_dir / "pose.npz")
    return LandmarkTimeSeries(
        timestamps=z["timestamps"],
        landmarks=z["landmarks"],
        fps=float(z["fps"]),
        detection_rate=float(z["detection_rate"]),
    )


def voice_bpm(trace_dir: Path) -> float | None:
    """Voice-channel tempo, replayed through the real pipeline."""
    try:
        bundle, meta = replay_bundle(trace_dir)
        params = analyze(meta.get("media", "replay.mp4"), bundle=bundle, use_pose=False)
    except Exception as exc:  # noqa: BLE001 - diagnostic, report and continue
        print(f"    (voice replay failed: {type(exc).__name__}: {exc})", file=sys.stderr)
        return None
    tempo = getattr(params, "tempo", None)
    bpm = getattr(tempo, "bpm", None)
    return float(bpm) if bpm else None


def family_ratio(gesture: float, voice: float) -> float | None:
    """The metric-level ratio linking the two BPMs, or None if unrelated."""
    for ratio in FAMILY:
        if abs(gesture / (voice * ratio) - 1.0) <= TOLERANCE:
            return ratio
    return None


def main() -> int:
    dirs = sorted(TRACES.glob("barre1-*"))
    if not dirs:
        print("no barre1 traces found", file=sys.stderr)
        return 1

    rows = []
    for d in dirs:
        words = json.loads((d / "whisper.json").read_text()).get("words", [])
        lts = load_pose(d)
        g = analyze_gesture(lts)
        v = voice_bpm(d)
        gb = g.dominant_bpm
        rows.append({
            "clip": d.name[len("barre1-"):],
            "voiceless": len(words) <= VOICELESS_MAX_WORDS,
            "words": len(words),
            "det": lts.detection_rate,
            "dur": g.duration,
            "events": len(g.event_times),
            "rate": g.event_rate,
            "windows": len(g.windows),
            "sig": len(g.significant_windows),
            "coverage": g.coverage,
            "period": g.dominant_period,
            "gesture_bpm": gb,
            "voice_bpm": v,
            "ratio": family_ratio(gb, v) if (gb and v) else None,
        })

    hdr = (f"{'clip':36s} {'V':>1s} {'det':>5s} {'dur':>6s} {'evt':>4s} {'/s':>5s} "
           f"{'win':>4s} {'sig':>4s} {'cov':>5s} {'gBPM':>6s} {'vBPM':>6s} {'ratio':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        per = f"{r['gesture_bpm']:6.1f}" if r["gesture_bpm"] else "     -"
        vb = f"{r['voice_bpm']:6.1f}" if r["voice_bpm"] else "     -"
        ra = f"{r['ratio']:6.2f}" if r["ratio"] else "     -"
        print(f"{r['clip']:36s} {'*' if r['voiceless'] else ' '} {r['det']:5.2f} "
              f"{r['dur']:6.1f} {r['events']:4d} {r['rate']:5.2f} {r['windows']:4d} "
              f"{r['sig']:4d} {r['coverage']:5.2f} {per} {vb} {ra}")

    print("\n(V * = voice-less take, <= 3 transcribed words)")

    # ---- prediction scoring -------------------------------------------------
    n = len(rows)
    rates = [r["rate"] for r in rows]
    g1 = all(r["events"] > 0 for r in rows) and float(np.median(rates)) >= 1.0
    sig_clips = [r for r in rows if r["sig"] > 0]
    periods = [r["period"] for r in rows if r["period"]]
    paired = [r for r in rows if r["ratio"] is not None or (r["gesture_bpm"] and r["voice_bpm"])]
    agreed = [r for r in paired if r["ratio"] is not None]
    voiceless = [r for r in rows if r["voiceless"]]
    voiced = [r for r in rows if not r["voiceless"]]

    print("\n=== prediction scorecard ===")
    print(f"G1 extraction    : {sum(1 for r in rows if r['events'] > 0)}/{n} clips with events, "
          f"median rate {np.median(rates):.2f}/s  -> predicted 22/22 & >=1.0/s : "
          f"{'HIT' if g1 else 'MISS'}")
    print(f"G2 periodicity   : {len(sig_clips)}/{n} clips with >=1 significant window "
          f"-> predicted >=12/22 : {'HIT' if len(sig_clips) >= 12 else 'MISS'}")
    med_p = float(np.median(periods)) if periods else float('nan')
    print(f"G3 level         : median dominant period {med_p:.2f}s "
          f"({60/med_p:.1f} BPM) -> predicted >1.2s : "
          f"{'HIT' if periods and med_p > 1.2 else 'MISS'}")
    frac = len(agreed) / len(paired) if paired else float('nan')
    print(f"G4 cross-channel : {len(agreed)}/{len(paired)} clips agree within a metric-level "
          f"family ({frac:.0%}) -> predicted <50% : "
          f"{'HIT' if paired and frac < 0.5 else 'MISS'}")
    if voiceless and voiced:
        vl_rate, v_rate = np.median([r["rate"] for r in voiceless]), np.median([r["rate"] for r in voiced])
        vl_cov, v_cov = np.median([r["coverage"] for r in voiceless]), np.median([r["coverage"] for r in voiced])
        print(f"G5 voice-indep   : voice-less median rate {vl_rate:.2f}/s cov {vl_cov:.2f} "
              f"vs voiced {v_rate:.2f}/s cov {v_cov:.2f}")

    Path("docs/research").mkdir(parents=True, exist_ok=True)
    Path("docs/research/w7-gesture-results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("\nwrote docs/research/w7-gesture-results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
