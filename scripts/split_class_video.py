#!/usr/bin/env python3
"""Cut a long ballet-class video into per-exercise section clips.

Reads an edit decision list (JSON), re-encodes each span for frame accuracy,
verifies contiguity and durations against the source, and writes the CSV +
summary.md companions.  The EDL itself is authored by hand from the analysis
artifacts (faster-whisper word timestamps, ffmpeg silencedetect, librosa RMS);
this script is the mechanical half only.

    python scripts/split_class_video.py cut  <edl.json>
    python scripts/split_class_video.py check <edl.json>   # QC only, no cutting
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

VIDEO_ARGS = ["-c:v", "libx264", "-crf", "19", "-preset", "medium", "-pix_fmt", "yuv420p"]
AUDIO_ARGS = ["-c:a", "aac_at", "-b:a", "192k"]


def tc(x: float) -> str:
    return f"{int(x // 60):02d}:{x % 60:05.2f}"


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def cut(edl: dict, root: Path) -> None:
    src = root / edl["source"]
    outdir = root / edl["output_dir"]
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(edl["segments"])
    for i, seg in enumerate(edl["segments"], 1):
        dst = outdir / seg["file"]
        dur = seg["end_sec"] - seg["start_sec"]
        print(f"[{i:2d}/{n}] {seg['file']:<44s} {tc(seg['start_sec'])}->{tc(seg['end_sec'])} ({dur:6.2f}s)", flush=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{seg['start_sec']:.3f}",
             "-i", str(src), "-t", f"{dur:.3f}", *VIDEO_ARGS, *AUDIO_ARGS, str(dst)],
            check=True)


def check(edl: dict, root: Path) -> dict:
    """Contiguity + duration QC.  Returns a report dict; prints failures."""
    src = root / edl["source"]
    outdir = root / edl["output_dir"]
    segs = edl["segments"]
    src_dur = probe_duration(src)

    problems = []
    # contiguity: each clip must start exactly where the previous ended
    for a, b in zip(segs, segs[1:]):
        if abs(a["end_sec"] - b["start_sec"]) > 1e-6:
            problems.append(f"gap/overlap between {a['file']} and {b['file']}: "
                            f"{a['end_sec']} != {b['start_sec']}")

    planned = sum(s["end_sec"] - s["start_sec"] for s in segs)
    lead = segs[0]["start_sec"]
    tail = src_dur - segs[-1]["end_sec"]

    actual, missing, drift = 0.0, [], []
    for s in segs:
        p = outdir / s["file"]
        if not p.exists():
            missing.append(s["file"])
            continue
        d = probe_duration(p)
        actual += d
        want = s["end_sec"] - s["start_sec"]
        if abs(d - want) > 0.25:
            drift.append(f"{s['file']}: planned {want:.2f}s, actual {d:.2f}s")

    for label, items in (("MISSING", missing), ("DURATION DRIFT >0.25s", drift),
                         ("CONTIGUITY", problems)):
        if items:
            print(f"\n!! {label}")
            for it in items:
                print(f"   {it}")

    total_bytes = sum((outdir / s["file"]).stat().st_size
                      for s in segs if (outdir / s["file"]).exists())
    report = {
        "source_duration_sec": round(src_dur, 2),
        "clips_planned_sec": round(planned, 2),
        "clips_actual_sec": round(actual, 2),
        "dropped_lead_in_sec": round(lead, 2),
        "dropped_tail_sec": round(tail, 2),
        "clips_expected": len(segs),
        "clips_present": len(segs) - len(missing),
        "total_bytes": total_bytes,
        "ok": not (missing or drift or problems),
    }
    print("\n".join([
        "",
        f"  source duration      {src_dur:9.2f} s  ({tc(src_dur)})",
        f"  clips planned        {planned:9.2f} s",
        f"  clips actual         {actual:9.2f} s",
        f"  dropped lead-in      {lead:9.2f} s",
        f"  dropped tail         {tail:9.2f} s",
        f"  planned + dropped    {planned + lead + tail:9.2f} s   "
        f"(= source? {abs(planned + lead + tail - src_dur) < 0.05})",
        f"  clips present        {len(segs) - len(missing)}/{len(segs)}",
        f"  total size           {total_bytes / 1e6:.0f} MB",
        f"  QC                   {'PASS' if report['ok'] else 'FAIL'}",
    ]))
    return report


def write_csv(edl: dict, root: Path) -> None:
    outdir = root / edl["output_dir"]
    cols = ["file", "exercise_number", "exercise", "type", "side", "start_sec", "end_sec",
            "start_tc", "end_tc", "duration_sec", "start_confidence", "end_confidence",
            "low_confidence", "boundary_note", "transcript_snippet"]
    with (outdir / "edit_decision_list.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s in edl["segments"]:
            w.writerow({c: s.get(c, "") for c in cols})


if __name__ == "__main__":
    mode, edl_path = sys.argv[1], Path(sys.argv[2])
    edl = json.loads(edl_path.read_text())
    # "source" and "output_dir" are relative to edl["root"], itself relative to
    # the EDL file.  The EDL conventionally lives inside the output directory,
    # so root defaults to "..".
    root = (edl_path.parent / edl.get("root", "..")).resolve()
    if mode == "cut":
        cut(edl, root)
    check(edl, root)
    write_csv(edl, root)
