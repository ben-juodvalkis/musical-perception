"""W4 (Ballet Barre 1 DEV ingestion) — read-only report over the frozen traces.

Replays every `evals/traces/barre1-*` trace offline (no models, no API key,
no media) and prints what the pipeline currently produces on the new
material, plus the two pre-registered checks:

  I2  counting-token fraction per clip (median across the batch)
  I5  left/right execution BPM agreement within 8% per exercise

Nothing here scores against truth: these clips have no case files and no
verified grids, so every number below is *observation*, never a gate.
Run from the repo root:  .venv/bin/python scripts/barre1-ingestion-report.py
"""

import json
import statistics
import warnings
from pathlib import Path

TRACES = Path("evals/traces")


def replay(trace_dir: Path):
    from musical_perception.analyze import analyze
    from musical_perception.evals.traces import replay_bundle

    bundle, meta = replay_bundle(trace_dir)
    use_pose = bool(meta.get("analyze_flags", {}).get("use_pose"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analyze(
            meta.get("media") or ("replay.mov" if use_pose else "replay.wav"),
            use_pose=use_pose,
            bundle=bundle,
        )
    return result, meta, [str(w.message) for w in caught]


def token_fraction(trace_dir: Path) -> tuple[float, int]:
    raw = json.loads(trace_dir.joinpath("gemini.json").read_text())["raw_response"]
    words = (raw if isinstance(raw, dict) else json.loads(raw))["words"]
    counted = sum(1 for w in words if w.get("marker_type") not in (None, "none"))
    return (counted / len(words) if words else 0.0), len(words)


def main() -> int:
    dirs = sorted(d for d in TRACES.glob("barre1-*")
                  if d.is_dir() and (d / "meta.json").is_file())
    if not dirs:
        print("no barre1-* traces found")
        return 1

    rows, fractions, guard_hits = [], [], []
    for d in dirs:
        result, meta, warns = replay(d)
        frac, n_words = token_fraction(d)
        fractions.append(frac)
        if warns:
            guard_hits.append((d.name, warns))
        nt = result.normalized_tempo
        rows.append({
            "id": d.name,
            "bpm": round(nt.bpm, 1) if nt else None,
            "meter": f"{result.meter.beats_per_measure}/{result.meter.beat_unit}"
                     if result.meter else None,
            "subdiv": result.subdivision.subdivision_type if result.subdivision else None,
            "exercise": (result.exercise.primary_exercise if result.exercise else None),
            "counts": result.structure.counts if result.structure else None,
            "words": n_words,
            "count_frac": round(frac, 3),
            "pose": d.joinpath("pose.npz").is_file(),
        })

    hdr = f"| {'clip':38} | {'BPM':>6} | meter | subdiv  | {'exercise':22} | cnts | words | count_frac | pose |"
    print(hdr)
    print("|" + "-" * (len(hdr) - 2) + "|")
    for r in rows:
        print(f"| {r['id']:38} | {str(r['bpm']):>6} | {str(r['meter']):5} | "
              f"{str(r['subdiv']):7} | {str(r['exercise'])[:22]:22} | "
              f"{str(r['counts']):>4} | {r['words']:>5} | {r['count_frac']:>10.3f} | "
              f"{'yes' if r['pose'] else 'NO':4} |")

    print(f"\nI2  median counting-token fraction: {statistics.median(fractions):.3f} "
          f"(n={len(fractions)}; predicted < 0.5)")

    print("\nI5  left/right execution BPM agreement (8% tolerance):")
    by_ex = {}
    for r in rows:
        name = r["id"]
        if "-execution-" not in name:
            continue
        ex = name.split("-execution-")[0]
        by_ex.setdefault(ex, {})[name.rsplit("-", 1)[-1]] = r["bpm"]
    agree = pairs = 0
    for ex, sides in sorted(by_ex.items()):
        left, right = sides.get("left"), sides.get("right")
        if left is None or right is None:
            print(f"  {ex:34} incomplete pair {sides}")
            continue
        pairs += 1
        rel = abs(left - right) / max(left, right)
        ok = rel <= 0.08
        agree += ok
        print(f"  {ex:34} L {left:6.1f}  R {right:6.1f}  "
              f"rel {rel:5.1%}  {'agree' if ok else 'DISAGREE'}")
    print(f"  -> {agree}/{pairs} pairs agree (predicted at least 5 of 7)")

    print("\nonset-vs-token sanity guard / replay warnings:")
    if not guard_hits:
        print("  none fired")
    for name, warns in guard_hits:
        for w in warns:
            print(f"  {name}: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
