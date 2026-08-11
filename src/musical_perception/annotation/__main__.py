"""
Tap-assist annotator CLI (rung 1):

    python -m musical_perception.annotation generate [--only ID ...] [--force]
    python -m musical_perception.annotation to-labels ID
    python -m musical_perception.annotation from-labels ID LABELS.txt [--verified]

`generate` pre-annotates a provisional beat grid per eval case via
peakRate. `to-labels`/`from-labels` round-trip through an Audacity label
track so the owner corrects beats by ear (rung 1.5); `--verified` is the
owner's act of flipping `provisional` off — never used by agent sessions.

Needs the [prosody,eval] extras (librosa/scipy/parselmouth + pyyaml).
"""

import argparse
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from musical_perception.annotation.grids import (
    BeatGrid,
    beats_from_label_text,
    load_grid,
    save_grid,
    to_label_text,
)
from musical_perception.annotation.peakrate import PeakRateParams, peak_rate_events

ANNOTATOR = "peakrate-tap-assist/1"
_VIDEO_SUFFIXES = {".mov", ".m4v", ".mp4", ".avi", ".mkv"}


def _load_audio(media: Path, sr: int):
    """Mono float audio at the analysis rate; video goes through ffmpeg."""
    import librosa

    if media.suffix.lower() in _VIDEO_SUFFIXES:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(media),
                 "-ac", "1", "-ar", str(sr), tmp.name],
                check=True,
            )
            y, _ = librosa.load(tmp.name, sr=sr, mono=True)
            return y
    y, _ = librosa.load(str(media), sr=sr, mono=True)
    return y


def _cmd_generate(args) -> int:
    from musical_perception.evals.cases import load_cases
    from musical_perception.evals.traces import _sha256_file

    params = PeakRateParams()
    grids_dir = Path(args.grids)
    cases = load_cases(Path(args.cases))
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {c.id for c in cases}
        if unknown:
            print(f"unknown case ids: {sorted(unknown)}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c.id in wanted]

    written, skipped, missing = [], [], []
    for case in cases:
        out_path = grids_dir / f"{case.id}.yaml"
        if out_path.is_file() and not args.force:
            skipped.append(case.id)
            print(f"  exists   {case.id} (--force to regenerate)")
            continue
        media = Path(case.media) if case.media else None
        if media is None or not media.is_file():
            missing.append((case.id, str(case.media)))
            print(f"  MISSING  {case.id}: media not on this machine ({case.media})")
            continue
        y = _load_audio(media, params.sr)
        events = [float(t) for t in peak_rate_events(y, params.sr, params)]
        grid = BeatGrid(
            clip=case.id,
            provisional=True,   # only the owner flips this (rung 1.5)
            beats=list(events),
            onsets=list(events),
            media=str(case.media),
            media_sha256=_sha256_file(media),
            annotator=ANNOTATOR,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            params=params.as_dict(),
            notes="peakRate suggestions — unverified; may include "
                  "subdivisions and spurious events",
        )
        path = save_grid(grid, grids_dir)
        written.append(case.id)
        print(f"  wrote    {path} ({len(events)} events)")

    print(f"\n{len(written)} written, {len(skipped)} skipped, "
          f"{len(missing)} missing media / {len(cases)} cases")
    if missing:
        print("missing media (owner: stage these, then re-run generate):")
        for cid, m in missing:
            print(f"  {cid}: {m}")
    return 0


def _grid_path(args) -> Path:
    return Path(args.grids) / f"{args.case_id}.yaml"


def _cmd_to_labels(args) -> int:
    grid = load_grid(_grid_path(args))
    out = Path(args.grids) / f"{args.case_id}.labels.txt"
    out.write_text(to_label_text(grid))
    print(f"wrote {out} ({len(grid.beats)} beats) — correct in Audacity, "
          f"then `from-labels {args.case_id} {out}`")
    return 0


def _cmd_from_labels(args) -> int:
    path = _grid_path(args)
    grid = load_grid(path)
    grid.beats = beats_from_label_text(Path(args.labels).read_text())
    if args.verified:
        grid.provisional = False
    save_grid(grid, Path(args.grids))
    state = "verified" if not grid.provisional else "still provisional"
    print(f"updated {path}: {len(grid.beats)} beats, {state}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m musical_perception.annotation")
    parser.add_argument("--cases", default="evals/cases")
    parser.add_argument("--grids", default="evals/grids")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="pre-annotate grids via peakRate")
    p_gen.add_argument("--only", nargs="*", default=None, help="case ids")
    p_gen.add_argument("--force", action="store_true", help="overwrite existing")
    p_gen.set_defaults(fn=_cmd_generate)

    p_to = sub.add_parser("to-labels", help="export an Audacity label track")
    p_to.add_argument("case_id")
    p_to.set_defaults(fn=_cmd_to_labels)

    p_from = sub.add_parser("from-labels", help="import corrected labels")
    p_from.add_argument("case_id")
    p_from.add_argument("labels")
    p_from.add_argument("--verified", action="store_true",
                        help="owner act: flip provisional off")
    p_from.set_defaults(fn=_cmd_from_labels)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
