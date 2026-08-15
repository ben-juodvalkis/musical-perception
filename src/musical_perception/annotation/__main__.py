"""
Tap-assist annotator CLI (rung 1):

    python -m musical_perception.annotation generate [--only ID ...] [--force]
    python -m musical_perception.annotation to-labels ID
    python -m musical_perception.annotation from-labels ID LABELS.txt [--verified]
    python -m musical_perception.annotation qc [ID ...]
    python -m musical_perception.annotation set-method ID {anchored,from_scratch}

`generate` pre-annotates a provisional beat grid per eval case via
peakRate. `to-labels`/`from-labels` round-trip through an Audacity label
track so the owner corrects beats by ear (rung 1.5); `--verified` is the
owner's act of flipping `provisional` off — never used by agent sessions.
`qc` runs the three ratified convention §4 checks (rung 2.5); `set-method`
records the anchored-vs-from-scratch provenance and, like `--verified`, is
an owner act — the cohort assignment is the owner's to state.

Needs the [prosody,eval] extras (librosa/scipy/parselmouth + pyyaml).
"""

import argparse
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from musical_perception.annotation.grids import (
    ANNOTATION_METHODS,
    BeatGrid,
    load_grid,
    load_grids,
    parse_label_text,
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
    grid.beats, grid.regions = parse_label_text(Path(args.labels).read_text())
    if args.verified:
        grid.provisional = False
    save_grid(grid, Path(args.grids))
    state = "verified" if not grid.provisional else "still provisional"
    tagged = f", {len(grid.regions)} tagged regions" if grid.regions else ""
    print(f"updated {path}: {len(grid.beats)} beats{tagged}, {state}")
    return 0


def _cmd_set_method(args) -> int:
    """Owner act: record how this grid was annotated (convention §2.4)."""
    path = _grid_path(args)
    grid = load_grid(path)
    grid.annotation_method = args.method
    save_grid(grid, Path(args.grids))
    print(f"updated {path}: annotation_method = {args.method}")
    return 0


def _cmd_qc(args) -> int:
    """The three ratified convention §4 checks over every grid present."""
    from musical_perception.annotation.qc import run_qc
    from musical_perception.evals.cases import load_cases

    grids = load_grids(Path(args.grids))
    if not grids:
        print(f"no grids under {args.grids}", file=sys.stderr)
        return 2
    labels = {
        c.id: c.expect.get("marking_bpm") for c in load_cases(Path(args.cases))
    }
    wanted = set(args.case_ids or grids)
    unknown = wanted - set(grids)
    if unknown:
        print(f"no grid for: {sorted(unknown)}", file=sys.stderr)
        return 2

    header = (
        f"{'clip':38s} {'state':5s} {'beats':>5s} {'phr':>4s} "
        f"{'BPM':>7s} {'inphr':>7s} {'label':>6s} {'Δ%':>7s} "
        f"{'minIOI':>7s} {'maxCV':>7s}  flags"
    )
    print(header)
    print("-" * len(header))
    flagged, findings = [], []
    for cid in sorted(wanted):
        r = run_qc(grids[cid], labels.get(cid))
        if r.findings:
            flagged.append(cid)
            findings.extend(r.findings)
        print(
            f"{cid:38s} {'prov' if r.provisional else 'ver':5s} "
            f"{r.n_beats:5d} {r.n_phrases:4d} "
            f"{_num(r.bpm_whole, 2, 7)} {_num(r.bpm_within_phrase, 2, 7)} "
            f"{_num(r.marking_bpm, 0, 6)} {_num(r.bpm_delta_pct, 2, 7)} "
            f"{_num(r.min_ioi_ratio, 3, 7)} {_num(r.max_phrase_cv, 3, 7)}  "
            f"{','.join(sorted({f.check for f in r.findings})) or '-'}"
        )
    print(f"\n{len(flagged)} of {len(wanted)} grids flagged")
    for f in findings:
        print(f"  {f.clip}: {f}")
    return 0


def _num(value, decimals: int, width: int) -> str:
    return f"{'':>{width}}" if value is None else f"{value:>{width}.{decimals}f}"


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

    p_qc = sub.add_parser("qc", help="run the convention §4 checks on grids")
    p_qc.add_argument("case_ids", nargs="*", help="default: every grid present")
    p_qc.set_defaults(fn=_cmd_qc)

    p_method = sub.add_parser(
        "set-method", help="owner act: record anchored vs from_scratch")
    p_method.add_argument("case_id")
    p_method.add_argument("method", choices=ANNOTATION_METHODS)
    p_method.set_defaults(fn=_cmd_set_method)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
