"""W17 — owner annotation round-trip for the granular timeline study.

The owner marks, by ear and eye and BEFORE seeing any machine output, where a
demonstration is actually in tempo and when he would commit as accompanist.
This script emits an Audacity label template for that pass and reads the
finished file back.

    python scripts/w17-owner-annotation.py emit  --clip barre6-frappe-demo
    python scripts/w17-owner-annotation.py read  --clip barre6-frappe-demo

Audacity label format is tab-separated `start<TAB>end<TAB>text`; a point label
has start == end. Vocabulary (only the first two are required):

  fullout           REGION - teacher dancing it full-out, genuinely in tempo
  commit            POINT  - the moment you would commit to a tempo

LIVE-TAP ALTERNATIVE (easier than selecting ranges while listening): drop POINT
labels during playback with Cmd+M and name them `<label>-start` / `<label>-end`.
Matching pairs are folded into regions on read, in time order, so
`fullout-start` at 3.0s + `fullout-end` at 29.2s becomes one fullout region.
An unmatched start or end is reported, never silently dropped.
  tempo=<bpm>       POINT or REGION - the tempo you would commit to
  marking           REGION - sketching the combination, not in tempo
  talking           REGION - explaining; no movement tempo to read
  cue=voice|feet|arm|breath|other   REGION - what you are reading tempo from

Unknown labels are preserved rather than rejected: the vocabulary is a
starting point and the owner may need words it does not have.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWN = {"fullout", "commit", "marking", "talking"}


def labels_path(clip: str) -> Path:
    return ROOT / "docs" / "research" / "w17" / f"{clip}.owner-windows.txt"


def emit(clip: str, force: bool) -> None:
    p = labels_path(clip)
    if p.exists() and not force:
        raise SystemExit(f"{p} exists; pass --force to overwrite")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "0.000000\t0.000000\tDELETE-ME see docs/research/w17/README.md for the vocabulary\n"
    )
    readme = p.parent / "README.md"
    if not readme.exists() or force:
        readme.write_text(__doc__.strip() + "\n")
    print(f"template -> {p}")
    print(f"vocabulary -> {readme}")
    print("\nImport into Audacity with File > Import > Labels, mark the clip, then")
    print("File > Export > Export Labels back to the same path.")


def read(clip: str) -> dict:
    p = labels_path(clip)
    if not p.is_file():
        raise SystemExit(f"no annotation at {p} - run `emit` first")
    regions, points, unknown = [], [], []
    for n, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise SystemExit(f"{p}:{n}: expected 3 tab-separated fields, got {len(parts)}")
        start, end, text = float(parts[0]), float(parts[1]), parts[2].strip()
        if text.startswith("DELETE-ME"):
            continue
        rec = dict(start=start, end=end, text=text)
        base = re.split(r"[=:]", text, maxsplit=1)[0].strip().lower()
        for suffix in ("-start", "-end"):          # live-tap pair form is valid
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if base not in KNOWN and not text.lower().startswith(("tempo", "cue")):
            unknown.append(text)
        (points if abs(end - start) < 1e-6 else regions).append(rec)

    # fold live-tapped `<label>-start` / `<label>-end` point pairs into regions
    paired, leftover, opens = [], [], {}
    for rec in sorted(points, key=lambda r: r["start"]):
        low = rec["text"].strip().lower()
        for suffix, is_open in (("-start", True), ("-end", False)):
            if low.endswith(suffix):
                base = low[: -len(suffix)]
                if is_open:
                    opens.setdefault(base, []).append(rec)
                elif opens.get(base):
                    o = opens[base].pop(0)
                    paired.append(dict(start=o["start"], end=rec["start"],
                                       text=base, from_pair=True))
                else:
                    leftover.append(rec)
                break
        else:
            continue
    unmatched = [r for v in opens.values() for r in v] + leftover
    regions.extend(paired)
    regions.sort(key=lambda r: r["start"])
    points = [p_ for p_ in points
              if not p_["text"].strip().lower().endswith(("-start", "-end"))]

    out = dict(clip=clip, regions=regions, points=points,
               unmatched_pair_labels=[r["text"] for r in unmatched],
               unknown_labels=sorted(set(unknown)))
    dest = p.with_suffix(".json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"{len(regions)} regions, {len(points)} points -> {dest}")
    for kind in ("fullout", "marking", "talking"):
        got = [r for r in regions if r["text"].lower().startswith(kind)]
        if got:
            total = sum(r["end"] - r["start"] for r in got)
            print(f"  {kind:<9} {len(got)} region(s), {total:.1f}s total")
    if any(p_["text"].lower().startswith("commit") for p_ in points):
        t = min(p_["start"] for p_ in points if p_["text"].lower().startswith("commit"))
        print(f"  commit    at {t:.2f}s")
    else:
        print("  commit    NOT MARKED - the study needs one")
    if paired:
        print(f"  paired    {len(paired)} live-tapped start/end pair(s) folded into regions")
    if unmatched:
        print(f"  WARNING   {len(unmatched)} unmatched start/end label(s): "
              f"{[r['text'] for r in unmatched]}")
    if unknown:
        print(f"  note: {len(unknown)} label(s) outside the vocabulary, kept as-is:")
        for u in unknown:
            print(f"           {u!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("emit", "read"))
    ap.add_argument("--clip", default="barre6-frappe-demo")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    emit(a.clip, a.force) if a.action == "emit" else read(a.clip)


if __name__ == "__main__":
    main()
