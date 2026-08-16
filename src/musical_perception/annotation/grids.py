"""
Beat-grid annotation format (rung 1; extended at rung 2.5) — one YAML
file per clip in `evals/grids/<case-id>.yaml`.

Two time lists with distinct contracts:

- `beats`   — the *annotation*: candidate beat times at vowel onsets. The
  tap-assist seeds it with the peakRate events; the owner corrects it
  (delete non-beats, nudge times) and flips `provisional` to false at
  rung 1.5. Until then the grid gates nothing.
- `onsets`  — frozen peakRate *evidence*: every voiced envelope-rise
  event the annotator heard. Never edited by the owner; feeds the
  onset-vs-token hallucination guard (ADR-016 clip-17) and rung-2
  debugging even after `beats` has been decimated to true beat times.

Rung 2.5 lifts the C6 limitation (convention (d′), §3) additively:
`regions` tags the three kinds of hole a flat time list cannot tell
apart, and `annotation_method` records the rung-1.5 anchored-vs-scratch
cohort per grid. Both are optional; `beats` is unchanged, so **every
format-1 grid stays valid with no edit** and the scorer sees the same
numbers it saw before.

Grids are new files under evals/ — the add-only ingestion carve-out.
Existing cases/traces/baseline stay untouched.
"""

from dataclasses import dataclass, field
from pathlib import Path

GRID_FORMAT = 2                     # written by this version
SUPPORTED_GRID_FORMATS = (1, 2)     # read: format 1 grids need no edit
_TIME_DECIMALS = 4  # 0.1 ms — far inside annotation precision

# The three holes convention (d′) names, which a flat time list cannot
# distinguish (the C6 limitation). Closed set: an unknown kind is an error.
REGION_KINDS = ("silent_beat", "free_time", "excluded_explanation")

# Regions whose gaps are explained, so QC must not read them as errors.
_SUPPRESSING_KINDS = frozenset(REGION_KINDS)

ANNOTATION_METHODS = ("anchored", "from_scratch")


@dataclass
class GridRegion:
    """A tagged time span running parallel to `beats` (never inside it)."""
    start: float
    end: float
    kind: str
    note: str = ""

    def overlaps(self, t0: float, t1: float) -> bool:
        """True when [t0, t1] shares any time with this region."""
        return t0 < self.end and t1 > self.start


@dataclass
class BeatGrid:
    """One clip's beat annotation plus its acoustic evidence."""
    clip: str                      # case id (== trace dir name)
    provisional: bool              # false only after owner verification
    beats: list[float] = field(default_factory=list)
    onsets: list[float] = field(default_factory=list)
    media: str | None = None
    media_sha256: str | None = None
    annotator: str = ""            # e.g. "peakrate-tap-assist/1"
    created_at: str = ""
    params: dict = field(default_factory=dict)
    notes: str = ""
    regions: list[GridRegion] = field(default_factory=list)
    annotation_method: str | None = None   # anchored | from_scratch | None


def _validate_times(name: str, values, clip: str) -> list[float]:
    times = [round(float(v), _TIME_DECIMALS) for v in values or []]
    if any(t < 0 for t in times):
        raise ValueError(f"grid {clip}: negative time in {name}")
    if times != sorted(times):
        raise ValueError(f"grid {clip}: {name} must be sorted ascending")
    return times


def _validate_regions(values, clip: str) -> list[GridRegion]:
    """Sorted, non-overlapping, known-kind spans. Empty stays empty."""
    regions = []
    for raw in values or []:
        if isinstance(raw, GridRegion):
            start, end, kind, note = raw.start, raw.end, raw.kind, raw.note
        else:
            start, end = raw.get("start"), raw.get("end")
            kind, note = raw.get("kind"), raw.get("note") or ""
        if kind not in REGION_KINDS:
            raise ValueError(
                f"grid {clip}: region kind {kind!r} not in {list(REGION_KINDS)}"
            )
        start = round(float(start), _TIME_DECIMALS)
        end = round(float(end), _TIME_DECIMALS)
        if start < 0:
            raise ValueError(f"grid {clip}: negative region start {start}")
        if end <= start:
            raise ValueError(
                f"grid {clip}: region {kind} end {end} must exceed start {start}"
            )
        regions.append(GridRegion(start=start, end=end, kind=kind, note=str(note)))
    ordered = sorted(regions, key=lambda r: r.start)
    if [r.start for r in regions] != [r.start for r in ordered]:
        raise ValueError(f"grid {clip}: regions must be sorted by start")
    for prev, nxt in zip(ordered, ordered[1:]):
        if nxt.start < prev.end:
            raise ValueError(
                f"grid {clip}: regions overlap at {nxt.start} "
                f"({prev.kind} ends {prev.end})"
            )
    return ordered


def _validate_method(value, clip: str) -> str | None:
    """anchored | from_scratch | None — None means 'not recorded'."""
    if value is None or value == "":
        return None
    if value not in ANNOTATION_METHODS:
        raise ValueError(
            f"grid {clip}: annotation_method {value!r} not in "
            f"{list(ANNOTATION_METHODS)}"
        )
    return str(value)


def save_grid(grid: BeatGrid, grids_dir: Path) -> Path:
    import yaml  # lazy: pyyaml lives in the [eval] extra

    grids_dir = Path(grids_dir)
    grids_dir.mkdir(parents=True, exist_ok=True)
    regions = _validate_regions(grid.regions, grid.clip)
    payload = {
        "format": GRID_FORMAT,
        "clip": grid.clip,
        "provisional": bool(grid.provisional),
        "media": grid.media,
        "media_sha256": grid.media_sha256,
        "annotator": grid.annotator,
        "annotation_method": _validate_method(grid.annotation_method, grid.clip),
        "created_at": grid.created_at,
        "params": grid.params,
        "beats": _validate_times("beats", grid.beats, grid.clip),
        "onsets": _validate_times("onsets", grid.onsets, grid.clip),
        "regions": [
            {"start": r.start, "end": r.end, "kind": r.kind, "note": r.note}
            for r in regions
        ],
        "notes": grid.notes,
    }
    path = grids_dir / f"{grid.clip}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
    return path


def load_grid(path: Path) -> BeatGrid:
    import yaml  # lazy: pyyaml lives in the [eval] extra

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or raw.get("format") not in SUPPORTED_GRID_FORMATS:
        raise ValueError(
            f"{path}: not a beat-grid file (format in {list(SUPPORTED_GRID_FORMATS)})"
        )
    clip = str(raw.get("clip") or "")
    if not clip:
        raise ValueError(f"{path}: missing clip id")
    if not isinstance(raw.get("provisional"), bool):
        raise ValueError(f"grid {clip}: provisional must be an explicit bool")
    return BeatGrid(
        clip=clip,
        provisional=raw["provisional"],
        beats=_validate_times("beats", raw.get("beats"), clip),
        onsets=_validate_times("onsets", raw.get("onsets"), clip),
        media=raw.get("media"),
        media_sha256=raw.get("media_sha256"),
        annotator=str(raw.get("annotator") or ""),
        created_at=str(raw.get("created_at") or ""),
        params=dict(raw.get("params") or {}),
        notes=str(raw.get("notes") or ""),
        regions=_validate_regions(raw.get("regions"), clip),
        annotation_method=_validate_method(raw.get("annotation_method"), clip),
    )


def load_grids(grids_dir: Path) -> dict[str, BeatGrid]:
    """{case_id: grid} for every grid file present. Missing dir → empty."""
    grids_dir = Path(grids_dir)
    if not grids_dir.is_dir():
        return {}
    grids = {}
    for path in sorted(grids_dir.glob("*.yaml")):
        grid = load_grid(path)
        if grid.clip != path.stem:
            raise ValueError(f"{path}: clip {grid.clip!r} != filename stem")
        grids[grid.clip] = grid
    return grids


# --- Audacity label-track round trip (the owner's correction surface) ----

def to_label_text(grid: BeatGrid) -> str:
    """Audacity label track: start<TAB>end<TAB>label.

    Beats are point labels (start == end) as before; regions are Audacity
    *region* labels (end > start) whose text is the region kind, so the
    owner drags them like any other region.
    """
    lines = [
        f"{t:.4f}\t{t:.4f}\tbeat-{i + 1}\n" for i, t in enumerate(grid.beats)
    ]
    lines += [
        f"{r.start:.4f}\t{r.end:.4f}\t{r.kind}\n"
        for r in _validate_regions(grid.regions, grid.clip)
    ]
    return "".join(lines)


_REGION_EPSILON_S = 0.001  # a drag this small is a mis-click, not a region


def parse_label_text(text: str) -> tuple[list[float], list[GridRegion]]:
    """Split corrected label text into beat times and tagged regions.

    A line is a region **iff** its label text is a known `REGION_KINDS`
    entry (then `end > start` is required). Any other line is a beat at
    its start time — which is exactly the old behaviour for the all-point
    `beat-N` tracks already exported. A dragged label carrying an
    unrecognized name is a loud error rather than a silent beat: that is
    the failure mode this format change could otherwise introduce into a
    verified correction pass.
    """
    beats: list[float] = []
    regions: list[GridRegion] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split("\t")
        if not parts or not parts[0]:
            continue
        start = round(float(parts[0]), _TIME_DECIMALS)
        end = round(float(parts[1]), _TIME_DECIMALS) if len(parts) > 1 else start
        label = parts[2].strip() if len(parts) > 2 else ""
        if label in REGION_KINDS:
            if end <= start:
                raise ValueError(
                    f"label line {lineno}: region {label!r} needs end > start "
                    f"(got {start} .. {end}) — drag it into a region in Audacity"
                )
            regions.append(GridRegion(start=start, end=end, kind=label))
            continue
        if end > start + _REGION_EPSILON_S:
            raise ValueError(
                f"label line {lineno}: dragged region labelled {label!r} is not "
                f"a known kind {list(REGION_KINDS)} — rename it or make it a point"
            )
        beats.append(start)
    return sorted(beats), sorted(regions, key=lambda r: r.start)


def beats_from_label_text(text: str) -> list[float]:
    """Parse corrected label text back to sorted beat times."""
    return parse_label_text(text)[0]
