"""
Beat-grid annotation format (rung 1) — one YAML file per clip in
`evals/grids/<case-id>.yaml`.

Two time lists with distinct contracts:

- `beats`   — the *annotation*: candidate beat times at vowel onsets. The
  tap-assist seeds it with the peakRate events; the owner corrects it
  (delete non-beats, nudge times) and flips `provisional` to false at
  rung 1.5. Until then the grid gates nothing.
- `onsets`  — frozen peakRate *evidence*: every voiced envelope-rise
  event the annotator heard. Never edited by the owner; feeds the
  onset-vs-token hallucination guard (ADR-016 clip-17) and rung-2
  debugging even after `beats` has been decimated to true beat times.

Grids are new files under evals/ — the add-only ingestion carve-out.
Existing cases/traces/baseline stay untouched.
"""

from dataclasses import dataclass, field
from pathlib import Path

GRID_FORMAT = 1
_TIME_DECIMALS = 4  # 0.1 ms — far inside annotation precision


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


def _validate_times(name: str, values, clip: str) -> list[float]:
    times = [round(float(v), _TIME_DECIMALS) for v in values or []]
    if any(t < 0 for t in times):
        raise ValueError(f"grid {clip}: negative time in {name}")
    if times != sorted(times):
        raise ValueError(f"grid {clip}: {name} must be sorted ascending")
    return times


def save_grid(grid: BeatGrid, grids_dir: Path) -> Path:
    import yaml  # lazy: pyyaml lives in the [eval] extra

    grids_dir = Path(grids_dir)
    grids_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": GRID_FORMAT,
        "clip": grid.clip,
        "provisional": bool(grid.provisional),
        "media": grid.media,
        "media_sha256": grid.media_sha256,
        "annotator": grid.annotator,
        "created_at": grid.created_at,
        "params": grid.params,
        "beats": _validate_times("beats", grid.beats, grid.clip),
        "onsets": _validate_times("onsets", grid.onsets, grid.clip),
        "notes": grid.notes,
    }
    path = grids_dir / f"{grid.clip}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
    return path


def load_grid(path: Path) -> BeatGrid:
    import yaml  # lazy: pyyaml lives in the [eval] extra

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or raw.get("format") != GRID_FORMAT:
        raise ValueError(f"{path}: not a beat-grid file (format {GRID_FORMAT})")
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
    """Audacity label track: start<TAB>end<TAB>label, one beat per line."""
    return "".join(
        f"{t:.4f}\t{t:.4f}\tbeat-{i + 1}\n" for i, t in enumerate(grid.beats)
    )


def beats_from_label_text(text: str) -> list[float]:
    """Parse corrected label text back to sorted beat times."""
    beats = []
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if not parts or not parts[0]:
            continue
        beats.append(round(float(parts[0]), _TIME_DECIMALS))
    return sorted(beats)
