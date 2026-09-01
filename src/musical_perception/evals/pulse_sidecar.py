"""
Pulse sidecars (W11, EVAL-CHANGE) — the acoustic pulse stream, frozen.

A trace freezes what the *models* said (whisper.json, gemini.json). The
rung-2 acoustic pulse stream is not a model output but a deterministic
function of the media, and the media is gitignored — so replaying it
today means re-deriving events from files that only exist on the runner
that recorded them. `pulse.json` closes that gap: derived evidence,
committed beside the trace it belongs to (Standing Lesson 9 — build the
replay path before betting on the channel).

Written under the owner-ratified sidecar carve-out (charter rule 2,
2026-08-28): ADD-only inside an existing trace directory, never
modifying any existing file, and only when the media on disk hashes to
the trace's stored `media_sha256`. That checksum is the whole argument
that a sidecar describes the same audio the trace was recorded from, so
the recorder refuses to write without it rather than warning and
proceeding.

Load-time verification is offline and cheap: the sidecar carries the
hash it was recorded against, and loading re-checks it against
meta.json. A sidecar that drifted from its trace raises instead of
silently feeding stale events to a scorer.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SIDECAR_NAME = "pulse.json"
SIDECAR_FORMAT = 1
EXTRACTOR = "acoustic-pulse/1"


class SidecarError(RuntimeError):
    """A sidecar exists but cannot be trusted for this trace."""


@dataclass(frozen=True)
class PulseSidecar:
    """One clip's frozen acoustic pulse stream."""
    events: list[float]
    params: dict
    extractor: str
    media_sha256: str

    @property
    def n_events(self) -> int:
        return len(self.events)


def sidecar_path(trace_dir: Path) -> Path:
    return Path(trace_dir) / SIDECAR_NAME


def _trace_media_sha(trace_dir: Path) -> str | None:
    meta = json.loads((Path(trace_dir) / "meta.json").read_text())
    return meta.get("media_sha256")


def trace_media_ref(trace_dir: Path) -> str | None:
    """The reference the trace itself uses for its media (W11-b).

    May be a repo path (`video/youtube/Frappe.mov`) or an opaque
    off-repo reference (`offrepo:barre1-A-s`). Either way it is already
    committed, so a sidecar that copies it names nothing new.
    """
    meta = json.loads((Path(trace_dir) / "meta.json").read_text())
    return meta.get("media")


def resolve_media_by_checksum(
    media_root: Path, wanted: dict[str, str]
) -> dict[str, Path]:
    """Find media by content, never by name (W11-b).

    `wanted` maps a trace's pinned `media_sha256` to the id it belongs
    to; the return maps those ids to the files that hash to them.

    Every regular file under `media_root` is hashed and a file whose
    digest is not wanted is **discarded on the spot** — its path is
    never returned, stored, logged, or put in an exception message. That
    is the entire point. The Barre-1 batch is split at the exercise
    level with DEV and held-out material sharing one directory, so
    listing that directory names the held-out exercises by complement
    (charter: "Containment is not agent-auditable — the only available
    audit is itself the leak"). Content-addressed lookup is the one form
    of search that cannot leak, because it never has to say what it
    rejected.
    """
    from musical_perception.evals.traces import _sha256_file

    found: dict[str, Path] = {}
    for path in sorted(Path(media_root).rglob("*")):
        if not path.is_file():
            continue
        try:
            digest = _sha256_file(path)
        except OSError:
            continue                    # unreadable; say nothing about it
        clip_id = wanted.get(digest)
        if clip_id is not None and clip_id not in found:
            found[clip_id] = path
    return found


def load_pulse_sidecar(trace_dir: Path) -> PulseSidecar | None:
    """The trace's pulse sidecar, or None when it has none.

    Raises SidecarError when the sidecar's recorded media hash disagrees
    with the trace's — the two describe different audio, and which one
    is right is not a question this layer may guess at.
    """
    trace_dir = Path(trace_dir)
    path = sidecar_path(trace_dir)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    recorded = payload.get("media_sha256")
    expected = _trace_media_sha(trace_dir)
    if expected is not None and recorded != expected:
        raise SidecarError(
            f"{trace_dir.name}: pulse.json was recorded against media "
            f"{recorded} but the trace pins {expected} — re-record the "
            f"sidecar against the trace's media"
        )
    return PulseSidecar(
        events=[float(t) for t in payload["events"]],
        params=payload.get("params") or {},
        extractor=payload.get("extractor", "unknown"),
        media_sha256=recorded,
    )


def verify_media(trace_dir: Path, media_path: Path) -> str:
    """Hash `media_path` and require it to match the trace's pin.

    Returns the verified hex digest; raises SidecarError otherwise. This
    is the carve-out's precondition — the recorder calls it before it is
    allowed to write anything.
    """
    from musical_perception.evals.traces import _sha256_file

    media_path = Path(media_path)
    expected = _trace_media_sha(trace_dir)
    if expected is None:
        raise SidecarError(f"{Path(trace_dir).name}: trace pins no media_sha256")
    if not media_path.is_file():
        raise SidecarError(f"{Path(trace_dir).name}: media not on this runner: {media_path}")
    actual = _sha256_file(media_path)
    if actual != expected:
        raise SidecarError(
            f"{Path(trace_dir).name}: media {media_path} hashes {actual} but "
            f"the trace pins {expected} — refusing to record a sidecar"
        )
    return actual


def record_pulse_sidecar(
    trace_dir: Path,
    media_path: Path,
    *,
    params=None,
    force: bool = False,
) -> PulseSidecar:
    """Extract the acoustic pulse stream for one clip and freeze it.

    Refuses to overwrite an existing sidecar unless `force`; refuses to
    write at all unless the media checksum matches the trace's pin.
    """
    from musical_perception.annotation.__main__ import _load_audio
    from musical_perception.evals.traces import _git_sha
    from musical_perception.precision.pulse import (
        AcousticPulseParams,
        acoustic_pulse_events,
    )

    trace_dir = Path(trace_dir)
    path = sidecar_path(trace_dir)
    if path.is_file() and not force:
        raise SidecarError(f"{trace_dir.name}: {SIDECAR_NAME} exists (--force to redo)")

    params = params or AcousticPulseParams()
    media_sha = verify_media(trace_dir, media_path)
    y = _load_audio(Path(media_path), params.peakrate.sr)
    events = [
        round(float(t), 4)
        for t in acoustic_pulse_events(y, params.peakrate.sr, params)
    ]
    payload = {
        "sidecar_format": SIDECAR_FORMAT,
        "extractor": EXTRACTOR,
        # The trace's own reference, not the path we were handed (W11-b).
        # Checksum equality already proves the two are the same bytes, so
        # copying the committed reference loses nothing and guarantees a
        # sidecar can never name a file its trace did not already name.
        # For the Barre-1 traces that reference is `offrepo:<case-id>`.
        "media": trace_media_ref(trace_dir) or str(media_path),
        "media_sha256": media_sha,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "params": params.as_dict(),
        "events": events,
    }
    path.write_text(json.dumps(payload, indent=1))
    return PulseSidecar(
        events=events, params=payload["params"],
        extractor=EXTRACTOR, media_sha256=media_sha,
    )
