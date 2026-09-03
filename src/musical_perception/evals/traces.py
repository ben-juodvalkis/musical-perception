"""
Frozen traces (ADR-009 tier 1): record model I/O once, replay forever.

Per clip: Whisper's timestamped words (whisper.json), Gemini's verbatim
response JSON plus the exact inputs it was sent (gemini.json), pose
landmarks (pose.npz, optional), and meta.json pinning identities/hashes.
Text-mostly and small — committable, unlike the media (gitignored).

Replay feeds the frozen raw JSON through the *current* parser and
recomputes onset tempo from the frozen words, so the fusion logic —
where this project's regressions actually happen — stays under test.
"""

import hashlib
import json
import subprocess
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from musical_perception.bundle import PerceptionBundle
from musical_perception.types import LandmarkTimeSeries, TimestampedWord

TRACE_FORMAT = 1


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def slugify(name: str) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in name.lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def make_recording_bundle(inner: PerceptionBundle, out_dir: Path) -> PerceptionBundle:
    """Wrap a bundle so every perception result is frozen to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def transcribe(audio_path: str):
        words = inner.transcribe(audio_path)
        payload = {"words": [asdict(w) for w in words]}
        (out_dir / "whisper.json").write_text(json.dumps(payload, indent=1))
        return words

    def analyze_media(media_path: str, *, onset_bpm=None, transcript_words=None):
        result = inner.analyze_media(
            media_path, onset_bpm=onset_bpm, transcript_words=transcript_words
        )
        payload = {
            "model": result.model,
            "raw_response": result.raw_response,
            "inputs": {
                "onset_bpm_sent": onset_bpm,
                "transcript_words_sent": list(transcript_words or []),
            },
        }
        (out_dir / "gemini.json").write_text(json.dumps(payload, indent=1))
        return result

    extract_landmarks = None
    if inner.extract_landmarks is not None:
        def extract_landmarks(video_path: str):
            series = inner.extract_landmarks(video_path)
            np.savez_compressed(
                out_dir / "pose.npz",
                timestamps=series.timestamps.astype(np.float32),
                landmarks=series.landmarks.astype(np.float32),
                fps=series.fps,
                detection_rate=series.detection_rate,
            )
            return series

    return PerceptionBundle(
        transcribe=transcribe,
        analyze_media=analyze_media,
        extract_landmarks=extract_landmarks,
    )


def write_meta(
    out_dir: Path,
    media_path: str,
    *,
    use_pose: bool,
    whisper_model_name: str,
    gemini_model: str,
) -> None:
    """Pin the identities that make a recording reproducible/attributable.

    Hashes describe the recording; tier-1 replay must never assert them
    against HEAD (a prompt change legitimately alters them — that is
    tier-2 drift material).
    """
    from musical_perception.perception import gemini as gemini_mod
    from musical_perception.perception import whisper as whisper_mod

    schema = {
        **gemini_mod._RESPONSE_SCHEMA,
        "properties": {
            **gemini_mod._RESPONSE_SCHEMA["properties"],
            "words": gemini_mod._words_schema(with_index=True),
        },
    }
    meta = {
        "trace_format": TRACE_FORMAT,
        "media": str(media_path),
        "media_sha256": _sha256_file(Path(media_path)),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "analyze_flags": {"use_pose": use_pose},
        "whisper": {
            "model_name": whisper_model_name,
            "initial_prompt_sha256": _sha256_text(whisper_mod.BALLET_VOCABULARY_PROMPT),
        },
        "gemini": {
            "model": gemini_model,
            "temperature": 0.0,
            "prompt_template_sha256": _sha256_text(gemini_mod._PROMPT_TEMPLATE),
            "response_schema_sha256": _sha256_text(json.dumps(schema, sort_keys=True)),
        },
    }
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=1))


# ADR-016 clip-17 guard: a fluent transcript over audio with little or no
# voiced acoustic support is the hallucination signature that once scored
# all-green. Thresholds are generous — real speech has at least as many
# voiced onsets as tokens (words have ≥1 syllable each).
GUARD_TOKEN_RATIO = 1.5
GUARD_TOKEN_SLACK = 8


def onset_token_mismatch(n_onsets: int, n_tokens: int) -> str | None:
    """Human-readable mismatch description, or None when consistent."""
    if n_tokens == 0:
        return None
    if n_onsets == 0:
        return f"{n_tokens} transcript tokens but 0 voiced acoustic onsets"
    if n_tokens > GUARD_TOKEN_RATIO * n_onsets + GUARD_TOKEN_SLACK:
        return (
            f"{n_tokens} transcript tokens vs {n_onsets} voiced acoustic "
            f"onsets (> {GUARD_TOKEN_RATIO}× + {GUARD_TOKEN_SLACK})"
        )
    return None


def _onset_guard(trace_dir: Path, n_tokens: int) -> None:
    """Run the onset-vs-token sanity check when a beat grid exists.

    Grids live at <evals_root>/grids/<trace-name>.yaml; their `onsets`
    list is the frozen peakRate evidence (annotation/grids.py). No grid,
    or no yaml available → the guard silently has nothing to check.
    """
    grid_path = trace_dir.parent.parent / "grids" / f"{trace_dir.name}.yaml"
    if not grid_path.is_file():
        return
    try:
        from musical_perception.annotation.grids import load_grid
        onsets = load_grid(grid_path).onsets
    except ImportError:  # pyyaml not installed — guard is best-effort
        return
    msg = onset_token_mismatch(len(onsets), n_tokens)
    if msg:
        warnings.warn(
            f"trace {trace_dir.name}: {msg} — transcription-hallucination "
            f"guard (ADR-016 clip-17); treat greens on this clip as "
            f"provisional until a human confirms the audio"
        )


def replay_bundle(trace_dir: Path) -> tuple[PerceptionBundle, dict]:
    """Build a PerceptionBundle that replays a frozen trace, plus its meta.

    Offline, deterministic, dependency-free: the raw Gemini JSON goes back
    through the current parser (parse_raw_response), and pose lands as the
    saved arrays. Loading also runs the onset-vs-token sanity guard against
    the clip's beat grid when one exists.
    """
    trace_dir = Path(trace_dir)
    meta = json.loads((trace_dir / "meta.json").read_text())
    whisper_payload = json.loads((trace_dir / "whisper.json").read_text())
    words = [TimestampedWord(**w) for w in whisper_payload["words"]]
    gemini_payload = json.loads((trace_dir / "gemini.json").read_text())
    _onset_guard(trace_dir, len(words))

    def transcribe(audio_path: str):
        return list(words)

    def analyze_media(media_path: str, *, onset_bpm=None, transcript_words=None):
        from musical_perception.perception.gemini import parse_raw_response

        frozen = gemini_payload["inputs"].get("onset_bpm_sent")
        if onset_bpm is not None and frozen is not None and abs(onset_bpm - frozen) > 0.1:
            warnings.warn(
                f"replay: recomputed onset_bpm {onset_bpm:.1f} != frozen "
                f"{frozen:.1f} — the rhythm layer changed since recording"
            )
        return parse_raw_response(gemini_payload["raw_response"], gemini_payload["model"])

    extract_landmarks = None
    pose_path = trace_dir / "pose.npz"
    if pose_path.is_file():
        def extract_landmarks(video_path: str):
            with np.load(pose_path) as data:
                return LandmarkTimeSeries(
                    timestamps=data["timestamps"].astype(float),
                    landmarks=data["landmarks"].astype(float),
                    fps=float(data["fps"]),
                    detection_rate=float(data["detection_rate"]),
                )

    # PP-1 replay seam: the frozen acoustic pulse stream, when W11/W11-c
    # recorded one for this clip. No sidecar -> no provider -> the rhythm
    # core is bit-for-bit its pre-PP-1 self on that row.
    pulse_events = None
    pulse_path = trace_dir / "pulse.json"
    if pulse_path.is_file():
        _frozen_pulse = [
            float(t) for t in json.loads(pulse_path.read_text())["events"]
        ]

        def pulse_events(audio_path: str):
            return list(_frozen_pulse)

    bundle = PerceptionBundle(
        transcribe=transcribe,
        analyze_media=analyze_media,
        extract_landmarks=extract_landmarks,
        pulse_events=pulse_events,
    )
    return bundle, meta
