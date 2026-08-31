"""
Gemini draw sidecars (W6-a, rung 5) — N classifications of one transcript.

ADR-011 measured 18, 18, 18, 32 BPM from four temperature-0 draws on
identical input, and Standing Lesson 4 has said "one draw is a coin
flip" ever since — but the pipeline has consumed exactly one draw all
along, because there was nowhere to put the others. `gemini-draws.json`
is that place: the per-draw classifications frozen beside the trace,
replayable offline, checksum-bound to the same media (Standing Lesson 9
— build the replay path before betting on the channel; `pulse.json` is
the precedent this file deliberately copies).

Written under the owner-ratified sidecar carve-out (charter rule 2,
2026-08-28): ADD-only inside an existing trace directory, never
modifying any existing file, and only where the media hashes to the
trace's stored `media_sha256`.

A draw is a list of `(index, marker_type, beat_number)` against a
SPECIFIC Whisper token sequence, so the payload also pins that
transcript's fingerprint. Indices into a different transcript are
silently wrong rather than loudly wrong — the one failure mode this
format can and does refuse.

Recording sidecars needs live model calls and is W6-b. This module
loads and mixes; nothing in the shipping path reads it yet.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from musical_perception.evals.pulse_sidecar import (
    SidecarError,
    _trace_media_sha,
)
from musical_perception.types import (
    BELIEF_CLASSES,
    MarkerBelief,
    TimestampedWord,
)

SIDECAR_NAME = "gemini-draws.json"
SIDECAR_FORMAT = 1


@dataclass(frozen=True)
class GeminiDraw:
    """One model's one classification of the pinned transcript.

    `labels` maps transcript index to a class in BELIEF_CLASSES; an
    index the draw never mentioned is `none` by omission, which is what
    the model means when it returns only the markers it found.
    """
    draw_id: str
    model: str
    params: dict
    labels: dict[int, str]
    beat_numbers: dict[int, int]


@dataclass(frozen=True)
class GeminiDrawsSidecar:
    """One clip's frozen ensemble of classifications."""
    draws: list[GeminiDraw]
    media_sha256: str | None
    transcript_sha256: str | None

    @property
    def n_draws(self) -> int:
        return len(self.draws)

    @property
    def models(self) -> list[str]:
        return sorted({d.model for d in self.draws})


def sidecar_path(trace_dir: Path) -> Path:
    return Path(trace_dir) / SIDECAR_NAME


def transcript_fingerprint(words: list[TimestampedWord]) -> str:
    """Hash the token sequence draws were made against.

    Text and start time, both: re-transcription can preserve the words
    and move the timing, and a draw's index is only meaningful against
    the sequence whose timings it will be joined to.
    """
    payload = "\n".join(
        f"{i}\t{w.word}\t{w.start:.3f}" for i, w in enumerate(words)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _parse_draw(raw: dict, position: int) -> GeminiDraw:
    labels: dict[int, str] = {}
    beat_numbers: dict[int, int] = {}
    for w in raw.get("words", []):
        idx = w.get("index")
        if idx is None:
            raise SidecarError(
                f"draw {raw.get('draw_id', position)}: a word carries no "
                f"index — draws are index-keyed against the pinned "
                f"transcript, and text matching is not a fallback here"
            )
        cls = w.get("marker_type") or "none"
        if cls not in BELIEF_CLASSES:
            raise SidecarError(
                f"draw {raw.get('draw_id', position)}: unknown class "
                f"{cls!r} at index {idx} (expected one of {BELIEF_CLASSES})"
            )
        labels[int(idx)] = cls
        if w.get("beat_number") is not None:
            beat_numbers[int(idx)] = int(w["beat_number"])
    return GeminiDraw(
        draw_id=str(raw.get("draw_id", f"draw-{position}")),
        model=str(raw.get("model", "unknown")),
        params=raw.get("params") or {},
        labels=labels,
        beat_numbers=beat_numbers,
    )


def load_gemini_draws(
    trace_dir: Path,
    *,
    words: list[TimestampedWord] | None = None,
) -> GeminiDrawsSidecar | None:
    """The trace's draws sidecar, or None when it has none.

    Raises SidecarError when the sidecar's media hash disagrees with the
    trace's, when `words` is supplied and does not match the pinned
    transcript, or when any index falls outside that transcript. Which
    of two disagreeing objects is right is not a question this layer may
    guess at — the pulse sidecar's rule, applied to the second sidecar.
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
            f"{trace_dir.name}: {SIDECAR_NAME} was recorded against media "
            f"{recorded} but the trace pins {expected} — re-record the "
            f"sidecar against the trace's media"
        )

    draws = [_parse_draw(d, i) for i, d in enumerate(payload.get("draws", []))]
    pinned = payload.get("transcript_sha256")

    if words is not None:
        actual = transcript_fingerprint(words)
        if pinned is not None and pinned != actual:
            raise SidecarError(
                f"{trace_dir.name}: {SIDECAR_NAME} pins transcript {pinned} "
                f"but this trace transcribes to {actual} — the draws' "
                f"indices address a different token sequence"
            )
        for d in draws:
            stray = [i for i in d.labels if not 0 <= i < len(words)]
            if stray:
                raise SidecarError(
                    f"{trace_dir.name}: draw {d.draw_id} indexes "
                    f"{stray[:5]} outside a {len(words)}-token transcript"
                )

    return GeminiDrawsSidecar(
        draws=draws,
        media_sha256=recorded,
        transcript_sha256=pinned,
    )


def beliefs_from_draws(
    draws: list[GeminiDraw],
    words: list[TimestampedWord],
    *,
    sub_vocab: set[str] | None = None,
) -> list[MarkerBelief]:
    """Mix N draws into one belief per transcript token.

    Each draw votes with weight 1/N for the class it assigned; a draw
    that never mentions a token votes `none`, which is what silence
    means in this format. The result is what `estimate_rhythm` takes as
    `marker_beliefs`.

    The token filters are the single-draw path's, kept identical on
    purpose: a token every draw calls `none` is dropped when its text is
    a subdivision vocable, or when a token carrying real marker mass
    already stands at its timestamp. The property the tests pin is that
    with ONE draw every stream the posterior reads — beat, sub, word,
    and the ladder's beat-number sequence — is identical to the one
    `beliefs_from_markers` produces from the same draw. The token ORDER
    differs (transcript order here, markers-then-words there) and
    nothing downstream reads it, which is why the pinned property is
    stream equality and not list equality.
    """
    from musical_perception.precision.posterior import (
        _SUB_VOCAB,
        _normalize_text,
    )

    vocab = _SUB_VOCAB if sub_vocab is None else sub_vocab
    if not draws:
        return []
    n = float(len(draws))

    probs: list[dict[str, float]] = []
    beat_nums: list[int | None] = []
    for i, _w in enumerate(words):
        counts: dict[str, float] = {}
        nums: list[int] = []
        for d in draws:
            cls = d.labels.get(i, "none")
            counts[cls] = counts.get(cls, 0.0) + 1.0
            if cls == "beat" and i in d.beat_numbers:
                nums.append(d.beat_numbers[i])
        probs.append({c: v / n for c, v in counts.items()})
        beat_nums.append(
            max(set(nums), key=nums.count) if nums else None
        )

    marker_ts = {
        round(w.start, 4)
        for w, p in zip(words, probs)
        if p.get("none", 0.0) < 1.0
    }

    beliefs = []
    for w, p, bn in zip(words, probs, beat_nums):
        pure_none = p.get("none", 0.0) >= 1.0
        if pure_none and (
            round(w.start, 4) in marker_ts
            or _normalize_text(w.word) in vocab
        ):
            continue
        beliefs.append(MarkerBelief(
            timestamp=w.start, probs=p, beat_number=bn, raw_word=w.word,
        ))
    return beliefs
