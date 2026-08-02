"""
Perception provider bundle.

The seam between analyze() and the DISPOSABLE perception layer (ADR-009).
The default bundle wraps the real model wrappers with lazy loading; the
eval harness substitutes a replay bundle built from frozen traces so
analyze() runs offline and deterministically.
"""

from dataclasses import dataclass
from typing import Callable

from musical_perception.types import (
    GeminiAnalysisResult,
    LandmarkTimeSeries,
    TimestampedWord,
)


@dataclass
class PerceptionBundle:
    """
    The three perception calls analyze() consumes.

    transcribe(audio_path) -> list[TimestampedWord]
    analyze_media(media_path, *, onset_bpm=None, transcript_words=None)
        -> GeminiAnalysisResult
    extract_landmarks(video_path) -> LandmarkTimeSeries, or None when pose
        is unavailable (audio-only trace, missing deps).
    """
    transcribe: Callable[[str], list[TimestampedWord]]
    analyze_media: Callable[..., GeminiAnalysisResult]
    extract_landmarks: Callable[[str], LandmarkTimeSeries] | None = None


def build_default_bundle(
    model=None,
    model_name: str = "large-v3-turbo",
    gemini_client=None,
    gemini_model: str = "gemini-2.5-flash",
) -> PerceptionBundle:
    """Bundle backed by the real wrappers, each loaded lazily on first call."""
    state = {"model": model, "client": gemini_client}

    def _transcribe(audio_path: str):
        from musical_perception.perception.whisper import load_model, transcribe
        if state["model"] is None:
            state["model"] = load_model(model_name)
        return transcribe(state["model"], audio_path)

    def _analyze_media(media_path: str, *, onset_bpm=None, transcript_words=None):
        from musical_perception.perception.gemini import analyze_media, load_client
        if state["client"] is None:
            state["client"] = load_client(model=gemini_model)
        return analyze_media(
            state["client"], media_path,
            onset_bpm=onset_bpm, transcript_words=transcript_words,
        )

    def _extract_landmarks(video_path: str):
        from musical_perception.perception.pose import extract_landmarks, load_model
        return extract_landmarks(load_model(), video_path)

    return PerceptionBundle(
        transcribe=_transcribe,
        analyze_media=_analyze_media,
        extract_landmarks=_extract_landmarks,
    )
