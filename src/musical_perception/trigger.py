"""
Analysis trigger: wake word + rhythm detection state machine.

KEEP — pure logic, no I/O. Decides when audio is worth sending to Gemini.
Two states: IDLE (wake word listening) and LISTENING (buffering + rhythm check).
When rhythm is confirmed, emits a TriggerEvent and returns to IDLE.

Not thread-safe — callers must synchronize externally if feeding audio
from a background capture thread.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from musical_perception.precision.rhythm import detect_onset_tempo
from musical_perception.types import (
    OnsetTempoResult,
    TimestampedWord,
    TriggerEvent,
    TriggerState,
)

# 80ms at 16kHz = 1280 samples
CHUNK_SAMPLES = 1280


class WakeWordDetector(Protocol):
    """Protocol for wake word detection — any callable returning scores."""

    def __call__(self, audio_chunk: np.ndarray) -> dict[str, float]: ...


class Transcriber(Protocol):
    """Protocol for transcription — takes audio bytes, returns words."""

    def __call__(self, audio: np.ndarray) -> list[TimestampedWord]: ...


class AnalysisTrigger:
    """
    Streaming trigger that gates Gemini analysis behind wake word + rhythm detection.

    State machine:
        IDLE  →  (wake word)  →  LISTENING  →  (rhythm confirmed)  →  emit TriggerEvent → IDLE
          ↑                          |
          +←── (timeout, no rhythm) ←+

    Usage:
        trigger = AnalysisTrigger(detect_wakeword=..., transcribe=...)
        for chunk, ts in audio_stream:
            event = trigger.feed(chunk, ts)
            if event is not None:
                result = analyze_from_trigger(event)
    """

    def __init__(
        self,
        detect_wakeword: WakeWordDetector,
        transcribe: Transcriber,
        *,
        wakeword_threshold: float = 0.5,
        rhythm_confidence_threshold: float = 0.3,
        buffer_seconds: float = 30.0,
        post_wake_timeout: float = 60.0,
        rhythm_check_interval: float = 3.0,
    ):
        """
        Args:
            detect_wakeword: Callable that takes int16 audio chunk, returns
                {model_name: confidence} dict.
            transcribe: Callable that takes int16 audio array, returns
                list[TimestampedWord].
            wakeword_threshold: Confidence above which wake word is detected.
            rhythm_confidence_threshold: Minimum onset tempo confidence to trigger.
            buffer_seconds: Max audio to buffer while listening.
            post_wake_timeout: Seconds after wake word before giving up.
            rhythm_check_interval: Seconds of audio between rhythm checks.
        """
        self._detect_wakeword = detect_wakeword
        self._transcribe = transcribe
        self._wakeword_threshold = wakeword_threshold
        self._rhythm_confidence_threshold = rhythm_confidence_threshold
        self._buffer_seconds = buffer_seconds
        self._post_wake_timeout = post_wake_timeout
        self._rhythm_check_interval = rhythm_check_interval

        self._state = TriggerState.IDLE
        self._wake_timestamp: float = 0.0
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_duration: float = 0.0
        self._last_rhythm_check: float = 0.0

    @property
    def state(self) -> TriggerState:
        return self._state

    def reset(self) -> None:
        """Return to IDLE state, clearing all buffers."""
        self._state = TriggerState.IDLE
        self._audio_buffer.clear()
        self._buffer_duration = 0.0
        self._wake_timestamp = 0.0
        self._last_rhythm_check = 0.0

    def feed(self, audio_chunk: np.ndarray, timestamp: float) -> TriggerEvent | None:
        """
        Feed an 80ms audio chunk (1280 int16 samples at 16kHz).

        Returns a TriggerEvent when analysis should begin, None otherwise.
        """
        if audio_chunk.dtype != np.int16:
            raise ValueError(
                f"audio_chunk must be int16, got {audio_chunk.dtype}. "
                "OpenWakeWord expects 16-bit PCM, not normalized float32."
            )
        if self._state == TriggerState.IDLE:
            return self._handle_idle(audio_chunk, timestamp)
        else:
            return self._handle_listening(audio_chunk, timestamp)

    def _handle_idle(
        self, audio_chunk: np.ndarray, timestamp: float
    ) -> TriggerEvent | None:
        """IDLE: check for wake word only."""
        scores = self._detect_wakeword(audio_chunk)
        if any(score >= self._wakeword_threshold for score in scores.values()):
            self._state = TriggerState.LISTENING
            self._wake_timestamp = timestamp
            self._audio_buffer.clear()
            self._buffer_duration = 0.0
            self._last_rhythm_check = timestamp
        return None

    def _handle_listening(
        self, audio_chunk: np.ndarray, timestamp: float
    ) -> TriggerEvent | None:
        """LISTENING: buffer audio, periodically check for rhythm."""
        # Buffer the audio
        self._audio_buffer.append(audio_chunk)
        chunk_duration = len(audio_chunk) / 16000.0
        self._buffer_duration += chunk_duration

        # Check timeout
        elapsed = timestamp - self._wake_timestamp
        if elapsed > self._post_wake_timeout:
            self.reset()
            return None

        # Check buffer overflow
        if self._buffer_duration > self._buffer_seconds:
            self.reset()
            return None

        # Periodically check for rhythm
        if timestamp - self._last_rhythm_check < self._rhythm_check_interval:
            return None
        self._last_rhythm_check = timestamp

        # Transcribe accumulated audio
        audio_array = np.concatenate(self._audio_buffer)
        words = self._transcribe(audio_array)

        if not words:
            return None

        # Check for rhythmic speech
        onset_tempo = detect_onset_tempo(words)
        if onset_tempo is None:
            return None
        if onset_tempo.confidence < self._rhythm_confidence_threshold:
            return None

        # Rhythm confirmed — emit trigger event
        audio_bytes = audio_array.tobytes()
        event = TriggerEvent(
            audio_segment=audio_bytes,
            words=words,
            onset_tempo=onset_tempo,
            timestamp=timestamp,
        )
        self.reset()
        return event
