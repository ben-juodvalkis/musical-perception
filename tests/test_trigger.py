"""Tests for the analysis trigger state machine. No audio or models needed."""

import numpy as np

from musical_perception.trigger import AnalysisTrigger, CHUNK_SAMPLES
from musical_perception.types import TimestampedWord, TriggerState


def _silent_chunk():
    """80ms of silence at 16kHz."""
    return np.zeros(CHUNK_SAMPLES, dtype=np.int16)


def _no_wakeword(chunk):
    """Mock detector that never detects a wake word."""
    return {"hey_jarvis": 0.0}


def _always_wakeword(chunk):
    """Mock detector that always detects a wake word."""
    return {"hey_jarvis": 0.9}


def _once_wakeword():
    """Mock detector that detects wake word on first call only."""
    called = [False]
    def detect(chunk):
        if not called[0]:
            called[0] = True
            return {"hey_jarvis": 0.9}
        return {"hey_jarvis": 0.0}
    return detect


def _rhythmic_words():
    """Words that form a clear rhythmic pattern at ~120 BPM."""
    interval = 0.5  # 120 BPM
    return [
        TimestampedWord(word=str(i + 1), start=i * interval, end=i * interval + 0.1)
        for i in range(12)
    ]


def _no_rhythm_words():
    """Words with highly irregular spacing — not rhythmic (CV > 0.4)."""
    return [
        TimestampedWord(word="so", start=0.0, end=0.2),
        TimestampedWord(word="we're", start=0.3, end=0.5),
        TimestampedWord(word="going", start=0.6, end=0.8),
        TimestampedWord(word="to", start=2.5, end=2.6),
        TimestampedWord(word="do", start=2.7, end=2.9),
        TimestampedWord(word="tendus", start=4.5, end=5.0),
        TimestampedWord(word="and", start=7.0, end=7.1),
        TimestampedWord(word="then", start=7.2, end=7.4),
    ]


def _mock_transcribe_rhythmic(audio):
    """Mock transcriber that returns rhythmic words."""
    return _rhythmic_words()


def _mock_transcribe_no_rhythm(audio):
    """Mock transcriber that returns non-rhythmic words."""
    return _no_rhythm_words()


def _mock_transcribe_empty(audio):
    """Mock transcriber that returns no words."""
    return []


def test_starts_idle():
    """Trigger starts in IDLE state."""
    trigger = AnalysisTrigger(
        detect_wakeword=_no_wakeword,
        transcribe=_mock_transcribe_empty,
    )
    assert trigger.state == TriggerState.IDLE


def test_idle_no_wakeword():
    """IDLE + no wake word → stays IDLE, no event."""
    trigger = AnalysisTrigger(
        detect_wakeword=_no_wakeword,
        transcribe=_mock_transcribe_empty,
    )
    event = trigger.feed(_silent_chunk(), 0.0)
    assert event is None
    assert trigger.state == TriggerState.IDLE


def test_idle_to_listening():
    """IDLE + wake word → transitions to LISTENING."""
    trigger = AnalysisTrigger(
        detect_wakeword=_always_wakeword,
        transcribe=_mock_transcribe_empty,
    )
    event = trigger.feed(_silent_chunk(), 0.0)
    assert event is None  # No event on transition — need rhythm first
    assert trigger.state == TriggerState.LISTENING


def test_listening_to_triggered():
    """LISTENING + rhythmic speech → emits TriggerEvent, returns to IDLE."""
    trigger = AnalysisTrigger(
        detect_wakeword=_once_wakeword(),
        transcribe=_mock_transcribe_rhythmic,
        rhythm_check_interval=0.0,  # Check every chunk
    )

    # First chunk: wake word detected → LISTENING
    trigger.feed(_silent_chunk(), 0.0)
    assert trigger.state == TriggerState.LISTENING

    # Second chunk: rhythm detected → TRIGGERED → event emitted → back to IDLE
    event = trigger.feed(_silent_chunk(), 1.0)
    assert event is not None
    assert len(event.words) == 12
    assert event.onset_tempo is not None
    assert event.onset_tempo.confidence >= 0.3
    assert event.timestamp == 1.0
    assert trigger.state == TriggerState.IDLE


def test_listening_timeout():
    """LISTENING + timeout → returns to IDLE without event."""
    trigger = AnalysisTrigger(
        detect_wakeword=_once_wakeword(),
        transcribe=_mock_transcribe_no_rhythm,
        post_wake_timeout=5.0,
        rhythm_check_interval=0.0,
    )

    # Wake word detected
    trigger.feed(_silent_chunk(), 0.0)
    assert trigger.state == TriggerState.LISTENING

    # Feed chunks without rhythm, past timeout
    event = trigger.feed(_silent_chunk(), 6.0)
    assert event is None
    assert trigger.state == TriggerState.IDLE


def test_listening_no_rhythm():
    """LISTENING + non-rhythmic speech → stays LISTENING (within timeout)."""
    trigger = AnalysisTrigger(
        detect_wakeword=_once_wakeword(),
        transcribe=_mock_transcribe_no_rhythm,
        post_wake_timeout=60.0,
        rhythm_check_interval=0.0,
    )

    trigger.feed(_silent_chunk(), 0.0)
    assert trigger.state == TriggerState.LISTENING

    event = trigger.feed(_silent_chunk(), 1.0)
    assert event is None
    assert trigger.state == TriggerState.LISTENING


def test_listening_empty_transcription():
    """LISTENING + empty transcription → stays LISTENING."""
    trigger = AnalysisTrigger(
        detect_wakeword=_once_wakeword(),
        transcribe=_mock_transcribe_empty,
        rhythm_check_interval=0.0,
    )

    trigger.feed(_silent_chunk(), 0.0)
    event = trigger.feed(_silent_chunk(), 1.0)
    assert event is None
    assert trigger.state == TriggerState.LISTENING


def test_multiple_cycles():
    """Can trigger multiple times in sequence."""
    trigger = AnalysisTrigger(
        detect_wakeword=_always_wakeword,
        transcribe=_mock_transcribe_rhythmic,
        rhythm_check_interval=0.0,
    )

    # First cycle: wake word → listening → rhythm → triggered → idle
    trigger.feed(_silent_chunk(), 0.0)
    assert trigger.state == TriggerState.LISTENING
    event1 = trigger.feed(_silent_chunk(), 1.0)
    assert event1 is not None
    assert trigger.state == TriggerState.IDLE

    # Second cycle: same trigger can fire again
    trigger.feed(_silent_chunk(), 2.0)
    assert trigger.state == TriggerState.LISTENING
    event2 = trigger.feed(_silent_chunk(), 3.0)
    assert event2 is not None
    assert trigger.state == TriggerState.IDLE


def test_buffer_overflow_resets():
    """Exceeding buffer_seconds resets to IDLE."""
    trigger = AnalysisTrigger(
        detect_wakeword=_once_wakeword(),
        transcribe=_mock_transcribe_empty,
        buffer_seconds=0.1,  # Very small buffer (100ms)
        rhythm_check_interval=0.0,
    )

    # Wake word → LISTENING (no audio buffered yet)
    trigger.feed(_silent_chunk(), 0.0)
    assert trigger.state == TriggerState.LISTENING

    # First LISTENING chunk: buffers 80ms (under 100ms limit)
    trigger.feed(_silent_chunk(), 0.08)
    assert trigger.state == TriggerState.LISTENING

    # Second LISTENING chunk: buffers 160ms total > 100ms → reset
    event = trigger.feed(_silent_chunk(), 0.16)
    assert event is None
    assert trigger.state == TriggerState.IDLE


def test_reset():
    """Manual reset returns to IDLE."""
    trigger = AnalysisTrigger(
        detect_wakeword=_always_wakeword,
        transcribe=_mock_transcribe_empty,
    )

    trigger.feed(_silent_chunk(), 0.0)
    assert trigger.state == TriggerState.LISTENING

    trigger.reset()
    assert trigger.state == TriggerState.IDLE


def test_wakeword_threshold():
    """Custom threshold is respected."""
    def detect_low(chunk):
        return {"hey_jarvis": 0.3}  # Below default 0.5

    trigger = AnalysisTrigger(
        detect_wakeword=detect_low,
        transcribe=_mock_transcribe_empty,
        wakeword_threshold=0.5,
    )

    trigger.feed(_silent_chunk(), 0.0)
    assert trigger.state == TriggerState.IDLE  # 0.3 < 0.5 threshold

    # With lower threshold, same score should trigger
    trigger2 = AnalysisTrigger(
        detect_wakeword=detect_low,
        transcribe=_mock_transcribe_empty,
        wakeword_threshold=0.2,
    )
    trigger2.feed(_silent_chunk(), 0.0)
    assert trigger2.state == TriggerState.LISTENING


def test_rhythm_check_interval():
    """Rhythm is only checked after rhythm_check_interval seconds."""
    transcribe_calls = [0]
    def counting_transcriber(audio):
        transcribe_calls[0] += 1
        return _rhythmic_words()

    trigger = AnalysisTrigger(
        detect_wakeword=_once_wakeword(),
        transcribe=counting_transcriber,
        rhythm_check_interval=5.0,
    )

    trigger.feed(_silent_chunk(), 0.0)  # Wake word
    assert trigger.state == TriggerState.LISTENING

    # Feed chunk only 1s later — should not transcribe yet
    trigger.feed(_silent_chunk(), 1.0)
    assert transcribe_calls[0] == 0

    # Feed chunk 5s later — should transcribe and trigger
    event = trigger.feed(_silent_chunk(), 5.1)
    assert transcribe_calls[0] == 1
    assert event is not None


def test_trigger_event_contains_audio():
    """TriggerEvent includes the buffered audio as bytes."""
    trigger = AnalysisTrigger(
        detect_wakeword=_once_wakeword(),
        transcribe=_mock_transcribe_rhythmic,
        rhythm_check_interval=0.0,
    )

    chunk = _silent_chunk()
    trigger.feed(chunk, 0.0)
    event = trigger.feed(chunk, 1.0)

    assert event is not None
    # Should contain 1 chunk (the LISTENING chunk; wake word chunk is not buffered)
    expected_bytes = CHUNK_SAMPLES * np.dtype(np.int16).itemsize
    assert len(event.audio_segment) == expected_bytes
