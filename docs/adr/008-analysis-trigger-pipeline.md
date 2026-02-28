# ADR-008: Analysis Trigger Pipeline

**Date:** 2026-02-28
**Status:** Accepted

## Context

The system currently operates in batch mode: a teacher records audio/video,
then runs `python -m musical_perception file.wav`. In production, the mic and
camera will be always-on during class. Without a gate, every second of audio
would hit the Gemini API — expensive and unnecessary, since most class time is
silence, music playing, or conversational speech.

We need a lightweight, local pipeline that decides *when* to send audio to
Gemini for full analysis.

## Decision

### Three-stage local trigger

```
Mic stream (always on, 16kHz PCM)
    ↓
[OpenWakeWord] — "Hey Maestro" detected? (~5MB, <1% CPU, always listening)
    ↓ (triggered → start buffering audio)
[Whisper base.en] — transcribe the buffered segment
    ↓
[detect_onset_tempo()] — rhythmic counting confirmed? (existing rhythm.py)
    ↓ (confirmed)
[Gemini] — full analysis of the counting segment
```

Each stage is cheaper than the next, filtering out non-analysis audio as early
as possible. The wake word detector runs continuously at negligible cost. Whisper
only runs after the wake word. Gemini only runs after rhythmic counting is
confirmed.

### State machine

```
IDLE  →  (wake word)  →  LISTENING  →  (rhythm confirmed)  →  emit TriggerEvent → IDLE
  ↑                          |
  +←── (timeout/overflow) ←──+
```

Two states only (IDLE and LISTENING). When rhythm is confirmed, a `TriggerEvent`
is emitted and the machine returns to IDLE — there is no separate TRIGGERED
state because the trigger is an event, not a durable condition.

- **IDLE:** Only wake word detector runs. Minimal CPU.
- **LISTENING:** Wake word detected. Whisper transcribes buffered audio.
  `detect_onset_tempo()` checks for rhythmic speech periodically. On rhythm
  confirmation, emits a `TriggerEvent` containing the audio segment, Whisper
  transcription, and onset tempo — all pre-computed so downstream `analyze()`
  doesn't redo the work.
- **Timeout/overflow:** If no rhythm within `post_wake_timeout` seconds or
  audio buffer exceeds `buffer_seconds`, returns to IDLE.

Not thread-safe — callers must synchronize externally if feeding audio from a
background capture thread.

### Wake word choice

OpenWakeWord (MIT, ~5MB, <1% CPU) with a pre-trained placeholder model for now.
Two-word phrases ("Hey Maestro") work better than single words. A custom model
should be trained before production use via the OpenWakeWord Colab notebook.

### Architecture

- `trigger.py` — KEEP layer, pure state machine logic with dependency injection
  for wake word detection and transcription. Fully testable with mocks.
- `perception/wakeword.py` — DISPOSABLE wrapper around OpenWakeWord, same
  pattern as `whisper.py` and `gemini.py`.
- `TriggerState` enum and `TriggerEvent` dataclass in `types.py`.

## Consequences

- New optional dependency group: `pip install -e ".[trigger]"` adds
  `openwakeword>=0.6`
- 13 state machine tests using mock detectors (no audio or models needed)
- 3 integration tests for the wakeword wrapper (auto-skipped without
  openwakeword installed)
- `analyze.py` and `__main__.py` are **not modified** — the trigger is a
  separate entry point that feeds into the existing pipeline

## Follow-up roadmap

1. **Train "Hey Maestro" custom model** — Use OpenWakeWord Colab notebook
   (~1 hour) to generate a model trained on synthetic speech. Two-word phrases
   produce significantly better accuracy than single words.

2. **Live mic capture loop** — Add a `stream.py` entry point using
   pyaudio or sounddevice that reads from the mic, feeds chunks to
   `AnalysisTrigger`, and calls `analyze()` on trigger events.

3. **VAD layer** — Add Silero VAD (~5MB) between the mic and wake word detector
   to skip silence entirely. Reduces power consumption for always-on use.

4. **Wire TriggerEvent into analyze()** — Modify `analyze()` to accept
   pre-computed Whisper words and onset tempo from `TriggerEvent`, avoiding
   double work (Whisper + onset already ran during the LISTENING phase).

5. **UI feedback** — Surface trigger state changes to the user
   ("Listening...", "Analyzing...") so the teacher knows the system heard them.

6. **Camera trigger** — Extend the trigger to also monitor pose landmarks for
   movement patterns that signal exercise start (e.g., preparation position).
