# ADR-001: Upgrade Audio Perception (Phase 1)

**Date:** 2026-02-24
**Status:** Accepted

## Context

The audio perception pipeline used OpenAI Whisper for transcription with its
native word timestamps, and Praat/Parselmouth for prosodic feature extraction.
Two issues:

1. **Whisper's word timestamps are approximate.** They come from the attention
   mechanism, not forced alignment. For rhythmic speech where timing precision
   matters (computing BPM from word onsets), this introduces error.

2. **No stress detection.** The counting signature relied entirely on Praat's
   numeric pitch/intensity to infer emphasis (`weight_placement`). A learned
   stress detection model could complement or validate these measurements.

3. **Temp file leak in prosody.py.** The temp WAV file created for Praat was
   never cleaned up.

## Decisions

### 1. Replace Whisper with WhisperX (with fallback)

WhisperX adds wav2vec2-based forced alignment on top of Whisper, producing
tighter word boundaries. The improvement matters because `precision/tempo.py`
computes BPM directly from word onset timestamps — tighter boundaries mean
more accurate tempo.

**Implementation:** `perception/whisper.py` tries `import whisperx` first. If
available, it runs transcription then forced alignment via `whisperx.align()`.
If WhisperX is not installed, it falls back to plain Whisper with native word
timestamps. The `load_model()` / `transcribe()` API is unchanged — callers
don't know which backend is active.

**Dependencies:**
- `pip install -e ".[whisper]"` now installs `whisperx>=3.1`
- `pip install -e ".[whisper-fallback]"` installs `openai-whisper` for
  environments where WhisperX's heavier dependency tree is unwanted

### 2. Add WhiStress for stress detection

WhiStress (Interspeech 2025) extends Whisper with a stress detection head,
outputting per-token binary stress labels (1 = stressed, 0 = unstressed).
This complements Praat's numeric F0/intensity with learned emphasis labels.

**Implementation:** New module `perception/whistress.py`, a thin DISPOSABLE
wrapper around `WhiStressInferenceClient`. Wired into the pipeline via
`analyze(detect_stress=True)` and exposed in the CLI via `--stress`. Results
stored in `MusicalParameters.stress_labels`.

**Not pip-installable.** WhiStress requires cloning the repo and downloading
weights manually. This is acceptable — it's an optional enrichment, not a
required dependency.

### 3. Keep Praat/Parselmouth

No model currently outputs numeric pitch (Hz) and intensity (dB) per word.
Multimodal LLMs are qualitative, not quantitative:

- GPT-4o scores 53–59% on stress detection (near random; humans: 92.6%)
- Gemini cannot return pitch contours
- Praat gives exact F0 and intensity at 10ms resolution, for free

Praat stays until models catch up.

### 4. Fix temp file cleanup in prosody.py

The temp WAV created with `delete=False` was never removed. Fixed with
`os.unlink()` in a `try/finally` block after Praat loads the file.

## Consequences

- **Better word boundaries** when WhisperX is installed, directly improving
  tempo accuracy. Transparent fallback when it isn't.
- **Optional stress labels** available as a new signal for downstream
  consumers. Not yet integrated into `CountingSignature` computation — that's
  a future decision once WhiStress is evaluated on counting speech.
- **No breaking changes.** `MusicalParameters.stress_labels` defaults to
  `None`. All existing tests pass unchanged.
- **Heavier dependency tree** when using WhisperX (pulls in faster-whisper,
  transformers, pyannote-audio). The fallback path keeps the light option
  available.

## Files Changed

- `perception/whisper.py` — rewritten for WhisperX with Whisper fallback
- `perception/whistress.py` — new module (DISPOSABLE)
- `perception/prosody.py` — temp file cleanup fix
- `analyze.py` — added `detect_stress` parameter
- `types.py` — added `stress_labels` field to `MusicalParameters`
- `__main__.py` — added `--stress` CLI flag
- `pyproject.toml` — `whisper` group now installs whisperx, added
  `whisper-fallback` group
