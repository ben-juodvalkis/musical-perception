# Musical Perception

Python package that extracts structured musical parameters from audio input.
Part of the AI accompanist system — this is the perception + precision layers,
without any playback.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
python -m musical_perception audio/your-file.aif
```

## Package Structure

```
src/musical_perception/
├── types.py              # All data types + MusicalParameters schema
├── analyze.py            # Main entry point: analyze(audio_path)
├── precision/            # KEEP — pure math, rarely changes
│   ├── tempo.py          # BPM from timestamps
│   ├── subdivision.py    # Duple/triplet classification
│   └── signature.py      # Counting signature computation
├── perception/           # DISPOSABLE — thin model wrappers
│   ├── whisper.py        # Whisper transcription
│   └── prosody.py        # Praat pitch/intensity extraction
└── scaffolding/          # SCAFFOLDING — AI will replace
    ├── markers.py        # Word classification (beat/and/ah)
    └── exercise.py       # Exercise type detection
```

## Architecture Labels

- **KEEP**: Precision math and signal processing. Pure functions. Test thoroughly.
- **DISPOSABLE**: Perception wrappers around AI models. Will be swapped.
  Don't build elaborate abstractions.
- **SCAFFOLDING**: Hand-built analysis that multimodal AI will replace.
  Good enough for now. Don't invest further.

## Key Types

- `MusicalParameters` — the stable output schema (the contract)
- `TempoResult` — BPM + confidence + raw intervals
- `SubdivisionResult` — duple/triplet/none + confidence
- `CountingSignature` — prosodic weight profile
- `TimestampedWord` — word + start/end time (from transcription)
- `TimedMarker` — classified rhythmic marker with beat association

## Running Tests

```bash
pytest
```

Tests for precision code use hardcoded data (no audio files, no models needed).

## Adding New Word Patterns

When Whisper transcribes subdivision words unexpectedly, add them
to the word lists in `scaffolding/markers.py`:
- `AND_WORDS` — for "and" sounds
- `AH_WORDS` — for "ah" sounds (Whisper often hears "the", "da", "ta")

## Dependencies

Core (always installed): numpy

Optional groups:
- `pip install -e .`            — precision math only
- `pip install -e ".[whisper]"` — add Whisper transcription
- `pip install -e ".[prosody]"` — add prosody extraction
- `pip install -e ".[all]"`     — everything
