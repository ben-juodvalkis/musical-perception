# Musical Perception

Python package that extracts structured musical parameters from audio input.
Part of the AI accompanist system — this is the perception + precision layers,
without any playback.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
export GEMINI_API_KEY=your-key-here
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
│   ├── signature.py      # Counting signature computation
│   └── dynamics.py       # Movement quality from pose landmarks
└── perception/           # DISPOSABLE — thin model wrappers
    ├── whisper.py        # Whisper transcription (word timestamps)
    ├── prosody.py        # Praat pitch/intensity extraction
    ├── whistress.py      # WhiStress stress detection
    ├── gemini.py         # Gemini multimodal analysis (words, exercise, meter, quality, structure)
    └── pose.py           # MediaPipe pose estimation
```

## Architecture Labels

- **KEEP**: Precision math and signal processing. Pure functions. Test thoroughly.
- **DISPOSABLE**: Perception wrappers around AI models. Will be swapped.
  Don't build elaborate abstractions.

## How It Works

Whisper owns word **timestamps**. Gemini owns word **classification** (beat/and/ah)
and qualitative analysis (exercise, meter, quality, structure). The merge step
in `analyze.py` pairs Gemini classifications with Whisper timestamps to produce
`TimedMarker` objects, which feed the precision layer for tempo and subdivision.

## Key Types

- `MusicalParameters` — the stable output schema (the contract)
- `TempoResult` — BPM + confidence + raw intervals
- `SubdivisionResult` — duple/triplet/none + confidence
- `CountingSignature` — prosodic weight profile
- `TimestampedWord` — word + start/end time (from transcription)
- `TimedMarker` — classified rhythmic marker with beat association
- `GeminiAnalysisResult` — bridge type from Gemini (words + exercise + meter + quality + structure, no timestamps)

## Running Tests

```bash
pytest
```

Tests for precision code use hardcoded data (no audio files, no models needed).

## Dependencies

Core (always installed): numpy

Required:
- `GEMINI_API_KEY` environment variable (get one at https://aistudio.google.com/apikey)

Optional groups:
- `pip install -e .`            — precision math only
- `pip install -e ".[whisper]"` — add Whisper transcription
- `pip install -e ".[prosody]"` — add prosody extraction
- `pip install -e ".[gemini]"`  — add Gemini multimodal analysis
- `pip install -e ".[all]"`     — everything

## Usage

```bash
python -m musical_perception video/your-file.mov
python -m musical_perception audio/your-file.aif --signature --pose
```

Flags:
- `--signature` — extract counting signature (requires prosody deps)
- `--stress` — detect stress labels (requires WhiStress)
- `--pose` — estimate pose from video (requires pose deps, video only)
