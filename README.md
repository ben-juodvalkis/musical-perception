# musical-perception

Extract structured musical parameters from audio and video — tempo, subdivision,
exercise type, meter, quality, and counting signature — for the AI accompanist system.

This package is the **perception + precision layers**. It takes audio or video
input and outputs structured musical information. No playback.

## Install

```bash
pip install -e ".[all]"
```

## Usage

```python
from musical_perception import analyze

result = analyze("path/to/counting.wav")
print(f"Tempo: {result.tempo.bpm} BPM ({result.tempo.confidence:.0%} confidence)")
print(f"Subdivision: {result.subdivision.subdivision_type}")
```

With Gemini for exercise detection and qualitative analysis:

```python
result = analyze("path/to/video.mov", use_gemini=True)
print(f"Exercise: {result.exercise.display_name}")
print(f"Meter: {result.meter['beats_per_measure']}/{result.meter['beat_unit']}")
print(f"Quality: {', '.join(result.quality['descriptors'])}")
```

### CLI

```bash
# Whisper + scaffolding (no API key needed)
python -m musical_perception audio/counting.wav

# Gemini multimodal analysis (requires GEMINI_API_KEY)
python -m musical_perception video/class.mov --gemini
```

## Output: MusicalParameters

```
tempo              — BPM, confidence, raw intervals
subdivision        — duple / triplet / none
meter              — beats per measure, beat unit (via Gemini)
exercise           — detected exercise type (scaffolding or Gemini)
quality            — musical character descriptors (via Gemini)
counting_signature — prosodic weight profile
structure          — phrase counts, sides (via Gemini)
```

## Architecture

Three layers, separated by how long the code will last:

- **Precision** (`precision/`): Pure math. Tempo calculation, subdivision
  analysis, signature computation. Stable, well-tested, rarely changes.
- **Perception** (`perception/`): Thin wrappers around AI models — Whisper
  for timestamps, Gemini for qualitative analysis. Disposable — will be
  swapped as better models emerge.
- **Scaffolding** (`scaffolding/`): Hand-built word classification and exercise
  detection. Replaced by Gemini when `--gemini` is used; kept as fallback.

### Gemini Integration

When `--gemini` is enabled, a single Gemini API call provides exercise
detection, meter, quality descriptors, and structure — replacing the
scaffolding layer. Whisper still runs for word timestamps; the precision
layer still computes tempo and subdivision from those timestamps.

Requires a `GEMINI_API_KEY` (get one at https://aistudio.google.com/apikey).
Set it in your environment or in a `.env` file.

See [ADR-002](docs/adr/002-gemini-integration.md) for design rationale.

## Development

```bash
pip install -e ".[all,dev]"
pytest
```
