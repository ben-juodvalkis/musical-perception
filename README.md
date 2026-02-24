# musical-perception

Extract structured musical parameters from audio — tempo, subdivision,
exercise type, and counting signature — for the AI accompanist system.

This package is the **perception + precision layers**. It takes audio input
and outputs structured musical information. No playback.

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

## Output: MusicalParameters

```
tempo              — BPM, confidence, raw intervals
subdivision        — duple / triplet / none
meter              — beats per measure, beat unit (future)
exercise           — detected exercise type (scaffolding)
quality            — musical character descriptors (future)
counting_signature — prosodic weight profile
structure          — counts, sides (future)
```

## Architecture

Three layers, separated by how long the code will last:

- **Precision** (`precision/`): Pure math. Tempo calculation, subdivision
  analysis, signature computation. Stable, well-tested, rarely changes.
- **Perception** (`perception/`): Thin wrappers around AI models (currently
  Whisper). Disposable — will be swapped as better models emerge.
- **Scaffolding** (`scaffolding/`): Hand-built analysis (exercise detection,
  word classification) that multimodal AI will replace.

## Development

```bash
pip install -e ".[all,dev]"
pytest
```
