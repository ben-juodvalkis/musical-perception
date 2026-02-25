# musical-perception

Extract structured musical parameters from audio and video — tempo, subdivision,
exercise type, meter, quality, and counting signature — for the AI accompanist system.

This package is the **perception + precision layers**. It takes audio or video
input and outputs structured musical information. No playback.

## Install

```bash
pip install -e ".[all]"
```

Requires a `GEMINI_API_KEY` (get one at https://aistudio.google.com/apikey).
Set it in your environment or in a `.env` file.

## Usage

```python
from musical_perception import analyze

result = analyze("path/to/video.mov")
print(f"Tempo: {result.tempo.bpm} BPM ({result.tempo.confidence:.0%} confidence)")
print(f"Subdivision: {result.subdivision.subdivision_type}")
print(f"Exercise: {result.exercise.display_name}")
print(f"Meter: {result.meter.beats_per_measure}/{result.meter.beat_unit}")
print(f"Quality: articulation={result.quality.articulation:.2f}")
```

### CLI

```bash
python -m musical_perception video/class.mov
python -m musical_perception audio/counting.wav --signature --pose
```

## Output: MusicalParameters

```
tempo              — BPM, confidence, raw intervals
subdivision        — duple / triplet / none
meter              — beats per measure, beat unit
exercise           — detected exercise type
quality            — articulation, weight, energy (0.0–1.0)
counting_signature — prosodic weight profile
structure          — phrase counts, sides
```

## Architecture

Two layers, separated by how long the code will last:

- **Precision** (`precision/`): Pure math. Tempo calculation, subdivision
  analysis, signature computation, dynamics from pose. Stable, well-tested,
  rarely changes.
- **Perception** (`perception/`): Thin wrappers around AI models — Whisper
  for word timestamps, Gemini for word classification and qualitative analysis,
  MediaPipe for pose estimation. Disposable — will be swapped as better
  models emerge.

Whisper owns word **timestamps**. Gemini owns word **classification** and
qualitative analysis (exercise, meter, quality, structure). The merge step
in `analyze.py` pairs Gemini's classifications with Whisper's timestamps
to produce markers that feed the precision layer for tempo and subdivision.

See [ADR-005](docs/adr/005-remove-scaffolding.md) for the removal of the
original scaffolding layer.

## Development

```bash
pip install -e ".[all,dev]"
pytest
```
