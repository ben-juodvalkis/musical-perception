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

## Autonomous Research Loop

Agent sessions working on the rhythm-core reset (ADR-016) MUST read
[docs/research/agent-charter.md](docs/research/agent-charter.md) before
doing anything: it defines the CURRENT RUNG, the rules (agent/* branches
only; `evals/cases/`, `evals/traces/`, `evals/baseline.json`, and the
scorer code are untouchable in pipeline rungs; never run `evals bless`),
and the goal ladder. Every session appends a dated entry to
[docs/research/RESEARCH-LOG.md](docs/research/RESEARCH-LOG.md) before
finishing — including negative results and blocked states.

## How It Works

Whisper owns word **timestamps** (and the tokenization: a ballet-vocabulary
`initial_prompt` is on by default, `large-v3-turbo` is the default model).
Gemini owns word **classification** (beat/and/ah) and qualitative analysis
(exercise, meter, quality, structure). `analyze.py` sends Whisper's indexed
transcript to Gemini, which classifies those exact tokens; the merge is an
index lookup producing `TimedMarker` objects, which feed the precision layer
for tempo and subdivision (see ADR-010).

Gemini also provides **per-phrase quality** — each phrase in a combination is rated
on articulation, weight, and energy, and flagged as primary or transitional. The
aggregate `QualityProfile` is computed from primary phrases only, filtering out
port de bras breaks and transitions that would skew the overall character.

## Key Types

- `MusicalParameters` — the stable output schema (the contract)
- `TempoResult` — BPM + confidence + raw intervals
- `SubdivisionResult` — duple/triplet/none + confidence
- `TempoCandidate` — one member of a raw pulse's metric-level family
  (`NormalizedTempo.alternates`; primary selection ignores it — ADR-014)
- `CountingSignature` — prosodic weight profile
- `TimestampedWord` — word + start/end time (from transcription)
- `TimedMarker` — classified rhythmic marker with beat association
- `PhraseQuality` — per-phrase quality ratings (articulation, weight, energy, primary flag)
- `QualityProfile` — aggregate quality dimensions + optional per-phrase breakdown
- `GeminiAnalysisResult` — bridge type from Gemini (words + exercise + meter + quality + structure, no timestamps)

## Running Tests

```bash
pytest
```

Tests for precision code use hardcoded data (no audio files, no models needed).

Tests answer "is it broken?"; **evals** answer "is it better?" — the harness
(tiers 0–1: synthetic sweep + frozen-trace replay) lives in
`src/musical_perception/evals/` per [ADR-009](docs/adr/009-evaluation-harness.md):

```bash
python -m musical_perception.evals run --suite tier0,tier1   # score everything
python -m musical_perception.evals bless                     # promote run to baseline
python -m musical_perception <clip> --record-traces          # freeze a new trace
```

Cases live in `evals/cases/*.yaml` (field names are a strict subset of
[Vision 08 §8.2](docs/vision/08-benchmark-and-shadow-mode.md)); traces in
`evals/traces/`; the blessed baseline is `evals/baseline.json` +
[docs/evals/baseline.md](docs/evals/baseline.md). The tier-1 pytest gate
fails on ANY outcome change vs the baseline — improvements too; re-bless
to carry the delta. Perception changes are judged by the eval delta, not
by a hand-run on one file.

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
