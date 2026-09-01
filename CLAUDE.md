# Musical Perception

Python package that extracts structured musical parameters from audio input.
Part of the AI accompanist system — this is the perception + precision layers,
without any playback.

## Talking to Ben about this work

Ben directs this research; he is a dance musician, not a coder. He can
judge whether an approach is right — he cannot read a diff to find out.
The documents in this repo (charter, ledger, eval specs) are written in a
dense house style **for agents**, and they stay that way. **What you say
back in chat is not written in that style.** Translate.

Every reply:

- **Lead with what the system now does differently**, in one or two plain
  sentences, before any metric, file, or term of art. "It now finds the
  beat where you'd tap on the frappé clips, but still loses it on
  triplets" — not "R@tac 0.349→0.719 on the step_names slice."
- **Every number gets a so-what.** Direction (better or worse), size in
  human terms ("about 3 clips out of 25", "roughly half a beat late"),
  and whether it's big enough to act on. A bare delta is not a result.
- **Expand jargon on first use each session**, then the short form is
  fine. Never send a paragraph built only out of repo vocabulary.
- **Explain by behavior, not by code.** No function names, no
  file-by-file tours, no syntax. Say what the pipeline does to audio now
  that it didn't before.
- **Detail on request, not by default.** Offer the per-clip table; don't
  dump it.
- **Answer the question he asked, in his words.** Don't silently
  translate it into repo vocabulary and answer that one instead.

None of this softens the findings. A negative result stays a negative
result, stated plainly — plain language means clearer, not vaguer, and
never rounder in his favor.

### House vocabulary, in plain language

- **rung / workstream (W-number)** — a numbered stage of the research
  plan; a workstream is one session's experiment within it.
- **the ledger** — `RESEARCH-LOG.md`, the running diary of every
  experiment, failures included.
- **pre-registration** — writing down what would count as success
  *before* running the experiment, so the result can't be reinterpreted
  after the fact.
- **bless / baseline** — freezing today's scores as the official "where
  we are." Only Ben does this; agents never self-bless.
- **case / trace / beat grid** — a test clip plus its expected answers /
  a frozen recording of a past run so it can be re-scored without the
  media / a hand-tapped list of where the beats actually fall.
- **DEV vs SEALED split** — clips agents may tune against, vs clips only
  Ben scores, kept back so nothing gets quietly fitted to them.
- **pulse / tactus** — the beat you'd tap or clap.
- **onset** — the moment a sound starts.
- **asynchrony** — how early or late our beat sits against the human tap,
  in milliseconds; negative means early.
- **precision / recall / F** — of the beats we called, how many were
  real / of the real beats, how many we caught / one blended score.
  `P_lc`, `R@tac`, `F_lc` are those three measured against hand-tapped
  beats, with sub-beat syllables not counted against us.
- **Acc1 / Acc2** — tempo correct / tempo correct *or* off by a metric
  factor (double or half time).
- **OE1 / OE2** — how far the tempo is off, measured in octaves: 0 is
  exact, ±1 is double or half speed.
- **posterior / lattice / arbitration** — the machinery that weighs
  competing readings of the beat and commits to one.
- **confidence / calibration / ECE** — whether the system's stated
  confidence matches how often it is actually right. High ECE means it
  is confidently wrong.
- **abstention / coverage** — how often it declines to answer, and on
  what share of clips it does answer.
- **provisional vs verified** — labels Ben hasn't checked yet (they are
  reported but never decide pass/fail) vs ones he has.

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
│   ├── dynamics.py       # Movement quality from pose landmarks
│   ├── pulse.py          # Acoustic pulse extractor (peakRate + nuclei regions, rung 2)
│   └── posterior.py      # Factored rhythm posterior — bar-pointer lattice (rung 4, ADR-017)
├── perception/           # DISPOSABLE — thin model wrappers
│   ├── whisper.py        # Whisper transcription (word timestamps)
│   ├── prosody.py        # Praat pitch/intensity extraction
│   ├── whistress.py      # WhiStress stress detection
│   ├── gemini.py         # Gemini multimodal analysis (words, exercise, meter, quality, structure)
│   └── pose.py           # MediaPipe pose estimation
└── annotation/           # Beat-grid tooling (rung 1)
    ├── peakrate.py       # peakRate vowel-onset detector (Oganian & Chang)
    ├── grids.py          # beat-grid YAML format + Audacity label round trip
    └── __main__.py       # tap-assist CLI (generate / to-labels / from-labels)
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
  (`NormalizedTempo.alternates`; primary selection ignores it — ADR-014;
  carries posterior mass as `weight` since ADR-017)
- `GroupingLevel` — one rung of the grouping ladder above the beat
  (`NormalizedTempo.grouping_levels`, ADR-017; a silent rung is absent,
  not zero)
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
python -m musical_perception.evals run --suite tier0,tier1,stage1  # score everything
python -m musical_perception.evals bless                     # promote run to baseline
python -m musical_perception <clip> --record-traces          # freeze a new trace
python -m musical_perception.annotation generate             # provisional beat grids
python -m musical_perception.evals record-pulse              # freeze pulse.json sidecars
```

Cases live in `evals/cases/*.yaml` (field names are a strict subset of
[Vision 08 §8.2](docs/vision/08-benchmark-and-shadow-mode.md)); traces in
`evals/traces/`; beat grids in `evals/grids/` (see
[docs/evals/beat-grids.md](docs/evals/beat-grids.md) — provisional grids
never gate anything); the blessed baseline is `evals/baseline.json` +
[docs/evals/baseline.md](docs/evals/baseline.md). Each trace directory
also carries a `pulse.json` **sidecar** — the rung-2 acoustic pulse
stream, frozen so it replays without the (gitignored) media, see
[docs/evals/pulse-sidecars.md](docs/evals/pulse-sidecars.md); the
`stage1-peakrate` suite scores that stream instead of word starts, gates
nothing like `stage1`, and its numbers carry an anchoring caveat
documented there. A case's `maturity` key
(`provisional` | `verified`, default `verified`) says whether its truth
labels were owner-verified; provisional rows are scored and reported in
their own slice but gate nothing — see
[docs/evals/case-maturity.md](docs/evals/case-maturity.md). The tier-1 pytest gate
fails on ANY outcome change vs the baseline — improvements too; re-bless
to carry the delta. The stage1 suite (pulse P/R/F + signed asynchrony vs
grids) pins no outcomes; tier-1 reporting also carries Acc1/Acc2 and
OE1/OE2 tempo metrics (informational), plus the W12 **factored meter
slice** (`meter_division` + `meter_grouping`, reported beside
`meter_triple` and gating nothing — see
[docs/evals/factored-meter.md](docs/evals/factored-meter.md)). Perception changes are judged by
the eval delta, not by a hand-run on one file.

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
