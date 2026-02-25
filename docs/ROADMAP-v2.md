# Roadmap v2: Learn & Accompany

Revision of the original roadmap, incorporating research from Phase 3
exploration (MediaPipe pose estimation experiments, Gemini numeric quality
output, and architectural discussions about real-time adaptation).

Supersedes the Phase 3 section of `ROADMAP.md`. Phases 1 and 2 in that
document remain accurate as historical record.

---

## Two-Phase Runtime Model

The system operates in two distinct phases at runtime, mirroring how a human
pianist works:

### Phase 1: Learn

Runs **once before playback starts**. All heavy analysis happens here.

The teacher demonstrates and counts an exercise. The system watches and
listens, using every available tool to understand what's about to happen.
The output is a complete `MusicalParameters` — the pianist's understanding
of what to play and how to play it.

| Tool | What it produces | Speed |
|------|-----------------|-------|
| Whisper/WhisperX | Word-level timestamps | ~2x real-time |
| Praat/Parselmouth | Pitch + intensity contours | Instant |
| WhiStress | Per-word stress labels | ~2x real-time |
| Gemini | Exercise ID, meter, structure, numeric quality | ~5-10s API call |
| MediaPipe BlazePose | 33 landmarks × 30 FPS | Real-time on CPU |
| Precision math | BPM, subdivision, counting signature, dynamics | Instant |

After Learn completes, the playback engine has everything it needs to start.

### Phase 2: Accompany

Runs **continuously during playback**. Only lightweight tools stay active.

The pianist is playing. The system keeps watching and listening for changes,
but it is NOT re-analyzing from scratch. It produces **adjustments** relative
to the learned baseline.

| Tool | What it watches for |
|------|-------------------|
| MediaPipe BlazePose | Movement tempo drift, energy changes, quality shifts |
| Audio listener | Verbal cues ("faster", "other side", "stop"), tempo changes |

Everything else is off. Gemini doesn't re-run. Whisper doesn't re-transcribe.
The precision layer might recompute a tempo delta from new movement timestamps,
but the full analysis pipeline is dormant.

---

## The Quality Model

### Six Dimensions, Numeric

Quality is represented as six float values (0.0–1.0), not word descriptors.
Word descriptors ("legato", "sustained") are human-readable but useless to a
playback engine — it would have to map them back to numbers anyway.

```python
@dataclass
class QualityProfile:
    smoothness: float    # 0 = sharp/staccato, 1 = flowing/legato
    energy: float        # 0 = gentle/soft, 1 = explosive/powerful
    groundedness: float  # 0 = aerial/elevated, 1 = earth-connected
    attack: float        # 0 = soft onset, 1 = percussive onset
    weight: float        # 0 = light/buoyant, 1 = heavy/pressing
    sustain: float       # 0 = quick movements, 1 = held positions
```

Reference calibration for each dimension:

| Dimension | 0.0 end | 0.5 mid | 1.0 end |
|-----------|---------|---------|---------|
| smoothness | frappé (0.1) | tendu (0.5) | port de bras (0.9) |
| energy | relevé balance (0.2) | — | grand allegro (1.0) |
| groundedness | sauté (0.1) | tendu (0.6) | grand plié (1.0) |
| attack | adagio (0.1) | — | frappé (0.9) |
| weight | balancé (0.2) | — | grand plié (0.9) |
| sustain | petit allegro (0.1) | — | adagio (0.9) |

These six were chosen because they map to concrete musical decisions a pianist
makes: articulation (smoothness, attack), dynamics (energy, weight), pedaling
(sustain), and register/voicing (groundedness). The set will evolve as the
playback engine matures.

### Two Sources, One Output

Both Gemini and MediaPipe can produce quality numbers. They measure the same
things but from different vantage points:

| | Gemini | MediaPipe |
|---|---|---|
| **How** | Semantic understanding of video | Joint position math |
| **Strength** | Knows what a fondu *should* feel like | Measures what *actually* happened |
| **Weakness** | May idealize (rates smoother than reality) | Needs exercise context to interpret |
| **Example** | smoothness: 0.7 | jerk ratio: 6.1 → smoothness: ~0.4 |

A new KEEP module (`precision/dynamics.py`) synthesizes both into the final
`QualityProfile`. The synthesis strategy is TBD (weighted average, Bayesian
prior + update, MediaPipe override when high-confidence), but the key property
is: **the playback engine sees one set of numbers, not two opinions.**

### Quality in the Accompany Phase

During accompaniment, `QualityProfile` provides the baseline. Real-time
adjustments are deltas on the same dimensions:

```python
@dataclass
class Adjustment:
    tempo_factor: float = 1.0     # 1.05 = "5% faster than baseline"
    energy_delta: float = 0.0     # +0.2 = "more energy than baseline"
    smoothness_delta: float = 0.0
    # ... other quality dimensions as needed
    cue: str | None = None        # "other_side", "stop", "prep", etc.
```

The playback engine holds the learned `MusicalParameters` and applies the
latest `Adjustment` on top. Each adjustment is relative, small, and cheap
to produce.

---

## Updated Type Design

### MusicalParameters (the contract)

The stable interface between perception and playback. Produced once during
Learn. Everything upstream fills it in; everything downstream consumes it.

```python
@dataclass
class MusicalParameters:
    # What to play
    exercise: ExerciseDetectionResult | None
    meter: Meter | None                      # was dict | None
    structure: PhraseStructure | None        # was dict | None

    # How to play it — from voice
    tempo: TempoResult | None
    subdivision: SubdivisionResult | None
    counting_signature: CountingSignature | None

    # How to play it — synthesized from all sources
    quality: QualityProfile | None           # was dict | None

    # Evidence (intermediate data, useful for debugging/display)
    words: list[TimestampedWord]
    markers: list[TimedMarker]
    stress_labels: list[tuple[str, int]] | None
```

Notable changes from current:
- `quality` becomes `QualityProfile` (six floats, not word descriptors)
- `meter` becomes a proper `Meter` dataclass (not dict)
- `structure` becomes a proper `PhraseStructure` dataclass (not dict)
- Future: `dynamics` evidence from MediaPipe may be stored separately for
  debugging, but the synthesized result lives in `quality`

### New Types

```python
@dataclass
class Meter:
    beats_per_measure: int   # 2, 3, 4, or 6
    beat_unit: int           # 4 (quarter note) or 8 (eighth note)

@dataclass
class PhraseStructure:
    counts: int              # Total counts in one full phrase (16, 32)
    sides: int               # 1 (one-sided) or 2 (both sides)

@dataclass
class QualityProfile:
    smoothness: float        # 0-1
    energy: float            # 0-1
    groundedness: float      # 0-1
    attack: float            # 0-1
    weight: float            # 0-1
    sustain: float           # 0-1
```

---

## Implementation Plan

### Phase 3a: Solidify Types and Quality Model

Low-risk changes to the existing codebase. No new dependencies.

1. **Add `QualityProfile`, `Meter`, `PhraseStructure` to `types.py`.**
   Replace the `dict | None` stubs in `MusicalParameters`.

2. **Update Gemini schema to request numeric quality.**
   The prompt already works (tested: Gemini returns calibrated 0-1 floats
   with exercise-aware reasoning). Update `_RESPONSE_SCHEMA` and
   `_parse_response()` in `gemini.py`.

3. **Update `analyze.py` and `__main__.py`** to produce and display the
   new types.

4. **Tests:** Extend `test_gemini_merge.py` with tests for the new types.
   No API key needed — test with hardcoded data like existing tests.

### Phase 3b: Add Pose Estimation

New DISPOSABLE perception modules.

1. **`perception/pose.py`** — MediaPipe BlazePose wrapper.
   - Input: video file path
   - Output: landmark time series (numpy arrays)
   - Handles the model download, video frame iteration, and landmark
     extraction. Thin wrapper, no analysis.

2. **`precision/dynamics.py`** — KEEP module. Pure math.
   - Input: landmark time series (from any pose model)
   - Output: `QualityProfile` (the six numeric dimensions)
   - Computes smoothness from jerk, energy from velocity, groundedness
     from hip height, etc.
   - Source-agnostic: works with MediaPipe today, any 33-point pose
     model tomorrow.

3. **Update `precision/dynamics.py` to synthesize Gemini + pose.**
   Takes Gemini's `QualityProfile` and pose-derived `QualityProfile`,
   produces a single merged `QualityProfile`. Strategy TBD but likely:
   Gemini as prior, MediaPipe as evidence that adjusts.

4. **Wire into `analyze.py`.**
   When input is video: run pose estimation, compute dynamics, merge
   with Gemini quality. When input is audio-only: use Gemini quality
   as-is (no pose data available).

### Phase 3c: Real-Time Accompaniment (future)

Design the Accompany phase. This is where MediaPipe runs continuously
and the system produces `Adjustment` deltas.

1. **Define `Adjustment` type** in `types.py`.

2. **`accompany.py`** — new orchestration module (like `analyze.py` is
   for Learn). Runs the event loop: MediaPipe frames in, adjustments out.

3. **Audio listener** — streaming keyword detection for verbal cues.
   Might be a streaming Whisper instance, a lightweight wake-word model,
   or Gemini's streaming API. Research needed.

4. **Change detectors** — modules that watch continuous signals and emit
   adjustments when something meaningful changes (tempo drift beyond
   threshold, energy shift, verbal cue detected).

This phase depends on the playback engine existing to consume the output.

---

## Module Map (target state)

```
DISPOSABLE (perception — thin wrappers, will be swapped):
  perception/whisper.py       Transcription + alignment → TimestampedWord[]
  perception/prosody.py       Praat pitch/intensity → contours
  perception/whistress.py     Stress detection → labels
  perception/gemini.py        Multimodal analysis → exercise, quality, meter, structure
  perception/pose.py          MediaPipe → landmark time series     ← NEW

KEEP (precision — pure math, tested, source-agnostic):
  precision/tempo.py          BPM from any timestamped events
  precision/subdivision.py    Duple/triplet from interval ratios
  precision/signature.py      Prosodic weight profile
  precision/dynamics.py       Quality profile from any motion signal ← NEW

SCAFFOLDING → DELETE (when Gemini fully trusted):
  scaffolding/markers.py      → gemini.py word classification
  scaffolding/exercise.py     → gemini.py exercise detection

ORCHESTRATION:
  analyze.py                  Learn phase — runs all tools, produces MusicalParameters
  accompany.py                Accompany phase — runs lightweight tools, produces Adjustments ← FUTURE
```

---

## Open Questions

1. **Synthesis strategy for Gemini + MediaPipe quality.** Weighted average?
   Gemini as prior with MediaPipe evidence? MediaPipe override when its
   confidence is high? Needs experimentation with more video samples.

2. **Which MediaPipe signals map to which quality dimensions?** Initial
   mapping (jerk → smoothness, velocity → energy, hip Y → groundedness)
   worked for fondu. Needs validation across exercise types.

3. **What triggers an Adjustment during Accompany?** Threshold-based
   change detection? Fixed polling interval? Event-driven from specific
   signals? Depends on playback engine latency requirements.

4. **Audio listener for verbal cues.** Streaming Whisper? Wake-word model?
   Gemini streaming API? Different latency/accuracy tradeoffs.

5. **Optimal video FPS.** MediaPipe runs at 30 FPS easily, but fast
   exercises at 120 BPM have beats every 0.5s. Is 30 FPS overkill for
   quality measurement (vs timing)? Could downsample to save CPU during
   Accompany.

6. **More video samples needed.** All experiments so far are from one
   18-second clip. Need to test across: different exercises (plié, tendu,
   frappé, adagio), different camera angles, different body types, and
   different demonstration styles (marking vs full-out).

---

## Research Notes

### MediaPipe Findings (from Phase 3 exploration)

- **Detection rate:** 534/534 frames (100%) on the test video
- **Upper body landmarks** (shoulders, hips, wrists): ~100% visibility
- **Lower body landmarks** (knees): 46-73% visibility (side-view occlusion)
- **Hip Y range:** 0.507–0.565 (small but measurable for fondu)
- **Jerk ratio:** 6.1x — indicates sharper transitions than Gemini suggested
- **Processing speed:** real-time on M1 Max CPU, no GPU needed

### Gemini Numeric Quality (tested)

Gemini reliably produces calibrated 0-1 floats when given:
- Exercise-specific reference examples in the schema descriptions
- An explicit instruction to rate what it sees, not what the exercise
  should ideally look like

Example output for the test video (battement fondu):
```
smoothness: 0.7, energy: 0.4, groundedness: 0.8,
attack: 0.4, weight: 0.5, sustain: 0.7
```

### Key Insight: Gemini Idealizes, MediaPipe Measures

Gemini rated smoothness at 0.7 ("fluid arm transitions"). MediaPipe's jerk
ratio of 6.1 suggests actual smoothness closer to 0.4 (sharp transitions
between sustained positions). Both are "correct" — Gemini describes the
character of the exercise, MediaPipe describes the execution. The precision
layer needs to decide which matters more for the pianist.

---

## Sources

All sources from the original ROADMAP.md remain relevant. Additional:

- MediaPipe PoseLandmarker Tasks API: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- Gemini structured output with numeric schemas: https://ai.google.dev/gemini-api/docs/structured-output
