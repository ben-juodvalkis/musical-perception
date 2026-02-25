# ADR-003: Typed Quality Model (Phase 3a)

**Date:** 2026-02-25
**Status:** Accepted

## Context

Phase 2 (ADR-002) introduced Gemini's qualitative output as `dict | None`
stubs in `MusicalParameters` — meter, quality, and structure were untyped
dictionaries. This was deliberate: get Gemini working first, solidify types
later.

The roadmap (ROADMAP-v2.md) defines a six-dimension numeric quality model
that a playback engine can consume directly. Word descriptors ("legato",
"sustained") are human-readable but useless to a playback engine — it would
have to map them back to numbers anyway.

Phase 3b (pose estimation) will need `QualityProfile` to exist as a stable
type so that `precision/dynamics.py` can synthesize Gemini + MediaPipe quality
into a single output. This phase creates that type.

### Validation

Gemini reliably produces calibrated 0-1 floats when given exercise-specific
reference points in the schema descriptions. Tested against an 18-second
développé video:

```
smoothness: 0.80, energy: 0.40, groundedness: 0.70,
attack: 0.20, weight: 0.30, sustain: 0.90
```

These numbers are sensible: développé is smooth, sustained, with soft onset
and light weight — distinct from the earlier fondu test (`smoothness: 0.7,
attack: 0.4, sustain: 0.7`) which had sharper transitions and less sustain.

## Decisions

### 1. Three new dataclasses in `types.py`

```python
@dataclass
class Meter:
    beats_per_measure: int   # 2, 3, 4, or 6
    beat_unit: int           # 4 or 8

@dataclass
class PhraseStructure:
    counts: int              # 16, 32, etc.
    sides: int               # 1 or 2

@dataclass
class QualityProfile:
    smoothness: float        # 0 = sharp/staccato, 1 = flowing/legato
    energy: float            # 0 = gentle/soft, 1 = explosive/powerful
    groundedness: float      # 0 = aerial/elevated, 1 = earth-connected
    attack: float            # 0 = soft onset, 1 = percussive onset
    weight: float            # 0 = light/buoyant, 1 = heavy/pressing
    sustain: float           # 0 = quick movements, 1 = held positions
```

### 2. Replace `dict | None` stubs with typed fields

Both `GeminiAnalysisResult` and `MusicalParameters` changed from:

```python
meter: dict | None       # {beats_per_measure: int, beat_unit: int}
quality: dict | None     # {descriptors: list[str]}
structure: dict | None   # {counts: int, sides: int}
```

to:

```python
meter: Meter | None
quality: QualityProfile | None
structure: PhraseStructure | None
```

This is a breaking change to the `MusicalParameters` contract. Any downstream
consumer that accessed `result.quality["descriptors"]` or
`result.meter["beats_per_measure"]` must switch to attribute access
(`result.quality.smoothness`, `result.meter.beats_per_measure`).

### 3. Gemini schema requests numeric quality, not word descriptors

The `_RESPONSE_SCHEMA` quality section changed from an array of descriptor
strings to six `NUMBER` fields with exercise-specific calibration examples
in each description. The prompt instructs Gemini to rate what it actually
observes, not what the exercise should ideally look like.

Calibration reference points (from ROADMAP-v2):

| Dimension | 0.0 end | 0.5 mid | 1.0 end |
|-----------|---------|---------|---------|
| smoothness | frappé (0.1) | tendu (0.5) | port de bras (0.9) |
| energy | relevé balance (0.2) | — | grand allegro (1.0) |
| groundedness | sauté (0.1) | tendu (0.6) | grand plié (1.0) |
| attack | adagio (0.1) | — | frappé (0.9) |
| weight | balancé (0.2) | — | grand plié (0.9) |
| sustain | petit allegro (0.1) | — | adagio (0.9) |

### 4. Parsing defaults to 0.5 for missing dimensions

If Gemini omits a quality dimension (unlikely with `required` in the schema,
but defensive), `_parse_response()` defaults to 0.5 (midpoint). This is
neutral and won't bias the playback engine.

## Consequences

- **`MusicalParameters.quality` is now numeric** — downstream consumers get
  six floats they can use directly for articulation, dynamics, pedaling, and
  register decisions
- **Breaking change** for any code that accessed quality/meter/structure as
  dicts. No external consumers exist yet, so impact is zero.
- **Phase 3b unblocked** — `QualityProfile` is the target type for
  `precision/dynamics.py` to synthesize Gemini + MediaPipe quality
- **Six dimensions may evolve** — the roadmap notes this set will change as
  the playback engine matures. Adding a dimension is additive (new field with
  default); removing one is breaking.

## Files Changed

- `types.py` — added `Meter`, `PhraseStructure`, `QualityProfile`; updated
  `GeminiAnalysisResult` and `MusicalParameters` from `dict | None` to typed
- `perception/gemini.py` — updated `_RESPONSE_SCHEMA` (six numeric fields
  replacing descriptor array), `_PROMPT` (calibration instructions),
  `_parse_response()` (constructs dataclasses instead of dicts)
- `__main__.py` — updated CLI display for attribute access and six quality
  dimensions
- `tests/test_gemini_merge.py` — added 5 tests for new types and their
  integration with merge logic

## Picking Up Phase 3b

The next developer should read `ROADMAP-v2.md` Phase 3b. The key tasks:

1. **`perception/pose.py`** (DISPOSABLE) — MediaPipe BlazePose wrapper.
   Input: video path. Output: landmark time series (numpy arrays).

2. **`precision/dynamics.py`** (KEEP) — Pure math. Input: landmark time
   series. Output: `QualityProfile`. Computes smoothness from jerk, energy
   from velocity, groundedness from hip height, etc.

3. **Synthesis** — `dynamics.py` takes Gemini's `QualityProfile` + pose-derived
   `QualityProfile`, produces a single merged `QualityProfile`. Strategy TBD
   (see ROADMAP-v2 Open Questions).

4. **Wire into `analyze.py`** — when input is video, run pose + dynamics +
   merge. When audio-only, use Gemini quality as-is.

The `QualityProfile` type is ready. The open question is the synthesis
strategy (weighted average vs Bayesian prior vs MediaPipe override).
