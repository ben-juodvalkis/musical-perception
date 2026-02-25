# ADR-002: Replace Scaffolding with Gemini (Phase 2)

**Date:** 2026-02-24
**Status:** Accepted

## Context

The scaffolding layer (`markers.py`, `exercise.py`) uses hand-built word lists
and pattern matching for word classification and exercise detection. These are
fragile:

1. **`markers.py` requires constant tuning.** Every new accent or voice
   produces Whisper transcription variants that need to be added to word lists.

2. **`exercise.py` maintains a large dictionary** of French ballet terms and
   their misheard equivalents ("grand batman", "frapeze", etc.).

3. **Both are labeled SCAFFOLDING** — explicitly meant to be replaced by a
   multimodal model.

Experiments with Gemini showed it correctly classifies rhythmic words from
audio context (not just text matching) and identifies exercises from combined
audio + video.

### Experiment Results

Three configurations tested with an 18-second dance teacher video:

| Config | Audio tokens | Caught speech | Words |
|--------|-------------|---------------|-------|
| 2.5 Flash (video) | 576 | All including "Tell me when to go" | 24 |
| 3.1 Pro (video only) | 0 | Missed conversational speech | 21 |
| 3.1 Pro (video + separate audio) | 446 | All + caught "a" subdivision | 25 |

Key findings:
- Schema-enforced JSON works perfectly — no parsing failures
- Exercise identification varies across runs — not reliable enough to replace
  the precision layer's tempo calculation
- BPM estimates vary wildly (96–127 across runs)
- Gemini 3.x models have a known audio regression when processing video files

## Decisions

### 1. Add `perception/gemini.py` as DISPOSABLE wrapper

Thin module: upload media, send structured prompt, parse JSON response.
Same load/analyze pattern as `whisper.py`.

### 2. Whisper owns timestamps and tempo; Gemini owns qualitative analysis

Gemini cannot provide word-level timestamps. The precision layer needs
timestamps for BPM and subdivision. Whisper + scaffolding markers always run
to produce `TimedMarker` list with timestamps — this feeds the precision layer
for tempo and subdivision.

Gemini provides:
- **Exercise detection** — exercise type identified from audio + video context
- **Meter** — time signature (e.g. 3/4, 4/4, 6/8)
- **Quality** — 2–5 musical/movement style descriptors (e.g. "sustained",
  "flowing", "marcato")
- **Structure** — phrase length in counts and number of sides
- **Word classifications** — rhythmic role of each word (beat/and/ah/none)
- **Counting structure** — qualitative observation of counting pattern

This means `--gemini` does NOT eliminate the Whisper dependency. Whisper
timestamps remain the source of truth for the precision layer.

### 3. Always extract and send audio separately from video

Gemini 3.x has a known regression where audio tokens = 0 when processing
video files. Extracting audio via ffmpeg and sending it alongside the video
ensures reliable audio processing across model versions. Even for 2.5 Flash
(which handles video audio natively), sending both is harmless.

### 4. Bridge type `GeminiAnalysisResult`, not reuse `TimedMarker`

`TimedMarker` requires a `timestamp: float`. Forcing 0.0 or NaN would corrupt
downstream precision math. A separate `GeminiWord` type makes the "no
timestamps" constraint visible at the type level.

`GeminiAnalysisResult` carries all Gemini output: word classifications,
exercise detection, counting structure, meter, quality, and structure. The
`analyze()` function unpacks the qualitative fields directly into
`MusicalParameters`.

### 5. Gemini's BPM and subdivision estimates are logged, not used

The precision layer's tempo calculation from actual word timestamps is more
reliable than Gemini's qualitative BPM estimate. Gemini's `counting_structure`
is stored in the result but does not feed into `MusicalParameters`.

### 6. Default to Gemini 2.5 Flash

Despite 3.1 Pro being newer, 2.5 Flash is the better default:
- Reliably processes audio from video files (no workaround needed)
- Stable (not preview)
- Cheaper (~$0.03 per 18s clip)
- Catches all speech content

The `gemini_model` parameter allows overriding for experimentation.

### 7. Merge logic preserved for future use

`_merge_gemini_with_timestamps()` in `analyze.py` implements sequential word
alignment between Gemini classifications and Whisper timestamps. It is not
used in the current flow (Whisper + scaffolding markers own the `TimedMarker`
list), but is retained and tested for a potential future where Gemini's word
classifications replace scaffolding markers entirely.

## Consequences

- **Exercise detection improves** — Gemini understands dance exercises from
  audio + video context, not just pattern matching against a word list
- **New qualitative fields** — meter, quality descriptors, and structure are
  now populated when using `--gemini`, filling in previously stubbed fields
  in `MusicalParameters`
- **Whisper still required** for timestamps when precision math is needed
- **New dependency** — `google-genai` SDK + API key. Optional via `[gemini]`
  dependency group.
- **API cost** — ~$0.03 per 18s clip with 2.5 Flash
- **Scaffolding NOT deleted yet** — remains as fallback for users without
  a Gemini API key

## Files Changed

- `perception/gemini.py` — new module (DISPOSABLE)
- `types.py` — added `GeminiWord`, `GeminiCountingStructure`,
  `GeminiAnalysisResult` bridge types; added `meter`, `quality`, `structure`
  fields to `MusicalParameters`
- `analyze.py` — added `use_gemini` parameter, Gemini qualitative path,
  merge logic (retained but unused)
- `__main__.py` — added `--gemini` CLI flag, display for meter/quality/structure
- `pyproject.toml` — added `[gemini]` dependency group
- `tests/test_gemini_merge.py` — unit tests for merge logic
- `scripts/try_gemini*.py` — experiment scripts used during evaluation
