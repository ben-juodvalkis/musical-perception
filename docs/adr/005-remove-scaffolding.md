# ADR-005: Remove Scaffolding Layer (Phase 4)

**Date:** 2026-02-25
**Status:** Accepted

## Context

ADR-002 replaced the scaffolding layer (hand-built word lists and pattern
matching) with Gemini multimodal analysis, but kept scaffolding as a fallback
for users without a Gemini API key. In practice:

1. **Scaffolding was never truly optional.** Even with `--gemini`, `analyze.py`
   called `extract_markers()` from scaffolding unconditionally to produce the
   `TimedMarker` list that feeds tempo and subdivision computation. Gemini
   provided exercise/meter/quality/structure but not the markers themselves.

2. **The merge function existed but was unused.** `_merge_gemini_with_timestamps()`
   was implemented and tested (12 tests in `test_gemini_merge.py`) but never
   called in the production pipeline. It produces the same `list[TimedMarker]`
   type as `extract_markers()`.

3. **Gemini classifications are more accurate.** The scaffolding word lists
   required constant tuning per accent/voice (e.g., Whisper hearing "the"
   instead of "ah"). Gemini understands language context and doesn't need
   word lists.

4. **Maintaining two paths adds complexity for no benefit.** Every change to
   the marker pipeline required testing both the scaffolding and Gemini paths.

## Decisions

### 1. Remove scaffolding entirely

Delete `scaffolding/markers.py`, `scaffolding/exercise.py`, and their tests.
No fallback path — Gemini API key is required.

### 2. Wire `_merge_gemini_with_timestamps()` into the production pipeline

This already-tested function now produces the `TimedMarker` list that feeds
`calculate_tempo()` and `analyze_subdivisions()`. The merge step pairs Gemini's
word classifications (no timestamps) with Whisper's timestamps (no
classification) via sequential word alignment.

### 3. Remove `--gemini` CLI flag

Gemini is always used. The `use_gemini` and `detect_exercise_type` parameters
are removed from `analyze()`. The `gemini_client` and `gemini_model` parameters
remain for pre-loading support.

### 4. Derive signature marker types from Gemini merge

The `--signature` path previously called `classify_marker()` from scaffolding
to pre-classify words for prosody feature extraction. Now it derives marker
types from the Gemini merge results via timestamp lookup:

```python
marker_lookup = {m.timestamp: m.marker_type for m in markers}
marker_types = [marker_lookup.get(w.start) for w in words]
```

## Consequences

- `GEMINI_API_KEY` is now required to run analysis (was optional before)
- The precision-only install (`pip install -e .`) still works for math functions
  but `analyze()` requires Gemini deps
- 13 scaffolding tests removed; 12 Gemini merge tests remain and cover the
  same merge logic that now runs in production
- Architecture simplifies from three layers (KEEP/DISPOSABLE/SCAFFOLDING)
  to two (KEEP/DISPOSABLE)

## Files Changed

- `src/musical_perception/analyze.py` — rewired to use Gemini merge for markers
- `src/musical_perception/__main__.py` — removed `--gemini` flag
- `CLAUDE.md` — updated structure, removed scaffolding references
- Deleted: `src/musical_perception/scaffolding/` (3 files)
- Deleted: `tests/test_markers.py`, `tests/test_exercise.py`
