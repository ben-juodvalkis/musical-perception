# ADR-006: Onset-Based Tempo Detection and BPM Normalization

**Date:** 2026-02-28
**Status:** Accepted

## Context

When ballet teachers count using step names instead of numbers ("tendu front
brush through" vs "1 2 3 4"), Gemini classifies all words as `"none"` and the
Gemini-based tempo pipeline produces no usable result. Testing on
`Exercise 1 Demo.m4v` (a tendu exercise in 3/4 at ~117 BPM) confirmed:

- The old prompt only recognized counted numbers as beats → 0-3 markers detected
- Gemini's BPM estimate (40.5) was at the measure level, not the beat level
- The rhythmic signal IS present in Whisper's word onset timestamps

Additionally, ballet class tempos almost always fall in the 70–140 BPM range at
the beat level. Raw BPM values outside this range indicate the detector locked
onto the wrong level of the metric hierarchy (subdivision or measure).

## Decision

### 1. Onset-based tempo detector (`precision/rhythm.py`)

A new KEEP precision module that estimates tempo from word onset regularity,
independent of Gemini word classification. Algorithm:

- Sliding-window analysis (3s window, 0.5s step) over word onset times
- Windows with low coefficient of variation (CV < 0.4) are "rhythmic"
- Overlapping rhythmic windows are merged into consolidated sections
- Final BPM via duration-weighted median across sections
- Confidence from coverage, consistency, regularity, and histogram agreement

This runs as a **parallel signal** alongside Gemini-based tempo — both always
appear in `MusicalParameters`. The onset detector runs first (needs only Whisper
timestamps), and its BPM feeds into the Gemini prompt as a calibration hint.

### 2. Updated Gemini prompt

The prompt now instructs Gemini to recognize step names spoken in rhythm as
beats (not just counted numbers), and includes the onset-detected BPM as
context to help Gemini calibrate its analysis.

### 3. BPM normalization (`normalize_tempo()`)

A precision function that snaps any raw BPM into the 70–140 range by
multiplying or dividing by 2 or 3. Returns both the normalized BPM and a
multiplier indicating what transformation was applied. The pipeline picks the
best raw BPM (preferring onset tempo when confident), normalizes it, and stores
the result in `MusicalParameters.normalized_bpm`.

## Results on Exercise 1 Demo (ground truth: 3/4 at ~117 BPM)

| Signal | Before | After |
|--------|--------|-------|
| Onset-based BPM | n/a | 115.1 (1.6% error) |
| Gemini beats detected | 0–3 | 12–14 |
| Gemini raw BPM | 40.5 | 65.1 |
| Normalized BPM | n/a | 115.1 |

## What this does NOT solve

**Meter detection.** Gemini consistently reports 4/4 instead of the correct 3/4.
This is a musically defensible ambiguity — 3/4 at 120 BPM and 4/4 at 40 BPM
with triplet subdivision produce the same rhythmic surface. Resolving this
requires accent pattern analysis (louder/longer on beat 1 vs 2 vs 3) which is
a distinct problem. See GitHub issue for tracking.

## Consequences

- `MusicalParameters` gains three new fields: `onset_tempo`, `normalized_bpm`,
  `tempo_multiplier`
- Downstream consumers should prefer `normalized_bpm` for tempo
- The onset detector has zero AI dependencies (numpy only)
- Gemini prompt changes are backward-compatible (still works with number counting)
