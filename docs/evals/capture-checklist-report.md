# Capture Checklist — Completion Report

**To:** whoever designed `capture-checklist.xlsx` / `teacher-letter.md`
**From:** the processing session that ran clips 1–24, 2026-08-06 to 2026-08-08
**Status:** All 24 rows recorded, QC'd, labeled, traced, eval-cased, and
blessed into `evals/baseline.json`. `pytest` green throughout.

## Headline: the checklist found a real, provably underdetermined defect

The most important thing to come out of this corpus isn't a single bug —
it's a new architecture-level finding, written up as
**[ADR-014](../adr/014-tempo-metric-level-ambiguity.md)** (proposed, not yet
implemented).

`normalize_tempo()` snaps any raw BPM reading into a 70–140 "ballet class"
band by multiplying or dividing by 2 or 3 (ADR-006/007). It's a hard,
single-answer rule. Two clips on this sheet broke it in a way that's more
interesting than a normal miss:

- **Clip 12** (half-tempo, ★): Ben marked a genuinely slow, metronome-confirmed
  60 BPM. The raw onset reading came back at 62.2 BPM, 100% coverage,
  **CV=0.00** — essentially exact. The band-snap doubled it to 124.4 anyway,
  because 62.2 sits just under the 70 floor.
- **Clip 13** ("the most valuable clip on this sheet"): a fast frappé
  combination at a confirmed 160 BPM. Raw onset read 161.8 BPM, 100%
  coverage, CV=0.05 — again essentially exact — and got halved to 80.9
  because it sits just above the 140 ceiling.

Same defect, mirrored on both sides of the band. But the real finding is
sharper than "the band is miscalibrated": clip 12 and the tier-0 synthetic
case `t0-3-4-half` (a *different*, already-known-failing scenario —
marking at half speed while the true tempo stays fast) produce
**audio-identical raw signals**. A perfectly regular slow reading is
consistent with both "this genuinely is slow" and "this is half-tempo
marking of something faster," and there is no way to tell them apart from
onset regularity alone — the information just isn't in the pulse. That's
exactly the tension your checklist's clip 12 instructions were fishing for
("write BOTH numbers... the real class tempo and the slower speed you
actually spoke at").

ADR-014 proposes reporting the metric-level candidate family (the raw
reading, ×2, ×3, ÷2, ÷3 — whichever are musically sane) instead of
silently collapsing to one answer, so a genuinely-slow or genuinely-fast
true tempo is at least *discoverable* even when it isn't picked as
primary. It's a design doc only right now — no code changed, primary
selection is untouched, so nothing in the corpus regressed because of it.

## Coverage

All 24 rows filled in: Recorded/BPM/Counts/Take-notes columns complete,
one commit per clip on `main`, each with its own eval case under
`evals/cases/` and frozen trace under `evals/traces/`. A handful of
process notes worth knowing about:

- **Clip 17** (vocables) got re-recorded mid-session — the first export had
  two overlapping tracks. Its case file documents *both* takes: the
  overlapping export had Whisper hallucinate a clean, timestamped
  "one and two three..." transcript onto pure vocable sounds (wrong text,
  timing survived, scored all-green by accident); the corrected export
  exposed something worse — Whisper collapses genuine non-lexical bursts
  into a single un-timestamped token with zero rhythmic granularity.
- Two bounce files landed with corrupted filenames in the source folder
  (a truncated leading "ri" on clip 23, a trailing space on clip 24) —
  flagged and copied in under the correct name each time rather than
  guessed at.
- Clip 20's count label got corrected mid-flow (16 → 8) after a self-catch;
  worth double-checking any "8 or 16, write which" style rows for similar
  slips if this checklist gets reused.

## Findings by category (beyond ADR-014)

- **Sparse/silent markers (clips 5–8, step_names Part 1).** Ben's natural
  marking habit leaves some counts wordless. Gemini has no concept of a
  skipped count and numbers every *spoken* word as consecutive, so sparse
  markers read as a slower tempo and undercount the phrase — this shows up
  across every meter tested (2/4, 3/4, 4/4, 6/8), so it's a `step_names`
  property, not a meter-specific one.
- **Waltz-precedence family (clips 2, 6, 13, 20)** — the 3/4-vs-triplet
  ambiguity ADR-006/007/013 already track. Clip 20 landed a near-miss:
  the marker path's confidence came in at 59%, one point under ADR-013's
  60% arbitration gate. Worth another data point if that threshold gets
  revisited — ADR-013 itself calls it "conservative and untuned."
- **Phrase-grouping is blind to multi-cycle combinations (clips 18, 19,
  24).** Four rounds of 1–8 read as four separate 8-count phrases, not one
  32-count combination — true even when every count is spoken cleanly, so
  it isn't a marker-density artifact. Clip 24 is a partial bright spot:
  the onset detector's rhythmic-section boundary correctly excluded a
  genuinely free, out-of-tempo coda from the metered phrase, even though
  tempo within the correctly-bounded section was still wrong.
- **The "trap" clip worked (clip 22).** Numbers used purely as
  embedded quantities ("we take two tendus, then one more...") with zero
  alignment to the real beat produced *zero* Gemini beat markers and a
  correctly abstained count, rather than false confidence. Good defensive
  behavior, worth protecting in future changes.
- **Delivery extremes broke word segmentation directly, not via the
  band-snap.** Sustained/legato adagio (clip 14) and quiet/mumbled
  end-of-day voice (clip 23) both starved the onset detector of usable
  word boundaries — legato has no clear gaps, mumbled speech gets
  wholesale dropped by Whisper rather than mistranscribed. Full-voice
  energetic allegro (clip 21) showed the mirror problem: real tempo
  *drift* (rushing ahead of the click from excitement) that reads as an
  ADR-014-adjacent "which tempo is truth" question rather than a
  straightforward detection bug.

## Suggested next steps

1. Implement ADR-014's candidate-family reporting — the design is written,
   verification plan is in the doc, and clips 12/13 are ready-made
   acceptance cases.
2. Revisit ADR-013's 60%/8-beat arbitration thresholds using clip 20's
   near-miss as a data point.
3. A phrase-grouping fix (clips 18/19/24) and a sparse-marker/beat-grid
   tolerance fix (clips 5–8) look like the two next-highest-value KEEP
   changes after ADR-014 — both are systematic across multiple clips
   rather than one-off misses.

Every finding above traces back to a specific `evals/cases/*.yaml` note
and a frozen trace under `evals/traces/` if you want to dig into the raw
timestamps and marker classifications yourself.
