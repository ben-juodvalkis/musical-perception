# ADR-013: Band-Aware Tempo Arbitration

**Date:** 2026-08-07
**Status:** Accepted

## Context

`interpret_meter()` picked its primary tempo by a hard rule from ADR-006:
onsets win whenever their confidence clears 0.3; the marker path is pure
fallback. That precedence was correct when Gemini's beat markers were
sparse garbage (0–3 usable beats). ADR-010's index-keyed merge quietly
changed the world: markers became dense and trustworthy — and the stale
precedence started losing to them.

Two rig/real clips showed the same signature. On the waltz fixture
(ground truth 90 BPM, 3/4, counted "ONE-and-ah"), the marker path
recovered **90.8 BPM at 0.92 confidence from all 32 count words** and
lost to the onset path reading the swung triplet syllables at 215.9 →
final answer 108/4-4/duple, wrong on every axis. Frappé shows the family
resemblance (markers near truth, onsets at another level), though weaker.

## Decision — and a rejected first design

**Rejected: confidence-weighted arbitration.** "Markers win when more
confident" breaks ADR-007's issue-10 case: a *measure-level* marker
stream (Gemini at 40 while onsets correctly read 115) is also regular and
confident. Confidence cannot arbitrate across metric levels. The
existing test suite caught this within minutes of implementing it.

**Accepted: band-aware arbitration.** The 70–140 band *is* the
definition of beat level (ADR-006), so band membership is the
discriminator:

> When the onset reading sits **outside** 70–140 (syllable or measure
> level) and a dense, regular marker tempo (confidence ≥ 0.6, ≥ 8 beats)
> sits **inside** it, the markers are the beat-level signal and become
> primary. Whenever onsets already read inside the band, ADR-006/007
> behavior — including the issue-10 cross-ratio meter correction — is
> preserved bit-for-bit.

The synthetic builder gained a `swing` knob (subdivision syllables drift
late, beats stay on grid — how humans actually count), and a tier-0 case
pinning the waltz shape forever.

## Results (pre-registered, then measured)

| Row | before | after |
|---|---|---|
| rig-waltz tempo | wrong (108) | **correct (90.8)** |
| rig-waltz meter triple | wrong, credit 0 (4/4 duple @108) | **equivalent_reading, credit 0.5** (4/4 triplet @90.8 ≡ 3/4 @90 — the slow-triple/fast-waltz interchangeability teachers use, now scored as such) |
| t0-4-4-clean-triplet (day-one ÷2/÷3 defect) | wrong | **correct** — KNOWN_FAILING shrinks to {t0-3-4-half} |
| tier-0 tempo sweep | 23/24 | **25/25** |
| tier-1 tempo | 4/7 | **5/7** |
| frappé tempo | wrong (74.3, metric_level_div2) | unchanged — explicitly out of scope: its onset reading is *inside* the band, so arbitration cannot help; needs accent evidence (Track 3) or the unused `counting_structure.estimated_bpm` (163.2 vs truth ~160 on that clip) admitted as a signal |
| everything else | — | no outcome changes |

## Consequences

- The remaining meter gap on the waltz (4/4-triplet vs 3/4) is the
  benign notation ambiguity; only accent periodicity can resolve it, and
  for playback it may not need resolving. Whether `equivalent_reading`
  should count as fully correct for the exercise-success metric is now a
  live scoring-policy question for Vision 08 §8.3.
- `counting_structure.estimated_bpm` — documented "unreliable, unused"
  in types.py — was near-truth on both frappé (163.2 vs ~160) and
  grande-battement (104 vs ~104). Two data points argue for admitting it
  as a vote with its own gate; that is a separate eval-gated change.
- Arbitration thresholds (0.6 confidence, 8 beats) are conservative and
  untuned; the corpus program (docs/evals/capture-checklist.md) is what
  will justify moving them.
