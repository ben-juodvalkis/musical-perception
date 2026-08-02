# ADR-012: Counts from Evidence Fusion

**Date:** 2026-08-02
**Status:** Accepted

## Context

ADR-011 tightened the *definition* of `structure.counts` and honestly
recorded that a better prompt improved the answer distribution without
pinning it (post-acceptance live runs still flipped 32↔18 on the verified
grande-battement case), and concluded the field needs an algorithm, not
more prompt English.

The frozen traces then showed why no prompt can win: **teachers mostly
don't count through the phrase.** On grande battement the spoken numbers
are quantities ("we take *two*… *one* more… *five, six*"); on frappé the
teacher speaks no numbers at all — forty markers, all step names. There
is no count-through for the model to read. But the traces also showed the
right answer already present in evidence the pipeline ignored: a
five-estimator prototype against the four labeled cases found every truth
recoverable from *some* combination of Gemini's two structural reads
(`structure.counts`, `counting_structure.total_counts`), the beat-marker
tally × subdivision, and the marked span crossed with each tempo
hypothesis — with no single estimator sweeping all four.

## Decision

New KEEP module `precision/structure.py` owns `counts`, in ADR-007's
one-coherent-answer style. `estimate_counts()` detects the counting
regime from the markers themselves:

1. **Numeric counting** (≥4 count words and ≥60% of beats numeric):
   phrase length is read from the spoken cycle — the value the count
   reaches before restarting ("1..8, 1..8" → 8). Inconsistent cycle
   maxima abstain.
2. **Step-name marking:** every available estimator is snapped to
   musical phrase lengths {6, 8, 12, 16, 24, 32, 48, 64, 96} and cast as
   a vote; the two tempo hypotheses cast a single span vote when they
   agree (within 5%), so redundant readings can't fake independence.
   **Two agreeing signals commit; a tie or a lone voice abstains.**

`PhraseStructure.counts` becomes `int | None` — `None` is the estimator
declining to guess. Gemini keeps `sides` (stable and correct throughout).
Snap ties break to the smaller phrase (under-playing is recoverable;
overrunning the phrase end is not). No estimator weighting: with two
labeled long combinations, learned weights would be curve-fitting —
agreement-or-abstain until the corpus grows (13 · Corpus & Capture).

## Results (baseline delta, frozen replay)

| counts | before | after |
|---|---|---|
| correct | 3 | 3 |
| wrong | 1 (frappe: 20) | **0** |
| abstained | 0 | 1 (frappe — genuine 2–2 evidence tie) |
| accuracy on committed | 0.75 | **1.0** |

Sole outcome change vs the prior baseline: `frappe.counts: wrong →
abstained`. The grande-battement instability ADR-011 documented is now
structurally absorbed: when live Gemini flips to 18, the tally (35→32)
and total_counts (34→32) outvote it — unit-tested as the 18-flip
scenario.

## Consequences

- The engine downstream must handle `counts=None` as "ask the teacher /
  wait" — which is the interaction design's preference anyway
  (Vision 07: silence over false starts; one legitimate question).
- Frappé-class combinations (long, fast, step-name-marked, no numbers)
  remain unanswered rather than wrong. The designed escape is better
  evidence, not better weights: accent-periodicity onsets (Track 3) will
  make span×tempo trustworthy at the right metric level, and corpus
  growth makes the vote pool deeper.
- The numeric regime reads the *cycle*, not the phrase: a 32-count
  phrase counted as four 8s reads as 8 unless other evidence lifts it.
  On rig fixtures (phrase = cycle) this is exact; on real classes
  teachers rarely number-count whole phrases, so the vote regime
  dominates there. Documented limitation, revisit with corpus data.
