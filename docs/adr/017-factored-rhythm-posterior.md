# ADR-017: The Factored Rhythm Posterior

**Date:** 2026-08-28
**Status:** Accepted — landed 2026-08-28 by explicit owner override of the
pre-registered W5 gate (tempo tied 20/29 instead of improving; the four
tempo losses accepted as diagnosed genuine-trades against four wins, with
meter_triple 12→13 and ECE 0.1998→0.1815 both net-improved). The override
is the owner's act, recorded in the 2026-08-28 rulings ledger entry —
sessions never argue past their own pre-registration.

## Context

ADR-016 committed rung 4 to a joint posterior replacing the
normalize/interpret/arbitrate stack. Three bodies of evidence shaped what
was actually built:

- **The owner's factored-meter direction (2026-08-26):** playing class,
  the perceived facts are pulse, division (each beat splits in 2 or 3),
  and grouping (a ladder of levels above the pulse — the bar one rung,
  the eight-count phrase another). "Meter" as a single label is a
  notation-level encoding, not a perceptual fact.
- **W2 (rung 3, negative):** bar-level accent periodicity is mostly
  absent in this corpus; the strongest periodicity sits at the count
  phrase; salience templates cannot separate 2/4 from 4/4 or 3/4
  from 6/8 as labels.
- **W9:** the hard 70–140 fold was replaced by soft level selection,
  and the residual gap was named: some level decisions (52-vs-61.5)
  carry no acoustic evidence in the current streams at all.

## Decision

`precision/posterior.py`: a bar-pointer lattice — the forward algorithm
on the Krebs-2015 state space (integer frames per beat at 50 fps,
deterministic pointer advance, tempo drift ±1 state at beat crossings
under an exponential log-ratio cost) with Whiteley-2006 per-frame
Poisson emissions over two replayable evidence classes (classified beat
markers, support-discounted; residual word onsets, mass-capped as a
nuisance template). W9's log-normal tempo prior is applied once over
the lattice; nothing folds anywhere.

**There is no meter variable in the state space, and no division axis:**

- **Division** is decided by sub-syllable counts per beat, vetted by
  timing CONSISTENCY (per-rank circular concentration ≥ 0.6, at least
  three positioned subs). Measured fact, twice confirmed: real duple
  "and"s cluster at one swung phase (0.61–0.77), triplets at two
  (~0.55 and ~0.9) — nowhere near the ideal 1/2, 1/3, 2/3, so
  position-vs-ideal classification and any joint division axis are
  falsified (the axis also hands the tempo search fine combs that eat
  dense streams a level down). Gemini's claim is a fallback only where
  nothing can be measured — never a pass-through (closes W9-b).
- **Grouping** is a per-level ladder (`GroupingLevel`) read out from
  counting cycles and boundary gaps; a silent rung is honest output.
- **The time-signature label is derived late**, only for the contract
  surface, with Gemini's claim as one vote.
- **Confidence is the posterior mass of the ±8% tempo window** — the
  probability the committed answer is one the scorer accepts — and
  commitment is the window-mass Bayes decision. Below 0.20 the answer
  is abstention.
- **ADR-014's alternates finally carry real weights** (window masses).
  On the landing run the truth sits inside the reported family on 5 of
  9 wrong tempo rows (baseline: 0 of 9); family-level Acc2@8% is 0.793
  vs 0.690 committed — the remaining problem is selection, and the
  selection information (junk-vs-signal in the marker stream) is
  largely absent from the current traces.
- **Evidence-poor and low-support streams fall back to
  `interpret_meter`** (fewer than 4 classified beats / 8 events, or
  beat-stream support below 0.6 — rhythm.py's own rhythmicity boundary),
  with division still measured on the fallback path.

Contract additions, all additive: `NormalizedTempo.grouping_levels`,
`TempoCandidate.weight`, `GroupingLevel`.

## Consequences

- Tier-0's driver (`evals/synthetic.py`) still exercises
  `interpret_meter` — untouchable eval code in a pipeline rung — so
  tier-0 pins the legacy path byte-identically. Follow-up EVAL-CHANGE:
  point the tier-0 driver at the new core; only after that can W8
  retire `interpret_meter`.
- The falsification ledger (nine mechanisms by which wrong-tempo
  hypotheses extract credit from data they do not explain) is the
  design constraint set for every future rhythm-core change; see the
  2026-08-28 W5 entries.
- The named path to the tempo wins this ADR did not deliver: the
  acoustic pulse channel as a third evidence class (W11 sidecars), and
  ensembled semantics (W6) turning marker classification into a
  distribution.
