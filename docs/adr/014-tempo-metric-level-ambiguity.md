# ADR-014: Report the Tempo Metric-Level Family Instead of Collapsing It

**Date:** 2026-08-08
**Status:** Accepted — implemented 2026-08-09. **Partially superseded
2026-08-28 (rung M / W9):** the "leave the primary alone, report the
family alongside it" compromise below was the right call while the fold
was unexamined. W9 replaced the hard 70–140 snap with level selection
under a soft log-normal tempo prior, so clip 12 (62.2 BPM) and clip 13
(161.8 BPM) are now correct *as primaries*, not merely discoverable as
alternates. The family and everything else in this ADR stand unchanged;
only the claim that the primary answer is frozen no longer holds. See
the 2026-08-28 RESEARCH-LOG entry.

## Context

`normalize_tempo()` (`precision/tempo.py`) snaps any raw BPM into 70–140 by
multiplying or dividing by 2 or 3, and `interpret_meter()` derives meter and
subdivision purely from *which* transform was applied (ADR-006/007). This is
a hard, single-answer rule: there is no way for it to say "62 BPM, and that's
correct" — a raw reading outside the band is always force-fit into it.

**Capture-checklist clip 12** (`rig-numbers-4-4-60-halftempo`, recorded
2026-08-08) is a clean counter-example. Ben marked a real 8-count phrase at a
metronome-confirmed, genuinely slow 60 BPM (the "quick reminder in class"
voice — an exercise he'd normally run at 120). The onset detector read the
tempo essentially perfectly: 62.2 BPM at 100% coverage, CV=0.00. The pipeline
still doubled it to 124.4, because 62.2 sits just under the 70 floor. Tempo,
meter, and the eval both went red for a reading that was already right.

This is not a new defect — the tier-0 synthetic suite already carries a
`KNOWN_FAILING` sibling, `t0-3-4-half` (`evals/synthetic.py`,
`tests/test_evals_tier0.py`), which simulates a teacher marking *every other
beat* of a faster underlying tempo. Doubling recovers the right BPM there but
assigns the wrong meter (its 4/4 sibling `t0-4-4-half` passes only because
doubling happens to reconstruct the correct meter too — a coincidence, not a
proof).

**The sharper finding:** clip 12 and `t0-3-4-half` are two *different real
phenomena that produce audio-identical raw signals*:

- **Genuinely slow** (clip 12): the true tempo *is* 60 BPM. Every beat is
  spoken. The raw onset reading is already correct and should not be touched.
- **Half-tempo marking** (`t0-3-4-half`, and the "teacher marked half-tempo,
  pianist doubled" example already anticipated in
  [Vision 08 §8.2](../vision/08-benchmark-and-shadow-mode.md)): the true
  tempo is faster; the teacher deliberately speaks on every other beat. The
  raw reading should be doubled to recover the intended pulse.

From onset regularity alone — coverage, CV, confidence — **these two cases
are indistinguishable.** A perfectly regular 62 BPM word stream is equally
consistent with both stories. `normalize_tempo()` currently picks one
universal answer (always assume the second story) via a fixed 70–140 band.
That band is a reasonable *prior* — ADR-006 chose it because ballet class
tempos usually do cluster there — but it silently fails exactly when someone
marks legitimately outside it, and no amount of better regularity math fixes
that: the missing information isn't in the pulse.

**Scope note:** this ADR is about faithfully recovering the *measured*
marking tempo (whichever metric level actually matches what was spoken) —
not about predicting `performance_bpm` (what a pianist would actually play).
Vision 08 §8.2 already tracks `marking_bpm` vs `performance_bpm` as a
separate, deliberately-not-1:1 pair, sourced empirically from accompanied
recordings. This ADR does not touch that gap; it only stops the precision
layer from corrupting `marking_bpm` itself when the true marking tempo falls
outside the comfort band.

## Decision

**Stop collapsing the metric-level family to one answer. Report it.**

`normalize_tempo()` currently returns `(normalized_bpm, multiplier)` for a
single winner. Replace/extend it with a function that computes the full
family of musically-sane candidates for a raw BPM — not just the first one
that lands in band — each carrying the meter/subdivision that its multiplier
implies (the same derivation table `interpret_meter()` already uses):

```python
@dataclass
class TempoCandidate:
    """One member of the metric-level family for a raw tempo reading."""
    bpm: float
    meter: Meter
    subdivision: str
    multiplier: int         # relation to raw_bpm, same encoding as today
    in_comfort_band: bool   # True if bpm sits in the 70-140 ballet-class range


@dataclass
class NormalizedTempo:
    bpm: float                            # primary — selection unchanged (see below)
    meter: Meter
    subdivision: str
    confidence: float
    raw_bpm: float
    tempo_multiplier: int
    alternates: list[TempoCandidate] = field(default_factory=list)  # NEW
```

**Primary selection stays exactly what it is today** (prefer the in-band
member, preferring ×2 over ×3, then ÷2, then ÷3, else raw unchanged). This
keeps the change additive and non-regressive: every currently-blessed case
(including `t0-4-4-half` and every green rig clip) gets the same primary
answer as before. `alternates` is populated with the other family members —
generated over a broad absolute range (e.g. 20–400 BPM, not just 70–140) so
a genuinely-slow or genuinely-fast true tempo still shows up as a candidate
even though it isn't picked as primary.

Concretely, clip 12 would report:
```
primary:    124.4 BPM, 4/4, none   (multiplier=2, in_comfort_band=True)
alternates: [62.2 BPM, 4/4, none   (multiplier=1, in_comfort_band=False)]
```
— today's answer, unchanged, but the correct reading is now *discoverable*
instead of discarded. Downstream consumers (or a future scoring change) that
know which octave is actually playable can pick from `alternates`; consumers
that ignore the new field see no behavior change.

## What this does NOT solve

**Which candidate is "true" is still not decidable from audio regularity
alone** — that's the whole point above. This ADR makes the ambiguity visible;
it does not resolve it. Two follow-on directions, deliberately left as future
work rather than bundled into this change:

- **Exercise-type tempo priors.** The pipeline already classifies exercise
  type with usable confidence (Plié 95%, Petit Allegro 80%, etc., visible in
  every rig trace). Adagio/plié work is conventionally slow, petit
  allegro/tendu faster, grand allegro faster still — a real disambiguating
  signal pure onset regularity can't provide. Selecting the primary by
  exercise-conditioned plausibility instead of (or in addition to) the fixed
  70–140 band is the natural next step, but needs its own tempo-range table
  and its own eval evidence before it should move the default.
- **Scoring policy.** `score_meter_triple` already gives partial credit for
  musically-equivalent (meter, bpm, subdivision) triples via `_surface()`
  (ADR-007's "the accompanist does not care" principle), but `score_tempo`
  still scores a clean `metric_level_x2` answer as flatly wrong. Once
  `alternates` exists, the scorers could check it directly — matching a
  ground-truth label against *any* reported family member, not just the
  primary — which would also make plain tempo scoring philosophically
  consistent with how meter/subdivision equivalence already works. This is
  a scoring-policy question for Vision 08 §8.3, not this ADR.

Also out of scope: predicting `performance_bpm` from `marking_bpm` (see
Scope note above) — that gap needs accompanied-recording ground truth per
Vision 08 §8.2/13, not smarter precision-layer math.

## Verification Plan

Once implemented, this is judged by eval delta (CLAUDE.md discipline), not a
hand-run:

- `tests/test_evals_tier0.py`: `t0-3-4-half` and `t0-4-4-half` should show
  the correct member present in `alternates` even though the primary/`KNOWN_FAILING`
  status is unchanged (this ADR alone does not flip them green — see Future
  Work above).
- `evals/cases/rig-numbers-4-4-60-halftempo.yaml` (clip 12): primary answer
  unchanged (still red — expected, since selection logic isn't changing),
  but the trace should show 62.2/60 BPM surfaced in `alternates`.
- Full `tier0,tier1` sweep + `pytest -q`: no outcome changes anywhere else —
  this is additive to the schema and must not touch primary selection for
  any existing case.
- New unit tests in `tests/` covering the candidate-family generation
  (multiple valid members, ordering, `in_comfort_band` flags) directly,
  the same way ADR-007 added multiplier-path tests for `interpret_meter()`.

## Consequences

- `NormalizedTempo` gains `alternates: list[TempoCandidate]` — additive,
  backward compatible.
- `MusicalParameters`' stable contract (CLAUDE.md) is preserved; this is a
  field addition, not a breaking change.
- Precision layer (`tempo.py`) stays pure math, no AI dependency — consistent
  with its KEEP label.
- Does not fix any currently-red eval row by itself; it makes the corpus's
  existing octave-ambiguity failures (clips 6, 8, 12; `t0-3-4-half`)
  legible instead of silent, and creates the hook (`alternates`) that the
  exercise-prior and scoring-policy follow-ons need to land.

## Measured results (2026-08-09)

**What shipped.** `tempo_family()` (`precision/tempo.py`) generates the
family over 20–400 BPM at multipliers 1/2/3/−2/−3, each member built by
`_derive_metric_reading()` — the derivation table extracted from
`interpret_meter()`, so primary and candidates cannot drift apart.
`NormalizedTempo.alternates` (`types.py`) carries the family minus the
member that *is* the primary. The eval harness gained a non-gating
`truth_in_family` measure: when a tempo scores wrong, it records whether
the expected BPM sits within the same 8% tolerance anywhere in
primary+alternates. Outcomes keep their old semantics — a wrong answer
whose family holds the truth is still wrong.

**Primary unchanged, as required.** `pytest -q`: 167 passed, 3 skipped.
Tier-0's known-failure set is still exactly `{t0-3-4-half}`. The
`tier0,tier1` sweep prints *no outcome changes vs baseline*, and a
field-by-field diff of the re-blessed `evals/baseline.json` against the
previous one shows the outcome maps byte-identical in both suites, with
`truth_in_family` / `truth_in_family_n` the only added keys.

**Family recall — the number this ADR bought.** Tier-1 tempo: 15 correct,
13 wrong, 1 abstained (unchanged); of the 13 wrong, **3 carried the truth
in their family**:

| case | truth | primary | recovered as |
|---|---|---|---|
| `rig-numbers-4-4-60-halftempo` (clip 12) | 60 | 124.4 | **62.2** (×1, out of band) |
| `rig-names-2-4-160-long` (clip 13) | 160 | 80.9 | **161.8** (×1, out of band) |
| `frappe` | 160 | 74.3 | 148.6 (×2) |

Both pre-registered acceptance cases land exactly as predicted. Tier-0
reports `—` for the column: its tempo accuracy is 1.0, so no row was
eligible to be checked.

**The split is the finding.** 3 of 13 tempo failures are *selection*
failures — the measurement was right and the band prior discarded it, and
an exercise-type prior over `alternates` could plausibly fix them. The
other 10 are *measurement* failures: the truth is nowhere in the family
(e.g. `rig-names-3-4-90-clean`, truth 90, family
`[130.2, 65.1, 195.3, 32.5, 21.7]`), so no amount of smarter selection
helps them. That bounds the follow-on work honestly — the exercise-prior
direction addresses at most a quarter of the current tempo reds.

**One limit worth recording.** `t0-3-4-half` stays failing for the reason
the ADR predicted, but the family does not even make its defect
discoverable: its primary tempo (96.0) is already correct, and the
correct *triple* (3/4 @96) is not among the alternates, because meter is
still derived one-answer-per-level (×2 always implies 4/4). The family
resolves tempo-level ambiguity only; meter-derivation ambiguity is a
separate, still-collapsed decision.
