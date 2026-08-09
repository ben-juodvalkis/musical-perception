# ADR-016: Rhythm-Core Reset — the n=30 Retrospective and the Acoustic-First Redesign

**Date:** 2026-08-09
**Status:** Accepted — direction of record for the next phase. This ADR is to
the rhythm core what [ADR-009](009-evaluation-harness.md) was to evaluation:
the plan, not the diff. Implementation lands in follow-on ADRs, each gated
under ADR-015's typed gates. It also records a posture decision: **the project
is in research mode — accuracy first, cost and latency deliberately
unconstrained** until the numbers justify optimizing.

## Context — the retrospective

After [ADR-015](015-onset-measurement-robustness.md)'s override, the owner
asked the step-back question: knowing everything the 24-clip capture programme
and fifteen ADRs have taught us, how would we build this if we were starting
over? This ADR records the answer. The short form: the system solves exactly
one case — a teacher counting **numbers**, in 4/4, one 8-count phrase — and
essentially nothing else, and every major failure traces to one founding
assumption (*the teacher's words are the beat*) plus one prior enforced as a
rule (*snap everything into 70–140*). The right move is not to restart the
repo. It is to restart the **rhythm core** — one signal chain and one fusion
module — while keeping the harness, the corpus, the contract, and the process
that produced this unusually honest evidence trail.

### Where the blessed baseline stands

All numbers from `evals/baseline.json` (generated 2026-08-09, blessed under
ADR-015's override). Tier-1 corpus: 30 cases — 26 Ben-voiced recordings
(including the full 24-clip checklist) plus 4 YouTube clips; one teacher
carries 87% of the corpus, all of it English, none of it accompanied.

| field | correct / wrong / abstained | committed accuracy | v1 gate ([Vision 08 §8.3](../vision/08-benchmark-and-shadow-mode.md)) |
|---|---|---|---|
| tempo | 16 / 12 / 1 | 0.571 | ≥ 0.95 |
| meter_triple | 10 / 18 / 1 | 0.357 | ≥ 0.90 |
| counts | 12 / 9 / 7 | 0.571 (mean credit 0.429) | ≥ 0.90 |

Tier-0 synthetic stays 25/25 tempo, 24/25 meter — the precision math is fine
on inputs that match its assumptions. The structure inside the tier-1 numbers
is more instructive than the toplines:

- **Fully green checklist clips — every pinned field correct — 5 of 24**, and
  all five are `numbers`-counted (`rig-numbers-4-4-104-clean`, `-prep`,
  `-duple`, `rig-numbers-4-4-80-triplet`, `rig-numbers-6-8-100-clean`). Zero
  step-name clips, zero vocables, zero mixed.
- **Counting style, not musical difficulty, is the dominant variable.** The
  step_names slice reads tempo 0.357, meter_triple **0.077** (1 of 13), counts
  0.143 — against numbers at 0.643 meter and 0.786 counts. Meanwhile the
  "hard" musical stressors all pass *when counted in numbers*: 6/8, triplet
  and duple subdivision, prep counts, mid-phrase interruption.
- **Meter output is a 4/4 prior, not a perception.** Of 8 non-4/4 checklist
  clips, one is correct (the numbers-counted 6/8). All six 2/4 and 3/4 clips
  read 4/4 — including `rig-numbers-2-4-120-clean`, where tempo read 119.7
  against a 120 truth and counts were 8/8, and the meter *still* came back
  4/4. Two of the misses collect 0.5 `equivalent_reading` credit, which is
  musically defensible and should not obscure that genuine meter recognition
  is 1-for-8.
- **Subdivision fails in one direction only.** Every failure is a hallucinated
  `duple` on step-name material — a false-positive gate on unstructured
  speech, not a discrimination failure (the three clips with real subdivision
  labels all pass). It is the largest single source of
  correct-tempo-correct-meter-still-red rows.
- **Phrase structure never survives length.** Of 7 checklist rows whose true
  phrase exceeds one 8-count, one is correct — and that one
  (`rig-names-4-4-96-allegro`) is luck-flagged in its own case notes. The
  estimator reads the counting *cycle* (8), not the phrase (32) —
  [ADR-012](012-counts-from-evidence-fusion.md)'s documented limitation, now
  measured at 1-of-7.
- **Confidence does not know when the system is wrong.** Mean tempo confidence
  is 0.802 on correct rows and 0.751 on wrong ones; the five most confident
  wrong answers (0.82–0.89) all outrank the least confident correct answer
  (0.58). ECE improved to 0.291 under ADR-015's support term — better, still
  poor.
- **Only 3 of the 12 wrong tempo rows are recoverable downstream.**
  [ADR-014](014-tempo-metric-level-ambiguity.md)'s `truth_in_family` splits
  them: 3 selection failures where the measurement was right and the band
  discarded it (`rig-numbers-4-4-60-halftempo`, `rig-names-2-4-160-long`,
  `frappe`) — the other 9 the truth is nowhere in the family.

### Five findings

**F1 · We built a word-rhythm system for an accent-rhythm problem.**
[Vision 05](../vision/05-perception-strategy.md) said it before the capture
programme confirmed it: *"Marking is not speech that happens to be rhythmic;
it is a percussion track wearing words."* The pipeline computes tempo from
Whisper word onsets and meter from which band multiplier fired — so vocables
collapse entirely (`rig-vocables-4-4-100-clean`: a 9.5 s vocable phrase became
a single un-timestamped token; all four fields abstain), quiet marking loses
half its words (12 tokens for a 16-count phrase), legato loses boundaries (19
words for 32 beats), and step-name IOIs land between metric levels. The
failures stratify by *what the ASR can segment*, not by what is musically
hard. The same clip 17 also produced the most alarming datum in the corpus:
its first (badly exported) audio had Whisper hallucinate a cleanly-timestamped
"one and two…" onto pure vocables and the row **scored all-green** — caught
only by a human re-export, not by any check.

**F2 · The 70–140 band is a prior that was enforced as a rule.** Introduced in
[ADR-006](006-onset-tempo-and-normalization.md) as an empirical convenience,
promoted to load-bearing logic in
[ADR-007](007-coherent-metric-interpretation.md) (meter is *derived from which
multiplier applied*), used as the arbitration key in
[ADR-013](013-tempo-arbitration.md), indicted in ADR-014. The smoking gun is
`rig-numbers-4-4-60-halftempo`: the detector read 62.2 BPM at CV = 0.00 and
100% coverage — essentially perfect — and the band doubled it to 124.4, taking
tempo, meter, and the eval row red for a measurement that was already right.
The band is hardcoded as four separate literals in `precision/tempo.py` and
once more in English inside the Gemini prompt. A prior applied as
post-processing destroys correct answers exactly when reality sits outside it.

**F3 · We arbitrated when we should have fused.** Two tempo readings (word
onsets; Gemini-classified markers) have only ever been *arbitrated*, and
`interpret_meter()` now carries three stacked generations of arbitration rule
— ADR-006's precedence, ADR-007's ratio-≈3 special case, ADR-013's band-aware
reversal — plus a `multiplier=3` that means two different things in two
branches. Each ADR was locally rational; the stack is patch-on-patch. And the
recurring failure mode of the whole history is *a fix in one layer silently
invalidating a precedence rule in another*: ADR-010's dense markers broke
ADR-006's precedence (→ ADR-013); ADR-006's prompt fix disarmed ADR-007's
ratio check on the very clip that motivated it. Nothing detects that a
heuristic's precondition has evaporated except a red eval row.

**F4 · A single temperature-0 LLM draw is a coin flip, and we already proved
the fix.** [ADR-011](011-phrase-structure-definition.md) measured temp-0
Gemini flipping bimodally on identical input (18, 18, 18, 32).
[ADR-012](012-counts-from-evidence-fusion.md)'s answer — outvote it with
independent evidence — is the best architectural move in the repo, and it was
applied only to `counts`. The general principle (treat the model as a
*sampler*; consume distributions, not draws) was never generalized to meter,
tempo opinion, exercise, or quality.

**F5 · The harness set the gradient.** Vision 05 pre-registered the remedies
in July — raw-audio onsets, accent periodicity for meter, exercise priors,
per-teacher calibration — and marked them "to build." They are still unbuilt;
ADRs 013–015 are all repairs to the word-onset path that Vision 05 §5.1 had
already declared topped out. The mechanism is visible in hindsight: frozen
traces made the Whisper path replayable in under a second, while the acoustic
channel had no replay story, so every iteration gravitated to the measurable
path. ([Vision 09](../vision/09-risk-register.md)'s R2 tripwire literally
cannot fire — it is conditioned "after the accent module ships.") The lesson
is structural, not moral: **replay support for a channel is destiny.** The
reset therefore builds the new channels' trace format first.

### Two permanent results

The retrospective's most valuable outputs are proofs, not bugs.
**ADR-014:** a genuinely-slow 60 and a half-tempo-marked 120 are
audio-identical — the disambiguating information is not in the pulse.
**ADR-015:** one level down, a long interval on a swung waltz is "one
stretched beat" or "two beats" and nothing in the intervals says which. No
onset algorithm resolves these. The answer must be imported from outside the
pulse — accent structure, exercise identity, word semantics, teacher history —
or the system must ask, or abstain. Any redesign that does not treat outside
evidence as first-class is re-litigating a settled question.

### What earned its keep

Unchanged by this ADR, explicitly: the eval harness and case format, the
frozen-trace mechanism, the scorers (octave-aware tempo credit, abstention
accounting, Wilson intervals, ECE), the capture programme (8 of 24 clips
produced findings that changed a design doc — the highest-yield artifact in
the project), abstention as product policy, ADR-010's single-owner
tokenization, ADR-012's vote-or-abstain shape, the `MusicalParameters`
contract, and ADR-015's typed gates. The process — pre-registration, honest
ledgers, on-the-record overrides — is the asset the reset spends.

## Decision

**Posture: research mode.** Accuracy is the only optimization target. Large
models, multi-model ensembles, repeated sampling, forced alignment, and human
annotation are all in budget; latency, cost, and production hardening are out
of scope until the accuracy numbers justify them. The ADR-008 trigger/wakeword
path — never wired to an entry point — leaves scope entirely. The Feb-2026
model comparison predates two model generations and is re-run, not trusted.

Six commitments:

**1 · The acoustic pulse channel becomes primary.** Syllable-nuclei / onset
detection on the raw teacher-mic audio — energy and spectral flux, with a
per-onset **salience vector** (intensity, duration, F0 prominence) from the
Praat path that already exists behind `--signature`. No ASR in the pulse loop:
the channel behaves identically on numbers, step names, vocables, and humming.
Whisper words demote to *evidence* — numbers are gold when present, and
ADR-012's regime detection remains their consumer. This is Vision 05 §5.2
channel 1, executed at last.

**2 · Meter comes from accent periodicity, not band multipliers.**
Autocorrelation of onset salience at lags 2 / 3 / 4 / 6 (Vision 05 §5.3 —
"ONE-two-three ONE-two-three is a peak at lag 3; no words involved"). 4/4
becomes an inference, never a default. This targets the worst number on the
board: 0.077 step-name meter, 1-of-8 non-4/4 recognition.

**3 · One joint posterior replaces the arbitration stack.** A hypothesis space
over (beat period, phase, meter, subdivision, phrase length); every channel —
acoustic onsets, salience periodicity, count words, marker classifications,
pose periodicity, exercise-type priors — contributes a likelihood; the 70–140
comfort band and the exercise table enter as **priors, not post-hoc snaps**.
Retired when this lands: `normalize_tempo()`'s hard snap, `interpret_meter()`'s
three-generation stack, and `subdivision.py`'s mean-count heuristic with its
dead `avg_ratios` schema. ADR-014's `alternates` family becomes the posterior's
support with real weights; **confidence becomes posterior mass** (the
structural fix for F-finding "confidence does not know"); **abstention becomes
an entropy threshold** instead of scattered per-module heuristics. The two
permanent underdetermination results surface naturally as bimodal posteriors —
which is the honest output, and what the one-word question in
[Vision 07](../vision/07-interaction-design.md) exists to resolve.

**4 · The semantic channel gets ensembled.** N ≥ 5 draws across ≥ 2 model
families for the Gemini-class call; downstream consumes distributions
(per-token beat-label agreement rates, meter vote shares), never a single
draw. Per-draw responses are frozen in the trace format so the ensemble
replays offline. This generalizes F4 beyond `counts`.

**5 · Ground truth and scoring go layered.** Hand-tapped beat grids for all 30
existing recordings (the raw audio is local-only — traces freeze Whisper and
Gemini output, not sound — so stage-1 work runs against the original
recordings, which exist). The harness gains **stage-level scoring**: pulse
recall/precision against the tapped grid, scored separately from inference
accuracy — the distinction ADR-015 had to reconstruct by hand, made a first-
class metric. Plus the clip-17 guard: an acoustic-onset-count vs
transcript-token-count consistency check, so a transcription hallucination can
never again score green without tripping an alarm.

**6 · Corpus before optimization.** In priority order: (a) other voices,
weighted toward step names and vocables — the styles the system is 0-for, at
n=14 and n=1 respectively; (b) **accompanied classes**, where beat-tracking
the pianist yields `performance_bpm` labels mechanically — the
marking-tempo → performance-tempo pair was pre-registered in
[Vision 08 §8.2](../vision/08-benchmark-and-shadow-mode.md) as a novel
research result and has zero data points to date; in research mode it is the
primary research question, not an annotation nicety; (c) seal one teacher and
one class as holdouts immediately ([Vision 13 §13.7](../vision/13-corpus-and-capture.md),
specified and never done); (d) grow toward n≈140 (±5 points, §13.8). Until
then, ADR-015's amendment stands: knife-edge rows gate nothing.

## Falsification plan — sequenced so each step can kill the next

1. **Annotate** beat grids for the 30 recordings; add stage-level scoring and
   the onset-vs-token guard to the harness. (A day of tedium; it converts
   every future red row from forensics into a lookup.)
2. **Build the acoustic extractor; measure pulse recall against the grids.**
   Pre-registered bar: it must beat Whisper-word onsets decisively on the
   step_names and vocables slices. If it cannot, the bottleneck is
   perception-model-class rather than architecture — the reset stops here, and
   the [Vision 10](../vision/10-pivots.md) P2 posture (corpus + methods as the
   contribution) strengthens accordingly.
3. **Accent-periodicity meter votes against the 24 checklist clips.** Bar:
   non-4/4 recognition must beat 1-of-8, and the numbers-counted 2/4 and 3/4
   clips (currently 0-for-3 with clean pulses) are the must-move rows.
4. **The joint posterior lands; the stack retires.** Gated as a measurement
   change under ADR-015's typed gates: net improvement on the primary metric
   AND ECE, zero undiagnosed regressions, every regression classified.
5. **Ensemble the semantic channel; re-freeze per-draw traces.**
6. **New capture** (voices, accompanied classes, sealed holdouts) — funded by
   steps 2–3 proving the bet, not before.

Steps 1–5 run entirely on recordings already on hand.

## Consequences

- Every documented failure class maps to a structural owner, not a patch:
  meter-collapses-to-4/4 → commitment 2; hallucinated duple → 3 (subdivision
  inferred jointly, with sparse-evidence prior for `none`); band-snap
  selection failures → 3 (band as prior); phrase-cycle-vs-phrase → 3 (phrase
  length in the hypothesis space) plus 6a; segmentation loss on
  vocables/legato/quiet → 1 (pulse without ASR); the clip-17 false green → 5;
  uninformative confidence → 3 (posterior mass).
- Costs, honestly: the acoustic bet is *unfalsified, not validated* — vocables
  at n=1 is exactly why step 2 is a kill-test, not a milestone. A joint
  posterior is a bigger swing than any patch to date and will be harder to
  debug than the stack it replaces; stage-level scoring (step 1) is the
  mitigation, and it lands first for that reason. And n=30 still caps what any
  gate can prove — which is why capture is a commitment, not a follow-on.
- The KEEP/DISPOSABLE taxonomy gains its missing third label: **RETIRED** —
  kept in history, deleted from the tree. Scheduled for retirement with
  commitment 3: the ADR-008 trigger/wakeword path, the legacy text-matching
  merge, the six Feb-2026 one-off scripts, and `subdivision.py`'s dead fields
  (≈1,200 unreachable lines against ≈2,000 reachable). "Tried it, kept it
  anyway" has cost more reading than it saved.
- What this ADR does *not* claim: that the redesign reaches the v1 gates. It
  claims the current architecture demonstrably cannot — 0.357 meter with the
  meter signal unbuilt, 0.077 on the counting style real teachers actually use
  — and that the replacement is the one the project's own vision documents
  specified, now sequenced behind kill-tests so it is measured, not believed.

## Addendum (2026-08-09) — the literature check

A four-track survey of the beat-tracking / meter-induction / speech-rhythm /
evaluation literature was run the day this ADR was accepted:
[docs/research/voice-as-drum-review.md](../research/voice-as-drum-review.md).
Nothing in it contradicts the six commitments; five things sharpen them:

- **Commitment 1** gets a validated event definition: the perceptual beat of
  a syllable is the peak of the envelope's rate of change ("peakRate" ≈
  vowel onset ≈ P-center), and ASR word starts carry a 0–150 ms
  word-dependent early bias — so step 1's beat grids must be annotated at
  vowel onsets, not word starts.
- **Commitment 3**'s joint posterior has a direct ancestor family
  (bar-pointer models, Whiteley 2006 → Krebs 2015) with exact sub-second
  inference at our clip lengths and published meter-discrimination results;
  the recommended implementation is specified in the review's Review 3 §(a).
- The 70–140 replacement is parameterized by the resonance literature:
  log-Gaussian prior, T₀ ≈ 100–110 BPM, σ ≈ 1.2–1.4 octaves, applied at
  level selection only.
- `truth_in_family` is the field's **Accuracy-2** metric; adopting it plus
  the continuous octave-error **OE2** sharpens the eval gates and makes
  results comparable to twenty years of published numbers.
- No published system and no public dataset does beat/meter on rhythmic
  speech — the corpus (commitment 6) is a first-of-kind contribution, which
  strengthens the Vision 10 P2 posture.
