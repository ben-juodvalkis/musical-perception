# Beat-grid annotation convention

**Status:** Owner-ratified 2026-08-11 (rung 1.5, branch
`agent/rung-1.5-grid-verification`). Authority: the owner is the human ground
truth; agent sessions are scribe and tool-runner only.

This document is the contract for what a beat time in `evals/grids/<id>.yaml`
*means*. It was settled **before** the first clip was annotated, because
changing it later invalidates every clip already verified. Format mechanics and
the peakRate recipe live in [beat-grids.md](beat-grids.md); this file covers the
judgment calls that format cannot express.

Every ruling below was decided from corpus evidence, not from principle alone —
the supporting numbers are in the rung-1.5 ledger entry in
[RESEARCH-LOG.md](../research/RESEARCH-LOG.md).

## 0. The anchor rule (fixed, not a ruling)

**Beat times sit at vowel onsets (P-centers), never word starts.** Standing
Lesson 1; [review-1 §2.9](../research/review-1-onsets-pcenters.md); restated by
the owner at rung 1.5. ASR word starts carry a 0–150 ms word-dependent early
bias, so a grid anchored to them would encode the very error the grids exist to
measure.

## 1. Rulings

### (a) Grid level — tactus only

`beats` contains **one entry per felt tactus beat**. "plié" gets one mark, not
two. "one-and-a" gets one mark, not three.

Syllable-level evidence is not discarded: it remains in the grid's frozen,
machine-derived `onsets` list, which is **never edited and never verified**. Two
lists, two epistemic statuses — `beats` is human truth at the tactus level,
`onsets` is unverified acoustic evidence at (roughly) the syllable level.

*Why:* the tactus grid is what rungs 3 (meter votes) and 4 (bar-pointer)
actually consume, and the format was designed for it — `grids.py` describes the
owner's job as "delete non-beats" and expects `beats` to be "decimated to true
beat times". Correction work is then deletion, which is far faster and more
reliable than hand-placing labels at vowel onsets.

*Known cost, accepted:* rung 2's extractor is a syllable-nuclei detector, so
scoring its native output against a tactus grid charges it for events it is
correct to emit. **This is fixed in the metric, not in the ground truth** — see
§2.

### (b) Silent beats — vocalized only

Annotate **only beats that were actually voiced.** Where the pulse continued
through silence, place no mark; instead describe the silent stretch in the
grid's free-text `notes`.

*Why:* the label round trip cannot tag a beat (§3), so a mixed grid would
contain detectable and undetectable references with no way to separate them at
scoring time — every silent beat becomes an untraceable permanent false negative
for any acoustic extractor, deflating recall by a per-clip amount nobody can
recover. Vocalized-only keeps the acoustic recall ceiling at 1.0, so rung-2
margins read as detection quality.

*Not lost:* full metric grids are **derivable later** from a verified grid plus
tempo — the human pass is the expensive, unrecoverable part, and it is captured.
Standing Lesson 6 ("silence is evidence") stays unscored until a rung whose
explicit deliverable is a taggable grid format. **That rung was 2.5, and ruling
(g) below now says where the evidence counts** — at the rung-4 tier only, never
in stage-1, which this ruling continues to govern unchanged.

### (c) Prep counts IN, framing talk OUT

In-tempo prep ("five, six, seven, eight") **is annotated** — those are real
metric beats carrying the tempo. Framing speech ("okay, from the top", "ready?")
is **not**.

*Note the contract boundary:* `rig-numbers-4-4-104-prep`'s case notes say the
prep is deliberately **not** part of the phrase for the `counts` field. That is
the counts contract; the grid is a different contract. Prep beats are in the
grid and still outside the counted phrase — no contradiction.

### (d) Explanation speech — SUPERSEDED by (d′) on 2026-08-14

*Original ruling (v1):* explanation speech contributes no beats, categorically,
even where it sounds in tempo. Justified on reproducibility.

**Clips annotated under v1:** `rig-numbers-4-4-104-explained`,
`rig-names-4-4-104-explained`. Both have their explanation stretches excised
(8.198→13.907 s and 8.180→13.397 s).

**Owner ruling 2026-08-14: no re-annotation needed — they are already correct
under (d′).** Asked to judge those windows by ear, the owner confirmed the
pulse was *not voiced* through them. (d′) makes explanation *eligible*; it does
not require marks where nothing was voiced, which remains ruling (b). Zero beats
in those windows is therefore the right answer under both v1 and (d′), and the
two clips are consistent with the rest of the corpus rather than grandfathered.

### (d′) Explanation speech IS eligible — owner amendment, 2026-08-14

**Annotate every beat you hear voiced on the pulse, whether you are counting it
or talking through it.** Framing talk that is not in tempo ("okay, from the
top") still carries nothing, and genuinely free-time material carries nothing
(see below).

*Why the amendment (charter rule 9 — evidence against a standing rule):*

1. **The pulse demonstrably continues through explanation.** In the two v1
   clips, the excised explanation gaps span almost exactly whole numbers of
   beats: 5.709 s = 9.92 beat periods (10 beats, off by 44 ms) and 5.217 s =
   9.08 (9 beats, off by 44 ms). Landing within 44 ms of the grid after ~5.5 s
   of talking is not chance — the teacher kept the pulse and resumed in phase.

   > **Owner correction (2026-08-15) — argument 1 WITHDRAWN as evidence.**
   > The speaker (the owner) testifies the phase-aligned re-entries were
   > **coincidence**: restarts were ad lib ("I just restarted whenever I
   > felt like it"), not a held pulse. Under random restart the two-gap
   > phase agreement had p ≈ 2%, so the inference was reasonable — but the
   > speaker's direct account of his own production outweighs a 1-in-40
   > coincidence, and it is withdrawn. (One curiosity recorded without
   > authority: trained musicians often re-enter in tempo without feeling
   > they counted, so "ad lib" and "in phase" are not always
   > contradictions — but tags must never assume this.) **(d′) stands on
   > arguments 2–4 alone.** Tagging consequences: unmarked talking
   > stretches default to `free_time` (no pulse), per owner ruling;
   > phase-aligned re-entry is a *flag for human attention only*, never an
   > automatic held-pulse tag — this exact case proves the auto-rule would
   > false-positive; whether any future teacher holds pulse through talk
   > is an empirical per-clip question the owner rules on.
2. **It is the point of the system.** An accompanist must hold tempo *through*
   the teacher talking. A grid that deletes those beats grades the wrong thing.
3. **On teaching video, explanation is the majority of the audio**, not a brief
   aside. Under v1 a video grid covers a small fraction of its clip and any
   extractor that correctly finds beats in the rest is charged for all of them
   as false positives — measuring the convention, not the extractor.
4. **It removes a judgment step rather than adding one.** v1 required
   classifying speech as counting-vs-explaining before annotating. (d′) leaves
   only the question the annotator must answer anyway: *did I voice a beat on
   the pulse here?* The original reproducibility argument for v1 was therefore
   backwards.

**Three kinds of hole, none distinguishable in the file** (the C6 limitation,
§3): (i) in-tempo beats that went unvoiced — ruling (b) silent beats;
(ii) free-time material (out-of-time demonstration, rubato coda) where no
metric beat exists at all; (iii) framing talk. Only the grid `notes` record
which is which, and silent-beat arithmetic must exclude free-time regions or it
will count them as skipped beats.

### (e) Transitions — annotate continuously

Annotate the pulse **through** transitional material (port de bras breaks,
balances, the `frappe` balance section) wherever it is voiced. Do not exclude
transitional phrases from the grid.

*Why:* grids record reality; slicing primary from transitional is the scorer's
job, and the pipeline already models it (Gemini's per-phrase primary flag,
`QualityProfile`). Excluding regions here would be an untagged exclusion — the
same failure mode as (b), where a detector firing correctly is scored wrong with
no way to diagnose it.

### (f) Phrase-final lengthening — annotate what was heard

Mark the beat **where it actually occurred**, including where phrase-final
lengthening stretches it. Do not regularize toward an isochronous grid.

*Why:* grids record reality. Standing Lesson 5 (censor or down-weight boundary
intervals) is an instruction to the **scorer**, not to the annotator. Verified
grids will therefore contain genuinely non-isochronous intervals at phrase
boundaries, and any consumer assuming isochrony must do its own censoring.

### (g) Silent-beat crediting — owner-ratified 2026-08-18

**Stage-1 scoring is unchanged: vocalized-only, exactly as ruling (b)
leaves it.** No tag written under format 2 moves a stage-1 number, and
none may. Rung 2's verdict rests on those figures, so a crediting rule
that shifted them would silently re-open a settled kill-test.

**Crediting happens instead at the rung-4 / inference tier, as a separate
CONTINUATION metric.** It asks one question per tagged `silent_beat`
gap: when the voice returns, is the system still in phase? It is scored
as phase coherence across the gap **against the verified beats on the far
side of it** — real annotations, never invented ones.

**No beat is ever placed inside a gap.** Not by hand, not by
interpolation, not by inference from the surrounding tempo. A silence
stays a silence in the file permanently. This is the load-bearing clause:
the moment anything fills a gap, the grid stops recording what the human
produced and starts recording what a model guessed, and the corpus's
whole claim to being ground truth goes with it.

**`free_time` credits nothing, anywhere.** No metric pulse existed, so
there is nothing to hold or lose — neither success nor failure. This
covers the out-of-time coda, the demonstration tail, and every stretch of
talking the annotator did not mark (2026-08-15 owner ruling: unmarked
talking defaults to `free_time`).

**CONTINUATION reports per-clip, never as a bare average across gaps, and
pins no gate until its coverage is broad enough to support one.**
Rationale, and precedent: the rung-2 gate (§2.3, condition 2) already
refused a margin carried by one or two clips — "A margin carried by one
or two clips does not pass." As of ratification the corpus holds 33
tagged `silent_beat` gaps across 4 clips, and 19 of them (58%) are on
`adr006-exercise-1-demo` alone, so a gap-averaged CONTINUATION would be
one clip wearing a corpus's name. Until coverage is broad, it is
informational — the posture `stage1` already occupies.

*Deliberately not specified:* how phase coherence is computed. That is
rung 4's to pin, against grids that exist, with a reference
implementation — the lesson of the rung-2 gate, which was pre-registered
against provisional grids and had to be declared void and re-expressed
(§2). A convention that pre-specifies a metric it has not built repeats
that mistake.

*What (g) does not do:* it does not change ruling (b), does not add a
beat to any grid, and does not alter any committed number. It states
where the evidence in `silent_beat` tags is allowed to count, and where
it is not.

## 2. Rung-2 gate (re-expressed and BLESSED by the owner, 2026-08-14)

The pre-registered margins (+15 points step_names, +30 points vocables) were
set against *provisional* grids and are void: they named a step_names baseline
of 0.483 that no longer exists, and rung 1's belief that video clips were the
easy slice was an artifact of machine-generated references (video macro F
0.621 → 0.299 against verified truth).

Per the owner's **Option 2** ruling, this gate is derived from **only** the
ratified convention, the verified-grid baseline, and the already-adopted
metrics. **No candidate extractor has been built, run, or inspected.** The
derivation below is auditable from committed artifacts.

### 2.1 Metrics

**recall-at-tactus (R@tac)** — fraction of verified tactus beats matched by a
predicted event within ±70 ms, one-to-one. Unchanged from stage-1 recall;
renamed because the reference is now human tactus truth.

**level-collapsed precision (P_lc)** — predictions falling inside one
inter-beat interval (slot boundaries at midpoints between consecutive verified
beats) are clustered and charged **once**. A cluster containing an on-beat
match is a true positive; a cluster with no on-beat member, and any prediction
outside the annotated span, is a false positive. `P_lc = TP clusters / total
clusters`. This stops a syllable-rate detector being charged for sub-tactus
events it is correct to emit (ruling (a)'s accepted cost, fixed in the metric
rather than the ground truth) while still penalising firing where no beat
exists and firing at the wrong time.

**F_lc** — harmonic mean of R@tac and P_lc.

**Pinned edge semantics (folded from the rung-2 P0 validity gate,
2026-08-14 — the single 24-variant interpretation that reproduces every
committed number in §2.2):** cluster slots are beat-centered with
boundaries at midpoints between consecutive verified beats; the annotated
span extends half the **median** inter-beat interval beyond the first and
last beat, and predictions outside it are charged as **individual** false
positives; a cluster is a TP iff it contains a prediction one-to-one
matched to a beat at ±70 ms by the frozen matcher; per-clip
F_lc = harmonic mean of R@tac and P_lc; slice rows are per-clip macros.
Reference implementation: `scripts/rung2_kill_test.py`. A metric that
exists only as prose is not yet a metric — future gates ship with
reference code.

*Honesty note:* level-collapsing raises the **baseline** too — overall F 0.383
→ F_lc 0.452, and `rig-numbers-4-4-80-triplet` 0.469 → 0.882, because
Whisper's 48 triplet tokens collapse into 16 slots. The new metric raises the
bar rather than lowering it.

### 2.2 Baseline (whisper-word-starts, 28 verified grids, macro per slice)

```
                  n     R@tac   P_lc    F_lc
ALL              28     0.449   0.506   0.452
numbers          14     0.568   0.604   0.577
step_names       13     0.349   0.363   0.343
vocables          1     0.062   1.000   0.118
```

Slices are by the case's `count_style` tag; the three verified video clips
fall inside `numbers` (adr010) and `step_names` (adr006-exercise-1-demo,
frappe). `mixed` has no verified member and gates nothing.

### 2.3 The gate

Rung 2 passes if **all four** hold:

1. **Primary — step_names recall.** `R@tac ≥ 0.499` on the step_names slice
   (baseline 0.349, i.e. **+0.15 absolute**), the slice where the word channel
   is weakest and the acoustic claim is strongest.
2. **Consistency, not just the mean.** R@tac improves on **≥ 9 of the 13**
   step_names clips. A margin carried by one or two clips does not pass.
3. **Vocables — decisive, and n=1.** On `rig-vocables-4-4-100-clean`,
   `R@tac ≥ 0.60` **and** `P_lc ≥ 0.50` (baseline R@tac 0.0625 — Whisper emits
   one token for the whole phrase). This is a tenfold recall requirement on a
   **single clip**, which is the entire vocables slice; it must never be quoted
   as a slice average. It is the cleanest available test of whether an acoustic
   channel earns its place.
4. **No regression elsewhere.** numbers-slice `F_lc ≥ 0.527` (baseline 0.577
   − 0.05). The acoustic channel may not buy step_names by breaking numbers.

### 2.4 Reading the results

- **F, P and R are trustworthy.** The self-consistency measurement found beat
  *identification* perfectly reproducible (24/24 beats, both passes), so these
  metrics carry no measurable annotator variance.
- **Signed-asynchrony differences below ~25 ms are inside annotator noise** and
  may not be claimed as results.
- **Cohort offset.** **25** grids are seed-anchored (corrected 2026-08-14
  from the originally stated 21 — the four session-2 audio grids were also
  anchored), 3 are from-scratch, and the methods differ by ~20 ms
  systematically. Asynchrony comparisons must be within-cohort or must
  correct for it. F/P/R are unaffected.
- Per charter rule 5, a documented **negative result** with per-clip evidence
  satisfies rung 2 as fully as a pass, and ends it (ADR-016: the reset stops
  and P2 strengthens).

## 3. Tooling constraint behind (b) and (e)

*Original text (v1), kept because rulings (b), (d′) and (e) were decided under
it:* `beats_from_label_text()` parses **only the start time** of each Audacity
label and discards the label text. The schema is a flat list of floats. **A grid
beat therefore cannot carry a tag** — silent-vs-voiced, prep-vs-body,
primary-vs-transitional are all unrepresentable. Every annotation decision
reduces to a binary: in the list, or not. Any future convention needing a tag
requires a format + tool change, i.e. a new EVAL-CHANGE rung.

> **LIFTED at rung 2.5 (2026-08-16) — for *spans*, not for individual beats.**
> Grid format 2 adds an optional `regions` list tagging the three holes (d′)
> names: `silent_beat`, `free_time`, `excluded_explanation`. The Audacity round
> trip carries them as region labels, and the §4 QC checks are suppressed
> inside them. See [beat-grids.md](beat-grids.md).
>
> **What this does and does not change.** The format now *permits* a tag; the
> rulings above are unchanged and still govern what the annotator marks —
> (b) is still vocalized-only, and a `silent_beat` region describes a stretch
> the annotator did not mark, it does not add beats. Changing a ruling is owner
> business (charter rule 9), and none is proposed here. Individual beats still
> carry no tag: prep-vs-body and primary-vs-transitional remain
> unrepresentable, so ruling (c)'s and (e)'s reasoning stands as written.

## 4. Per-clip workflow (rung 1.5)

1. **Prepare.** Extract 16 kHz mono wav to a local working dir (mandatory for
   the four video clips, so Audacity handling is uniform); run
   `python -m musical_perception.annotation to-labels <id>`.
2. **Correct.** Owner opens the wav in Audacity, *File > Import > Labels* with
   `evals/grids/<id>.labels.txt`, deletes non-beats, nudges to vowel onsets,
   adds missed beats, then *File > Export Labels*.
3. **Commit the correction.** On the owner's explicit confirmation only:
   `python -m musical_perception.annotation from-labels <id> <file> --verified`.
   `--verified` is an owner act; agent sessions never apply it unprompted.
4. **Correction stats.** Diff verified `beats` against the grid's frozen
   `onsets`: `n_kept`, `n_deleted`, `n_added`, median |nudge| of survivors.
   These measure peakRate's pre-annotation quality and set rung-2 expectations.
5. **Grid-implied-BPM check.** Compute BPM from the verified beats (60 / median
   inter-beat interval) and compare against the case label's `marking_bpm`.
   **Disagreement > 4% is flagged** — in the grid's `notes` and the ledger — and
   resolved by the owner. It is a cross-check on the annotation, not on the case
   file: `evals/cases/` is untouchable this rung. (4% is the field-standard
   Acc1 window and the human-tapping noise floor, Standing Lesson 7.)
6. **Observations.** Owner's per-clip notes go into the grid's `notes` field and
   the session log.

### Self-consistency measurement (calibration clips)

The **two calibration clips are annotated twice**, in independent passes, and
the two passes are scored against each other with the stage-1 matcher. The
result — agreement rate and median |Δt| between passes — is recorded in the
ledger as this corpus's **intra-annotator reliability**, the noise floor beneath
which no rung-2 result can claim significance.

Never run `annotation generate --force` on a corrected grid: it overwrites
`beats` and destroys the human pass.

### QC checks — owner amendment, 2026-08-14

Ratified from rung-1.5 evidence (report §6.7): the step-5 BPM-vs-label
check **false-passed** (+3.51%) on a grid carrying three spurious labels
and a missing beat; what actually caught that error — and three further
owner-export errors — were two checks not in the ratified workflow:

- **Minimum-IOI check:** flag any inter-beat interval implausibly short
  for the clip's tempo (a spurious double-mark signature).
- **Within-phrase IOI-spread check:** flag excess IOI variance computed
  *within* phrases, suppressed in free-time regions (a missing-beat or
  stray-label signature that per-clip medians hide).

Both are **required** parts of the per-clip workflow from this date,
alongside (not replacing) the BPM-vs-label check.

**Implemented at rung 2.5 (2026-08-16):**

```bash
python -m musical_perception.annotation qc [ID ...]
```

Frozen thresholds, pre-registered before the checks were ever run — a
misfiring threshold is proposed for owner ratification, never quietly
retuned: min-IOI < 0.5 × the clip's median IOI; within-phrase IOI CV
> 15%, over phrases split at intervals above 1.75 × the median, minimum 3
intervals, CV = population sd / mean; BPM-vs-label > 4% as before. Both
amendment checks are suppressed inside `regions` (§3) — the missing
capability that forced the ad-hoc runs.

*Validation against the human record:* the implementation reproduces 24 of
the 25 grid-implied BPM figures the owner recorded by hand in grid `notes`
(±0.02 BPM). The 25th, `rig-names-4-4-104-coda`, reproduces exactly once
its out-of-time coda is tagged `free_time`; so does
`rig-names-4-4-63-adagio`'s recorded 15.0% within-phrase CV once its six
unvoiced beats are tagged. Two of those six gaps compress to 1.67–1.72 ×
the median under rubato — **below** the 1.75 × break ratio — so tagging,
not threshold tuning, is what makes the check correct on rubato material.

## 5. Provenance

Provisional grids are peakRate suggestions and measure *words-vs-peakRate*, not
*words-vs-truth*. Only grids with `provisional: false` participate in typed-gate
decisions; stage1 reports `aggregate_provisional` and `aggregate_verified`
separately, and this rung's job is to move all 30 clips into the second bucket.
