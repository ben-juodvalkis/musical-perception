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
explicit deliverable is a taggable grid format.

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
`rig-names-4-4-104-explained`. Both have their explanation stretches excised.
They are consistent with each other and flagged here; re-annotating them under
(d′) is an owner decision, not yet taken.

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

## 2. Rung-2 gate re-expression (§6 of the decision brief)

The pre-registered rung-2 margins (+15 points step_names, +30 points vocables)
were set against *provisional* grids and shift with this convention. Per the
owner's ruling — **Option 2** — the gate is re-expressed at the **end of rung
1.5**, under these constraints:

1. Derived from **only** the ratified convention, the verified-grid baseline
   numbers, and the already-adopted metrics.
2. **No candidate peeking** — no rung-2 extractor performance may inform the
   threshold. Pre-registration is preserved by construction.
3. Intended shape: **recall-at-tactus + level-collapsed precision** (a syllable
   detector is not charged for correct sub-tactus events).
4. **Vocables must remain a decisive-win requirement.** Whisper produces one
   token for the entire `rig-vocables-4-4-100-clean` phrase; any acoustic
   channel that does not win decisively there has not earned the reset.
5. Owner-blessed before the CURRENT RUNG pointer advances to 2.

Recorded as a charter note at
[agent-charter.md](../research/agent-charter.md) rung 2 so no future session
reads the old margins as final.

## 3. Tooling constraint behind (b) and (e)

`beats_from_label_text()` (`src/musical_perception/annotation/grids.py`) parses
**only the start time** of each Audacity label and discards the label text. The
schema is a flat list of floats. **A grid beat therefore cannot carry a tag** —
silent-vs-voiced, prep-vs-body, primary-vs-transitional are all unrepresentable.
Every annotation decision reduces to a binary: in the list, or not.

Any future convention needing a tag requires a format + tool change, i.e. a new
EVAL-CHANGE rung.

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

## 5. Provenance

Provisional grids are peakRate suggestions and measure *words-vs-peakRate*, not
*words-vs-truth*. Only grids with `provisional: false` participate in typed-gate
decisions; stage1 reports `aggregate_provisional` and `aggregate_verified`
separately, and this rung's job is to move all 30 clips into the second bucket.
