# Research Agent Charter — the rhythm-core reset loop

**Date:** 2026-08-09
**Status:** Active. Governs every autonomous session working on
[ADR-016](../adr/016-rhythm-core-reset.md)'s reset. Companion documents:
[voice-as-drum literature review](voice-as-drum-review.md) (the textbook),
[RESEARCH-LOG.md](RESEARCH-LOG.md) (the memory),
[agent-environment.md](agent-environment.md) (cloud setup).

> **CURRENT RUNG: M — the marathon (COMMISSIONED 2026-08-14 at the
> rung-2 verdict: PASS).** Rung 2 blessed and merged — all four §2 gate
> conditions held with wide margins (step_names R@tac 0.349→0.719 with
> 12/13 clips improved and zero losses; vocables 0.062→0.875; numbers
> F_lc 0.577→0.926). Sessions now execute Rung M's per-session contract:
> one increment on the highest-ranked non-BLOCKED workstream. The owner
> reviews in weekly batches and clears the BLOCKED queue; every hard
> rule (eval files, provisional-never-gates, HELD-OUT/SEALED, no
> self-blessing) is unchanged.

## Mission

Maximize honest performance of the perception + precision pipeline on the
benchmark by executing ADR-016's falsification plan. Research posture:
accuracy first; cost and latency out of scope. "Honest" means: DEV-split
improvements that follow the rules below, survive owner-scored SEALED
milestones, and are reported in the house style (pre-registration, luck
flags, classified regressions).

## Session boot sequence (every session, before any work)

1. Read this charter in full.
2. Read [RESEARCH-LOG.md](RESEARCH-LOG.md) — at minimum the Standing
   Lessons section and the last five entries — **as it stands on
   `origin/agent/marathon`, not only main's copy**
   (`git show origin/agent/marathon:docs/research/RESEARCH-LOG.md`;
   owner-ratified 2026-08-30, converging with the same night's A3-30):
   a completed-but-unmerged workstream is invisible from main by
   construction — two sessions built W11 on the same night for exactly
   this reason. Any workstream carrying a RESULTS entry on the branch
   is COMPLETE-pending-review and is not taken again. Never re-attempt
   an approach the ledger records as falsified without new evidence.
3. Read `docs/evals/baseline.md` (and `evals/baseline.json`) for the
   current blessed state.
4. Confirm the CURRENT RUNG above and its prerequisites. If prerequisites
   are unmet, write a BLOCKED ledger entry saying exactly what is needed
   and stop.
5. Work the rung. 6. Before ending: ledger entry appended, branch pushed.

## Fitness function

- **Primary:** tier-1 committed accuracy per field (tempo, meter_triple,
  counts) on the DEV split; stage-1 pulse F-score against beat-grid
  annotations once rung 1 lands.
- **Secondary:** OE2 distribution shrinking; ECE not worsening; abstention
  behaving as designed (coverage reported, never punished as wrong).
- **Never optimized:** anything on the SEALED split (owner-scored only,
  off-repo); knife-edge rows near tolerance (ADR-015: they gate nothing at
  this corpus size).

## Data splits

- **DEV = all 30 current tier-1 cases.** Every one of them has been in the
  iteration loop since capture, so none can be retroactively sealed — that
  is the honest label, not a choice.
- **SEALED is built only from future capture:** at least one third of every
  new batch, including at least one entire recording session/teacher, held
  **off-repo on the owner's main machine — never on the Air runner, never
  in this repository** (see [agent-environment.md](agent-environment.md)).
  Sealed clips have no case files, no traces, no filenames in this
  repository. An agent session must never
  request, reconstruct, or reason about sealed content; encountering a
  reference to it is a reason to stop and flag.
- **Ingestion protocol for new capture:** freeze traces → agent
  pre-annotates beat grids (peakRate suggestions, marked `provisional`) →
  **owner** verifies/taps (ground truth is human) → owner assigns
  DEV/SEALED at ingestion, before any iteration touches the batch.
  Annotation targets **vowel onsets** (P-centers), never word starts
  ([Review 1 §2.9](review-1-onsets-pcenters.md)).
- **Ballet Barre 1 sections (re-assigned 2026-08-14, amending the
  2026-08-09 all-DEV ruling per the rung-1.5 report §8.1):** split
  **before any ingestion**, which cannot be redone later.
  - **HELD-OUT (~one third, weak seal), split at the EXERCISE level
    (amended 2026-08-14):** the batch is 12 exercises, each with demo +
    execution videos; demo and execution of one exercise are near-
    duplicate data, so the unit of splitting is the exercise, never the
    file (Vision 13's class-level-split rule applied). The owner draws
    **4 of the 12 exercises at random — one from each quarter of the
    barre order (1–3, 4–6, 7–9, 10–12), which is public metadata, never
    by content** — and **moves ALL files of those exercises off the Air
    to the main machine** before the first ingestion session exists. The
    list lives on the main machine, never in this repository. "Weak"
    because the source is public and thus reconstructable in principle;
    the property being protected is *never iterated on*, which physical
    removal provides. **Containment is not agent-auditable (W0
    2026-08-27, PROPOSED):** the Barre-1 DEV media still lives on the
    runner at `video/youtube/Ballet Barre 1`, and with the batch split
    8 DEV / 4 HELD-OUT, *listing that directory names the held-out four
    by complement*. Agents must therefore **never enumerate it** — the
    only available audit is itself the leak. Confirmation that the four
    exercises left the Air can come only from the owner, as a dated
    one-line attestation appended to the ledger. **Attested 2026-08-28**
    (name check on the Air by the owner's eyes only; see the ledger
    entry of that date). The enumeration prohibition stands forever.
  - **DEV (the remainder):** ingested per the protocol above; truth
    labels agent-proposed `provisional` until owner-verified.
  - The four existing video demos stay as they are (frozen traces,
    blessed baseline); new material adds alongside, never replaces.
  - **SEALED (strong) is unchanged:** only the owner's own future
    capture, per the bullet above. HELD-OUT and SEALED are distinct
    labels and must not be conflated in reports.

## Rules of engagement

1. **Branches only.** All work on `agent/rung-<N>-<slug>`. Never push
   `main`. Never run `evals bless` — blessing is the owner's act.
   *Sole ratified exception (owner, 2026-08-24):* the nightly runner may
   push `main` carrying **only** `logs/run-summaries.md`, after its pull;
   a push touching any other file voids the carve-out.
2. **Eval integrity.** Never **modify** any existing file under
   `evals/cases/`, `evals/traces/`, or `evals/baseline.json`, and never
   touch the scorer/harness code (`src/musical_perception/evals/`) in a
   pipeline rung. Eval *infrastructure* changes (new metrics, new suites)
   happen only in rungs/workstreams whose explicit deliverable is eval
   infrastructure (W1 is one), flagged **EVAL-CHANGE** in the report,
   never bundled with pipeline changes. **Add-only carve-out for
   ingestion:** creating NEW case and trace files for new material is
   permitted and expected — every agent-authored label ships with
   `maturity: provisional`. **Add-only carve-out for sidecars
   (owner-ratified 2026-08-28):** in an EVAL-CHANGE increment, agent
   sessions may ADD new derived-evidence files inside existing trace
   directories (e.g. `pulse.json`) — never modifying any existing file,
   with the source media checksum-verified against the trace's stored
   hash and byte-identical suite output proven before merge. **Provisional rows never gate anything and
   are always reported as a separate slice**; only owner-verified rows
   participate in typed-gate decisions. Verification (owner corrects/
   confirms labels and grids, flips `maturity: verified`) is an owner
   act, requested via a BLOCKED note.
3. **Pre-registration.** Before implementing, write the expected
   flips/deltas and reasons into the ledger entry; score the predictions
   honestly in the report (ADR-015 discipline).
4. **Typed gates (ADR-015).** Logic changes: zero-regression. Measurement
   changes: diagnosed-regression — net improvement on the primary metric
   AND ECE, zero undiagnosed regressions, every regression classified
   fake-green-lost / genuine-trade / knife-edge.
5. **Negative results are deliverables.** A documented dead end with
   per-clip evidence satisfies a rung as fully as a win, and ends it.
6. **One bounded change per session.** No opportunistic refactors outside
   the rung's scope; park discoveries in the ledger's Backlog notes.
7. **House honesty style.** Luck flags, disclosed retries, "the totals hid
   X" — follow the ADR record's example. A green earned by chance is
   annotated as such in the case notes. **Redaction by default (A5-30,
   owner-ratified 2026-08-30):** agent-authored repo text does not quote
   transcript lines verbatim when they name steps — paraphrase or use
   opaque ids, the same containment posture as the enumeration ban.
8. **Blessing is human.** Recommend, don't bless.
9. **Charter conflicts.** If evidence says a rung or rule is wrong, stop
   and write a ledger entry proposing the amendment — never silently
   deviate. Charter changes are owner-reviewed PRs.

## The goal ladder

Each rung below has prerequisites, deliverables, and a ready-to-paste
`/goal` condition (for interactive or `claude -p` sessions). Scheduled
runs — the Air's nightly `scripts/air-nightly.sh` or a cloud overflow
routine — use the runner-agnostic standing contract in
[agent-environment.md](agent-environment.md), which delegates to the
CURRENT RUNG's condition here. Conditions are contracts:
measurable end state + stated proof in the transcript + constraints +
turn bound. The evaluator judges only what the session surfaces, so the
proof clauses are mandatory, not decorative.

### Rung 0 — owner: runner setup + data logistics *(no /goal — checklist)*

Per [agent-environment.md](agent-environment.md): the **Air** configured
as primary runner (dedicated account, persistent env via
`scripts/cloud-setup.sh --live`, DEV recordings synced, power settings,
`scripts/air-nightly.sh` schedule staged); SEALED vault on the owner's
main machine; repo privacy verified before any recording is committed;
one supervised dry-run of rung 1 completed before the schedule is
enabled. Cloud overflow (DEV audio committed via the `.gitignore`
exception, environment secrets/network, overflow routine) is Path B —
optional, later.

### Rung 1 — annotation tooling + stage-level scoring (EVAL-CHANGE rung)

Prereq: rung 0. Deliverables: a beat-grid annotation format (per-clip beat
times at vowel onsets, with a `provisional` flag); a tap-assist tool that
pre-annotates grids via peakRate for owner correction; a `stage1` eval
suite reporting pulse precision/recall/F and signed asynchrony per clip
against grids; Acc2 (aliasing `truth_in_family`) + OE1/OE2 added to tier-1
reporting; the onset-count-vs-token-count sanity guard on trace loading;
docs updated.

```
/goal Per docs/research/agent-charter.md rung 1: a beat-grid annotation
format, a peakRate-based tap-assist annotator, and a stage1 eval suite
(pulse precision/recall/F + signed asynchrony per clip) exist; provisional
grids are generated for all 30 DEV clips and marked provisional=true;
tier-1 reporting includes Acc2 and OE1/OE2; the onset-vs-token sanity
guard runs on trace load; pytest is fully green — proven by the complete
pytest and `python -m musical_perception.evals run --suite tier0,tier1,stage1`
outputs in the transcript. Constraints: no file under evals/cases/ or
evals/traces/ is modified and evals/baseline.json is untouched (prove with
`git diff --stat main` output whenever completion is claimed); all work
committed on branch agent/rung-1-stage-scoring; a dated entry appended to
docs/research/RESEARCH-LOG.md. Or stop after 35 turns.
```

*(Rung 1.5 — owner: verify/correct the provisional grids for all 30 DEV
clips using the tap-assist tool; flip `provisional` to `verified`. Grids
are not ground truth until this happens.)*

### Rung 2 — peakRate acoustic extractor: the kill-test

> **VERDICT: PASS — blessed and merged 2026-08-14.** All four §2 gates
> held: step_names R@tac 0.719 (≥0.499), 12/13 clips improved with zero
> losses, vocables 0.875/0.875, numbers F_lc 0.926. Full results:
> `docs/research/rung2-kill-test.md`. Anchoring caveat stands: quote the
> from-scratch cohort for external magnitude claims.

Prereq: rung 1 blessed AND grids owner-verified. Deliverable: the
acoustic pulse channel ([Review 1 "steal this first"](review-1-onsets-pcenters.md)),
plus the Whisper-word-onset baseline scored on the same grids.

> **Gate re-expressed and BLESSED (2026-08-14, rule 9):** the margins in the
> condition below were pre-registered against *provisional* grids. Per the
> owner-ratified annotation convention (`docs/evals/annotation-convention.md`
> §6), the gate is re-expressed at the END of rung 1.5 — from the
> convention, the verified-grid baseline numbers, and the already-adopted
> metrics only (recall-at-tactus + level-collapsed precision is the
> intended shape; a decisive vocables win remains mandatory; no peeking at
> candidate-extractor performance). The owner blesses the re-expressed
> gate before the CURRENT RUNG pointer advances to 2. Until then, do not
> treat the margins below as final.

```
/goal Per docs/research/agent-charter.md rung 2: a peakRate acoustic pulse
extractor (envelope-derivative peaks, voiced-gated, syllable-nuclei
regions) exists in the precision layer and, on the owner-verified DEV beat
grids, its stage1 pulse F-score beats the Whisper-word-onset baseline by
at least 15 points on the step_names slice and 30 points on vocables, with
signed asynchrony reported — proven by the full stage1 eval output for
both extractors in the transcript. If after a genuine attempt the
extractor cannot beat the baseline, a ledger entry documenting the
negative result with per-clip evidence also satisfies this goal (ADR-016:
the reset stops here and P2 strengthens). Constraints: no file under
evals/cases/, evals/traces/, or src/musical_perception/evals/ modified;
evals/baseline.json untouched (prove with `git diff --stat main` when
claiming completion); branch agent/rung-2-acoustic-pulse; dated
RESEARCH-LOG.md entry appended. Or stop after 40 turns.
```

### Rung 2.5 — taggable grid format + ratified QC checks (EVAL-CHANGE rung)

Prereq: rung 2's verdict recorded. Must complete **before** any Ballet
Barre 1 ingestion (the video material is explanation-heavy, where the
current format's untagged holes are worst). Scheduled 2026-08-14 from the
rung-1.5 report §8.2–8.3, owner-ratified.

Deliverables: (1) lift the C6 limitation — an **additive-only** grid
extension distinguishing silent-beat, free-time, and excluded-explanation
regions: `beats` remains a flat time list so **all verified grids remain
valid untouched**; tags land in an optional parallel structure, with the
Audacity round trip extended accordingly. (2) Implement the two
owner-ratified QC checks (convention §4 amendment, 2026-08-14): minimum-IOI
and within-phrase IOI-spread, suppressed in free-time regions — the checks
that caught four real export errors while the BPM-vs-label check
false-passed. (3) Annotation-method metadata per grid (`anchored` vs
`from-scratch`, per the rung-1.5 cohort-offset finding). Gated as an
EVAL-CHANGE: no pipeline changes bundled; stage1 outputs on the 28
verified grids must be byte-identical before and after the format change.

### Rung 3 — accent-periodicity meter votes

> **VERDICT: NEGATIVE — owner-accepted 2026-08-24 (W2, run 2026-08-20).**
> Accent periodicity in this corpus sits at the count phrase (lag 8), not
> the bar — half the clips carry no significant periodic accent at any
> lag — and salience-clock templates cannot separate 2/4 from 4/4
> (r = 0.90) or 3/4 from 6/8 (r = 0.93) on any data. Family-level
> accuracy is **6/13 (chance)** — the entry's "9/13" was corrected
> against the committed artifact at review. **Accent periodicity folds
> into W5 as one observation channel; no standalone meter iteration.**
> Stale-number correction (amendment 5, owner-ratified 2026-08-24): the
> baseline's non-4/4 truth is **2-of-9**, and only two rows
> (`rig-numbers-2-4-120-clean`, `rig-numbers-3-4-90-clean`) were ever
> reachable by meter-only code — the "1-of-8 … 0-of-3" below is the
> stale pre-registration-era text, kept for the record.

Prereq: rung 2 blessed (positive). Deliverables: S-AMPH-style delta/theta
band phase reader + Povel–Essens/Parncutt salience-clock scoring
([Reviews 1 §3.4, 3 §2](review-3-beat-meter-models.md)), evaluated as
votes against the 24 checklist clips.

```
/goal Per docs/research/agent-charter.md rung 3: an accent-periodicity
meter module (delta/theta envelope phase + salience-clock scoring over
candidate (period, phase, duple/triple) hypotheses) exists and, on the DEV
split, non-4/4 meter recognition beats the blessed baseline's 1-of-8 with
the three numbers-counted 2/4 and 3/4 clips (currently 0-of-3) as
must-move rows, while 4/4 clips currently green stay green within ADR-015
typed-gate rules — proven by full tier-1 meter_triple results for baseline
vs new module in the transcript, every regression classified. A documented
negative result with per-clip evidence also satisfies the goal.
Constraints: eval files untouched (prove with `git diff --stat main`);
branch agent/rung-3-accent-meter; dated RESEARCH-LOG.md entry. Or stop
after 40 turns.
```

### Rung 4 — the joint posterior

Prereq: rungs 2–3 blessed. Deliverable: bar-pointer HMM (Krebs-2015 state
space, PIPPET/Poisson observation model per
[Review 3 §(a)–(b)](review-3-beat-meter-models.md)) replacing
`normalize_tempo`'s snap, `interpret_meter`'s stack, and `subdivision.py`'s
heuristic; log-Gaussian tempo prior (T₀≈100–110, σ≈1.2–1.4 octaves) and
exercise-conditioned priors at level selection only; posterior mass as
confidence; entropy abstention; ADR-014 alternates carrying posterior
weights. Gated as a **measurement change** (diagnosed-regression gate).
Read the review's top-5 papers first; write the pre-registration before
any code. **Owner direction 2026-08-26 (see the ledger entry of that
date):** the meter state variable is replaced by a factored
representation — division (duple/triple) as its own axis, grouping as a
per-level ladder (2/3/4/6/8/12…) with the bar one rung and the count
phrase another; the time-signature label is derived late, outside the
state space, only where a consumer needs notation.

```
/goal Per docs/research/agent-charter.md rung 4: a joint-posterior rhythm
core (bar-pointer state space over period, phase, meter, subdivision, with
an event-based salience observation model and soft priors) replaces the
normalize/interpret/arbitrate stack behind the existing MusicalParameters
contract, and on the DEV split it net-improves tier-1 tempo AND
meter_triple committed accuracy AND does not worsen ECE, with zero
undiagnosed regressions and every regression classified per ADR-015 typed
gates — proven by full before/after tier-0, tier-1, and stage1 outputs in
the transcript plus the pre-registered prediction scorecard. Constraints:
eval files untouched (prove with `git diff --stat main`); tier-0 tempo
stays 25/25; branch agent/rung-4-joint-posterior; dated RESEARCH-LOG.md
entry including the prediction scorecard. Or stop after 60 turns.
```

### Rung 5 — ensembled semantics *(live-perception rung: needs GEMINI_API_KEY)*

N ≥ 5 draws across ≥ 2 model families; per-draw frozen traces;
distributions consumed downstream (Cemgil-style: marker labels as observed
grid-assignment switches). Also re-runs the Feb-2026 model comparison.
Backlog note (2026-08-09): include an audio-native small local model
(Qwen2-Audio class) as a cheap third ensemble vote — measure it, don't
prioritize it (see [agent-environment.md](agent-environment.md) local
models policy). Condition to be finalized when rung 4's shape is known —
the meta-rung drafts it.

### Rung 6 — baselines benchmark *(independent; can run parallel any time after rung 1)*

The six-tool plan from [Review 4 §(a)](review-4-tools-baselines.md), run on
raw audio AND marker streams, scored with mir_eval/madmom evaluators on the
verified grids, reported as a comparison table in `docs/research/`.

```
/goal Per docs/research/agent-charter.md rung 6: the Review-4 baseline
benchmark is complete — librosa suite, Beat This!, madmom (min_bpm=40),
Essentia RhythmExtractor2013, and the syllable-nuclei hybrid each run on
all 30 DEV clips (raw audio and marker-stream conditions), scored against
the verified grids with F@70ms, CMLt, AMLt-with-triples, Acc1/Acc2, OE2 —
proven by the committed results document docs/research/baseline-benchmark.md
whose summary table appears in the transcript, with any tool that could
not be installed documented with the exact failure. Constraints: eval
files untouched (prove with `git diff --stat main`); branch
agent/rung-6-baselines; dated RESEARCH-LOG.md entry. Or stop after 35 turns.
```

### Rung 7 — the RETIRED sweep *(logic change: zero-regression gate)*

Delete the unreachable ~1,200 lines named in ADR-016 (trigger/wakeword
path, legacy text-merge, Feb-2026 scripts, dead subdivision fields) once
rung 4 lands. Proof: pytest green, tier suites byte-identical outcomes.

### Rung M — the marathon *(COMMISSIONED by the owner 2026-08-14, at the rung-2 PASS verdict)*

Staged autonomy delivered its gate: rungs 1→2 ran rung-by-rung with
daily blessings while the measuring instruments were built (rung 1),
ground truth was humanized (rung 1.5), and the kill-test decided the
strategy (rung 2: PASS). Rung M replaces per-rung blessing with
**batch review**:

- **Workstreams** (ranking refreshed at commissioning; the meta-rung
  re-ranks; status marks owner-ruled 2026-08-24): W1 = rung 2.5,
  taggable grid format + ratified QC checks — gates all ingestion;
  includes the vocables dropped-beats QC listen *(BLESSED 2026-08-24)* ·
  **W1.5 = provisional-slice eval infrastructure (COMMISSIONED
  2026-08-24, EVAL-CHANGE, ranked first among non-BLOCKED workstreams):**
  `maturity: provisional|verified` as a case-file key (default
  `verified`, existing 30 cases untouched in meaning); provisional rows
  excluded from `compare_outcomes`, the typed gates, and every headline
  aggregate; reported as their own slice with their own n; tag
  vocabulary gains the **accompaniment-only** condition (the six
  pianist-playing Barre-1 takes, owner ruling B5). Proof style as W1:
  byte-identical suite output on the existing corpus plus tests proving
  the exclusion. Gates W4 case files, W7 scoring, and all future
  capture ·
  W2 = rung 3, accent-periodicity meter votes *(COMPLETE 2026-08-20,
  negative — folded into W5)* · W3 = rung 6, baselines benchmark
  *(COMPLETE 2026-08-21; raw-condition remainder — 24 rows + optional
  BeatNet — **UNBLOCKED 2026-08-27**, W0 verified all 24 rig MP3s
  present in `audio/rig/` on the runner, closing C5; ranked 3)* ·
  W4 = Ballet Barre 1 DEV ingestion (requires W1; the 8 remaining
  exercises only — 4 are HELD-OUT on the owner's machine) *(traces
  frozen 2026-08-22; **case files UNBLOCKED 2026-08-27** — W1.5's
  `maturity` key is in the code at `evals/cases.py:27,53,144`, so
  ruling A6's objection is discharged; ranked 2)* · W5 = rung 4,
  joint posterior (requires W2's evidence; **OWNER-STARTED — scheduled
  sessions must never take this workstream**: it runs in an
  owner-attended session on the top-tier model; if W5 is next and
  unstarted, treat it as BLOCKED-on-owner and move on) *(**PHASE 1
  LANDED 2026-08-28 — by explicit owner override of the pre-registered
  gate** (tempo tied instead of improving; four diagnosed genuine-trade
  losses accepted for meter 12→13, ECE 0.1998→0.1815, weighted ADR-014
  families — see ADR-017 and the 2026-08-28 rulings entry). W5 remains
  OPEN for the pulse-fed continuation after W11, still owner-started)*
  · W6 = rung 5,
  ensembled semantics · W7 = pose/gesture channel prototyping on the
  Barre 1 video (after W4) *(velocity-minima prototype COMPLETE
  2026-08-23, negative — movement folds into W5 as a weak vote; if
  revisited: nod-kinematics / phrase-arrival segmentation first, per
  Review 5 and the W7 entry)* · W8 = rung 7, RETIRED sweep (after W5) ·
  parked from rung 2: relative (speech-band-level) nuclei silence floor
  for quiet clips *(EXECUTED as W2.5 2026-08-26 — hypothesis FALSIFIED:
  the floor discards 0 of 802 beats, and a faithful implementation
  changes no emitted event. The real defect was a fused-nucleus
  artifact; the one-event-per-nucleus rule was dropped instead.
  PROPOSED)* ·
  **W9 = tempo-estimator robustness, the pulse → BPM step (COMMISSIONED
  2026-08-26 by the owner; pipeline workstream; evidence in the owner
  probe ledger entry of that date).** `calculate_tempo`'s median-of-
  consecutive-gaps is destroyed by extra ticks in a way a periodicity
  estimator is not (owner probe: 11/23 vs 20/23 on rig clips against the
  filename metronome — indicative only; wrong stream, wrong metric, see
  the entry's "does NOT establish" list). Scope: both tempo arms
  (`calculate_tempo` and `detect_onset_tempo`) and the arbitration
  between them, measured on the **shipping path** with the **blessed
  tier-1 metrics**, reporting Acc1/Acc2/OE1/OE2 beside the
  committed-accuracy delta; the 70–140 band is a separate named
  question, since `interpret_meter` gates arbitration on it and the
  probe shows it turning a correct 60.9 BPM reading into 121.7.
  Standard typed gate; it moves tier-1 outcomes, so it needs an owner
  re-bless. **RANKED 1 by W0, 2026-08-27** (ratified at blessing,
  2026-08-28): re-derived from `evals/baseline.json` on the shipping path
  and the blessed metric, tier-1 committed tempo is **17/24 (0.708) when
  the truth lies inside 70–140 and 0/5 (0.000) when it lies outside**,
  every one of those five predictions folded back in-band. Five of the
  twelve tempo failures are that shape; the honest ceiling from band work
  alone is the three truth-in-family rows (0.586 → 0.690 = Acc2@8%), with
  the in-band 0.708 the thing at risk. The band is hard-coded at
  `precision/tempo.py:32-33`, `:124-125`, and inside `interpret_meter` at
  `:249-259`, where **both** arbitration arms are conjoined with
  `70.0 <= bpm <= 140.0` — confirming the probe's claim that it gates
  arbitration, not merely normalization. Standing Lesson 2 has named this
  exact band as an error since 2026-08-09.
  **COMPLETE 2026-08-28, BLESSED same day:** `normalize_tempo`'s hard
  fold replaced by MAP level selection under a log-normal prior derived
  from the band itself (+3 tempo, +1 meter_triple, +1 counts, ECE
  0.265→0.200, zero outcome-level regressions; six outcome changes
  blessed into the baseline). Negative finding with per-clip evidence:
  the band inside `interpret_meter`'s arbitration is a **level
  discriminator and STAYS** — deleting it costs the three marker-arm
  wins. Backlog **W9-b** parked: the derivation table passes Gemini's
  subdivision claim through even when the selected metric level differs
  from the one the observation was made under (`rig-names-2-4-160-long`
  meter credit 0.5→0.0, classified genuine-trade) — a natural W5
  absorbee.
  Policy: each
  session advances exactly one workstream — the highest-ranked one not
  BLOCKED; blocked workstreams get a BLOCKED ledger note (the owner's
  task queue) and the session moves on. The loop never idles while any
  workstream is open.
  **W11 = pulse sidecars (COMMISSIONED 2026-08-28 by owner ruling;
  EVAL-CHANGE):** record `pulse.json` sidecars (rung-2 acoustic pulse
  events) into all 30 existing trace directories under the ratified
  sidecar carve-out — 24 clips from the committed rig MP3s, 6 from the
  video clips' audio, each checksum-verified against the trace's
  `media_sha256`; loader exposes them; stage1 may gain a peakRate
  source as part of the same increment; byte-identical tier outcomes
  proven (nothing consumes the sidecars yet). Unblocks W5's
  continuation (the owner-probed between-beat discriminator, word-span
  de-confounded — see the 2026-08-28 rulings entry). ·
  **W12 = the factored meter slice (COMMISSIONED 2026-08-28 by owner
  ruling; EVAL-CHANGE):** a REPORTED-ONLY factored meter score beside
  meter_triple — division scored as measured (duple/triplet/none),
  grouping scored with duple-family credit ({2,4} both correct where
  truth is 2/4 or 4/4; exact bar informational), factored truth
  DERIVED from existing `meter`+`subdivision` labels (nothing
  relabeled), with the owner-ratified 6/8 mapping: pulse = the counted
  eighth (bpm label unchanged), accent-every-3 = grouping rung 3, the
  bar rung 6, division none. Gates NOTHING until a separate future
  owner ruling; both metrics reported side by side. Mapping table
  pre-registered in the increment before any pipeline comparison. ·
  **W6 — condition DRAFTED 2026-08-30 by the out-of-cadence W0 and
  SPLIT (A2-30, owner-ratified 2026-08-30): W6-a**, the consumption
  path — distributions through `posterior.py`'s marker evidence class,
  gated on one-hot byte-identity — *(COMPLETE 2026-08-30, accepted at
  the same day's second batch review; the flip-point finding stands as
  W6-b's design constraint: summed fractional belief is NOT voting, and
  a 1-of-5 dissent can buy a metric level on any clip with ≥12
  contested tokens)*; **W6-b**, the N ≥ 5 draws themselves, BLOCKED on
  `GEMINI_API_KEY` reaching the runner plus two owner decisions (cost
  ceiling; the second model family — the 2026-08-09 backlog note
  suggests a Qwen2-Audio-class local model), both DEFERRED by the owner
  (2026-08-30) to the 2026-09-03 W0. Conditions in the 2026-08-30 W0
  ledger entry §1. ·
  **W13 = the expert information-timing trace (COMMISSIONED 2026-08-30
  by the owner):** trace what the professional accompanist actually
  uses — the owner reports extracting the full picture from a few
  seconds of watching, listening and prior knowledge, by finding the
  moments where the teacher packs the most information and discarding
  the rest. Three components: **(a) the trace session** (owner-led,
  attended-agent-supported): one video the owner's brain has not
  already studied — NEVER a HELD-OUT exercise; a fresh clip or an
  unstudied DEV Barre-1 take — watched once real-time (state the
  playable answer and when it was known), then re-watched with
  scrubbing to timestamp every light-up moment with what it revealed,
  what was discarded around it, and a modality tag (heard / saw /
  knew-from-context); deliverable is a research memo mapping each
  moment onto pipeline hypotheses. **(b) The prefix-replay convergence
  twin** (agent-runnable, nightly-eligible, read-only over frozen
  traces): replay the pipeline on prefixes of each clip and chart when
  each field's answer converges to its final value — the machine's
  time-to-commitment curve, to lay against the owner's. **(c) The
  cue-nod capture question routed from W10** (owner ruling 2026-08-30):
  the trace records whether and when cueing gestures matter, informing
  what future capture must cover. Provenance: owner introspection
  2026-08-30, continuing the line that produced the factored-meter
  direction. ·
  **W14 = the commitment stopping rule (*PROPOSED 2026-08-31 by the
  out-of-cadence W0 — owner's to ratify*; **EXECUTED 2026-08-31 ahead of
  ratification, declared not slipped in** — a REPORTED-ONLY increment
  that pins no outcome, so a rejection costs only the artifact; see that
  session's pre-registration §0). **RESULT: negative and useful —
  neither family yields a tempo stopping rule at the pre-registered
  ≤0.10 premature ceiling (F1 best 0.370 having consumed 57% of the
  clip; F2 flat at ~0.93–0.97 across every θ), because
  `normalized_tempo.confidence` is *highest when the pipeline knows
  least* (median 1.000 at the first prefix, 0.780 on the full clip) and
  falls as evidence arrives. One honest win: `grouping` commits at 21%
  of span, never wrong on 29 verified clips, k=2. Two items parked —
  W14-b (the trajectory-shape family) and the confidence-calibration
  defect, which is a shipping-path finding W14's own scope forbade
  touching.** Original commission: W13(b) measured that the
  pipeline re-decides tempo a median of 5 times per clip and settles only
  after 60–88% of the clip is gone, so the missing piece is a stopping
  rule rather than a better estimator. REPORTED-ONLY, offline, key-free:
  record committed confidence at every prefix (an additive re-run of
  `scripts/w13b-prefix-replay.py`, existing published numbers must
  reproduce exactly), then score both stopping-rule families —
  k-stable-prefixes and confidence ≥ θ — for premature-commit rate and
  committed-at time, laid against the owner's W13(a) curve. The
  k-stable family is scoreable from the committed JSON today; the
  confidence family is **not** — no confidence is recorded at any prefix
  (verified 2026-08-31), which is why the re-run is step one. Condition
  in the 2026-08-31 W0 ledger entry §4. ·
  **W15 = the stated-structure channel (H1 re-scoped; *PROPOSED
  2026-08-31 by the out-of-cadence W0 — owner's to ratify*):** parse
  declarative structure announcements out of the frozen transcripts as a
  second meter channel, emitting a typed claim (beats-per-bar /
  repetitions / bars / unknown) with abstention — **the disambiguation
  is the deliverable, not the regex.** Sized before commissioning: the
  patterns fire on **7 of 52** traces, only **3 verified**, and on those
  three, reading the spoken number as the bar grouping agrees with truth
  on **1 of 3** (`plies-demo` yes; `exercise-1-demo` and
  `rig-mixed-4-4-104-quantities` no — a teacher naming "four counts" of a
  3/4 phrase is counting repetitions). REPORTED-ONLY, gates nothing,
  wired into no pipeline path (Standing Lesson 9); 1-of-3 is the bar it
  must clear. Condition in the 2026-08-31 W0 ledger entry §4. ·
  **Standing ranking (owner-ratified 2026-08-30, post-burst batch
  review; *amended-as-PROPOSED 2026-08-31 by the out-of-cadence W0*):**
  1. ~~W14~~ *(EXECUTED 2026-08-31, PROPOSED — awaiting owner review)* ·
  2. **W15** *(PROPOSED, untaken — now ranked 1 for a scheduled
  session)* · **W14-b** *(PROPOSED 2026-08-31 by the W14 increment: the
  trajectory-shape stopping rule — commit when the answer stops
  oscillating between metric levels, the failure mode W13(b) Finding 2
  actually described; scoreable from the same artifact now that
  `series_num` makes the trajectory replayable)* · W6-b BLOCKED on two
  owner decisions — cost and the second model family — both deferred to
  the 2026-09-03 W0 (blocker (i), the key reaching the runner, is
  discharged for owner-run local sessions; see the 2026-08-31 W0 §5) ·
  W5 continuation OPEN (owner-started; sidecars and factored slice now in
  place) · W11-b BLOCKED (barre1 media is `offrepo:`) · W8 BLOCKED (after
  W5's continuation and the tier-0 driver EVAL-CHANGE named in ADR-017) ·
  W1/W1.5/W2/W2.5/W3/W3r/W4/W6-a/W7/W9/W10/W11/W12/W13(a)/W13(b)
  COMPLETE. **Why the ranking was amended at all:** on 2026-08-31 every
  commissioned workstream was finished, owner-reserved, or blocked —
  the queue was empty for scheduled sessions, and would have stayed empty
  through 09-02.
  **Burst schedule note (owner, 2026-08-30):** the Air's 3×/day burst
  stays until the 2026-09-03 W0 review, which recommends keep or
  revert; the revert procedure and plist backup are recorded in the
  2026-08-28 air-service ledger entry. **Cadence ruling (A6-30
  response, owner 2026-08-30):** while the burst runs, the owner holds
  short attended reviews roughly every two days; the weekly W0 deep
  pass is unchanged.
  **W10 = nod-kinematics gesture channel *(COMPLETE 2026-08-30,
  negative — owner-accepted 2026-08-30: no beat-locked head movement
  on this corpus under three event definitions with a working positive
  control; head height is postural, not nodal, on a dancing body. The
  untested cue-nod hypothesis is ROUTED TO W13(c); movement remains a
  weak W5 vote per W7)*:** head-nod kinematics /
  phrase-arrival segmentation on the Barre 1 video — the owner's
  "nod-first" adoption (ruling A8, 2026-08-24), formerly unschedulable
  inside COMPLETE W7. Ranked last among open workstreams; a future W0
  may re-rank it. **W0 = the meta-rung, self-scheduling:** whenever
  the ledger's most recent meta-rung entry is older than 7 days (or none
  exists yet and the ledger carries ≥ 5 marathon entries — entries, not
  "sessions", per amendment 3, owner-ratified 2026-08-24), W0 outranks
  every other workstream — that session performs the weekly review (re-rank
  the workstreams, audit the BLOCKED queue, propose any charter
  amendments as a PROPOSED ledger entry, write a plain-language summary
  addressed to the owner) instead of a pipeline increment. **The 7-day
  clause is a floor, not exclusive (A1-30, owner-ratified 2026-08-30):**
  an out-of-cadence W0 is permitted when circumstances warrant, and it
  never resets the 7-day clock — the scheduled meta-rung stands.
  **Out-of-cadence W0 (*PROPOSED A1-30, 2026-08-30 — owner's to
  ratify*):** the 7-day clause says when W0 *outranks* everything; it is
  read here as a floor, not as exclusivity, so on a night when every
  other workstream is BLOCKED, W0 is the highest-ranked non-BLOCKED
  workstream and may be taken early rather than idling the loop. An
  out-of-cadence W0 **does not reset the 7-day clock** — the next
  scheduled meta-rung keeps its original date. Rule the other way and the
  alternative is explicit: consecutive BLOCKED notes until the trigger
  fires.
- **Per-session condition** (what the standing contract resolves to):
  one complete increment on one workstream, committed on
  `agent/marathon`, evidence shown by full command output, constraints
  verified (`git diff --stat main`; no existing eval file modified; new
  cases provisional-only), dated ledger entry appended — or a one-line
  ledger note that every workstream is BLOCKED. Or stop after 45 turns.
  **Writability precondition (owner, 2026-08-24, amendments 1–2):** the
  session's *first act* is a cheap write-and-commit probe; a session
  that cannot write satisfies no clause and stops immediately with
  whatever diagnostic it can surface, rather than burning turns.
- **Cadence:** owner batch review roughly weekly (plus whenever a BLOCKED
  queue item needs clearing); the meta-rung runs weekly and re-ranks.
- **Completion targets** (owner-editable; provisional rows excluded): on
  verified DEV rows with n ≥ 60 — tier-1 committed accuracy tempo ≥ 0.85,
  meter_triple ≥ 0.75, counts ≥ 0.75; stage-1 pulse F ≥ 0.85 on the
  step_names and vocables slices; ECE ≤ 0.15 — plus a **multifaceted
  proof**: an ablation table showing at least three independent evidence
  channels (acoustic pulse, ensembled semantics, accent-meter and/or
  pose) each contributing positive marginal accuracy. Completion is
  declared by a meta-rung report **co-signed by the owner**, never by a
  session alone. **Standing note (owner, 2026-08-24, amendment 4):**
  with 28 verified rows these targets are arithmetically unreachable;
  corpus growth (W4 + future capture) is the binding constraint, and
  "targets unmet" must never be read as "pipeline regressing" until
  n ≥ 60 verified rows exist.

### Meta-rung — weekly review

Read the entire ledger; re-rank the remaining backlog; propose charter
amendments and the next rung set as a PR the owner reviews. Never executes
pipeline work itself.

## Blessing protocol (owner)

Review the rung branch + ledger entry; optionally re-run suites locally;
merge to main; run `python -m musical_perception.evals run` then `bless`;
update the CURRENT RUNG pointer at the top of this charter; note the
blessing in the ledger entry's Status line. Rejection: one-line reason in
the ledger; the rung stays current with that guidance.
