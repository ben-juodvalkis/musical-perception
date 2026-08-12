# Research Agent Charter — the rhythm-core reset loop

**Date:** 2026-08-09
**Status:** Active. Governs every autonomous session working on
[ADR-016](../adr/016-rhythm-core-reset.md)'s reset. Companion documents:
[voice-as-drum literature review](voice-as-drum-review.md) (the textbook),
[RESEARCH-LOG.md](RESEARCH-LOG.md) (the memory),
[agent-environment.md](agent-environment.md) (cloud setup).

> **CURRENT RUNG: 1.5 — owner verification of the 30 provisional beat
> grids.** Agent sessions assist as scribe/tooling only (to-labels /
> from-labels, correction stats, ledger notes); `--verified` is applied
> only at the owner's word. Rung 2 unlocks when grids are verified.
> (Rung 1 blessed 2026-08-11, merged from agent/rung-1-stage-scoring —
> the supervised session. Rung 0 complete 2026-08-09. Autonomy mode:
> rung-by-rung with daily blessings through rung 2; the dormant Rung M
> below is commissioned — or not — at the rung-2 verdict.) The
> owner edits this line when a rung is blessed. Sessions execute the
> current rung and nothing else.

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
   Lessons section and the last five entries. Never re-attempt an approach
   the ledger records as falsified without new evidence.
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
- **Ballet Barre 1 sections (assigned 2026-08-09): all DEV.**
  Public-source class video handed to the loop by the owner, staged on
  the Air at `video/youtube/Ballet Barre 1/Sections` (gitignored — local
  media, never committed). Their truth labels are agent-proposed
  `provisional` until owner-verified. Sealing continues to come only
  from the owner's own future capture.

## Rules of engagement

1. **Branches only.** All work on `agent/rung-<N>-<slug>`. Never push
   `main`. Never run `evals bless` — blessing is the owner's act.
2. **Eval integrity.** Never **modify** any existing file under
   `evals/cases/`, `evals/traces/`, or `evals/baseline.json`, and never
   touch the scorer/harness code (`src/musical_perception/evals/`) in a
   pipeline rung. Eval *infrastructure* changes (new metrics, new suites)
   happen only in rungs/workstreams whose explicit deliverable is eval
   infrastructure (W1 is one), flagged **EVAL-CHANGE** in the report,
   never bundled with pipeline changes. **Add-only carve-out for
   ingestion:** creating NEW case and trace files for new material is
   permitted and expected — every agent-authored label ships with
   `maturity: provisional`. **Provisional rows never gate anything and
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
   annotated as such in the case notes.
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

Prereq: rung 1 blessed AND grids owner-verified. Deliverable: the
acoustic pulse channel ([Review 1 "steal this first"](review-1-onsets-pcenters.md)),
plus the Whisper-word-onset baseline scored on the same grids.

> **Gate re-expression pending (2026-08-11, rule 9):** the margins in the
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

### Rung 3 — accent-periodicity meter votes

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
any code.

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

### Rung M — the marathon *(DORMANT — commissioned only by the owner, at the rung-2 verdict)*

Staged-autonomy plan (owner decision pending, 2026-08-09): rungs 1→2 run
rung-by-rung with daily blessings, because the early rungs build the
measuring instruments (rung 1), require human ground truth (rung 1.5),
and end at the plan's strategic fork (rung 2's kill-test verdict). At
that verdict the owner decides whether to commission this rung for the
parallel-friendly middle stretch. Until then, sessions must treat this
section as inert.

When commissioned, Rung M replaces per-rung blessing with **batch review**:

- **Workstreams** (initial ranking; the meta-rung re-ranks): W1 = rungs
  3–4 pipeline work · W2 = Ballet Barre 1 ingestion (traces + provisional
  cases per the add-only carve-out) · W3 = rung 6 baselines · W4 = rung 5
  ensembles · W5 = pose/gesture channel prototyping on the Barre 1 video ·
  W6 = rung 7 cleanup. Policy: each session advances exactly one
  workstream — the highest-ranked one not BLOCKED; blocked workstreams
  get a BLOCKED ledger note (the owner's task queue) and the session
  moves on. The loop never idles while any workstream is open.
- **Per-session condition** (what the standing contract resolves to):
  one complete increment on one workstream, committed on
  `agent/marathon`, evidence shown by full command output, constraints
  verified (`git diff --stat main`; no existing eval file modified; new
  cases provisional-only), dated ledger entry appended — or a one-line
  ledger note that every workstream is BLOCKED. Or stop after 45 turns.
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
  session alone.

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
