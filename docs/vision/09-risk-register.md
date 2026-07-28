# 09 · Risk Register

The living list of what could sink this — each with evidence, mitigations,
residual risk, and **tripwires** (pre-committed signals that force a decision,
so bad news changes the plan instead of the narrative). Review after every
benchmark run and pilot week.

---

## R1 · Marking segmentation in live rooms — **the #1 risk**

**What:** Teachers interleave explanation, anecdote, correction, and counting.
The system must find the marking inside the talk without the teacher
performing for it.

**Evidence:** ADR-007 documented the failure directly (onset detector picking
up conversational speech; cross-signal checks collapsing) and its only
mitigation was teacher behavior change — which violates the brief. The
landscape confirms the general form (device-directed speech detection in open
rooms) is unsolved industry-wide ([03](03-landscape.md) §3.3).

**Mitigations:** teacher mic (SNR + identity); prosody-first features that
*distinguish* rhythmic vocalization from talk (regular onsets, compressed
pitch range, accent periodicity); eager-hypothesis + commit-on-cue (a wrong
segmentation with no cue plays nothing); class-state priors (a marking is
*expected* in a slot); silence-biased policy caps the blast radius at
"missed start."

**Diagnostic:** the **capture penalty** — clean-captured vs chaotic accuracy
([13 · Corpus & Capture](13-corpus-and-capture.md) §13.3) — separates this risk
from R2. A large gap means mic, VAD, and segmentation; no gap means the model
is the ceiling and the fix belongs to the meter plan instead.

**Residual:** Real. Some teachers barely vocalize.
**Tripwire:** if calibrated marking-segmentation IoU < 80% for a teacher after
3 shadowed classes → that teacher runs at ladder rung 2–3 permanently; if this
is >30% of pilot teachers → re-scope v1 to rung 2 as the default product
posture (still deletes the form).

## R2 · Meter inference

**What:** 3/4 vs 4/4 vs 6/8 from marking alone; the loudest failure mode.

**Evidence:** 2 of 4 correct on the repo's own files (ADR-007). Genuine
musical ambiguity exists (ADR-006's defensible-ambiguity analysis).

**Mitigations:** the three-vote plan (accent periodicity + priors + semantics,
[05](05-perception-strategy.md) §5.3); the one-word question floor; the
coherent-triple discipline already built.

**Residual:** Medium — humans ask "in three?" too; the metric is
question-rate, not perfection.
**Tripwire:** calibrated meter accuracy < 90% on the benchmark after the
accent module ships → escalate: collect 5× more corpus for the failing
marking styles and evaluate a fine-tuned classifier; meanwhile widen the
question band.

## R3 · Teacher idiosyncrasy (zero-shot)

**What:** counting in numbers vs names vs vocables vs near-silence; unbounded
in general.

**Mitigations:** don't do zero-shot ([02](02-reframes.md) Reframe 3):
calibration from 2–3 shadowed classes; idiom lexicon; class template; the
onboarding choreography ([07](07-interaction-design.md) §7.7) makes the
few-shot window a designed experience, not a caveat.

**Residual:** Low for teachers whose profile converges; the honest product
answer for the rest is a lower ladder rung.
**Tripwire:** if profiles fail to converge (exercise-success < 80% after 4
classes) for >30% of pilot teachers → the wedge narrows to syllabus mode
([10](10-pivots.md) P1) while perception matures.

## R4 · Marking tempo ≠ intended tempo

**What:** teachers mark approximately; human pianists compensate from
experience.

**Evidence:** known human-factors gap (practice literature); the corpus will
quantify it for the first time (`marking_bpm` vs `performance_bpm`,
[08](08-benchmark-and-shadow-mode.md) §8.2).

**Mitigations:** learned per-teacher (and per-exercise) `tempo_offset`; live
verbal nudges are one word and instant.

**Residual:** Low — tempo is the *most correctable* parameter mid-exercise.
**Tripwire:** if offset variance for a converged teacher still yields >8%
error on >15% of exercises → make the first 4 intro counts tempo-elastic
(settle onto the room like human pianists do) — an engine feature, spec'd
but not scheduled.

## R5 · Musicality ceiling

**What:** flat, mechanical playback reads as "karaoke class," not accompanist.

**Mitigations:** v1 standard is *excellent rehearsal pianist*
([01](01-north-star.md)); humanization templates as data
([06](06-performance-engine.md) §6.6); the comparison bar is the playlist,
which is *also* mechanical and additionally wrong-tempo and wrong-length.

**Residual:** Accepted for v1; revisit with style capture (§6.7).
**Tripwire:** pilot teachers rating music quality below their current
recordings → stop feature work, fix templates/library first; nothing else
matters if the sound is worse than the album.

## R6 · Center-work following

**What:** adagio breathing, grand-allegro catch — real accompanists adapt to
dancers; score-following research doesn't transfer (no score, no instrument).

**Mitigations:** barre-first scope; center runs steady-tempo (which is what
recorded music does today anyway — parity with baseline, not with pianist).

**Residual:** Deferred, honestly and explicitly.
**Tripwire:** none for v1 — revisit only after v1 gates pass.

## R7 · Trust destruction from one loud failure

**What:** a false start during a quiet correction is the story the teacher
tells everyone; one bad demo can end a studio relationship.

**Mitigations:** the asymmetric policy with target **zero false starts**
([07](07-interaction-design.md) §7.4); shadow-before-sound
([08](08-benchmark-and-shadow-mode.md) §8.5); rung demotion on any live false
start; the state light (the room can *see* it's attending, not guessing).

**Residual:** Managed by process, not model quality — deliberately.
**Tripwire:** any live false start → immediate rung drop + root cause before
re-promotion. Two in a pilot → freeze rung 1 across the fleet until fixed.

## R8 · Adoption friction (the non-technical risk)

**What:** teachers are tradition-bound; studios are cash-constrained; a
camera in the room raises eyebrows; the person who buys (studio owner) is not
always the person who must trust it (teacher).

**Mitigations:** shadow-mode onboarding costs the teacher nothing; rung 3
delivers value with zero AI risk exposure; privacy posture with deployed
precedent ([07](07-interaction-design.md) §7.8); price against the manual
apps, not against a pianist's salary.

**Residual:** Real; only pilots reveal it.
**Tripwire:** if pilot teachers use it at rung 3 but decline rungs 1–2 for a
full term despite passing gates → the product is the tap-plus-inference
assistant; embrace it ([10](10-pivots.md) P3 posture) and let full-auto ride
along as opt-in.

---

## Non-risks (deliberately closed)

| Non-risk | Why closed |
|---|---|
| Compute/cost | Reflex on laptop CPU; ~<$1 of API per class ([04](04-system-architecture.md) §4.7) |
| Core latency | Only start/stop need sub-second; the domain grants seconds for everything intelligent (§4.5) |
| Playback technology | Symbolic MIDI + modeled piano is commodity ([03](03-landscape.md) §3.4) |
| Classroom always-on acceptance | Commercial precedent exists (Merlyn, TeachFX) |
| Competitive rush | Niche empty as of 2026-07; watch-triggers defined in [03](03-landscape.md) §3.6; the corpus is the moat |

## Review cadence

- After every benchmark run: R1–R4 numbers vs tripwires. The tripwire
  thresholds above are harness fields, not judgement calls — the run report
  ([08](08-benchmark-and-shadow-mode.md) §8.4,
  [ADR-009](../adr/009-evaluation-harness.md)) emits each of them per slice, so
  a tripwire trips on its own instead of waiting to be noticed.
- After every pilot week: R5, R7, R8 qualitative check.
- Quarterly: landscape re-scan ([03](03-landscape.md) §3.6) and register
  refresh; retire risks with evidence, add new ones with names and tripwires.
