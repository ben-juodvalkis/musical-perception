# 08 · Benchmark & Shadow Mode

The measurement program — the part of the project that converts "is this
feasible?" from an opinion into a number that improves weekly. Principle
([01 · North Star](01-north-star.md)): **measured, not believed.** No
capability claim without a benchmark number; no live sound without shadow
gates passed.

This is also the moat: the scan ([03 · Landscape](03-landscape.md) §3.5)
found **no dataset of ballet marking anywhere** — building one is both the
engineering foundation and a first-of-its-kind research asset.

The corpus below is the *destination* of the measurement program, not its
entry point. Measurement starts before a single class is recorded — see the
eval ladder (§8.4), specified in
[ADR-009 · Evaluation Harness](../adr/009-evaluation-harness.md).

---

## 8.1 The corpus

**Target:** 10–20 full real classes (≈200–400 exercises), spanning:

- ≥4 teachers (styles: numbers-counter, step-namer, vocable-user, minimal-marker)
- levels (children's / intermediate / advanced / professional)
- both accompanied classes (human pianist = gold labels for "what a
  professional chose") and recorded-music classes (playlist workflow =
  the baseline being replaced)
- ≥2 languages of instruction

**Capture rig:** teacher lapel mic (analysis channel), room mic (ambient),
wide camera (pose + cue), synchronized clocks. One-button start; zero
workflow intrusion (this is the same rig as production —
[04](04-system-architecture.md) §4.2).

**Three sources, one corpus.** The target above is source C — real classes,
chaotic, the only thing a gate may be scored on. Two cheaper sources sit under
it and do different jobs ([13 · Corpus & Capture](13-corpus-and-capture.md)):

| | What | Job |
|---|---|---|
| **A · Rig** | metronome-locked counting, factorial across count style × meter × interleaved explanation × marking tempo; labels exact and free | localize *which module* broke |
| **B · Clean capture** | a real teacher teaching normally, into a proper mic in a quiet room | the honest ceiling |
| **C · Chaotic** | real classes as they are — the corpus target above | **the gate** |

The rule that keeps this honest: **the clean sources can fail you but never
pass you.** A and B block merges as regression suites; only C certifies a
capability. The B-vs-C gap is itself a diagnostic — the **capture penalty** —
that separates "segmentation and mic" failures (R1) from "the model is the
ceiling" failures ([13](13-corpus-and-capture.md) §13.3). Clean means clean
*capture*, never clean *teaching*: a teacher who counts more neatly for the
camera measures a system nobody can ship (§13.4).

## 8.2 Annotation schema (per exercise)

```yaml
exercise_id: 2026-07-15-classA-07
slot: frappe                      # class-position ground truth
marking: {start: 512.3, end: 538.9}   # the teacher's demonstration span
cue_time: 549.1                   # the "aaand—" moment (or absence)
meter: 2/4                        # what the music was / should be
performance_bpm: 104              # what was played / requested (not marking tempo!)
marking_bpm: 112                  # measured from the marking (→ tempo_offset data)
counts: 32
sides: both
character: {articulation: 0.2, weight: 0.4, energy: 0.7}
music_used: {piece: "...", source: pianist|app|album}
interruptions: [{t: 561.0, type: correction}]
verbal_commands: [{t: 610.2, text: "other side"}]
notes: "teacher marked half-tempo, pianist doubled"   # the gold anomalies
```

The `marking_bpm` vs `performance_bpm` pair across a corpus is, by itself, a
novel research result: nobody has quantified the marking-tempo gap.

This schema is also the **eval case format**. The lower tiers of the ladder
(§8.4) write cases that are a strict subset of it — an `expect:` block naming
whichever fields are known — so annotating a real class only *adds cases*;
nothing about the harness changes as the corpus arrives. Each case also carries
the difficulty `tags:` of [13](13-corpus-and-capture.md) §13.6, so "clean vs
chaotic" is a query rather than a folder — and it is filled in *in the room*,
not retroactively (§13.5).

## 8.3 Metrics

**Per-component:**

| Metric | Definition | v1 gate |
|---|---|---|
| Meter accuracy | coherent-triple meter correct (ADR-007 discipline) | ≥ 90% |
| Tempo accuracy | \|predicted − performance_bpm\| / performance_bpm ≤ 8% | ≥ 95% |
| Counts/structure | exact phrase-length match | ≥ 90% |
| Exercise/slot ID | top-1 slot correct | ≥ 90% (with tracker) |
| Marking segmentation | IoU of detected vs annotated marking span ≥ 0.7 | ≥ 90% |
| Cue detection | detected within ±400 ms of annotated cue | ≥ 95%, **0 false cues** |
| Command spotting | grammar intents recalled | ≥ 95% |

**End-to-end (the ones that matter):**

| Metric | Definition | v1 gate |
|---|---|---|
| **False starts** | would-play events outside a valid cue | **0 per class** |
| Exercise success | meter ∧ tempo ∧ length all correct with ≤1 question | ≥ 85% of exercises |
| Question rate | one-word questions per class | ≤ 4, falling with calibration |
| Start latency | cue → first note | ≤ 500 ms |
| Stop latency | teacher speech → silence | ≤ 300 ms |

Report **zero-shot vs calibrated** for every metric — the gap between those
two columns is Reframe 3's entire claim, made falsifiable.

### Scoring discipline

Three rules turn the tables above from headline numbers into decision-grade
ones (full rationale in [ADR-009](../adr/009-evaluation-harness.md)):

1. **Abstention is a separate outcome, not a wrong answer.** The product's
   policy is silence over false starts ([07](07-interaction-design.md) §7.4)
   and one legitimate question when genuinely split
   ([05](05-perception-strategy.md) §5.3). A scorer that counts "did not
   commit" as "wrong" optimizes directly against that policy. Every metric
   reports **correct / wrong / abstained** plus coverage, a risk–coverage
   curve, and a calibration error (ECE) over confidence bins — which is what
   makes "well-calibrated uncertainty" and the 0.80 / 0.55 decision thresholds
   falsifiable rather than asserted.
2. **Score musically.** Tempo error separates *metric-level* failures (exactly
   ×2, ×3, ÷2, ÷3 out — the ADR-006/007 subject) from genuine tempo error;
   meter is scored as the coherent `(meter, bpm, subdivision)` triple with
   partial credit for musically equivalent readings; quality is scored on
   ordering (rank correlation) as well as absolute error.
3. **The gate is the worst slice, above a minimum n.** Report every metric per
   teacher, count style, exercise slot, language, and accompanied-vs-recorded
   — a mean hides that meter is strong on numbers-counters and weak on vocable
   users. Quote intervals: 90% on 30 cases is ±11%, worthless against a 90%
   gate; ±5% at p≈0.9 needs n≈140 exercises and ±3% needs n≈350. That is the
   arithmetic behind §8.1's 200–400-exercise target, and until n suffices a
   gate is explicitly provisional — the full ladder of what each corpus size
   entitles you to claim is [13](13-corpus-and-capture.md) §13.8.

## 8.4 The eval ladder (and the replayer)

`analyze.py` batch mode **is** the replayer — this is why it never gets
deleted. Harness: cases in → per-exercise `MusicalParameters` (and, once the
[engine](06-performance-engine.md) exists, `PerformancePlan`) out → scored
against expectations → dashboard (a static HTML report per run, tracked over
time). Every PR that touches perception shows its delta; the DISPOSABLE layer
finally has an objective swap test.

The corpus is only the top of that harness. Four tiers share one case format
and one scorer library, and each is runnable the day it is written:

| Tier | What it measures | Cost | Cadence |
|---|---|---|---|
| **0 · Synthetic** | KEEP math on generated timelines from a known (meter, BPM, subdivision) triple, corrupted with jitter, dropped counts, interleaved explanation | free, seconds | every PR |
| **1 · Frozen traces** | Fusion logic — merge, `interpret_meter`, quality synthesis — replayed from recorded Whisper/Gemini/pose output | free, deterministic | every PR |
| **2 · Live perception** | The real stack: accuracy vs labels **and** drift vs the frozen trace | API spend | nightly; on any prompt or model change |
| **3 · Corpus** | Everything above, §8.3 as written | annotation | at gates |
| **4 · Shadow** | Policy in live rooms — false starts, latency, question rate (§8.5) | a class | per shadowed class |

Tier 1 carries the most weight: ADR-006 and ADR-007 were both *fusion* bugs,
not model bugs, and frozen traces test exactly that layer offline and for free.
Tier 2 is where a silent Gemini version bump becomes visible — drift against
the previous trace is a first-class number, distinct from accuracy.

### Labels before the corpus

Two label sources need no annotation program and can run now:

- **Synthesized markings** (source A, [13](13-corpus-and-capture.md) §13.2).
  Metronome-locked counting at a known BPM and meter gives perfect labels at
  zero cost, permuted across the matrix that actually breaks the pipeline:
  numbers vs step names vs vocables vs near-silence × 2/4, 3/4, 4/4, 6/8 × with
  and without interleaved explanation. Not real rooms, so never a gate — but it
  localizes failures precisely, which real rooms do not.
- **Accompanied recordings — free `performance_bpm`.** Where a live pianist
  plays, the piano audio *is* the ground truth for what the teacher wanted:
  beat-track it and `performance_bpm` falls out mechanically, while the marking
  that precedes it gives `marking_bpm`. The novel pair of §8.2 — and the
  `tempo_offset` prior of [05](05-perception-strategy.md) §5.6 — from public
  recordings, without a human annotator. This is why source C can start before
  any filming does ([13](13-corpus-and-capture.md) §13.5).

Every run emits one JSON (metrics, per-case rows, and the hashes that make it
reproducible: git SHA, model ids, prompt hash, trace version) plus the HTML
diff against baseline. The hand-pasted results tables in ADR-006/007 become
generated output.

## 8.5 Shadow mode

Tier 4 of the ladder: production rig, production pipeline, **zero sound**. During a real class the
system runs the full lifecycle silently; afterwards it emits the report:

> **10:14 · frappé (slot 6, P=0.93)** — would have started at 10:14:32.1
> (cue detected), 2/4 polka "Anitra" at 104 (marking 112 × offset 0.93),
> 32 counts both sides. *Teacher used:* album track 11 at ~100.
> ✅ meter ✅ tempo (+4%) ✅ length ✅ cue ⚠️ would have asked "polka or march?"

Shadow mode simultaneously produces: the trust artifact (teachers see it
*understanding* before they hear it play), the calibration profile
([05](05-perception-strategy.md) §5.6), fresh corpus data, and the go-live
evidence. It is the answer to "how do you deploy an unproven system into a
live classroom": **you don't — you deploy a silent one.**

## 8.6 Go-live gates (per teacher)

Advance the [ladder](07-interaction-design.md) §7.6 only on evidence. These
are per-teacher slices by construction — which is the general rule (§8.3):
**a gate is cleared by the worst slice, not the mean.**

- **→ Rung 3 (tap):** 2 shadowed classes; music/engine quality accepted by
  the teacher; stop-reflex verified live.
- **→ Rung 2 (voice):** shadow exercise-success ≥ 80% calibrated; command
  spotting ≥ 95%; 0 false cues across all shadowed classes.
- **→ Rung 1 (full-auto):** shadow exercise-success ≥ 90%; cue detection
  ≥ 95% with 0 false cues; question rate ≤ 4/class and falling.
- **Demotion:** any live false start → drop a rung immediately, investigate,
  re-earn.

## 8.7 Data governance

Explicit teacher + studio consent for retention; students outside analytic
scope (camera framing + pose-only retention by default); per-teacher profiles
are theirs — exportable and deletable; corpus contributions
anonymized/consented separately if ever published. Precedent that this posture
works commercially: TeachFX/Merlyn ([03](03-landscape.md) §3.3).

## 8.8 The research asset

If pursued ([10 · Pivots](10-pivots.md) P2), the corpus + tasks form a
publishable benchmark ("marking comprehension": segmentation, meter, tempo,
structure, character from teacher demonstration), and the onset/accent methods
are novel against both the beat-tracking and speech-prosody literatures. A
fine-tuned small local model trained on the corpus is the likely endgame for
the perception layer — the DISPOSABLE wrappers' final swap.
