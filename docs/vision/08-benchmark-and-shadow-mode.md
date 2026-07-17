# 08 · Benchmark & Shadow Mode

The measurement program — the part of the project that converts "is this
feasible?" from an opinion into a number that improves weekly. Principle
([01 · North Star](01-north-star.md)): **measured, not believed.** No
capability claim without a benchmark number; no live sound without shadow
gates passed.

This is also the moat: the scan ([03 · Landscape](03-landscape.md) §3.5)
found **no dataset of ballet marking anywhere** — building one is both the
engineering foundation and a first-of-its-kind research asset.

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

## 8.4 The replayer

`analyze.py` batch mode **is** the replayer — this is why it never gets
deleted. Harness: corpus in → per-exercise `MusicalParameters` (and, once the
[engine](06-performance-engine.md) exists, `PerformancePlan`) out → scored
against annotations → dashboard (a static HTML report per run, tracked over
time). Every PR that touches perception shows its benchmark delta; the
DISPOSABLE layer finally has an objective swap test.

## 8.5 Shadow mode

Production rig, production pipeline, **zero sound**. During a real class the
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

Advance the [ladder](07-interaction-design.md) §7.6 only on evidence:

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
