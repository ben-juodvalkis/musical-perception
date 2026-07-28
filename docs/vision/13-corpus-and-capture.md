# 13 · Corpus & Capture

[08 · Benchmark](08-benchmark-and-shadow-mode.md) defines the measurement
program and [ADR-009](../adr/009-evaluation-harness.md) the harness that runs
it. This document is the **data program**: where cases come from, how they are
captured, labeled, tagged, and split — the part that decides whether any of
those numbers mean anything.

One rule governs the whole document, because the easy/hard split has a famous
failure mode — the clean set quietly becomes the number you quote:

> **The clean set can fail you, but it can never pass you.** Diagnostics come
> from controlled data; gates come from chaotic data.

---

## 13.1 Three sources

Not two. The middle tier is usually collapsed into one of its neighbors, and
it is the one that tells you *why*.

| | What it is | Labels | Job |
|---|---|---|---|
| **A · Rig** | Metronome-locked counting, scripted, factorial. Not a class — someone counting into a phone in an afternoon | free and exact | Localize **which module** broke |
| **B · Clean capture** | A real teacher teaching normally, with a proper mic, framing, and a quiet room | cheap — captured in the room (§13.5) | The honest **ceiling** |
| **C · Chaotic** | Phone in the corner, kids, music bleed, overlapping speech, accents, second languages | expensive to annotate — or free from accompanied classes (§13.5) | **The gate** |

A feeds tiers 0–1 of the eval ladder ([08](08-benchmark-and-shadow-mode.md)
§8.4); B and C feed tier 3. Gates in §8.3 and §8.6 are scored on C only. B and
A may block a merge as regression suites — they may never certify a capability.

**C does not require filming and can start now.** Public full-length classes
with a live pianist are chaotic real-world data whose tempo labels come free
from the piano audio (§13.5). Filming effort is best spent on B, which is the
one thing the internet cannot supply.

## 13.2 The rig matrix (source A)

A's value is not that it is easy. It is that **only one variable moves at a
time** — which no real footage, however clean, can offer. Design it as a
factorial sweep over the dimensions the record says break the pipeline:

| Dimension | Levels |
|---|---|
| Count style | numbers · step names · vocables ("da-da-POM") · near-silent marking |
| Meter | 2/4 · 3/4 · 4/4 · 6/8 |
| Explanation | none · before the count · interleaved with the count |
| Marking tempo | at intended tempo · half-tempo · rushed |
| Prep | "5-6-7-8" · "and—" · none |

A sparse but balanced sample of that grid — ~24–30 clips — is enough to answer
"is meter broken on vocables, or broken everywhere?" in an afternoon, at zero
annotation cost, because the BPM and meter are known before recording starts.
These clips are also the permanent regression fixtures: they never expire, and
they cost nothing to re-run.

## 13.3 The capture penalty

The gap between B and C is itself a metric. Name it, track it, and read it:

| B (clean) | C (chaotic) | Reading | Where the fix lives |
|---|---|---|---|
| high | high | Shipping-grade for this metric | — |
| high | low | **Capture and segmentation dominate** (R1) | mic, VAD, marking segmentation, cue detection |
| low | low | Capture is not your problem; the model or the fusion is the ceiling | the meter plan ([05](05-perception-strategy.md) §5.3), priors, calibration |
| low | high | Measurement bug | the harness, the labels, or the split |

This is the diagnostic neither set produces alone, and it is the substantive
argument for building both rather than picking one.

## 13.4 Clean capture ≠ clean teaching

The trap in "film it carefully for legibility" is that legibility gets
delivered by the *teacher*. The moment a teacher counts more cleanly than they
normally would because a camera is running, source B measures a system nobody
can ship — and that is precisely ADR-007's rejected "instruct the teachers"
mitigation and [01 · North Star](01-north-star.md) principle 3.

So the split is strict:

- **Clean is the capture.** Teacher lapel mic on its own channel; room mic for
  ambience; wide camera framing the teaching space; quiet room; slate + clap
  for A/V sync; the same rig as production
  ([04](04-system-architecture.md) §4.2).
- **Never the teaching.** Film on a day the teacher would be teaching anyway,
  with the instruction to teach exactly as always. No ritual phrases, no
  cleaner counting, no separating explanation from counting.

If a B recording contains a teacher visibly performing for the machine, it is
tagged `performed: true` and reported separately — a real category, but not one
that can clear a gate.

## 13.5 Label at capture time

Retro-annotating video costs 5–10× what labeling in the room costs, and it is
the step that usually kills corpus projects. Three sources, cheapest first:

1. **Ask, between exercises.** Twenty seconds per exercise: meter, tempo, how
   many counts, one side or both. That is [§8.2](08-benchmark-and-shadow-mode.md)'s
   schema, filled in by the only person who knows the answer for certain.
2. **Film accompanied classes — the labels play themselves.** Where a live
   pianist works, the piano audio *is* the ground truth for what the teacher
   wanted: beat-track it and `performance_bpm` falls out mechanically, while
   the marking preceding it gives `marking_bpm`. That is the novel pair of
   §8.2 and the `tempo_offset` prior of [05](05-perception-strategy.md) §5.6 —
   with no annotator at all. It also applies to public recordings, which is why
   source C can start before any filming does.
3. **Model-assisted pre-annotation, human-verified.** Only after 1 and 2, only
   with the rule from [ADR-009](../adr/009-evaluation-harness.md) — no model
   grades a field it produced — and with inter-annotator agreement measured on
   a subset, so the corpus's own noise floor is a published number rather than
   an assumption.

Record the class order and slot boundaries as you go; the class-state tracker
([05](05-perception-strategy.md) §5.5) needs the sequence, and it is free at
capture time and painful later.

## 13.6 Tags, not buckets

"Clean" and "chaotic" are folders, and folders cannot express the cell you will
actually want to query — *clean audio, vocable counting*. Difficulty is a
**query over per-case tags**, using the `tags:` block the case format already
carries:

```yaml
tags:
  source: rig | clean | chaotic
  teacher: t03
  slot: frappe
  count_style: numbers | step_names | vocables | minimal
  explanation: none | before | interleaved
  lang: en
  accompanied: true            # a pianist is playing → free tempo labels
  snr_band: high | mid | low
  marking_seconds: 14
  performed: false             # teacher visibly adapting to the camera
```

Every metric slices on these ([08](08-benchmark-and-shadow-mode.md) §8.3), and
the gate is the worst slice above a minimum n. New difficulty axes are added as
tags, never as new folders.

## 13.7 Splits, and the slice you seal on day one

- **Split at the class level, never the clip level.** Same teacher, same
  combination, same room across a split leaks everything.
- **Hold out one entire teacher and one chaotic class from the very first
  recording session.** The held-out teacher is the only honest zero-shot number
  this project will ever have, and per-teacher calibration
  ([05](05-perception-strategy.md) §5.6) makes leaking it especially seductive.
- **Calibration is evaluated leave-one-class-out** within a teacher, otherwise
  the profile is being scored on the classes that built it.
- The sealed slice runs at gates only — never inside the iteration loop.

## 13.8 Sizing, and what a number is allowed to mean

| Exercises scored | 95% interval at p≈0.9 | What that supports |
|---|---|---|
| 20 (one class) | ±13 points | Ranking fixes; finding what is obviously broken |
| 40 (two classes) | ±9 points | Direction of travel between runs |
| 140 | ±5 points | A defensible capability claim |
| 350 | ±3 points | v1 gates at the stated thresholds |

Starting set: ~24–30 rig clips, **one** clean-captured class (~20 exercises),
3–5 chaotic classes. That is enough to rank fixes and to publish an honest
baseline — and not enough to claim a capability. The first baseline report says
so in its own headline, and every number ships with its interval.

## 13.9 Consent, on filming day

The governance posture is [08 §8.7](08-benchmark-and-shadow-mode.md); the
operational form is one page signed before the camera comes out: explicit
teacher and studio consent for retention, camera framed on the teaching space,
students out of analytic scope with pose-only retention by default, per-teacher
data exportable and deletable, and separate consent for anything that might
later be published ([10 · Pivots](10-pivots.md) P2). Collaborators who might
co-design or host this are listed in [12](12-collaborators.md).

## 13.10 Filming-day checklist

1. Consent signed; teacher briefed to teach normally, not clearly.
2. Lapel mic on the teacher, own channel, levels checked; room mic running.
3. Camera wide on the teaching space; slate + clap at start for sync.
4. Class order logged as it happens (slot names, rough timestamps).
5. Between exercises: meter, tempo, counts, sides — twenty seconds, every time.
6. If a pianist is present, capture their audio cleanly; those labels are free.
7. Before leaving: back up, and tag which class is the sealed one (§13.7).
