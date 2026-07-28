# ADR-009: Evaluation Harness

**Date:** 2026-07-28
**Status:** Accepted (harness not yet built — sequenced as
[11 · Roadmap](../vision/11-roadmap.md) Track 2)

## Context

[08 · Benchmark & Shadow Mode](../vision/08-benchmark-and-shadow-mode.md) states
*what* to measure — metrics, v1 gates, shadow reports — and
[01 · North Star](../vision/01-north-star.md) principle 5 makes it binding:
**measured, not believed.** What does not exist is the machinery that produces
a number, and doc 08 presumes an asset we do not have: 10–20 annotated real
classes. Waiting for the corpus means no measurement for months, during which
Track 3 ([11 · Roadmap](../vision/11-roadmap.md)) — the meter work, the crux —
would be steered by vibes. Gate A2 asks for the honest zero-shot baseline
*first*.

The repo's own record shows what happens without a harness. ADR-006 and ADR-007
each ship with a 4-row results table pasted into the ADR body, hand-run on
files that are gitignored. Those tables are already evals — untracked,
unreproducible, and not comparable run to run. ADR-007's own "2 of 4 correct"
is the most valuable number in the repo and nothing recomputes it.

Three different kinds of thing need measuring here, and they do not share
machinery:

| Kind | Example | Correct instrument |
|---|---|---|
| Deterministic math (KEEP) | `interpret_meter`, `normalize_tempo` | Assertions — unit + property tests, per-PR |
| Probabilistic perception (DISPOSABLE) | Gemini meter, Whisper onsets | Aggregate accuracy on labeled cases, with confidence intervals |
| System policy | false starts, question rate, latency | Event scoring over replayed/shadowed sessions |

Asserting a single probabilistic case is a flake generator; reporting an
average without a slice breakdown hides that meter is fine on numbers-counters
and broken on vocable users. The harness has to treat all three honestly.

## Decision

Build the eval program as a **four-tier ladder** sharing one case format and
one scorer library. Every tier is runnable the day it is written; each tier
feeds the next; only the top tier needs the corpus.

### Tier 0 · Synthetic — pure math, no audio, no API, every PR

Extend today's hardcoded-data tests into generated ones. A fixture builder
emits a marker/onset timeline from a known `(meter, bpm, subdivision)` triple
with controlled corruption — timing jitter, dropped counts, extra onsets from
interleaved explanation, prep counts, half-tempo marking — and the harness
asserts recovery within tolerance across a sweep, not on one example.

This measures the KEEP layer end to end with zero recording cost, and it is the
only tier fast and free enough to gate every PR. The corruption knobs are
deliberately the ADR-007 failure modes; the accent-periodicity module of
[05](../vision/05-perception-strategy.md) §5.3 arrives with its tier-0 sweep on
day one.

### Tier 1 · Frozen traces — the load-bearing tier

Record model I/O once, replay forever. Per clip, freeze: Whisper's
`list[TimestampedWord]`, Gemini's raw response JSON, pose landmarks (`.npz`).
These are small, text-mostly, and committable — unlike the media, which stays
gitignored.

With traces frozen, `analyze()` runs offline, deterministically, for free, in
under a second — and what it exercises is exactly the fusion logic:
`_merge_gemini_with_timestamps`, `interpret_meter`, `dynamics.synthesize`.
**Both ADR-006 and ADR-007 were fusion bugs, not model bugs.** This tier is
where the regressions this project actually suffers get caught, and it makes
the KEEP/DISPOSABLE boundary a testable seam rather than a labeling
convention.

It requires one small refactor: `analyze()` currently imports `transcribe`,
`analyze_media`, and `extract_landmarks` as module functions inside its body,
so the existing `model=` / `gemini_client=` parameters cannot redirect them.
Introduce a provider bundle —

```python
@dataclass
class PerceptionBundle:
    transcribe: Callable[[str], list[TimestampedWord]]
    analyze_media: Callable[..., GeminiAnalysisResult]
    extract_landmarks: Callable[[str], LandmarkTimeSeries] | None = None
```

— defaulting to the real wrappers, with `ReplayBundle.from_trace(path)` for
evals. ~15 lines; it is the whole difference between an eval suite and a
shell script. Pair it with `--record-traces` on the CLI so every real run
donates a fixture.

### Tier 2 · Live perception — the swap test, nightly

Re-run the real Whisper/Gemini/pose stack over the clip set and report **two
distinct numbers**:

- **accuracy** vs gold labels — is the new model/prompt better?
- **drift** vs the frozen trace — what changed, on cases whose answers did not?

`scripts/compare_models.py` and `compare_extra.py` are the prototypes of this
and get folded in. Drift is the number that matters for a DISPOSABLE layer: a
silent Gemini version bump currently has no detector at all. Costs API money,
so: nightly, plus on any prompt or model-id change, never on every PR.

### Tier 3 · Corpus benchmark — doc 08 as written

Annotated real classes, the §8.3 metric table, the §8.6 go-live gates, the
static HTML dashboard per run. Nothing about the harness changes between tier 1
and tier 3 — **the case format is a strict subset of the
[§8.2](../vision/08-benchmark-and-shadow-mode.md) annotation schema from the
start**, so annotating a class only adds cases. Tiers 0–2 exist to make tier 3
a data problem, not an engineering problem.

### Tier 4 · Shadow mode

Same scorers, streaming input, sliced per teacher; the "would have played"
report of §8.5 is a rendering of a scored run. Adds the event-level policy
metrics (false starts, latency, question rate) that only a timeline can carry.

---

## Cross-cutting rules

These apply at every tier and are the substance of the decision.

**1 · One case, one scorer library.** `evals/cases/*.yaml`, keyed by id:

```yaml
id: adr007-plies-demo
input: {trace: traces/plies-demo/, media: "video/plies demo.m4v"}
tags: {teacher: t03, slot: plies, count_style: numbers, lang: en, source: youtube}
expect:
  meter: {beats_per_measure: 3, beat_unit: 4}
  performance_bpm: 118
  subdivision: none
  counts: 32
```

`expect` fields are optional — a case may pin tempo only. Scorers live in one
module and are shared by tiers 0–4; the dashboard, the tests, and the shadow
report all call the same functions.

**2 · Score musically, not with `==`.** Three comparators carry the domain:

- *Tempo:* relative error with an explicit **octave/metric-level check** — an
  answer that is exactly ×2, ×3, ÷2, or ÷3 off is a `metric_level` failure, not
  a `tempo` failure. That distinction is the entire subject of ADR-006/007 and
  collapsing it into "wrong BPM" destroys the signal that tells you which
  module to fix.
- *Meter:* score the **coherent triple** `(meter, bpm, subdivision)` as one
  item (ADR-007's discipline), with partial credit for musically equivalent
  readings — 3/4 at 120 and 4/4 at 40 with triplets produce identical sound,
  and the accompanist does not care.
- *Quality:* per-dimension MAE plus a within-±0.2 hit rate, **and** Spearman
  correlation across the corpus. Absolute quality values from a model are not
  trustworthy; the ordering (adagio smoother than frappé) is what the engine
  consumes and what should be gated.

**3 · Abstention is a first-class outcome — the most important rule here.**
The product policy is silence over false starts ([07](../vision/07-interaction-design.md)
§7.4) and one legitimate question ([05](../vision/05-perception-strategy.md)
§5.3). A harness that scores "did not commit" as "wrong" therefore optimizes
directly against the product. Every metric reports **correct / wrong /
abstained** with coverage, plus a risk–coverage curve and a calibration error
(ECE) over confidence bins.

This makes two otherwise-unfalsifiable claims measurable: that the 0.80/0.55
decision thresholds are placed correctly, and that the system has
"well-calibrated uncertainty" rather than accurate guessing. Doc 08 §8.3 now
carries these as scoring discipline alongside the accuracy table.

**4 · Bootstrap gold labels before the corpus exists.** Three sources, cheapest
first (the capture and labeling program that produces them is
[13 · Corpus & Capture](../vision/13-corpus-and-capture.md)):

- *Synthesized markings.* Metronome-locked counting recorded (or TTS'd) at a
  known BPM and meter gives perfect labels at zero annotation cost, and can be
  permuted across the exact matrix that breaks the pipeline: numbers vs step
  names vs vocables vs near-silence × 2/4, 3/4, 4/4, 6/8 × with and without
  interleaved explanation. Twenty of these produce an honest zero-shot meter
  number *this week*. They are not real rooms and must never be the gate — but
  they localize failures precisely, which real rooms do not.
- *Accompanied class recordings — free `performance_bpm`.* In any class with a
  live pianist, the piano audio **is** the gold label for what the teacher
  wanted; beat-tracking it (librosa) yields `performance_bpm` mechanically,
  while the marking that precedes it yields `marking_bpm`. That pair is doc
  08's novel research result and the `tempo_offset` prior of §5.6, obtainable
  from public recordings without a single human annotation.
- *Model-assisted pre-annotation with human verification*, with inter-annotator
  agreement measured on a subset so the eval's own noise floor is known.

**5 · No model grades its own homework.** No field may be scored by the model
family that produced it — an obvious trap when Gemini is both the system under
test and the most convenient judge. Where a judge is unavoidable (qualitative
character), pin its version per run and report judge-vs-human agreement as a
published number before trusting it.

**6 · Slices are the gate; the mean is a headline.** Report every metric per
teacher, count style, exercise slot, language, and accompanied-vs-recorded.
Gate on the **worst slice above a minimum n**, which is what §8.6's per-teacher
gating already implies.

**7 · Split at the class level, and hold out an unseen teacher.** Clip-level
splits leak — same teacher, same combination, same room — and per-teacher
calibration makes that leakage especially seductive. Calibration is evaluated
leave-one-class-out per teacher; the only honest zero-shot number comes from
the unseen-teacher split. A sealed test split runs at gates only, never in the
iteration loop.

**8 · State n, and quote intervals.** A 90% claim on 30 cases has a ±11% 95%
interval — worthless against a 90% gate. For ±5% at p≈0.9 you need n≈140
exercises; for ±3%, n≈350. Doc 08's 200–400-exercise corpus target is the right
order of magnitude, and this is why. Until n is sufficient, the harness reports
intervals and the gate stays explicitly provisional.

**9 · Every run is an artifact.** One JSON per run (metrics, per-case rows, and
the hashes that make it reproducible: git sha, model ids, prompt hash, trace
version) under `evals/runs/`, plus a static HTML report diffing against
baseline. PRs touching perception show the delta. ADR results tables stop being
hand-pasted and start being generated.

**10 · Wire the tripwires.** [09 · Risk Register](../vision/09-risk-register.md)
already names thresholds (R2: calibrated meter < 90%; R1: IoU < 80%; R4: >8%
error on >15% of exercises). The harness emits exactly those fields so a
tripwire trips automatically instead of being noticed.

---

## CI policy

| Tier | Trigger | Cost | Failure means |
|---|---|---|---|
| 0 Synthetic | every PR | free, <5 s | block |
| 1 Frozen traces | every PR | free, <30 s | block on regression vs baseline |
| 2 Live perception | nightly + on prompt/model change | API $ | report drift; investigate |
| 3 Corpus | per milestone / gate | annotation | gate decisions (§8.3, §8.6) |
| 4 Shadow | per shadowed class | — | rung promotion (§8.6) |

## First increment (what to build now)

1. `evals/` package — case schema + loader, the scorer module (tempo with
   metric-level classification, coherent-triple meter, counts, quality, the
   abstention/calibration accounting), JSON+HTML reporter,
   `python -m musical_perception.evals run --suite tier0`.
2. `PerceptionBundle` seam in `analyze()` + `--record-traces`.
3. Synthetic suite: ~24 metronome-locked cases spanning the style × meter ×
   interleaved-explanation matrix.
4. Port ADR-006's and ADR-007's hand-run tables into tier-1 cases — traces
   frozen from those exact files. The first regression suite is four rows the
   project already has.
5. Publish the baseline table, whatever it says. That is Gate A2's deliverable.

## Consequences

- The KEEP/DISPOSABLE split gains the objective swap test doc 08 §8.4 promises;
  `analyze.py` batch mode becomes the replayer, as intended.
- Measurement starts weeks before the corpus, and none of it is thrown away —
  tiers 0–2 remain the fast loop after tier 3 exists.
- Corpus collection stays on the critical path for *claims*, not for *progress*.
- Cost: one refactor of `analyze()`, a new package, and the discipline of
  labeling before building.
- The suite absorbs the consequences: the ladder and the pre-corpus label
  sources land in [08](../vision/08-benchmark-and-shadow-mode.md) §8.4, the
  scoring discipline in §8.3, the sequencing in
  [11](../vision/11-roadmap.md) Track 2, and the tripwire wiring in
  [09](../vision/09-risk-register.md).
- Risk: synthetic cases are not real rooms. Mitigated by never gating on tier 0
  and by the unseen-teacher split at tier 3.
