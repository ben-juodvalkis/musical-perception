# 05 · Perception Strategy

How the system understands the teacher. The strategy in one line: **demote ASR
from spine to evidence, promote prosody and priors, and let per-teacher memory
close the gap.** This document specifies the channels, the meter plan, the
exercise-prior table, the class-state tracker, and the calibration profile.

---

## 5.1 Why the current pipeline tops out

The pipeline today is ASR-first: Whisper produces word timestamps; Gemini
classifies words; even the "classification-free" onset tempo
(`precision/rhythm.py`) runs on *Whisper word onsets*. The record
(ADR-006/007) shows exactly where that ceiling is:

- Step-name counting broke word classification (fixed by prompt, fragile).
- Mixed explanation+counting speech pollutes onset windows.
- Meter lands ~50% — and wrong meter is the loudest possible failure.
- "Da-da-POM," hummed marking, and non-English idiom fall outside ASR's model
  of the signal entirely.

Marking is not speech that happens to be rhythmic; it is **a percussion track
wearing words**. The professional literature confirms teachers are trained to
encode tempo/meter/quality into the marking deliberately ("mark the first 8
counts in the exact tempo you want"). The signal is designed to be read — but
by ears, not by a lexicon.

## 5.2 The channel inventory (target state)

| Channel | Extracts | Role | Status |
|---|---|---|---|
| **Raw-audio onsets** (energy/spectral flux, VAD-gated, teacher mic) | Pulse; IOI series | **Primary rhythm channel** — replaces Whisper onsets as input to `rhythm.py` (input-agnostic by design; the KEEP bet pays a third time) | To build (reflex layer) |
| **Accent features** (intensity, duration, F0 prominence per onset) | Accent salience series | **The meter signal** (§5.3); extends `signature.py`'s prosodic-weight work | Partially exists (Praat path) |
| **ASR (streaming + Whisper)** | Words as *evidence*: commands, numbers, exercise names, French terms | Demoted. Numbers are gold when present — they label beats and phrase position ("5-6-7-8" reveals prep convention and phrase starts) | Exists; role change |
| **Gemini (multimodal)** | Exercise semantics, per-phrase quality, structure, character | Semantics only, never clocks (ADR-006's `estimated_bpm: unreliable` stays policy) | Exists |
| **Pose** (MediaPipe class) | Marking movement rhythm (velocity peaks → secondary tempo evidence), cue gesture, barre/center scene, quality dynamics | Secondary evidence + cue detection | Exists (`pose.py`, `dynamics.py`) |
| **Priors** (exercise table §5.4 + class-state §5.5) | Prior distributions per hypothesis field | Constrains everything | To build |
| **Per-teacher memory** (§5.6) | Calibration profile | Converts zero-shot to few-shot | To build |

All channels feed one **hypothesis record**, maintained continuously during
LEARNING:

```python
@dataclass
class ExerciseHypothesis:
    exercise: Distribution[str]        # over exercise types
    meter: Distribution[Meter]         # jointly with subdivision (coherent triple)
    tempo_bpm: Gaussian                # performance tempo (after teacher offset)
    counts: Distribution[int]          # 16 / 32 / 64, per side
    character: QualityProfile          # + per-phrase when available
    sides: int
    confidence: float                  # gates the interaction rung (07)
```

`NormalizedTempo`'s coherent-triple discipline (ADR-007) is preserved: meter,
BPM, and subdivision commit together or not at all.

## 5.3 The meter plan (the crux)

Meter gets three votes and a floor:

1. **Accent periodicity (new precision module, KEEP).** Per-onset accent
   salience `a_i = w₁·intensity_z + w₂·duration_z + w₃·F0-prominence_z` —
   exactly the features `signature.py` already measures. Autocorrelate the
   accent series at candidate lags (2, 3, 4, 6 beats): "ONE-two-three
   ONE-two-three" is a peak at lag 3. No words involved; works on vocables and
   any language.
2. **Priors.** The exercise table (§5.4) weighted by the class-state tracker
   (§5.5): a marking in the rond-de-jambe slot starts life mostly 3/4.
3. **Semantics.** Gemini's meter opinion and any lexical evidence ("balancé"
   implies 3/4; "polka" implies 2/4).

Fused into a posterior over coherent (meter, BPM, subdivision) triples via an
extended `interpret_meter()`. Then the **decision policy**:

| Posterior of top triple | Action |
|---|---|
| ≥ 0.80 | Commit silently |
| 0.55 – 0.80 | Ask one word at ARMED ("Waltz?") — budgeted per [07](07-interaction-design.md) |
| < 0.55 | Stay silent; rung 2/3 recovery |

The floor matters: even human accompanists ask "in three?" A system allowed
one word when genuinely split does not need superhuman meter perception —
it needs well-calibrated *uncertainty*, which the benchmark
([08](08-benchmark-and-shadow-mode.md)) measures directly: abstention is scored
as its own outcome and calibration error is reported per confidence bin
(§8.3), so these two thresholds are tuned against a curve rather than chosen.

## 5.4 The exercise-prior table

Seed values from accompanist practice; **every cell is a prior, not a rule**,
and per-teacher calibration (§5.6) overrides all of it. Tempo is the musical
beat (quarter, or dotted-quarter in compound meters) — aligned with the
70–140 counting-pulse band the normalization work already targets.

**Barre** (canonical order):

| Slot | Exercise | Meter prior | Beat BPM | Counts↔bars | Character (art./weight/energy) | Genre exemplars |
|---|---|---|---|---|---|---|
| 1 | Pliés | 3/4 slow ·5 / 4/4 adagio ·5 | 58–80 | 1 count = 1 beat | legato / grounded / calm | sarabande, slow waltz, nocturne |
| 2 | Battement tendu | 4/4 ·5 / 2/4 ·3 / 3/4 ·2 | 96–126 | 1 = 1 | crisp / mid / moderate | march, gavotte, tango |
| 3 | Battement dégagé (jeté/glissé) | 2/4 ·6 / 4/4 ·4 | 112–144 | 1 = 1 | sharp / light / bright | polka, galop |
| 4 | Rond de jambe à terre | 3/4 ·8 / 6/8 ·2 | 126–168 (waltz ¼) | **1 count = 1 bar** | flowing / mid / calm | waltz, barcarolle |
| 5 | Battement fondu | 3/4 slow ·5 / 4/4 ·5 | 58–76 | 1 = 1 (or 1 = 1 bar slow waltz) | melting legato / sustained / gentle | slow waltz, andante |
| 6 | Battement frappé | 2/4 ·7 / 4/4 ·3 | 100–126 | 1 = 1 | staccato / accented / sharp | polka, character 2/4 |
| 7 | Petit battement | 2/4 ·8 | 116–144 | 1 = 1 | light / crisp / quick | polka, moto perpetuo |
| 8 | Rond de jambe en l'air | 2/4 ·5 / 3/4 ·5 | 104–132 | 1 = 1 | rounded / mid / moderate | waltz, 2/4 |
| 9 | Adagio | 4/4 ·4 / 3/4 ·3 / 12/8 ·3 | 46–66 | 1 = 1 (long counts) | sustained / weighted / calm | adagio, barcarolle, aria |
| 10 | Grand battement | 4/4 march ·5 / 3/4 grand ·5 | 88–116 | 1 = 1 (battement per 1–2 counts) | powerful / grounded / strong accent | march, polonaise, grand waltz |

**Center** (typical order):

| Slot | Exercise | Meter prior | Beat BPM | Counts↔bars | Character | Genre exemplars |
|---|---|---|---|---|---|---|
| 11 | Adagio (center) | as barre adagio, grander | 46–66 | 1 = 1 | sustained / expansive | adagio, aria |
| 12 | Tendu / small center | as barre tendu | 96–126 | 1 = 1 | crisp | march, tango |
| 13 | Pirouettes | 3/4 waltz ·6 / 4/4 ·4 | 120–160 (waltz ¼) | often 1 = 1 bar | poised, strong prep | waltz, coda |
| 14 | Petit allegro | 2/4 ·5 / 6/8 ·5 | 108–132 (¼ or ♩.) | 1 = 1 | sparkling / light / quick | polka, gigue |
| 15 | Medium allegro | 3/4 ·5 / 2/4 ·5 | 116–144 | mixes | buoyant / bigger | waltz, galop |
| 16 | Grand allegro | 3/4 grand waltz ·7 / 6/8 ·3 | 160–200 (¼) | **1 count = 1 bar** (counts ~54–66/min) | soaring / powerful / big | grand waltz, mazurka |
| 17 | Coda / manège | 2/4 ·8 | 120–144 | 1 = 1 | driving / brilliant | coda, galop |
| 18 | Révérence | 3/4 ·5 / 4/4 ·5 | 60–84 | 1 = 1 | gracious / calm | slow waltz, hymn-like |

**The counts↔bars column is load-bearing.** Dancers count in dancer's-eights
that map to bars differently per genre (one count per waltz *bar* in rond de
jambe and grand allegro; one count per *beat* in tendu). This is precisely the
metric-level ambiguity ADR-006/007 fought (raw 40 BPM vs pulse 120): the table
turns it from a per-clip puzzle into a per-genre lookup. The performance
engine consumes the same mapping ([06](06-performance-engine.md) §6.5).

## 5.5 The class-state tracker

A Bayes filter over class position. States: the exercise slots above, plus
`talk`, `stretch`, `break`. Transition prior: canonical order (strong
self-loop + forward-step mass, small skip mass), overridden by the teacher's
learned `class_template`. Evidence, each with a learned reliability weight:

- spoken exercise names / French terms (ASR),
- the marking analyzer's running (meter, tempo, character) posterior scored
  against each slot's priors,
- elapsed class time and per-slot duration statistics,
- scene evidence from pose (at barre vs center; facing; group formation).

Output: `P(slot)` — consumed as the prior in §5.3, by repertoire preselection
(the selector can warm-load candidates for the *next* likely slot during
ATTEND), and by the shadow-mode report.

The tracker is pure math over model outputs → `precision/` (KEEP), with
perception adapters supplying evidence.

## 5.6 The per-teacher calibration profile

Learned from shadowed classes ([08](08-benchmark-and-shadow-mode.md));
read at class start; updated after every class.

```yaml
teacher: <id>
tempo_offset:            # marking→performance multiplier, e.g. ×1.06
  default: 1.06
  per_exercise: {grand_battement: 1.00, adagio: 1.10}
count_style: numbers | step_names | vocables | mixed
idiom_lexicon:           # vocables → rhythmic role
  "pom": accent          # "da-da-POM" → beat-3 accent
  "and-a": triplet_pickup
meter_habits:            # exercise → meter distribution overrides
  fondu: {3/4: 0.8, 4/4: 0.2}
structure_habits:
  default_counts: 32
  en_croix: common
  sides: both, right_first
cue_signature:           # the teacher's personal "aaand—"
  mean_duration_ms: 620
  pitch_contour: rising
  gesture: right_arm_lift
command_vocab:           # personal stop/go phrases
  stop: ["thank you", "okay hold on"]
  go: ["music please", "let's go"]
music_prefs:
  avoid: [ragtime]
  favor: [chopin-esque waltzes]
  repeat_tolerance_days: 3
class_template:          # this term's observed slot sequence + per-slot stats
  - {slot: plies, counts: 32, meter: 3/4, tempo: 66}
  - ...
```

Two or three shadowed classes populate everything above to useful precision —
teachers are habitual, which is the entire point of
[Reframe 3](02-reframes.md). The profile is also the honest answer to
"which teachers will this work for": ones whose profile converges. That is
measurable, per teacher, before going live.

## 5.7 What stays true from the current architecture

- **Precision stays input-agnostic and grows** (accent periodicity, state
  tracker, calibration math — all pure, testable, hardcoded-data tests).
- **Perception wrappers stay disposable** — streaming ASR vendors, Gemini
  versions, pose models will all be swapped; nothing above depends on which.
- **`MusicalParameters` remains the contract**, now produced with a posterior
  and consumed by a real engine.
- **Whisper-owns-timestamps dies as a slogan** but survives as a role: ASR
  timestamps still anchor *lexical* events (numbers, commands) — they just no
  longer carry the rhythm channel alone.
