# 04 · System Architecture

The whole system, end to end: hardware, the two-tier runtime, the exercise
lifecycle, and where the existing codebase slots in. Companion documents
specify the perception internals ([05](05-perception-strategy.md)), the
playback engine ([06](06-performance-engine.md)), and the interaction policy
([07](07-interaction-design.md)).

---

## 4.1 Design rules

1. **Two latency domains, never mixed.** Anything that must happen inside the
   music's flow (stop, start-on-cue, tempo nudge) is **reflex**: local,
   <300 ms, zero cloud dependency. Anything that can take seconds (marking
   analysis, repertoire choice) is **deliberation**: cloud or local LLM,
   eager, cancellable. A cloud outage degrades intelligence, never control.
2. **Eager hypotheses, late commitment.** Deliberation runs *while the teacher
   marks*; the reflex layer's cue detection commits the current best
   hypothesis. No "record, then analyze, then respond" batching in live mode.
3. **Timing stays local.** Word/onset timestamps, pose, beat clock — all
   on-device. Cloud models see semantics, not clocks (they are bad at clocks;
   see the Gemini Live 1 FPS / 2-min A/V limits in
   [03 · Landscape](03-landscape.md), and ADR-006's "estimated_bpm —
   unreliable, logged but not used").
4. **Silence-biased arbitration.** Every ambiguous decision resolves toward
   not playing. See [07 · Interaction Design](07-interaction-design.md).

## 4.2 Hardware (v1 reference rig)

| Item | Choice | Why |
|---|---|---|
| Teacher audio | Wireless lapel/headset mic (~$150) | Converts far-field research into engineering: clean VAD, speaker identity for free, marking prosody at high SNR. Human parity: the pianist also positions to hear the teacher |
| Room audio | The device's mic (ambient) | Secondary: room energy, music echo cancellation reference |
| Camera | Single wide-angle, teacher-framed, 30 FPS | Pose + gesture cues; 30 FPS on-device pose is commodity |
| Compute | Laptop-class (M-series or equiv.), or mini-PC | Runs all reflex components + pose on CPU; cloud for deliberation |
| Output | Studio speaker (existing) | — |

Explicitly deferred: mic arrays / no-mic far-field, multi-camera, custom
hardware. (A studio Disklavier rendering the MIDI on a real acoustic piano is
the flagship embodiment someday — not v1.)

## 4.3 The two tiers

```
                        ┌──────────────────────────────────────────────┐
                        │              DELIBERATION (seconds)          │
                        │  streaming marking analysis · class-state    │
                        │  tracker · repertoire selection · plan       │
                        │  compilation · per-teacher calibration       │
                        │  (Gemini / cloud LLM / local LLM)            │
                        └───────▲──────────────────────────┬───────────┘
                hypotheses,     │                          │ PerformancePlan
                evidence        │                          ▼
┌───────────────┐   ┌───────────┴───────────┐   ┌──────────────────────┐
│ teacher mic   ├──▶│    REFLEX (<300 ms)   │──▶│  PERFORMANCE ENGINE  │
│ room mic      │   │ VAD · speaker ID ·    │   │ MIDI scheduler ·     │
│ camera (pose) ├──▶│ keyword spotter ·     │   │ piano render ·       │
└───────────────┘   │ cue detector ·        │   │ vamp/endings/button ·│
                    │ onset/beat clock ·    │   │ instant stop         │
                    │ stop arbiter          │   └──────────────────────┘
                    └───────────────────────┘
                          all local, no cloud on the critical path
```

**Reflex components** (always on, laptop CPU):

- **VAD + teacher speaker ID** — is the teacher speaking? (Lapel mic makes
  this nearly trivial.)
- **Keyword spotter** — the small command grammar of
  [07 · Interaction Design](07-interaction-design.md): stop/again/other
  side/faster/slower/music-please. Local streaming ASR or dedicated KWS.
- **Cue detector** — the "aaand—" preparation cue: elongated vowel + pitch
  rise + inhale on the mic; preparatory arm/posture on camera. Replaces the
  ADR-008 wake word. Commits the pending hypothesis.
- **Onset/accent extractor** — raw-audio pulse and accent features streamed to
  deliberation ([05 · Perception](05-perception-strategy.md)).
- **Beat clock + stop arbiter** — during playback: tracks the musical
  position; kills sound <300 ms when the teacher speaks over the music
  (with the [07](07-interaction-design.md) grace rules).

**Deliberation components** (eager, seconds):

- **Streaming marking analyzer** — maintains running hypotheses
  {exercise, meter, tempo, counts, character} as the marking unfolds.
- **Class-state tracker** — Bayesian position-in-class distribution
  ([05](05-perception-strategy.md)).
- **Repertoire selector + plan compiler** — `MusicalParameters` →
  `PerformancePlan` ([06](06-performance-engine.md)).
- **Calibration store** — per-teacher profile, read at class start, updated
  after class ([08 · Benchmark](08-benchmark-and-shadow-mode.md)).

## 4.4 The exercise lifecycle (runtime state machine)

Supersedes the two-state ADR-008 trigger machine, whose wake-word entry is
replaced by marking/cue detection.

```
        teacher talks (not rhythmic)                       ┌────────────┐
   ┌──────────────── ATTEND ◀──────────────────────────────┤ post-class │
   │                    │                                  └────────────┘
   │   marking detected │ (rhythmic vocalization + priors)
   │                    ▼
   │                 LEARNING ── streaming analysis; hypotheses update
   │                    │        continuously; may pre-select repertoire
   │   marking ends     │
   │                    ▼
   │                 ARMED ───── best hypothesis + compiled plan held;
   │                    │        confidence gates the rung:
   │                    │        high → wait for cue; medium → ask one
   │                    │        word; low → stay silent (rung 2/3)
   │   cue detected     │
   │   ("aaand—")       ▼
   │                 PERFORMING ─ intro at exact tempo → body → ending
   │                    │         reflex listens: stop/nudge/again
   │        teacher     │
   │        speaks /    ▼
   │        combination ends
   └───────────────  RESOLVE ─── button + clean stop; log outcome;
                        │        "again"/"other side" → PERFORMING
                        ▼        (same plan, repeat/transpose)
                     ATTEND
```

Notes:

- **ATTEND ≠ idle.** The class-state tracker keeps updating from ambient
  evidence (exercise names spoken, time elapsed, barre/center camera cues).
- **ARMED is where the asymmetric policy lives.** No cue → no sound, ever.
  A missed cue recovers verbally ("music, please" → treat as cue).
- **PERFORMING keeps deliberation out of the loop.** Only reflex signals
  (stop, nudge, again) touch running music — the ROADMAP-v2
  Learn/Accompany insight, preserved.

## 4.5 Latency budget

| Path | Budget | Mechanism |
|---|---|---|
| Teacher speaks over music → silence | <300 ms | Local VAD + stop arbiter |
| Cue → first intro note | <500 ms | Plan pre-compiled in ARMED; scheduler starts on commit |
| Command ("slower") → tempo change | <1 beat | Reflex keyword → engine nudge |
| Marking → ARMED with plan | ≤ marking duration + ~2 s | Eager streaming analysis; the teacher's natural 3–15 s walk-to-position absorbs the tail |
| One-word question → resume | ~1–2 s | Local TTS, pre-rendered phrasings |

The generous budget (20–60 s of marking + the gap) is the structural luck of
this domain: **the only sub-second decisions are start and stop.** Everything
intelligent gets seconds.

## 4.6 Where the existing codebase lands

| Existing | Disposition |
|---|---|
| `precision/tempo.py`, `subdivision.py`, `rhythm.py`, `signature.py` | **Keep** — the input-agnostic design pays again: same math, new event sources (raw-audio onsets, pose events) |
| `precision/` (new) | **Add** accent-periodicity meter module; class-state tracker; calibration math ([05](05-perception-strategy.md)) |
| `perception/whisper.py` | Keep, demoted: words become *evidence* (commands, numbers, exercise names), not the rhythm channel |
| `perception/gemini.py` | Keep for semantics (exercise, quality, structure); never for clocks; move to streaming calls in LEARNING |
| `perception/pose.py` + `precision/dynamics.py` | Keep — quality evidence + cue/gesture features |
| `trigger.py` + `perception/wakeword.py` | **Retire the wake word**; the state machine generalizes into §4.4's lifecycle. The dependency-injection test pattern carries over |
| `analyze.py` (batch) | Keep forever — it is the **benchmark replayer** ([08](08-benchmark-and-shadow-mode.md)) |
| `MusicalParameters` | The contract holds; gains a consumer (`PerformancePlan` compiler, [06](06-performance-engine.md)) |

New top-level modules (target): `attend.py` (lifecycle + reflex wiring),
`engine/` (performance engine), `state/` (class-state + calibration store),
`shadow.py` (shadow-mode recorder/reporter).

## 4.7 Cost and footprint

Reflex + pose: laptop CPU, no GPU. Deliberation: pennies per exercise
(~$0.03–0.09 per analyzed clip at current Gemini Flash pricing), ~20 exercises
per class → **under a dollar per class**. Storage: MIDI library is megabytes;
class recordings (for shadow/benchmark) are the only real storage consumer.

## 4.8 Failure modes and degradation

| Failure | Behavior |
|---|---|
| Cloud unreachable | Reflex + engine still fully functional; fall back to prior-based defaults per exercise slot + verbal confirmation ("Adagio in four?") — class continues |
| Mic battery dies | Detect signal loss; announce once; ladder rung 3 (tap) until restored |
| Confidence collapse mid-class | Drop a rung ([07](07-interaction-design.md)); never guess loudly |
| Wrong piece committed | "Stop — waltz" → new plan <2 s; log as a benchmark miss |
