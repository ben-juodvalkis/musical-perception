# 06 · Performance Engine

The missing half of the system. `MusicalParameters` is a contract with no
consumer: nothing plays. This document specifies the engine that does — the
repertoire library, the plan compiler, the runtime controls, and the rendering
path. Design verdict from the landscape ([03](03-landscape.md) §3.4):
**symbolic MIDI first**; time-stretched recordings as stopgap; generative
audio rejected for structural work.

---

## 6.1 The musical obligations

Derived from what class actually requires of an accompanist:

| Obligation | Meaning | Feasible with |
|---|---|---|
| Exact tempo | Any BPM, not a menu of recordings | MIDI ✅ · stretch ~0.8–1.3× ⚠️ · generative ❌ |
| Exact length | 16/32/64 counts, per side, en croix | MIDI (repeats/cuts at phrase bounds) ✅ · recordings ❌ |
| The intro | 2 chords or 4 counts ("aaand—") at performance tempo, per teacher habit | MIDI ✅ |
| Square phrasing | Cadence lands on the final count of each 8/16 | MIDI (by construction) ✅ |
| Vamp till ready | Hold a 1–2 bar loop if the room isn't set | MIDI ✅ |
| First/second endings | "Other side" repeats with a proper turnaround | MIDI ✅ |
| The button | A 1–2 chord tag confirming the end | MIDI ✅ |
| Instant stop | <300 ms, damped, not a click-off | MIDI (damper + fast release) ✅ |
| Live tempo nudge | ±10% mid-piece without artifacts | MIDI ✅ · stretch ⚠️ |
| Character | Articulation/weight/energy matching `QualityProfile` | MIDI (velocity/articulation/pedal maps) ✅ |

Every ❌ above is why the playlist workflow taxes teachers today — and why
recordings can be a *tempo* stopgap but never the length/structure solution.

## 6.2 The library

**Size:** ~30 pieces for the MVP demo; **100–200 for v1** — the working
repertoire of a human accompanist. Coverage target: every (meter × tempo band
× character) cell demanded by the prior table
([05](05-perception-strategy.md) §5.4), with ≥3 pieces per common cell so
repetition can be avoided.

**Sourcing:** public-domain class literature (the classical class-piano
repertoire: waltzes, polkas, marches, adagios, mazurkas, galops, tangos, rags),
original arrangements, transcriptions. Each piece arranged *for class*: square
8-bar phrases, clear cadences, marked vamp bars, marked ending variants.

**Format:** one MIDI file + one metadata sidecar per piece.

```yaml
id: waltz-grand-017
title: "Grand Waltz in E-flat"
meter: 3/4
tempo_range: [152, 200]        # musical beat BPM
counts_per_bar: 1              # dancer count ↔ bar mapping (see §6.5)
phrase_bars: 8                 # square unit
sections: [A, A, B, A]         # each = phrase_bars
vamp: bars 0-1                 # loopable intro/hold
endings: {first: bars 31-32, final: bars 31-33+button}
character: {articulation: 0.7, weight: 0.6, energy: 0.9}   # QualityProfile space
genres: [grand-waltz]
exercise_affinity: [grand_allegro, pirouettes]
provenance: PD-arrangement
```

`character` lives in the same `QualityProfile` space perception produces —
selection is nearest-neighbor in the space the system already measures, which
is why quality was made numeric in the first place (ROADMAP-v2).

## 6.3 The plan compiler

`compile(params: MusicalParameters, profile: TeacherProfile) → PerformancePlan`

1. **Select** candidate pieces: filter by coherent (meter, subdivision);
   score by tempo-range fit, character distance, exercise affinity,
   teacher `music_prefs`, and a no-repeat window (don't replay a piece within
   N classes; never twice in one class).
2. **Adapt length:** map requested counts → bars via `counts_per_bar`;
   realize with section repeats/cuts at phrase boundaries only; choose
   endings (both-sides → first/second ending; single → final + button).
3. **Set tempo:** perception tempo × teacher `tempo_offset` (per-exercise
   override), clamped to the piece's `tempo_range`; if clamping would exceed
   ±8% of target, reject the piece and rescore.
4. **Prefix the intro** per teacher habit (2 chords vs 4 counts vs 2 bars).
5. Emit:

```python
@dataclass
class PerformancePlan:
    piece_id: str
    tempo_bpm: float
    intro: IntroSpec                 # style + count-in length
    body: list[SectionRef]           # realized section sequence
    ending: EndingSpec               # first/second/final+button
    character_map: RenderParams      # velocity/articulation/pedal targets
    fallbacks: list[str]             # next-best piece ids (instant swap)
```

Compilation is pure and fast (<50 ms): it runs speculatively in ARMED, and
`fallbacks` makes "stop — waltz" a <2 s swap.

## 6.4 Runtime control surface

The reflex layer ([04](04-system-architecture.md)) drives the engine through a
small verb set — everything in [07 · Interaction Design](07-interaction-design.md)
compiles down to these:

```
start(at_cue_time)        # begin intro so count 1 lands on schedule
stop(grace="damp")        # <300 ms damped stop, any position
pause() / resume()
nudge_tempo(factor)       # ±2% per "bit", ±6% per "much"; ramp over 1 bar
set_energy(delta)         # velocity/pedal remap, applied at next phrase
again()                   # restart body (same side): from the vamp
other_side()              # repeat with first→second ending logic
finish_from_here()        # jump to nearest ending at next phrase boundary
```

Rules: tempo changes ramp (never jump) across one bar; structural verbs
(`again`, `other_side`, `finish_from_here`) take effect at phrase boundaries;
`stop` is the only verb that acts mid-phrase, because it models the teacher
speaking — and the teacher always wins instantly.

## 6.5 Counts↔bars (shared convention)

The engine and perception share one mapping (defined per genre in the
[prior table](05-perception-strategy.md) §5.4): a "count" is the dancer's
counting pulse; `counts_per_bar` declares how it lands in the music (1 per
beat for tendus; 1 per bar for grand waltzes and rond de jambe). All
length arithmetic — "32 counts, both sides" — passes through this mapping.
This is the engine-side face of the metric-normalization work (ADR-006/007):
the 70–140 band is the *counting pulse*, and the piece metadata declares how
the music realizes it.

## 6.6 Rendering

- **Engine:** physical-modeling piano (Pianoteq-class; ~50 MB, real-time,
  continuous velocity/pedal response, Linux-friendly) or a quality sampled
  piano via a standard sampler. Either is commodity; the choice is taste.
- **Expression:** not flat MIDI — per-genre humanization templates (voicing
  balance, micro-timing envelopes, phrase-end breath on adagios, waltz
  bass-lightness), driven by `character_map`. Templates are data, not code:
  improving musicianship = editing templates, no releases.
- **Output chain:** local audio to studio speaker; total start latency budget
  <500 ms from cue commit ([04](04-system-architecture.md) §4.5).

**Stopgap path (optional):** for teachers attached to specific recordings, an
élastique/SoundTouch-class stretch of their own tracks inside 0.8–1.3× — with
the honest limitation that length/structure verbs don't apply. Ship only if
pilot teachers demand it; it dilutes the core advantage.

## 6.7 Growth path (not v1)

1. **Variation engine:** re-harmonization/re-figuration of library pieces for
   freshness (same skeleton, new surface) — symbolic transforms, fully
   structure-safe.
2. **Constrained symbolic generation** (MIDI-GPT/NotaGen-class) behind the
   same `PerformancePlan` contract with hard validators (bar count, cadence,
   tempo) and rejection sampling — variety without structural risk.
3. **Style capture:** learn a specific accompanist's voicing/figuration
   habits from their MIDI performances (with consent) as a rendering
   template.
4. **The Disklavier embodiment:** the engine's MIDI driving a real acoustic
   player piano in the studio — the flagship demo, in Yamaha's 2016
   AI-ensemble lineage.

## 6.8 Build order

1. Scheduler + renderer + `start/stop/nudge` on 3 hand-prepared pieces (the
   [90-day track 1](11-roadmap.md)).
2. Metadata schema + compiler on ~30 pieces; length realization + endings.
3. Humanization templates; selection scoring + no-repeat memory.
4. Library to 100+; character map tuning against pilot-teacher feedback.
