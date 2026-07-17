# 01 · North Star

**The product in one sentence:** a studio speaker with a camera and a
microphone that plays the right music for every exercise by itself — the
teacher teaches, and the room simply works, the way it does when a great
accompanist is at the piano.

---

## The magic moment

It is 9:47 AM, fourth exercise of barre. The teacher finishes marking a frappé
— "and a ONE, and a TWO, flex through THREE and…" — walks back toward the
mirror, scans the room, and says "aaand—". A crisp 2/4 at 104, two-chord
intro, first accent lands exactly with thirty tendu-flexed feet striking the
barre. Sixteen bars later the music cadences on the final count with a small
button. The teacher never looked at a phone. Nobody touched anything.

Then the teacher starts talking about wrists. The music is already silent —
it stopped the instant she drew breath to speak.

That is the whole product. Everything in this documentation suite exists to
make that minute real, reliable, and repeatable across a full class.

---

## What "seamless" means (and what it does not)

The requirement is **no manual interaction** — but the correct definition of
that bar is *human parity*, not telepathy:

| A human accompanist… | So the system… |
|---|---|
| Watches and listens to the marking; needs no announcement | Detects and analyzes marking without a wake word |
| Starts from the room's natural cue ("aaand—") | Treats the cue as the trigger; joins the existing protocol |
| Is told "waltz please," "slower," "again," "other side" | Accepts natural verbal input as first-class, not as failure |
| Asks "in three?" when the marking is ambiguous | May ask **one short question** when genuinely uncertain |
| Needs a few classes to sync with a new teacher | Calibrates per teacher from a few shadowed classes |
| Stops the moment the teacher speaks | Runs a local reflex layer with <300 ms stop latency |
| Never requires the teacher to change how they teach | Never requires counting "cleanly," numbers, or ritual phrases |

What seamless does **not** require: reading minds, zero-shot perfection on a
never-seen teacher, following dancers' bodies through an adagio, or improvising
like a conservatory pianist. Humans don't clear those bars either — or don't
need to.

## What we are explicitly not building (v1)

- **A Turing-test pianist.** The v1 musical standard is *excellent rehearsal
  pianist*: right piece, right meter, exact tempo, square phrases, clean
  endings. Rubato and breath come later, if ever.
- **A dancer-follower.** At barre, holding tempo *is* the job. Center-work
  adaptivity (breathing with an adagio, catching a grand allegro) is deferred
  research — see [09 · Risk Register](09-risk-register.md).
- **A generative-music system.** Playback is a curated symbolic library.
  Generation is a possible future behind the same interface — see
  [06 · Performance Engine](06-performance-engine.md).
- **A far-field no-hardware system.** v1 accepts a wireless teacher mic and a
  wide camera. See [04 · System Architecture](04-system-architecture.md).

---

## Who it is for

**First: the playlist majority.** The overwhelming share of ballet classes
worldwide run on recorded albums and a phone. Fixed tempi, fixed 32/64-count
tracks, the teacher DJ-ing between every exercise — and designing combinations
around the recordings they own, so the music constrains the pedagogy. Existing
apps (Ballegro, Ballet Class, BalletBox, Cadence — see
[03 · Landscape](03-landscape.md)) monetize this pain but still require hands:
the teacher enters tempo and bar count per exercise on a touchscreen. **Our
job is to delete that form.**

**Not first: schools with live accompanists.** Where a good pianist exists,
they are better than v1 of this system and should keep their job. The system
competes with the playlist, not the pianist. (The pianist remains the north
star and the calibration standard.)

## Product principles

1. **Join the room's protocol; never impose one.** The cue, the corrections,
   the "thank you" that ends an exercise — these already exist. The system
   learns the room; the room never learns the system.
2. **Silence over false starts.** Playing when the teacher is still talking is
   trust-destroying; a missed start costs three seconds and recovers naturally
   ("music, please"). Bias every threshold toward silence. Target: **zero
   false starts per class**, forever.
3. **The teacher never performs for the machine.** Any design that requires
   counting with numbers, separating explanation from counting, or speaking a
   ritual phrase has failed the brief (this was the honest lesson of
   ADR-007's "instruct the teachers" mitigation).
4. **Voice is the interface.** Whatever a teacher would say to a human
   accompanist is valid input; one-word questions back are valid output.
5. **Measured, not believed.** No capability claim without a number from the
   [benchmark](08-benchmark-and-shadow-mode.md). Shadow mode before sound.
   Go-live gates per teacher.
6. **Better than the playlist, every single class.** Any exercise where the
   system does worse than the teacher's old recording workflow is a bug with
   a name and a ticket.

## Success criteria by horizon

| Horizon | Success looks like |
|---|---|
| **One exercise** (demo) | Teacher marks naturally → correct meter, tempo ±8%, correct length → music starts on the cue → stops on speech. No touch. |
| **One class** (pilot) | Full barre hands-free with ≤2 one-word clarifications, zero false starts, teacher chooses to use it again tomorrow. |
| **One term** (product) | A calibrated teacher runs class with the ladder at rung 1–2 (see [07 · Interaction Design](07-interaction-design.md)); combinations are no longer designed around available recordings. |
| **One year** (mission) | Studios that could never afford an accompanist teach as if they had one. The benchmark corpus is the field's reference dataset. |

## Relationship to the rest of the suite

This document states the destination. [02 · Reframes](02-reframes.md) explains
the three framing moves that make it reachable;
[04](04-system-architecture.md)–[08](08-benchmark-and-shadow-mode.md) specify
the machine; [09](09-risk-register.md)–[11](11-roadmap.md) manage the journey.
The founding analysis is [FEASIBILITY-2026-07](../FEASIBILITY-2026-07.md).
