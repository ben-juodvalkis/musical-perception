# 02 · The Three Reframes

The fully automated accompanist is infeasible under its naive framing and
feasible under a corrected one. This document records the three framing moves,
because they are load-bearing: every architectural decision downstream
([04](04-system-architecture.md)–[07](07-interaction-design.md)) traces back to
one of them. When a future decision feels hard, check whether it is being made
under the naive framing.

---

## Reframe 1 — Voice is the interface, not a failure of automation

**Naive framing:** "No manual interaction" means the system must infer
everything from observation alone; any teacher input is a defeat.

**Why it fails:** It sets a bar no human accompanist meets. Teachers talk to
their pianists constantly — "waltz, please," "slower," "just the last eight" —
and pianists ask one-word questions back ("In three?"). Requiring telepathy
concentrates impossible demands on the perception layer: every ambiguity must
be resolved from a signal that is sometimes genuinely ambiguous (a marking that
fits both 3/4 and 4/4 fits both — even a human would ask).

**The reframe:** Set the interaction bar at **human parity**. Anything a
teacher naturally says to a human accompanist is legitimate input. One short
question per genuine ambiguity is legitimate output. "Manual interaction"
means *touchscreens, phones, and forms* — not speech, which is how the room
already works.

**What it unlocks:**

- The wake word ("Hey Maestro," ADR-008) disappears — it was *below* human
  parity; nobody says "hey pianist." Its replacement is **cue detection**: the
  "aaand—" + inhale + gesture that every teacher already produces is the
  natural trigger, and it is prosodically distinctive.
- Perception no longer needs zero-shot perfection. It needs to be a **good
  listener that takes correction gracefully** — and correction arrives through
  the same channel a human uses.
- The hardest inference (meter, see [05 · Perception](05-perception-strategy.md))
  gains a graceful floor: when the posterior is genuinely split, ask "Waltz?"
  Cost: one second. Human parity: total.

**Design consequences:** the verbal command grammar and question budget in
[07 · Interaction Design](07-interaction-design.md); the reflex-layer keyword
spotter in [04 · Architecture](04-system-architecture.md).

---

## Reframe 2 — The competitor is the playlist, not the pianist

**Naive framing:** Success means being indistinguishable from a professional
ballet accompanist.

**Why it fails:** It aims the project at the ~5% of rooms that already have the
best solution (a human) and judges v1 by its weakest dimension (musicianship),
while ignoring the actual global baseline. It also postpones shipping until
the last 10% of musicality is solved — the part that may never be.

**The reframe:** The realistic benchmark is **the teacher DJ-ing fixed
recordings from a phone** — which is how the overwhelming majority of classes
run. That baseline has: fixed tempi, fixed 32/64-count track lengths, a walk
to the speaker between every exercise, zero adaptivity, and a subtle
pedagogical tax — **teachers design combinations to fit the recordings they
own**. The music constrains the teaching.

The market has already voted on what it wants and priced it: the manual apps
teachers use (Ballegro, Ballet Class, BalletBox, Cadence, Cadance — see
[03 · Landscape](03-landscape.md)) let the teacher set **tempo and number of
bars per exercise, by hand, between exercises**. That interface *is* our
output schema (`MusicalParameters`), operated manually. The entire product,
stated in three words: **delete the form.**

**What it unlocks:**

- A shippable v1 standard: *excellent rehearsal pianist* — right piece, right
  meter, exact tempo, any length, clean endings, hands-free. Strictly better
  than the playlist on every axis, no rubato required.
- Barre-first scope. Barre is 20+ exercises back to back, each needing its own
  tempo and length — where the DJ tax is highest and where holding steady
  tempo (the thing machines are best at) is the musical job. Center-work
  adaptivity defers cleanly.
- An honest relationship with live accompanists: the system serves rooms that
  have none, rather than under-delivering in rooms that do.

**Design consequences:** repertoire and length-adaptation requirements in
[06 · Performance Engine](06-performance-engine.md); success criteria in
[01 · North Star](01-north-star.md); pivot P1 (syllabus mode) in
[10 · Pivots](10-pivots.md).

---

## Reframe 3 — Class is a script and teachers are habits

**Naive framing:** Perception must classify arbitrary marking from scratch,
for any teacher, on first contact — a zero-shot problem.

**Why it fails:** Zero-shot is exactly where the current pipeline's own data
shows the pain: meter at ~50% (ADR-007), onset detection confused by mixed
explanation-and-counting speech, Gemini classifying step-name counting as
non-rhythmic (ADR-006). Idiosyncrasy is unbounded in general — some teachers
count in numbers, some in step names, some in "da-da-POM," some barely
vocalize. No model, present or near-future, solves the general case.

**The reframe:** The general case never occurs. Three layers of structure
constrain every real classroom moment:

1. **The syllabus prior.** Ballet class order is near-canonical: pliés →
   tendus → dégagés → rond de jambe → fondus → frappés → petit battement →
   adagio → grand battement; then center (adagio → pirouettes → petit allegro
   → grand allegro → révérence). Each slot carries strong priors on meter,
   tempo range, character, and phrase length — writable once as a ~20-row
   table ([05 · Perception](05-perception-strategy.md) contains it).
2. **Class-state tracking.** The system maintains a running distribution over
   "where we are in class." Perception stops answering *"what exercise is
   this?"* and starts answering *"given that fondus usually follow rond de
   jambe, and the marking is legato at ~72 BPM, which of three likely
   hypotheses is this?"* Same models, transformed odds.
3. **Per-teacher memory.** A teacher's counting idiom, marking-tempo-to-
   performance-tempo offset, meter preferences, phrase habits, and this term's
   actual combinations are stable for weeks. Two or three shadowed classes
   yield a calibration profile that converts zero-shot into few-shot. Human
   parity again: a new human accompanist also needs a few classes with a new
   teacher — nobody calls that setup.

**What it unlocks:**

- The single cheapest reliability multiplier available — no new models, no new
  research, just priors and memory the current pipeline ignores.
- A principled home for the ambiguity that remains: the state tracker outputs
  a posterior; the posterior decides whether to commit silently, ask one word,
  or (rung 3) wait for a tap. Uncertainty becomes routing, not failure.
- Overfitting to a teacher stops being a methodological sin and becomes the
  product working as intended.

**Design consequences:** the exercise-prior table, class-state tracker, and
calibration-profile schema in [05 · Perception](05-perception-strategy.md);
shadow-mode calibration in [08 · Benchmark](08-benchmark-and-shadow-mode.md).

---

## The reframes in one line each

1. **Human parity, not telepathy** — speech is the interface the room already
   has.
2. **Delete the form** — beat the playlist and the touchscreen, not the
   pianist.
3. **Priors and memory over zero-shot** — the system is allowed to know what a
   ballet class is and who its teacher is.
