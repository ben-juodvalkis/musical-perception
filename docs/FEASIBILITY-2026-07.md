# Feasibility Study: The Fully Automated Ballet Accompanist

**Date:** 2026-07-17
**Question:** Is an automated ballet accompanist that works seamlessly from
camera and audio — no manual interaction from the teacher, just like a human
pianist — actually feasible and reasonable given today's technology?

---

## Verdict

**Yes — but only after three reframes.** Framed as "zero-shot, zero-setup,
indistinguishable from a human pianist for any teacher on day one," this is a
multi-year research program with genuinely unsolved sub-problems. Framed
correctly, it is a staged engineering project that one person can carry to a
working prototype with today's models, with exactly one load-bearing research
risk (robust marking comprehension across idiosyncratic teachers) — and that
risk can be measured, bounded, and fenced with fallbacks rather than solved
outright.

The three reframes:

1. **"No manual interaction" should mean "nothing a human accompanist wouldn't
   need."** Human pianists are talked to constantly — "waltz, please,"
   "slower," "from the last eight" — and they ask one-word questions back.
   Voice is not a crutch; it *is* the interface. Zero-interaction-ever is a bar
   even humans don't meet, and it makes the perception problem artificially
   impossible.
2. **The competitor is the playlist, not the pianist.** The realistic benchmark
   is the teacher DJ-ing fixed-tempo, fixed-length recordings from a phone —
   which is how the overwhelming majority of classes in the world run. That bar
   is beatable with today's technology. The Turing-level pianist stays the
   north star, not the MVP gate.
3. **Class is a script and teachers are habits — exploit both ruthlessly.**
   Ballet class order is near-canonical, each exercise carries strong
   meter/tempo/character priors, and an individual teacher reuses structure,
   idiom, and tempo preferences for weeks. Perception should disambiguate among
   a handful of prior-weighted hypotheses with per-teacher memory — not
   classify arbitrary input from scratch.

---

## 1. What the job actually is

Decomposing what a human class accompanist does, with a feasibility grade for
each piece as of mid-2026:

| # | Sub-task | What it requires | Feasibility today |
|---|----------|------------------|-------------------|
| 1 | **Track class protocol** — know pliés come first, grand battement last, barre then center | A state machine + priors; no AI | ✅ Trivial, and currently unexploited |
| 2 | **Comprehend the marking** — tempo, meter, phrase length, character, exercise from voice+body | The Learn phase this repo builds | ⚠️ Partially proven; the one real research risk |
| 3 | **Choose repertoire** — right genre, meter, tempo range, phrase structure, character; don't repeat | Tagged music library + retrieval | ✅ Straightforward engineering |
| 4 | **Perform** — intro ("aaand—") at exact tempo, square phrases, cadence at ends of 8s, clean button | Symbolic playback engine | ✅ Solved technology (MIDI + modeled piano) |
| 5 | **React in real time** — stop instantly when teacher talks, "again"/"other side," tempo nudges | Local reflex layer, <300 ms | ✅ Feasible if local; disaster if cloud-dependent |
| 6 | **Follow dancers** — breathe with an adagio, catch a grand allegro's landing | Score-following research (Music Plus One / Antescofo lineage) | ❌ Research; also *not needed at barre*, where holding tempo is the job |
| 7 | **Be part of the room** — rapport, taste, humor | — | ❌ Not automatable; also not the bar |

The critical observation: **rows 1, 3, 4, 5 are engineering, and none of them
exist in the system yet** (this repo is perception-only by design). Row 2 is
where all the research risk concentrates. Row 6 can be deferred entirely by
scoping to barre-first — which is also where recorded music hurts teachers
most, because barre is relentless (20+ exercises back to back, each needing its
own tempo and length).

---

## 2. What this codebase already proves — and what its own data says

Nine months of ADRs give unusually honest evidence. Summarizing the record:

**Proven:**
- **Tempo from clean counting works.** Onset-based detection hit 115.1 BPM
  against ~117 ground truth (1.6% error) where the classification-based path
  failed (ADR-006). The precision layer's input-agnostic design (KEEP/DISPOSABLE)
  has already paid off twice — exactly as intended.
- **Gemini is a workable qualitative analyst.** Exercise ID, per-phrase quality
  as calibrated floats, structure — for pennies per exercise. The
  Gemini-idealizes / MediaPipe-measures split (ROADMAP-v2) is a real insight.
- **The two-phase Learn/Accompany runtime model is correct.** It mirrors how
  accompanists actually work and cleanly separates latency domains.
- **The trigger-pipeline instinct is right** (ADR-008): always-on cheap local
  gate, escalate to expensive analysis only when warranted.

**The honest problems, from the repo's own results:**
- **Meter is ~50/50.** ADR-007's table: 2 of 4 real files got correct
  meter/subdivision. Wrong meter is also the *loudest possible failure* — a
  march played for a waltz combination is instantly disqualifying, where ±5%
  tempo error goes unnoticed. Meter is the crux.
- **Mixed speech breaks everything upstream.** When explanation and counting
  interleave (i.e., every real classroom), the onset detector picks up
  conversational speech and the cross-signal checks stop working (ADR-007,
  "noisy upstream signals").
- **The current mitigation is behavior change** — "teachers should count with
  numbers, separate explanation from counting, be rhythmically precise"
  (ADR-007). This is exactly the compromise the seamless-accompanist goal
  forbids. A system that requires the teacher to perform for it has already
  failed the brief.
- **ASR is still inside every loop.** Even the "classification-free" onset
  path runs on *Whisper word timestamps*, not audio onsets. Whisper is the
  single point of failure for a signal that isn't fundamentally lexical:
  marking is quasi-musical vocalization ("da-da-POM," step names, French,
  humming) — a percussion track wearing words, not speech that happens to be
  rhythmic.
- **"Hey Maestro" is below human parity.** Nobody says "hey pianist" before
  marking. The wake word was the right pragmatic gate for a prototype, but the
  natural cue already exists in the room: the marking itself, and the
  "aaand—" + inhale + gesture that every teacher produces before dancers move.

None of this says infeasible. It says: the remaining distance is concentrated
in (a) segmentation of marking from talk, (b) meter, (c) teacher idiosyncrasy —
and the codebase has correctly identified accent-pattern analysis as the
missing signal for meter but hasn't built it yet.

---

## 3. The three reframes, in full

### Reframe 1 — Voice is the interface, not a failure of automation

Set the interaction bar at **human parity**: anything a teacher naturally says
to a human accompanist is legitimate input, and one-word questions back are
legitimate output. Concretely:

- "Slower." / "A bit brighter." / "Again." / "Other side." / "Thank you" (= stop) —
  all natural, all trivially spottable by a local keyword model.
- When genuinely uncertain between 3/4 and 4/4, the system may ask **one word**:
  "Waltz?" Human accompanists ask exactly this, constantly. Cost: one second.
  Benefit: converts the hardest inference into a confirmation.
- The wake word disappears; **cue detection replaces it**. The "aaand—"
  preparation cue is prosodically distinctive (elongated vowel, pitch rise,
  inhale, arm gesture on camera) and it is *already the protocol* every dancer
  responds to. The system joins the room's existing protocol instead of
  imposing one.

This reframe dissolves the tyranny of zero-shot perfection. The system needs to
be a *good listener that takes correction graccefully* — not an oracle.

### Reframe 2 — Beat the playlist, not the pianist

Live accompanists play for a small percentage of the world's classes (major
schools, companies, conservatoires). Everyone else runs class on recorded
albums and a phone: fixed tempi, fixed 32/64-count tracks, the teacher walking
to the speaker between every exercise — and, most tellingly, **teachers design
combinations to fit the recordings they own**. The recording constrains the
pedagogy, not the reverse.

So the honest product claim for v1 is not "a virtual pianist." It is: **the
studio speaker that plays the right thing by itself** — any length, any tempo,
starts on your cue, stops when you talk, never needs your hands. That is
strictly better than the status quo for the playlist-majority, and it is
buildable now. Musicality beyond "excellent rehearsal pianist" (true rubato,
the lift before a grand allegro landing) is deferred without guilt, because the
playlist never had it either.

### Reframe 3 — Priors and memory do the heavy lifting

Three layers of structure that current zero-shot perception ignores:

1. **The syllabus prior.** Barre order is near-canonical (pliés → tendus →
   dégagés → rond de jambe → fondus → frappés → petit battement → adagio →
   grand battement; then center). A ~20-row table maps each exercise to strong
   priors on meter, tempo range, character, and typical phrase lengths. Written
   once, it constrains every downstream inference.
2. **Bayesian class-state tracking.** At any moment the system maintains a
   distribution over "where we are in class." Perception doesn't answer "what
   exercise is this?" from scratch — it answers "given that fondus usually come
   after rond de jambe, and I heard 'fondu,' and the marking is legato at ~72
   BPM, which of 3 likely hypotheses is this?" Same models, different question,
   dramatically better odds.
3. **Per-teacher memory.** A teacher's counting idiom, marking-tempo-to-
   performance-tempo offset (teachers mark approximately; humans compensate
   from experience), meter preferences per exercise, and this term's actual
   combinations are all stable for weeks. Two or three shadowed classes yield a
   calibration profile that converts "unsolvable zero-shot" into "easy
   few-shot." This has perfect human parity: a new human accompanist also
   needs a few classes to sync with a teacher. Nobody calls that setup.

---

## 4. Architecture consequences

### 4.1 Two-tier runtime: reflex and deliberation

The Learn/Accompany split (ROADMAP-v2) is right; sharpen it into two latency
domains:

- **Reflex layer — local, always-on, <300 ms, no cloud dependency ever:**
  VAD + teacher speaker-ID, keyword spotting (stop/again/other-side/
  faster/slower), "aaand—" cue detection, the beat clock during playback,
  instant stop when the teacher speaks over the music.
- **Deliberation layer — cloud or local LLM, seconds:** streaming marking
  analysis. Hypotheses form *while the teacher marks* (incremental, not batch);
  the cue commits the current best hypothesis. The 3–15 s a teacher naturally
  takes between finishing the marking and cueing ("walk to your spot… ready?
  aaand—") absorbs API latency — but only if analysis is eager, not
  triggered-after-the-fact.

### 4.2 Asymmetric error policy (the social contract)

The two failure modes are wildly asymmetric:

- **False start** (music when the teacher is still talking): mortifying,
  trust-destroying, remembered forever.
- **Missed start** (silence when music was expected): costs three seconds and
  has a *natural, graceful* recovery — "Music, please" — which is exactly what
  teachers say to distracted human pianists.

Therefore: bias hard toward silence, make verbal recovery first-class, and
count false starts as the primary reliability metric (target: zero per class).
This single policy decision is what makes full autonomy socially viable while
the perception layer matures.

### 4.3 Prosody-first perception (demote ASR)

Move the primary rhythm channel from Whisper word timestamps to the raw audio:
energy/spectral-flux onsets for pulse, and intensity + F0 + duration
periodicity for **accent patterns** — which is precisely the meter signal
ADR-006 named as missing ("ONE-two-three ONE-two-three" is an autocorrelation
peak at lag 3 in the accent series, no words required). Benefits:

- Works on "da-da-POM," hummed marking, step names, and any language — ballet
  is global and French-coded; an ASR-first system is an English-first system.
- Removes Whisper's segmentation failures from the critical path. ASR remains
  for what is genuinely lexical: commands, numbers when present, exercise
  names as *evidence* for the state tracker.
- The counting-signature work (Praat, prosodic weight) already points here;
  this is a promotion of an existing direction, not a new bet.

### 4.4 Playback: symbolic MIDI, not recordings, not generative audio

The accompanist's musical obligations — exact counts, square phrases, arbitrary
tempo, instant tempo change, vamp-till-ready, first/second endings, a two-bar
button, an instant clean stop — are trivial in symbolic MIDI and unavailable
in the alternatives. Time-stretched recordings artifact badly beyond ±10% and
can't change length; generative-audio models (Suno/Udio/Lyria class) cannot
*guarantee* "exactly 32 counts of 3/4 at 138 with a cadence on count 32," and
an accompanist that is right only most of the time about structure is unusable.

A working accompanist needs roughly what a human carries: an active repertoire
of ~100–200 pieces, tagged by meter, tempo range, character, genre, and
phrase structure. Public-domain ballet-class piano literature is abundant;
modern modeled/sampled pianos (e.g., physical-modeling engines) render with
real-time tempo and dynamics control at fully acceptable quality. Constrained
symbolic generation can add variety later behind the same interface. This is
the **missing half of the system** — `MusicalParameters` is a contract with no
consumer yet, and several key perception questions (how good is good enough?)
are unanswerable until something plays.

### 4.5 Accept the teacher mic

Far-field audio in a reverberant studio with 20 dancers is a research problem;
a $150 wireless lapel/headset mic on the teacher — already common in large
studios — makes speaker separation and VAD near-trivial and upgrades every
downstream component. This is the single cheapest reliability multiplier
available and a completely reasonable constraint for v1. (Human parity again:
the human accompanist also positions themselves where they can hear the
teacher.)

### 4.6 Shadow mode (the trust path)

Before the system ever makes a sound in a real class, it rides along silently:
teacher mic + wide camera in, and after class it produces a report — "for the
frappé at 10:14 I would have started *here*, a 2/4 polka at 104." Reviewed
against what the teacher actually used (or what a human accompanist actually
played), this yields:

- a measured **hit-rate** instead of a feasibility opinion,
- the per-teacher calibration profile,
- the benchmark corpus (see §6),
- and teacher trust before the first live note.

Go live per-teacher when shadow metrics clear a bar (suggested: ≥90% correct
meter, ≥95% tempo within ±8%, phrase length exact, zero would-be false
starts). Shadow mode converts an adoption gamble into a measurement.

### 4.7 The graceful-degradation ladder

Ship three rungs and let each teacher's reliability data choose the rung:

1. **Full-auto** — cue-detected start, inferred everything.
2. **Voice-assist** — system asks one-word questions when uncertain; teacher
   uses natural verbal commands.
3. **Glance/tap backstop** — a watch tap or tiny remote as physical start/stop
   of last resort (one bit of interaction, everything else automatic).

The ladder removes the all-or-nothing bet on the hardest AI problem, and rung 3
exists precisely so rungs 1–2 can be attempted without fear in a live room.

---

## 5. The honest risk register

What remains genuinely hard after all reframes — with mitigations, not
hand-waving:

| Risk | Severity | Mitigation | Residual |
|------|----------|------------|----------|
| Marking segmentation in live rooms (talk vs. count interleaved) | High | Teacher mic; cue detection; eager-hypothesis + commit-on-cue; silence-biased policy | Real. The #1 thing shadow mode must measure |
| Meter from ambiguous marking | High | Accent periodicity (§4.3) + exercise priors + one-word confirm | Medium — even humans ask "in three?" |
| Teacher idiosyncrasy, zero-shot | High | Don't do zero-shot: per-teacher calibration from 2–3 shadowed classes | Low for calibrated teachers; new-teacher onboarding is a feature, not a bug |
| Marking tempo ≠ intended tempo | Medium | Learned per-teacher offset; live verbal nudges | Low |
| Musicality ceiling (rubato, breath) | Medium | Accept "excellent rehearsal pianist"; the playlist bar never had it | Accepted, revisit later |
| Center-work following (adagio breathing, grand allegro) | Medium | Barre-first scope; center = steady-tempo mode initially | Deferred, honestly |
| Trust destruction from one bad failure | High | Shadow mode; false-start-zero policy; ladder rung 3 | Managed by process, not model quality |

And the non-risks worth naming: **compute and cost are solved** (all reflex
components run on a laptop-class CPU; multimodal analysis is pennies per
exercise, i.e. under a dollar per class), and **latency budgets fit** the
natural rhythm of a class as long as the reflex/deliberate split is respected.

---

## 6. The benchmark is the project

The highest-leverage artifact buildable right now is not a better model wrapper
— it is a **recorded-class benchmark**: 10–20 full real classes (teacher-mic
audio + wide video), annotated with ground truth per exercise (start cue time,
meter, performance tempo, phrase structure, character, what music was actually
used). The existing batch pipeline is already the replayer. Every architectural
claim in this document then becomes a number on a dashboard, per component and
end-to-end, and "is this feasible?" stops being a philosophy question.

The same corpus is simultaneously: the per-teacher calibration source, the
shadow-mode scorer, the fine-tuning set that eventually beats zero-shot
prompting, and — if research ambitions exist — a publishable contribution,
because **no marking-comprehension dataset exists anywhere**. The data moat is
the defensible asset; models will keep being swapped (the DISPOSABLE label has
been right three times already).

---

## 7. Pivot menu

Ranked options, with a recommendation:

- **P0 — Same vision, resequenced (recommended).** Hands-free class music for
  the playlist-majority: barre-first, teacher-mic, MIDI library, reflex layer,
  shadow-mode onboarding, degradation ladder. This *is* the original dream,
  entered through its feasible face. One person + this codebase can reach a
  magic-moment demo (teacher marks → right music starts on the cue) with one
  known teacher in months.
- **P1 — Syllabus mode first.** RAD/Vaganova/Cecchetti exam classes have
  *fixed* exercises with prescribed music character and known counts.
  Perception collapses to "which set exercise, which tempo variant, when to
  start" — an order of magnitude easier than free-work comprehension, with a
  real paying market (exam-prep studios, home practice). Shippable early;
  funds and feeds P0. Compatible with P0, not a replacement.
- **P2 — The dataset as the product/contribution.** If the deeper motivation is
  research: instrument accompanied classes, build the marking→music paired
  corpus and benchmark (§6), fine-tune small local models against it. This is
  the piece nobody else has.
- **P3 — Accompanist copilot (fallback posture).** If live-room autonomy stalls:
  the same perception stack drives a display/auto-play assistant with a human
  (teacher or pianist) holding veto. Weaker vision, real utility, keeps the
  data flowing.
- **P4 — Adjacent domains later.** Fitness/barre-fitness, figure skating,
  gymnastics floor: rhythmic-leader-plus-music domains with lower musical
  stakes and bigger markets, but commodity music cultures. Ballet remains the
  differentiated beachhead; these are expansions, not pivots.

The strong recommendation is **P0 with P1 inside it** (syllabus mode as the
first shipped rung) and **§6 as the immediate work**, because it de-risks
everything else.

---

## 8. What I would do next (90 days, three tracks)

1. **Build the missing half: a minimal performance engine.** ~30 tagged MIDI
   pieces, tempo scaling, vamp-till-cue, first/second endings, button, instant
   stop; modeled-piano rendering. Wire `MusicalParameters` → performance plan.
   Without a consumer, the perception layer cannot even define "good enough."
   Target: the magic-moment demo with one cooperative teacher.
2. **Stand up the benchmark + shadow recorder** (§6). Record real classes,
   annotate, replay the existing pipeline, publish the hit-rate dashboard.
   This turns the feasibility question into a weekly-improving number.
3. **Attack meter with the missing signal.** Accent-periodicity module
   (intensity/F0/duration autocorrelation at beat lags — pure precision math,
   KEEP), the exercise-prior table, and the class-state tracker. Measure the
   meter hit-rate jump on the benchmark from §6's corpus.

Explicitly deferred: dancer-following, generative music, far-field
no-mic operation, center-work adaptivity, any UI beyond the ladder.

---

## 9. Bottom line

A camera-and-mic accompanist that a teacher never touches is **feasible to
prototype now and reasonable as a project** — provided "seamless" is defined as
*human parity* (voice included) rather than *telepathy*, the first competitor
is the playlist rather than the pianist, and the system is allowed to know
what a ballet class is and who its teacher is. The perception bet this repo has
been making is the right bet, its own ADRs have correctly located the hard
parts, and the two structural gaps — no playback layer, ASR-first rhythm — are
both addressable with known technology. The one honest research risk is robust
marking comprehension in live rooms, and the path through it is not a better
model but a better process: priors, per-teacher calibration, silence-biased
turn-taking, and a shadow-mode benchmark that replaces belief with measurement.
