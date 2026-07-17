# 07 · Interaction Design

The social contract between the system and the room. The design principle
([02 · Reframes](02-reframes.md), Reframe 1): **join the room's existing
protocol at human parity — never impose a new one.** This document specifies
the cue, the voice grammar, the error policy, the question budget, the
degradation ladder, and privacy posture.

---

## 7.1 The room's existing protocol (what we join)

A ballet class already runs on a precise, universal interaction protocol:

1. **The marking** — teacher demonstrates + vocalizes; dancers *and pianist*
   read it. Nobody announces "I will now mark."
2. **The cue** — "aaand—" (elongated vowel, rising pitch, inhale, often an arm
   lift). Dancers prepare; the pianist starts the intro so count 1 lands on
   the downbeat after the cue.
3. **Corrections** — the teacher speaks; music stops instantly; class
   listens; a nod or "again" resumes.
4. **Flow words** — "again," "other side," "one more time," "from the top,"
   "last eight."
5. **The close** — "thank you" ends an exercise (this is real accompanist
   convention: *thank you = stop*).

Every trigger the system needs already exists in this protocol. Design task:
detect it, never extend it.

## 7.2 Cue detection (replaces the wake word)

The ADR-008 wake word ("Hey Maestro") is retired — below human parity. The
cue detector fuses:

- **Audio:** the "aaand—" prosodic template — elongated vowel (typ. 400–900 ms),
  rising F0 contour, followed by inhale; matched per-teacher via the
  `cue_signature` in the [calibration profile](05-perception-strategy.md) §5.6.
- **Vision:** preparatory posture/arm-lift from pose; teacher facing the room.
- **Context:** only ARMED state accepts a cue ([04](04-system-architecture.md)
  §4.4) — a cue with no marked exercise is ignored, killing a whole class of
  false positives.

Commit semantics: cue confirmed → `start(at_cue_time)` so that **count 1 lands
one beat after the cue at performance tempo** (the intro is elastic; the
downbeat is sacred). Verbal fallback is always live: "music, please" = cue.

## 7.3 The voice grammar

Small, closed, locally spotted (<300 ms), personalized by `command_vocab`:

| Intent | Default phrases | Engine verb ([06](06-performance-engine.md) §6.4) |
|---|---|---|
| Start | "music please", "let's go", cue | `start` |
| Stop | *any teacher speech over music* (see §7.4); "thank you", "hold on" | `stop` |
| Again same side | "again", "one more time" | `again` |
| Other side | "other side", "second side", "left" | `other_side` |
| Faster / slower | "(a bit / much) faster/slower" | `nudge_tempo(±2% / ±6%)` |
| Bigger / softer | "more energy", "gently" | `set_energy(±)` |
| Wrap up | "finish", "take it to the end" | `finish_from_here` |
| Override piece | "waltz", "something in three", "march" | recompile with constraint |

Rules: the grammar is **suggestions-closed** — unknown speech is never
interpreted as a command; it is just speech (which, over music, means stop).
Multilingual variants ship for the ballet lingua franca (English, French,
Russian) and extend via the profile.

## 7.4 The asymmetric error policy

The two failure modes are not symmetric, and the entire policy follows from
that:

| Failure | Cost | Recovery |
|---|---|---|
| **False start** (music while teacher talks) | Trust-destroying; remembered forever; the story teachers tell about the robot | None good |
| **Missed start** (silence when expected) | ~3 seconds; mildly awkward | Natural: "music, please" — exactly what teachers say to a distracted human pianist |

Policy consequences:

1. Every arbitration threshold biases toward silence. Target: **zero false
   starts per class**, tracked as the headline metric
   ([08](08-benchmark-and-shadow-mode.md)).
2. **Teacher speech over music always stops the music** — with one grace rule:
   counting *with* the music ("2… 3… stay!") and brief encouragements
   ("good!") are classified rhythm-aligned/short and do not stop playback;
   sustained off-beat speech does. When wrong, wrongness must fall toward
   stopping. (The general form of this problem — premature interruption
   during pauses — is unsolved in full-duplex research; the bias *sidesteps*
   it rather than pretending to solve it.)
3. Recovery phrases are first-class citizens, tested and fast, because they
   are the designed failure path — not an embarrassment.

## 7.5 The question budget

The system may speak. Sparingly:

- **Form:** one word, two at most, local TTS, at ARMED only — "Waltz?",
  "In three?", "Sixty-four?" Never during marking, never over music.
- **Budget:** at most ~1 question per exercise, and only inside the posterior
  band defined in [05](05-perception-strategy.md) §5.3 (0.55–0.80). Above:
  silent commit. Below: stay silent, wait for verbal direction.
- **Learning:** every answered question updates the calibration profile, so
  the same question is not asked twice in a week. A system that asks the
  same thing daily has failed; a system that asks like a good new pianist in
  week one and rarely by week three is *exactly on target*.

## 7.6 The degradation ladder

Three rungs, chosen **per teacher, per confidence, automatically** — with
manual pinning available:

| Rung | Behavior | When |
|---|---|---|
| 1 · Full-auto | Cue-detected start; inferred everything; questions within budget | Calibrated teacher, posterior healthy |
| 2 · Voice-assist | System never self-starts; teacher verbally cues ("music please"); system may ask its one word | New teacher mid-calibration; noisy room; confidence dip |
| 3 · Tap backstop | Watch tap / tiny remote = start-stop; all analysis still automatic (tempo, piece, length are still inferred) | Trust-building week one; mic failure; teacher preference |

The ladder is the honest answer to "what if the hard AI problem isn't solved
yet for this teacher": the product still deletes the form at rung 3 — the tap
replaces twenty touchscreen interactions with one gesture — and every class at
any rung generates the calibration data that raises the rung.

## 7.7 Onboarding choreography (per teacher)

1. **Class 0–2 · Shadow** ([08](08-benchmark-and-shadow-mode.md)): rig
   present, system silent, old workflow unchanged. Post-class report builds
   trust and the profile.
2. **Class 3 · Rung 3:** system plays, teacher taps. Zero perception risk
   exposed; music quality and stop-reflex earn their keep.
3. **Class 4+ · Rung 2 → 1** as gates pass ([08](08-benchmark-and-shadow-mode.md)
   §8.6). Any bad week drops a rung without ceremony.

The teacher never configures anything. The system's understanding is visible
only in its behavior — like a pianist's.

## 7.8 Privacy and consent posture

Always-on classroom capture has deployed precedent (Merlyn Mind, TeachFX —
[03 · Landscape](03-landscape.md) §3.3), and this system observes the
*teacher*, not students:

- Camera framed on the teaching space; pose landmarks (not video) retained by
  default; students out of analytic scope.
- Audio: teacher-mic stream is the analyzed signal; room mic is ambient
  reference, discarded after feature extraction by default.
- Recordings retained **only** in shadow/benchmark mode, with explicit
  teacher (and studio) consent, for calibration and evaluation
  ([08](08-benchmark-and-shadow-mode.md) §8.7).
- On-device by default; cloud calls carry audio segments of the *teacher's
  marking*, not continuous surveillance. A visible state indicator (small
  light: attending / armed / performing) keeps the room informed without a
  screen.
