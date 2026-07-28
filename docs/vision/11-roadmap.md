# 11 · Roadmap

Sequenced plan from today's codebase to a piloted product. This document sits
*above* [ROADMAP-v2](../ROADMAP-v2.md) (which remains the perception-layer
implementation plan — its Learn/Accompany model and quality work carry
forward) and encodes one meta-rule: **every phase ends with a measured gate,
not a vibe** ([08 · Benchmark](08-benchmark-and-shadow-mode.md)).

---

## Phase A · The 90 days (three parallel tracks)

### Track 1 — The missing half: performance engine MVP

*Why first: `MusicalParameters` has no consumer; "good enough perception" is
undefinable until something plays.*

1. MIDI scheduler + modeled-piano rendering; `start/stop/nudge_tempo` verbs;
   3 hand-prepared pieces (a waltz, a polka, an adagio).
2. Metadata schema + plan compiler ([06](06-performance-engine.md) §6.2–6.3);
   length realization (repeats/cuts, endings, button); library to ~30 pieces.
3. Wire `analyze() → compile() → play()` end-to-end on recorded clips.

**Gate A1 — the magic-moment demo:** one cooperative teacher, live room,
ladder rung 2 (verbal cue): teacher marks naturally → correct meter/tempo/
length → music on "music please" → stops on speech. Recorded on video.

### Track 2 — Benchmark + shadow recorder

*Why now: every subsequent claim needs the number.*

The harness comes first and does not wait for the corpus — the eval ladder
([08](08-benchmark-and-shadow-mode.md) §8.4, specified in
[ADR-009](../adr/009-evaluation-harness.md)) climbs in order:

1. **Harness + tiers 0–1 (weeks 1–2).** Case format and scorer library; the
   `PerceptionBundle` seam in `analyze()` so frozen Whisper/Gemini/pose traces
   replay offline; a synthetic suite across the style × meter ×
   interleaved-explanation matrix; ADR-006/007's hand-run tables ported as the
   first trace cases. Tiers 0–1 gate every PR from then on.
2. **Tier 2 + capture rig.** Live-vs-trace drift job folding in
   `scripts/compare_models.py`; one-button recorder (§8.1); record 6–10 real
   classes across ≥3 teachers, harvesting free `performance_bpm` labels from
   accompanied classes as they arrive.
3. **Tier 3.** Annotation tooling + schema (§8.2); annotate; dashboard with the
   §8.3 metrics and the abstention/calibration columns, zero-shot first.
4. **Tier 4.** Shadow-mode reporter (the "would have played" report, §8.5).

**Gate A2:** dashboard live; baseline (zero-shot) numbers published for all
metrics — *whatever they are*. The honest baseline is the deliverable. A
synthetic-plus-trace baseline lands in week two; the corpus columns follow it
rather than blocking it.

### Track 3 — Meter and the priors (the crux, attacked with the missing signal)

1. **Accent-periodicity module** (`precision/`): raw-audio onsets + per-onset
   intensity/duration/F0 salience → autocorrelation meter vote
   ([05](05-perception-strategy.md) §5.3). Pure math, hardcoded-data tests.
2. **Exercise-prior table + class-state tracker** (§5.4–5.5), fused into an
   extended `interpret_meter()`; posterior + decision thresholds.
3. **Calibration v0:** learn `tempo_offset`, meter habits, and the class
   template from Track 2's corpus; report calibrated-vs-zero-shot.

**Gate A3:** calibrated meter accuracy ≥ 85% and tempo-within-8% ≥ 90% on the
benchmark (v1 gates are 90/95 — Phase B closes the rest), with the
zero-shot→calibrated delta quantifying Reframe 3.

## Phase B · Live rooms (the following ~2 quarters)

1. **Reflex layer productionization:** cue detector (per-teacher signature),
   command grammar, stop arbiter with the counting-with-music grace rule
   ([07](07-interaction-design.md) §7.4); streaming (eager) analysis replacing
   batch in live mode; retire the wake word (ADR-008 superseded).
2. **Pilot cohort:** 3–5 teachers through the onboarding choreography
   (shadow → rung 3 → 2 → 1) with go-live gates enforced per teacher
   ([08](08-benchmark-and-shadow-mode.md) §8.6).
3. **Library to 100+ pieces;** humanization templates tuned on pilot
   feedback; selection memory (no-repeat, preferences).
4. **Syllabus mode** ([10 · Pivots](10-pivots.md) P1): set-exercise packs with
   certainty-priors; ship to exam-prep users as the first public rung.

**Gate B:** one full calibrated class run live at rung 1 with zero false
starts and **live** exercise-success ≥ 85%; ≥1 pilot teacher *chooses* the
system over their old workflow unprompted for a full week. (Promotion to
rung 1 itself still requires the *shadow* bar of
[08](08-benchmark-and-shadow-mode.md) §8.6 — ≥ 90% shadow exercise-success;
the live gate is deliberately looser because live rooms are harder than
replayed ones.)

## Phase C · Product (beyond)

- Fleet: multi-teacher studios, profile portability, the state-light hardware
  posture; pricing against the manual apps.
- Corpus to 20+ classes; publish the benchmark + methods if P2 posture is
  taken; evaluate fine-tuned local perception (the DISPOSABLE layer's final
  swap).
- Musicality: variation engine, style capture; center-work steady-mode polish.
- Re-scan landscape quarterly ([03](03-landscape.md) §3.6); revisit
  generation-behind-validators when a model can honor bar contracts.

## Explicitly deferred (with their tripwires elsewhere)

| Deferred | Where its return is defined |
|---|---|
| Dancer-following / center adaptivity | [09](09-risk-register.md) R6 — after v1 gates |
| Generative music | [03](03-landscape.md) §3.6 watch-trigger |
| Far-field no-mic operation | after R1 is measured & beaten with the mic |
| Custom hardware / Disklavier embodiment | flagship moment, not v1 |
| Any GUI beyond the state light | never, if the ladder holds |

## The one-sentence plan

Build the half that plays, measure the half that listens, attack meter with
accent priors and per-teacher memory, and let shadow mode carry the system
into live rooms one earned rung at a time.
