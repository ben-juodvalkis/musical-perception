# The AI Accompanist — Documentation

Documentation site for the **musical-perception** project and the system it
serves: an automated ballet-class accompanist that works from camera and
audio alone — the teacher teaches, the room simply works.

**Status (2026-07):** perception + precision layers exist in this repo
(tempo proven on clean counting; meter identified as the crux); the vision
suite below defines the path to the full system. Founding analysis:
[Feasibility Study 2026-07](FEASIBILITY-2026-07.md) — verdict: *feasible now,
under three reframes.*

---

## The Vision Suite

Read in order for the full argument, or jump by role below.

| # | Document | One line |
|---|---|---|
| 01 | [North Star](vision/01-north-star.md) | The magic moment, what "seamless" means (human parity), product principles, success criteria |
| 02 | [The Three Reframes](vision/02-reframes.md) | Voice is the interface · beat the playlist, delete the form · priors and memory over zero-shot |
| 03 | [Landscape (July 2026)](vision/03-landscape.md) | Empty niche, converged manual apps, component stack status, academic whitespace, watch-triggers |
| 04 | [System Architecture](vision/04-system-architecture.md) | Reflex/deliberation tiers, the exercise lifecycle, latency budgets, where existing code lands |
| 05 | [Perception Strategy](vision/05-perception-strategy.md) | Prosody-first channels, the meter plan, the exercise-prior table, class-state tracker, calibration profile |
| 06 | [Performance Engine](vision/06-performance-engine.md) | The missing half: MIDI library + schema, plan compiler, runtime verbs, rendering |
| 07 | [Interaction Design](vision/07-interaction-design.md) | Cue detection, voice grammar, asymmetric error policy, question budget, the degradation ladder, privacy |
| 08 | [Benchmark & Shadow Mode](vision/08-benchmark-and-shadow-mode.md) | The corpus, metrics and gates, the eval ladder (measurement before the corpus), shadow-before-sound, the data moat |
| 09 | [Risk Register](vision/09-risk-register.md) | Eight named risks with mitigations, residuals, and tripwires; the closed non-risks |
| 10 | [Pivot Menu](vision/10-pivots.md) | P0 hands-free player · P1 syllabus mode · P2 dataset · P3 copilot · P4 adjacent domains |
| 11 | [Roadmap](vision/11-roadmap.md) | 90-day tracks and gates, live-room phase, product phase, the deferred list |
| 12 | [Collaborators & Ecosystem](vision/12-collaborators.md) | Verified people and institutions bridging ballet and technology, with suggested first asks |
| 13 | [Corpus & Capture](vision/13-corpus-and-capture.md) | The data program: three sources (rig / clean / chaotic), the capture penalty, labeling in the room, tags, splits, sizing |

**By role:**

- *Builder:* 04 → 05 → 06 → 07 → 11
- *Strategist:* 01 → 02 → 10 → 11 → 12
- *Skeptic:* [Feasibility Study](FEASIBILITY-2026-07.md) → 03 → 09 → 08 → 13 →
  [ADR-009](adr/009-evaluation-harness.md)

## Engineering Record

- [Rig Capture Checklist](evals/capture-checklist.md) — the 24-clip recording program that grows the benchmark
- [Eval Baseline](evals/baseline.md) — generated tier 0–1 numbers (`python -m musical_perception.evals run` → `bless`); the honest Gate A2 table
- [Voice-as-Drum Literature Review](research/voice-as-drum-review.md) — what the beat-tracking / speech-rhythm / evaluation literature already knows (2026-08, companion to [ADR-016](adr/016-rhythm-core-reset.md))
- [Agent Charter](research/agent-charter.md) · [Research Log](research/RESEARCH-LOG.md) · [Agent Environment](research/agent-environment.md) — the autonomous research loop: goal ladder, rules of engagement, ledger, and cloud setup for the ADR-016 reset

- [Feasibility Study 2026-07](FEASIBILITY-2026-07.md) — the founding analysis
  (question → verdict → evidence)
- [ROADMAP](ROADMAP.md) — original perception roadmap (Phases 1–2 historical
  record)
- [ROADMAP-v2](ROADMAP-v2.md) — perception implementation plan
  (Learn/Accompany model, quality model); still current for the perception
  layer, now nested under [Vision 11 · Roadmap](vision/11-roadmap.md)

### Architecture Decision Records

| ADR | Decision |
|---|---|
| [001](adr/001-upgrade-audio-perception.md) | Upgrade audio perception (WhisperX, WhiStress, Praat) |
| [002](adr/002-gemini-integration.md) | Gemini multimodal integration |
| [003](adr/003-typed-quality-model.md) | Typed quality model |
| [004](adr/004-quality-model-redesign.md) | Quality model redesign |
| [005](adr/005-remove-scaffolding.md) | Remove scaffolding layer |
| [006](adr/006-onset-tempo-and-normalization.md) | Onset tempo detection + BPM normalization |
| [007](adr/007-coherent-metric-interpretation.md) | Coherent metric interpretation (BPM+meter+subdivision) |
| [008](adr/008-analysis-trigger-pipeline.md) | Analysis trigger pipeline (wake word — *superseded by cue detection, see [Vision 07](vision/07-interaction-design.md) §7.2*) |
| [009](adr/009-evaluation-harness.md) | Evaluation harness — the four-tier eval ladder that implements [Vision 08](vision/08-benchmark-and-shadow-mode.md) |
| [010](adr/010-transcript-authority.md) | Transcript authority + deterministic classification (index-keyed merge) |
| [011](adr/011-phrase-structure-definition.md) | Phrase-structure definition — the first eval-gated change (live-check 3/3) |
| [012](adr/012-counts-from-evidence-fusion.md) | Counts from evidence fusion — `precision/structure.py` owns counts, abstains on ties |
| [013](adr/013-tempo-arbitration.md) | Band-aware tempo arbitration — beat-level markers outvote off-level onsets |
| [014](adr/014-tempo-metric-level-ambiguity.md) | Report the tempo metric-level family instead of collapsing it — `alternates` + non-gating `truth_in_family` |
| [015](adr/015-onset-measurement-robustness.md) | Onset measurement robustness — grid-fit IOIs; kill criterion tripped, owner override on the record; typed gates for future measurement changes |
| [016](adr/016-rhythm-core-reset.md) | Rhythm-core reset — the n=30 retrospective and the acoustic-first redesign; research posture (accuracy over cost), six commitments, sequenced kill-tests |

## Serving this site

```bash
pip install mkdocs-material
mkdocs serve        # from the repo root → http://127.0.0.1:8000
```

(The docs are plain Markdown and read equally well on GitHub.)
