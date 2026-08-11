# Research Log — rhythm-core reset

Append-only ledger of every agent iteration on the
[agent charter](agent-charter.md)'s goal ladder. This file is the loop's
memory: sessions are stateless, the ledger is not. Newest entry last.
Never rewrite or delete past entries — corrections get their own entry.

**Entry template** (copy verbatim, fill every field; "n/a" is allowed,
blank is not):

```markdown
## YYYY-MM-DD · rung N · agent/<branch> · cloud|local
Attempted:
Pre-registered expectations:
Result: (eval deltas with numbers; prediction scorecard X/Y landed)
Regressions and classifications: (fake-green-lost / genuine-trade / knife-edge / none)
Lesson (durable, one paragraph):
Status: PROPOSED | BLESSED | REJECTED (reason) | DEAD-END | BLOCKED (needs)
```

## Standing Lessons

Distilled from the ADR record and the retrospective
([ADR-016](../adr/016-rhythm-core-reset.md)); every session reads these
before working. Add to this list only via a ledger entry that earns it.

1. **Words are not the beat.** ASR word timestamps carry a 0–150 ms
   word-dependent early bias vs the perceptual beat (vowel onset /
   peakRate). Never anchor a grid to word starts; never annotate ground
   truth at word starts.
2. **Priors are priors, not post-processing.** A hard fold (the old 70–140
   band) destroys correct out-of-band measurements. Apply priors at level
   selection, multiplicatively, never to the raw measurement.
3. **Levels vote, they don't average.** Mean-IOI across mixed 1×/2×/3×
   onsets lands between metric levels — the field abandoned it ~2000–2005.
   Use harmonic summing / ratio-reinforced clustering / grid regression.
4. **One temp-0 LLM draw is a coin flip** (ADR-011: 18,18,18,32 on
   identical input). Consume distributions or outvote with independent
   evidence (ADR-012), never trust a single draw.
5. **Phrase-final lengthening is expected structure, not noise** (Repp;
   Wightman). Censor or down-weight boundary intervals; never average them
   into tempo.
6. **Silence is evidence.** A hypothesis that predicts a strong beat where
   the teacher voiced nothing pays for it (Povel–Essens; PIPPET).
7. **Sub-4% eval disagreements are noise by construction** (human tapping
   CV is 3–5%). Knife-edge rows gate nothing (ADR-015 typed gates).
8. **A transcription hallucination once scored all-green** (clip 17). Any
   green earned without a verified perceptual chain is provisional until
   the sanity guard or a human confirms it.
9. **The harness sets the gradient** (retrospective F5). Whatever is
   replayable gets iterated; build the trace/replay path for a new channel
   before betting on the channel.
10. **Falsified ideas stay falsified.** Check this ledger before
    attempting anything — re-runs of dead ends need new evidence, not
    optimism.

---

## 2026-08-09 · rung — · (charter created) · cloud

Attempted: Charter, ledger, environment guide, and goal ladder created
from ADR-016 + the voice-as-drum literature review. No pipeline work.
Pre-registered expectations: n/a.
Result: Blessed baseline at creation time (git `15b8164`, blessed by
ADR-015 override): tier-1 committed accuracy tempo 0.571 (16/12/1),
meter_triple 0.357 (10/18/1), counts 0.571 (12/9/7); tier-1 ECE 0.291;
tier-0 tempo 25/25, meter 24/25. Fully-green checklist clips 5/24, all
numbers-counted; step_names meter slice 0.077. DEV = all 30 current
cases; SEALED = empty until new capture.
Regressions and classifications: n/a.
Lesson (durable): The loop's constraints are downstream of documented
incidents, not hypotheticals — see Standing Lessons 1–10.
Status: PROPOSED (awaiting rung 0 — owner data logistics).

## 2026-08-09 · rung — · (infrastructure decision) · cloud

Attempted: Runner architecture decided with the owner; docs updated
(agent-environment.md rewritten, charter rung 0 / rung 5 / ladder intro
amended, scripts/air-nightly.sh added).
Pre-registered expectations: n/a (decision, not experiment).
Result: Three-machine architecture — the always-on 16 GB MacBook Air is
the PRIMARY runner (data locality, persistent model weights and tool
venvs, free compute); cloud sessions are OVERFLOW (burst parallelism,
clean-room checks) behind Path B; the owner's main machine is the SEALED
vault and blessing desk, so DEV/SEALED separation stays physical on every
runner. Scheduled runs use one never-changing standing contract that
delegates to the charter's CURRENT RUNG pointer. Local-model policy:
specialized nets in now (Whisper, WavLM/DistilHuBERT SSL front-ends,
Silero VAD, MediaPipe); general local LLMs (7–27B) deferred as a
premature cost-for-accuracy trade under the research posture — and a 27B
at 4-bit (~14–16 GB) does not fit the 16 GB Air regardless; audio-native
small model logged as a rung-5 ensemble backlog note; graduation
condition for fine-tuning a small model on the corpus: ~140+ annotated
clips.
Regressions and classifications: n/a.
Lesson (durable): The runner is the hands, not the brain — plan usage is
identical on the Air and in the cloud; what the Air buys is state
(weights, venvs, recordings) surviving between runs, which is exactly
the tax ephemeral containers keep re-paying.
Status: BLESSED (owner decision in session, 2026-08-09).

## 2026-08-09 · rung — · (data staged; autonomy mode pending) · cloud

Attempted: Rung-0 data staging confirmed by the owner (DEV audio ready;
Ballet Barre 1 section videos at `video/youtube/Ballet Barre 1/Sections`
on the Air, assigned all-DEV). Charter amended: add-only ingestion
carve-out with `maturity: provisional|verified` case labels (provisional
rows never gate, always a separate slice); Rung M (marathon) drafted as
an explicitly DORMANT section — staged-autonomy plan is rung-by-rung
with daily blessings through rung 2, then an owner decision at the
rung-2 kill-test verdict on whether to commission the marathon.
CURRENT RUNG set to 1.
Pre-registered expectations: n/a (process decision).
Result: n/a.
Regressions and classifications: n/a.
Lesson (durable): The early rungs are inherently interactive — rung 1
builds the measuring instruments, rung 1.5 needs human ground truth,
rung 2 ends at a strategic fork. Autonomy is earned at exactly the
point where the work becomes parallel and the foundations are verified.
Status: PROPOSED (awaiting owner confirmation of staged-autonomy mode +
the supervised rung-1 session).
