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

## 2026-08-11 · rung 1 · agent/rung-1-stage-scoring · local

Attempted: The full rung-1 EVAL-CHANGE deliverable set. (1) Beat-grid
format `evals/grids/<case-id>.yaml` — editable `beats` + frozen peakRate
`onsets`, mandatory explicit `provisional` flag, media sha256 provenance,
frozen detector params recorded per grid ([docs/evals/beat-grids.md](../evals/beat-grids.md)).
(2) Tap-assist annotator `python -m musical_perception.annotation`
(peakRate per review-1 recipe #1: 300–3000 Hz band, 10 Hz zero-phase
envelope, derivative, 3·MAD prominence, 120 ms spacing, Praat voiced gate
±30 ms) with Audacity label round trip for owner correction; `--verified`
is owner-only. (3) `stage1` eval suite: pulse P/R/F at ±70 ms + signed
asynchrony per clip vs grids, provisional/verified aggregates split,
count_style slices, wired into `evals run --suite tier0,tier1,stage1`.
(4) Acc1/Acc2 (±4% standard + ±8% house, fixed family {⅓,½,1,2,3}) and
OE1/OE2 distributions added to tier-0/1 reporting (informational, never
gating). (5) Onset-vs-token guard on trace load (warn when tokens >
1.5×onsets+8 or onsets=0). Provisional grids generated for 24/30 DEV
clips (all with media on this machine), `provisional: true` everywhere.
Pre-registered expectations: written before implementation — P1 pooled
matched-pair asynchrony in [−120,−20] ms; P2 pooled F@70ms in
[0.35,0.65]; P3 |mean asynchrony| numbers < step_names; P4 Acc1@4%
0.35–0.55, Acc2@4% 0.45–0.65, ≥2 rows |OE2|∈[0.30,0.585]; P5 zero guard
triggers; P6 zero tier-0/1 outcome changes.
Result: Scorecard 3 full hits, 1 partial, 2 misses — both misses are
findings. P2 ✓ pooled F=0.391 (macro 0.373; P 0.425, R 0.362, 24 clips).
P4 ✓ Acc1@4% 0.393, Acc2@4% 0.500 (@8%: 0.571/0.679), OE1 median 0.0009,
|OE2| median 0.053, 6 rows with |OE2|∈[0.30,0.585] — the between-levels
mass ADR-016 predicted, now visible. P6 ✓ "no outcome changes vs
baseline". P1 partial: sign right, magnitude wrong — mean −12.7 ms
(median −16.4, sd 31.1): ±70 ms matching censors the strongly-early words
into non-matches (recall 0.362), so matched-pair asynchrony
under-measures the ASR lead; rung 2 must read recall AND asynchrony
together, not asynchrony alone. P3 ✗ reversed: numbers −15.9 ms vs
step_names −8.6 ms (n=11/11). P5 ✗ one trigger, classified benign:
rig-numbers-3-4-90-clean, 94 tokens vs 52 voiced onsets = dense triplet
counting ("one-and-a-…": 31 "and" + 31 "a"; unstressed schwas don't each
earn a voiced envelope rise) — threshold calibration data point, kept
sensitive on purpose (false positive costs one listen; false negative is
a silent green). The totals hid one headline: the vocables clip collapsed
to a single Whisper token (pred=1 vs 24 grid onsets, R=0.042) — the
strongest per-clip evidence yet that rung 2's acoustic channel is not
optional for vocables. pytest 193 passed / 3 skipped / 0 failed.
Regressions and classifications: none — tier-0/tier-1 outcomes byte-
identical to the blessed baseline; every addition is reporting-only.
Two disclosed non-eval accommodations: (a) tests/test_wakeword.py gained
a skipif for the missing tflite runtime (no macOS arm64 wheel — tests
errored on this machine before this branch; the trigger path is the
rung-7 RETIRED target); (b) the peakRate detector gained a conditional
degenerate-silence prominence floor, verified bit-identical on all 24
real clips vs the pure recipe (it exists so synthetic-silence tests can't
pass on filter ringing).
Lesson (durable, one paragraph): The instrument censors what it
measures — a ±70 ms matched-pair asynchrony can only ever report the
survivors, so a channel that is *very* early looks *mildly* early with
low recall; and provisional peakRate grids measure words-vs-peakRate,
not words-vs-truth, until rung 1.5 flips them. Separately, Whisper does
not merely mistime vocables, it deletes them (1 token for 24 voiced
onsets), and legitimate dense triplet counting reaches 1.8× tokens per
voiced onset — both now pinned as data, not anecdote.
Status: PROPOSED (owner: bless rung 1, then rung 1.5 — verify/correct
the 24 provisional grids via to-labels/from-labels). BLOCKED (needs):
6 DEV clips have no media on this machine, so no grids yet —
audio/counting/8-counts-2x.aif, audio/counting/8-counts-triple.aif,
video/youtube/Exercise 1 Demo.m4v, video/youtube/Frappe.mov,
video/youtube/plies demo.m4v, video/youtube/grande battement.mov; stage
them and re-run `python -m musical_perception.annotation generate`
(word-start-derived grids are forbidden by Standing Lesson 1, so this
gap is owner-only).
