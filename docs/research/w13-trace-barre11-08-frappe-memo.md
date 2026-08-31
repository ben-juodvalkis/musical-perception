# W13(a) memo — mapping the barre11-08 trace onto pipeline hypotheses

Companion to [w13-trace-barre11-08-frappe.yaml](w13-trace-barre11-08-frappe.yaml)
(owner-attended, 2026-08-30). Everything below was written **after** the
trace was locked; nothing here influenced it. The pipeline has still not
been run on this clip — every claim about pipeline behavior is
architectural (what the channels are built to see), not measured. Where a
claim is measurable, it is phrased as a hypothesis for W13(b) or a later
increment to test.

## The owner's convergence curve

From the trace, per field, in clip time:

| field | committed at | evidence |
|---|---|---|
| exercise type | ~3s | one spoken line, before any movement (heard) |
| meter (6, and the 6/4 call) | ~3s, reinforced from ~6s | same spoken line; then counting+movement fused |
| tempo (120) | within 6 counts of the ~6s demo start (~9–12s) | counting voice + movement, fused |
| quality (sharp, light) | same window as tempo | fused, plus filename prior |
| structure (8 sets of 6 + balance) | ~30–33s, with light counting through the middle | rhythm-break + spoken announcement |

Cross-check on his own numbers (not a measurement): 6 counts at his stated
120bpm is ~3s, so "sure of everything besides form" lands ~9s — consistent
with his pass-1 "within 10 seconds."

Two prior dynamics worth keeping:

- **One declarative line beat a stated prior instantly.** Pre-roll:
  "usually wouldn't be in groupings of 3 or 6 pulses." One sentence at ~3s
  flipped him to 6 with no visible hesitation, before any movement existed
  to corroborate it.
- **The tempo prior anchored; evidence narrowed it.** Pre-roll range
  100–120; commitment 120 — at the edge of the prior, reached only after
  the fused counting+movement window.

## Mark-by-mark mapping

### Mark 1 (~3s): the declarative announcement — exercise + meter from one line

| what it bought | current channel? | mapping |
|---|---|---|
| exercise type | **yes — current** | Gemini's `exercise` field reads the transcript; a line naming the step is exactly what it classifies from. |
| meter, by declaration | **partial** | Gemini's qualitative `meter` sees the transcript in principle. But the precision path treats spoken numerals only as *timing markers* (beat/and/ah → `TimedMarker` → `posterior.py`); a numeral used *declaratively* ("six counts…", before any rhythm exists) is a different evidence class the lattice never consumes. |

**Hypothesis H1 — stated-structure channel (current capture, new
analysis):** parse declarative count/set announcements in the existing
Whisper transcript ("N counts", "N sets", an announced balance) into direct
priors on `meter_grouping` and structure. Zero new capture; it is the
single highest-value-per-second evidence in this trace, and it is available
seconds before any pulse evidence can exist. W13(b) will show whether the
pipeline's meter answer effectively already converges this early (via
Gemini) or only after rhythmic evidence accumulates.

**Hypothesis H2 — exercise-conditioned priors (pure prior):** the owner's
pre-roll shows a rich prior keyed on nothing but the exercise name: feel,
character, plausible tempo range, phrase shape, and a *structured
uncertainty* (duple/triplet explicitly equal). The pipeline extracts
`exercise` but conditions nothing on it. A small exercise→prior table
(tempo range, character, division prior) is the pure-prior counterpart of
what the filename gave the owner. Note the trace also shows the risk case:
his grouping prior was *wrong* for this clip and one line of evidence had
to override it — so priors must stay soft.

### Mark 2 (~6s + 6 counts): the fused window — tempo, meter, quality

| what it bought | current channel? | mapping |
|---|---|---|
| tempo | **yes — current, twice** | counted markers (Whisper timestamps → `precision/tempo.py`) and the rung-2 acoustic pulse (`pulse.py` peakRate). The owner's 6-count time-to-commitment is the benchmark: how many seconds of demo does each channel need before its answer stops moving? That is literally W13(b)'s curve. |
| meter reinforcement | **partial — current** | the W12 factored slice (`meter_division` + `meter_grouping`) and the ADR-017 grouping ladder are the right shape; whether they *reinforce within 6 counts* is a W13(b) question. |
| quality | **partial — current, weakly fused** | Gemini per-phrase quality and pose-based `dynamics.py` both exist, but W7 ruled movement a weak vote and nothing fuses them tightly. The owner reports the carrier as "both fused" — one percept, not two votes. |

**Hypothesis H3 — audio-visual fusion for quality:** the expert does not
have separate audio-quality and video-quality opinions to average; the
fusion happens before judgment. Current architecture fuses late and weakly.
Needs design work, not new capture (both streams are already captured).

**Supporting observation for rung 2:** the owner *actively discards the
step names* during the demo — "more just getting the quality and rhythm."
During demonstration, speech is a rhythm-and-prosody carrier, not a text
stream — which is precisely the sub-lexical bet of the peakRate pulse
channel. The inversion matters: the pipeline's front end is
transcription-heavy everywhere, while the expert uses lexis only where it
is declarative (mark 1) and drops it where it is rhythmic (mark 2). One
front end, two consumption modes, switched by function.

### Mark 3 (30–33s): the end — rhythm-break as boundary detector

| what it bought | current channel? | mapping |
|---|---|---|
| total length / structure | **partial** | Gemini `structure` is qualitative; counting "8 sets of 6" needs the grouping ladder extended to phrase/set level (ADR-017 leaves rungs above the bar mostly silent). |
| ending = balance | **partial** | the announcement is H1's stated-structure channel again. |
| *that it ended at all* | **no — new analysis, current capture** | the owner's detector is the teacher **breaking rhythm**: "he wasn't keeping rhtyhm anymore." Nothing in the pipeline models cessation; every channel extracts what *is* periodic, none flags when periodicity *stops*. |

**Hypothesis H4 — pulse-dropout boundary detector (current capture, new
analysis):** end-of-exercise = the pulse stream (already frozen in
`pulse.json` sidecars) losing periodicity while speech continues, ideally
coincident with a declarative announcement (H1). Testable offline against
existing sidecars on ingested corpora — no new capture.

### Negative space (~12–30s): attention is front-loaded

The owner spent ~18 of 38 seconds nearly idle, holding only a low-rate
set counter. Per-field convergence times differ by an order of magnitude
(meter ~3s; tempo ~9–12s; structure ~33s), and the middle of the clip
carries information for exactly one field. The pipeline currently emits one
answer per clip with no notion of *when* each field became knowable —
W13(b)'s prefix-replay curve is the machine-side instrument, and this trace
is the human curve to lay it against. Downstream (rung 5), it licenses an
early-commit consumption pattern: commit tempo/meter/quality early, keep
only a cheap repetition counter running, re-engage on boundary evidence
(H4).

## The cue-nod question (W13(c), routed from W10)

**No mark in this trace touched a cueing gesture.** The carriers were: a
spoken declarative line (mark 1), counting voice fused with demonstrated
movement (mark 2), and a rhythm-break plus announcement (mark 3). Asked
directly what signaled the end, the owner named the rhythm-break — not a
nod, not a gesture.

What this does and does not mean: this was a **demo** clip — the teacher
marking the exercise for the room — not a live start where an accompanist
waits on a preparatory cue. The cue-nod hypothesis mostly lives at the
moment playing must *begin* in execution context, which a demo cannot
exhibit. So this trace records a genuine absence in the demo context
(consistent with W10's postural negative), while leaving the live-start
context untested. **Capture implication for W13(c):** the clip type that
can answer the cue-nod question is an execution take including the seconds
before the exercise starts — future capture should preserve pre-exercise
lead-in rather than trimming to movement.

## Summary of hypotheses

| id | claim | cost |
|---|---|---|
| H1 | declarative count/set/ending announcements → direct structure/grouping priors | current capture, new parsing |
| H2 | exercise-type → soft prior table (tempo range, character, division) | pure prior, trivial |
| H3 | quality needs early audio-visual fusion, not late weak voting | design, current capture |
| H4 | end-of-exercise = pulse-stream dropout + coincident announcement | current sidecars, new analysis |
| — | per-field time-to-commitment is the right lens; human curve now on file | W13(b) instrument |

## POST-TRACE COMPARISON (coda — run after the trace was locked)

Run at the owner's request in the same sitting, after commit `ddbee78`
locked the trace. One plain full-clip run (`python -m musical_perception
<clip>`), no `--record-traces`, nothing under `evals/` touched. Transcript
content in the pipeline's segment readout is paraphrased here per A5-30.
The owner's pass-1 answers are the expert reference; disagreement below is
a finding about the pipeline, and none of it fed back into the trace.

| field | owner (pass 1) | pipeline (full clip) | read |
|---|---|---|---|
| tempo | 120 bpm | **109.0** committed (Gemini-based 107.0 @ 31% conf, 27 beats; onset-based 217.2 @ 75% — ~2× the committed level) | 9% low; inside the owner's pre-roll 100–120 prior but below his commitment |
| meter | 6/4 | **4/4** | **miss** — see below |
| division | (duple; "sharp") | subdivision: none, 0% confidence | abstained |
| exercise | (from the ~3s line) | **Frappé, 100%** | hit |
| quality | sharp and light | articulation 0.24 (toward staccato), weight 0.20 (light), energy 0.65 | direction aligned on all three |
| structure | 8 sets of 6 + a balance | 24 counts, 2 sides | under-counted (8×6 = 48); no balance surfaced; "2 sides" for a one-side demo + balance |
| enough-at | ~10 s of clip time | no analogue — consumes all 37.8s before answering | exactly the W13(b) gap |

**The meter miss is H1's exact scenario, now observed.** The pipeline's own
segment readout shows it transcribed the ~3s declarative lead-in line (the
one that gave the owner the meter before any movement — its local
onset-rate estimate for that 2.0–5.0s stretch was 240 bpm, i.e. the line
was consumed as *timing material*), yet the meter answer came out 4/4. The
declarative "six" was in the transcript and no channel treated it as a
statement about grouping. That is the stated-structure channel's gap,
demonstrated on the first try.

Other segment-level notes (paraphrased content): the 5.5–18.0s demo window
read 217 bpm at CV 0.12 — a clean metric level sitting 2× above the
committed answer, consistent with the fast-6 feel; a 21.5–28.0s spoken
stretch naming steps read 126 bpm; the closing 32.5–37.0s phrase naming
steps read 210 bpm. The 30–33s rhythm-break the owner used as his end
detector falls in the gap between those last two segments — the onset
channel effectively *saw* the dropout (no coherent segment there) but
nothing consumed the absence (H4).

Scorecard against the trace's hypotheses: H1 directly evidenced (meter
miss with the declaration in hand); H4 circumstantially evidenced (the
break shows up as segment absence); H2/H3 not exercised by a single run.
The owner's convergence curve (meter ~3s, tempo ~9–12s, structure ~33s)
now has its machine counterpart pending — this run answers only *what*,
never *when*; W13(b) charts the when.
