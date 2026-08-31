# W10 · Nod kinematics and phrase-arrival segmentation

**Date:** 2026-08-30 · **Branch:** `agent/marathon` · **Status:** PROPOSED
(negative result) · Pre-registration and prediction scorecard:
[RESEARCH-LOG.md](RESEARCH-LOG.md), entries of 2026-08-30.

Reproduce (read-only over committed traces and grids — no media, no models,
no API key):

```bash
python scripts/w10-nod-kinematics-report.py
```

Per-clip numbers: [`w10-nod-results.json`](w10-nod-results.json).

---

## 1. What this asked, and why it was worth a night

[W7](RESEARCH-LOG.md) (2026-08-23) asked whether *limb* movement carries a
recoverable pulse and answered no, with a diagnosis — the periodicity it
found tracked its own minimum-IOI parameter — and a recommendation: *"the
next thing worth trying is not a better peak-picker but a different event
definition — a dancer places phrase arrivals on the beat, which is a
segmentation problem, not a periodicity problem."* The owner commissioned
W10 on 2026-08-28 to take exactly that step.

The event definitions come from the one paper that studies our exact event,
Bishop & Goebl (2018), *Beating time: How ensemble musicians' cueing
gestures communicate beat position and tempo* (via
[Review 5 §c](review-5-gaze-and-cueing-tools.md)): gesture **acceleration**
patterns indicate beat position, specifically peak acceleration, in leaders'
**head-nodding** gestures; and visual cues at **re-entry points after long
pauses** are especially salient. Two changes from W7 follow: the landmarks
are the head, not the limbs, and the test is alignment to owner-verified
beat grids rather than self-consistent periodicity.

## 2. Method

| | |
|---|---|
| **Signal** | Head centroid (MediaPipe nose 0, ears 7/8) by `nanmean`, divided by median shoulder-to-hip distance → torso-lengths. Vertical is the nod axis (image *y* grows downward, so a nod bottom is a *maximum*). |
| **E1 · peak acceleration** | Local maxima of \|vertical acceleration\| — the literature's primary claim, taken sign-agnostically because choosing a sign was not pre-declared. |
| **E2 · nod bottom** | Local maxima of vertical position — the conductor's ictus analogue. |
| **E3 · head speed minima** | W7's falsified definition moved from limbs to head, kept as the **control** that separates "the landmark set mattered" from "the kinematic quantity mattered". |
| **Event floor** | Prominence ≥ 3 × (1.4826 × MAD) of the driving signal, then greedy strongest-first thinning to a 0.20 s minimum IOI. The factor 3 is not swept: it is the value this repository's own ratified peakRate detector uses (`prominence_mad_k: 3.0` in every committed grid's `params`). |
| **Truth** | Owner-verified beat grids. `evals.stage1.score_pulse` imported read-only; no scorer code touched. |
| **Tolerance** | **Primary ±0.15 s**, declared a priori: a visual channel synchronising with an auditory one does not hold mir_eval's ±0.07 s window, and Bishop & Goebl's effects live at the 100 ms scale. The blessed ±0.07 s is reported beside it, never instead of it. |
| **Null** | Circular shift, 500 draws: rotate the event train modulo clip duration, preserving event count and every IOI, destroying only phase. |
| **α** | 0.05 / 3 = **0.0167**, Bonferroni over the three pre-declared definitions. |

**Evaluation set: n = 3 clips.** Exactly four committed traces carry both
`pose.npz` and a beat grid — `exercise-1-demo` (41 beats),
`grande-battement` (36), `frappe` (55), all owner-verified, plus
`plies-demo` (171 beats, `provisional: true`, reported as its own slice and
gating nothing). Per the W3-remainder lesson of 2026-08-29, every
clip-level conclusion below is **provisional-on-n**. The remaining 22
Barre-1 traces have no grids and are used here for coverage only — never
for an accuracy number.

## 3. Results

### 3.1 Alignment to verified grids — nothing, at either tolerance

Mean over the three verified clips; `null F` is the mean score of the same
event train under random phase rotation.

| definition | tol | mean F | mean null F | best p | significant at α = 0.0167 |
|---|---|---|---|---|---|
| E1 peak acceleration | ±0.15 s | **0.216** | 0.249 | 0.633 | no |
| E2 nod bottom | ±0.15 s | 0.012 | 0.031 | 0.693 | no |
| E3 head speed minima | ±0.15 s | 0.097 | 0.129 | 0.471 | no |
| E1 peak acceleration | ±0.07 s | 0.139 | 0.124 | 0.086 | no |
| E2 nod bottom | ±0.07 s | 0.012 | 0.014 | 0.347 | no |
| E3 head speed minima | ±0.07 s | 0.051 | 0.061 | 0.198 | no |

**Zero of 18 clip × definition × tolerance cells reaches significance**, and
at the primary tolerance every arm scores *below* its own null mean. The
best p in the whole verified table is 0.086 (E1, `frappe`, ±0.07 s) — above
even an uncorrected 0.05. The provisional `plies-demo` row agrees (best p
0.186) and changes nothing.

The apparent F of 0.216 for E1 is the reason the null is the deliverable
rather than the F: with ~50 events against 41 beats at a ±0.15 s window, a
*random* event train scores 0.249. Reporting 0.216 without its null would
have looked like a weak positive channel. It is not a weak positive; it is
slightly worse than chance.

### 3.2 The re-entry contrast could not be tested on this corpus

The pre-registration declared in advance that a contrast resting on fewer
than 8 re-entry beats would be reported as **underpowered rather than as a
result**. At the pre-registered 2.0 s gap the three verified clips contain
**6 re-entry beats total** (3 / 1 / 2). N4 is therefore **untested**, not
falsified.

The reason is structural and worth recording: the verified grids annotate
the *counted* stretches, and within a counted stretch the teacher does not
pause for two seconds. The re-entry moments Bishop & Goebl identify as the
salient ones are precisely the moments the grid does not cover.

A post-hoc sweep (labelled post-hoc; it is not a test, because a threshold
chosen after seeing the deltas is not a hypothesis) shows the contrast only
becomes powered at gaps ≤ 1.0 s, and points the *opposite* way from the
prediction — recall at re-entry 0.067 vs 0.231 interior at a 1.0 s gap. On
a channel already shown to sit at chance, that direction carries no
information either; it is reported for completeness.

### 3.3 The nod-bottom arm is dead on this corpus, and the reason is postural

E2 produced events on only 18 of 26 pose traces and a **median of 2 events
per clip**. This is not the W7 NaN failure — the eight zero-event clips
include five with < 0.5 % missing landmarks. The cause is that the head's
vertical *position* on this corpus is dominated by whole-body motion: the
robust scale of the vertical series runs 0.019–0.515 torso-lengths, and on
the travelling clips (`exercise-1-demo` 0.425, `barre1-B-el` 0.515) no
individual nod's prominence comes near 3 × MAD of a signal that already
contains a plié.

That is the substantive finding under the negative one: **on a dancing
body, head height is a postural signal, not a nod signal.** Double
differentiation is what makes E1 usable at all — acceleration high-passes
the slow postural component away, which is why E1 alone yields events on
26 of 26 clips (median 0.94 Hz) while E2 yields almost none.

### 3.4 Coverage is genuinely voice-independent — the one thing that held

E1 extracts at 0.81–1.07 Hz on the seven `execution-left` takes, the ones
W4 flagged as carrying ≤ 3 transcribed words, against a 0.95 Hz median
across all 22 Barre-1 clips. Movement does not stop when the teacher does.
The channel has coverage; what it lacks is content.

### 3.5 The controls, which are what make the negative believable

- **Positive control (N6).** A synthetic non-isochronous nod — drifting
  IOIs and two 2.5 s gaps, so the circular-shift null is not degenerate —
  is recovered at **F = 1.000** (E1 and E2) and **0.987** (E3), at
  p < 0.01 against the null. Pinned in `tests/test_nod.py`.
- **Null calibration (N7).** Phase-destroyed (rotated) event trains from
  the real clips, run back through the identical test, reject at
  **FPR 0.015–0.030** over 200 replicates at α = 0.05 — conservative, not
  liberal.
- **A design bug the positive control caught before any real data was
  scored.** The first floor was a rank over candidates ("keep the top half
  of the extrema"), which caps recall at 0.5 by construction: the perfectly
  recovered synthetic nod scored F = 0.615 and would have been read as a
  weak channel. Replaced with the prominence floor above. This is W7's
  lesson paying for itself a second time.
- **The null's known blind spot, stated rather than discovered.** Against a
  perfectly isochronous reference, a rotation by one beat period realigns,
  so the circular-shift null is weak by construction there. The real grids
  are not isochronous and the mean null F is printed for every cell so a
  reader can see when it bites.

## 4. Verdict

**W10 is a negative result.** Head-nod kinematics — under three
pre-declared event definitions, against owner-verified beat grids, at a
tolerance chosen generously in the channel's favour — carry no recoverable
beat-position information on this corpus. The honest claim is narrow:
*peak vertical head acceleration, nod bottoms, and head-speed minima on
these three verified clips do not align with counted beats better than a
phase-randomised event train.*

What this does **not** establish, stated so no future session over-reads
it: it does not test Bishop & Goebl's actual claim, which is about
**cueing gestures at entries** — a ballet demo video where the teacher
counts continuously through an exercise is not the ensemble-cue situation
their pianists were in, and §3.2 shows this corpus's grids do not cover
the moments where their effect lives. Testing that claim needs grids that
extend across the talking, or capture aimed at the cue-in itself.

**Recommendation:** do not iterate W10 standalone. Movement now has two
independent negatives (W7 limbs, W10 head) and one positive property
(voice-independent coverage, §3.4). Fold it into W5 as a weak vote at
most, exactly as W2 and W7 were folded, and do not spend a third night on
a better peak-picker. If the cueing question is wanted, it is a **capture**
question, not an algorithm question.
