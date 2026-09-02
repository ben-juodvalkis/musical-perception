# EB-1 — the estimator bake-off

**Owner-directed 2026-09-02 in session** ("yes let's do it"), with the
harmonic/subharmonic profile added at his request **before any code was
written**. REPORTED-ONLY: the winner is not adopted. Nothing under
`src/musical_perception/` changes; no file under `evals/` is created or
modified.

Follows [SW-1](sw1-steady-window-sweep.md) (the window axis: dead end on
demos), [pulse-next-step.md](pulse-next-step.md) (the diagnosis: nothing
separates beats from syllables) and
[Review 6](review-6-syncopation-and-pulse-reconstruction.md) (what the
literature offers, and the warning that our failures may not be
syncopation at all).

---

## Part 1 — PRE-REGISTRATION

**Committed before `scripts/eb1-estimator-bakeoff.py` exists.** `git log`
on this branch shows the order.

### The question

SW-1 varied *where in the clip* we look and found it does nothing on the
demos. This varies *what arithmetic we apply* to the same events. The
estimator is the remaining suspect: all four demo misses land 9–12 % off,
which is the signature of a median taken across a stream mixing the beat
rate with the syllable rate.

### Frozen search space

**Arm A — five estimators, one fixed event stream** (peakRate on the
clip's own audio, checksum-verified; identical events for every
estimator, so the arithmetic is the only variable).

| id | estimator | rule |
|---|---|---|
| `median-consec` | **control — what ships today** | 60 / median of consecutive IOIs |
| `all-pairs` | Inner-Metric-Analysis family | histogram of **all pairwise** onset distances (not just adjacent), peak picked as the period |
| `comb` | autocorrelation / comb | score candidate periods by summed event mass landing near grid points of that period, best over phase |
| `povel-essens` | clock induction | candidate clocks scored by counterevidence: events off clock ticks, and **silent** clock ticks, weighted as in Povel & Essens (1985) |
| `hopf` | nonlinear resonance | bank of canonical Hopf oscillators, Velasco & Large (2011) parameters: **289 oscillators log-spaced 0.25–16 Hz, α = 0, β₁ = −1, β₂ = −0.25, ε = 1**; period read from the peak of the steady-state amplitude profile |

All five then project into [70, 140] by ×/÷{2, 3} with the factor
reported per clip, exactly as SW-1 did, so the level rule is held constant
and only the period estimate varies.

**Arm B — off-the-shelf trackers on the eight demos**, which did not exist
when W3 benchmarked them. `essentia_re2013` and `librosa_plp` (W3's raw-audio
winners) and `beat_this`. Scored on **step-one tempo**, not pulse F.
Any tool that fails to install is reported BLOCKED by name, never as a
null result.

**Arm C — the harmonic/subharmonic resonance profile (owner-requested).**
For every clip, the energy at **f, 2f, 3f, f/2, f/3** relative to the
clip's true beat frequency, computed **twice**: from the linear comb
analysis and from the Hopf bank. Plus the location of the single dominant
peak expressed as a ratio to the true beat.

This is the regime diagnostic Review 6 §1 says has never been run:

- beat frequency carries energy **linearly** → not a missing-pulse clip;
  our problem is arithmetic, not absent information.
- beat frequency appears **only** in the nonlinear profile → genuine
  missing-pulse; the oscillator earns its place.
- dominant peak at a **non-integer** ratio to the beat → the clutter
  signature (a rate that is neither the beat nor a clean subdivision).

### Population and metrics

The **34-row step-one gating set**, unchanged. Pass = committed pulse
within ±8 % of the in-band truth, reported beside Acc2@8% and the
between-levels count, with the **same fixed odd/even split-half** as SW-1
(lexically sorted case ids, rows 1,3,5… vs 2,4,6…) so the two increments
are directly comparable.

Coverage facts, stated in advance so they are not discovered
conveniently: media is present for all 34; `adr006-8-counts-triple` has a
truth of 68.38 BPM, below the band, and **cannot pass under any estimator**;
`rig-vocables-4-4-100-clean` has 1 Whisper word but is unaffected here
because Arm A uses peakRate, not the trace.

### Pre-registered predictions

| # | prediction | reason |
|---|---|---|
| **E1** | `all-pairs` beats `median-consec` on the two sparsely-voiced demos (plié, rond-de-jambe) — both currently failed. | Their events are ON the grid, just not on every beat; the beat-to-beat distance survives in all-pairs and is destroyed by adjacent gaps. This is the sharpest single prediction here. |
| **E2** | `comb` beats `median-consec` on the full 34 by ≥ 3 rows. | Clutter is the dominant failure and a comb scores a period against the whole train instead of averaging gaps. |
| **E3** | `hopf` does **not** beat the best linear estimator on the full 34. | Review 6 §1: no clip in this corpus has been shown to be syncopated. Nonlinear resonance buys the missing-pulse case, and we predict we do not have one. |
| **E4** | On ≥ 6 of 8 demos the true beat frequency carries **non-trivial linear energy** (within 6 dB of the dominant peak). | Same reason: sparse-but-on-grid voicing puts the beat in the spectrum as a harmonic. If E4 fails, Review 6's §3 machinery becomes the priority and this prediction is the thing that says so. |
| **E5** | On ≥ 5 of 8 demos the dominant linear peak sits at a **non-integer** ratio (outside ±5 % of {⅓,½,1,2,3}) to the true beat. | The clutter signature: 2.6 events per beat with a varying count cannot land on a clean subdivision. |
| **E6** | **No** estimator passes both drift clips (frappé, fondu). | Drift is orthogonal to the estimator: frappé runs 139→132→165 and has no single true tempo. If one passes, it is luck and will be flagged as such. |
| **E7** | At least one Arm-B off-the-shelf tracker matches or beats our current 4/8 on the demos. | They are trained on real music with real syncopation. Against this: a talking teacher is far out of their domain, and W3 already showed them collapsing on marker streams. Genuinely uncertain — which is why it is written down. |
| **E8** | The best Arm-A estimator's odd/even half-gap exceeds 0.15. | 34 rows split 17/17. SW-1's stability criterion produced a three-way tie; this says the instrument is coarse and predicts it again. |
| **E9** | Containment: `git diff --stat origin/main` shows only `docs/research/` and `scripts/`; pytest green. | — |

**Stated in advance, and it is the main risk:** five estimators against 34
rows is enough freedom to crown a winner by noise. **The deliverable is a
diagnosis — which of clutter / sparse sampling / missing pulse / drift we
actually have — not an adoption.** Any winner is reported with its
half-gap beside it and adopted by nobody.

Late-added measurements, if any, are disclosed W2-reopen style.

---

## Part 2 — RESULTS (run 2026-09-02)

**Headline: the median was the single biggest defect in the tempo path.
Replacing it — with either all-pairs distances or a comb, on exactly the
same events — takes the gating set from 16 of 34 to 28 of 34, past the
blessed pipeline's 20, and collapses between-levels rows from 21 to 7.
But almost all of that is the owner's own recordings (12 → 24 of 26). On
the eight demos it moves 4 → 5. The demo is still unsolved, and the
resonance profile says why: on none of the eight demos is the beat the
dominant periodicity in the event stream. It is not a missing pulse — the
nonlinear oscillator finds nothing the linear methods miss, and comes
last. It is clutter, and on the sparse 3/4 clips it is the bar.**

Artifacts: `scripts/eb1-estimator-bakeoff.py`,
`scripts/eb1-arm-b-trackers.py`, `docs/research/eb1-estimator-bakeoff.json`,
`docs/research/eb1-arm-b-trackers.json`. Coverage: **34/34 rows, 0 skipped,
0 checksum mismatches.**

### Arm A — five estimators, identical peakRate events

| estimator | pass /34 | demo /8 | rig /26 | Acc2 | between-levels | odd | even | half-gap |
|---|---|---|---|---|---|---|---|---|
| `median-consec` *(control, ships today)* | 16 | 4 | 12 | 16 | **21** | 9 | 7 | 0.118 |
| **`all-pairs`** | **28** | 4 | **24** | 29 | **7** | 15 | 13 | 0.118 |
| **`comb`** | **28** | **5** | 23 | 28 | 8 | 16 | 12 | 0.235 |
| `povel-essens` | 27 | **5** | 22 | 28 | 8 | 14 | 13 | **0.059** |
| `hopf` | 19 | 2 | 17 | 19 | 20 | 10 | 9 | 0.059 |

Blessed pipeline, same 34 rows: **20 pass, between-levels 10 of 33.**

### Arm C — the regime diagnostic (the owner's question)

Dominant periodicity in the event stream, as a ratio to the true beat:

| demo | truth | dominant periodicity ÷ beat | reading |
|---|---|---|---|
| coupé-barre | 108 | **2.00** | syllable rate, exactly 2 per beat |
| dégagé | 110 | 2.13 | syllable rate, non-integer |
| frappé | 135 | 2.10 | syllable rate, non-integer |
| plié | 120 | 2.50 | syllable rate, non-integer |
| tendu | 102 | 2.85 | syllable rate, non-integer |
| fondu | 86 | 0.54 | ≈ half the beat rate (2-beat group) |
| rond-de-jambe | 96 | 0.35 | ≈ ⅓ the beat rate — **the 3/4 bar** |
| tendu-warmup | 112 | 0.35 | ≈ ⅓ the beat rate — **the 3/4 bar** |

**On 0 of 8 demos is the beat the strongest periodicity present.** The beat
sits 7.6–29.4 dB below the dominant peak in the linear profile, and lower
still in the nonlinear one.

**This answers Review 6 §1's open question: we are not in a missing-pulse
regime.** If we were, the Hopf bank would recover a pulse the linear
methods cannot. It does the opposite — 19/34, worst of the five, 2/8 on
demos. Two regimes are present instead, and they are different problems:

- **clutter** (5 clips): the strongest rhythm is the syllable rate, at
  2.0–2.85× the beat and *not a clean multiple*, so no ×/÷{2,3} projection
  can recover the beat from it.
- **bar-dominant sparse voicing** (rond-de-jambe, tendu-warmup): she voices
  beats 1 and 3 of a 3/4 bar, so the strongest periodicity is the **bar**,
  one third of the beat rate. Notably these are the two demos where a
  ×3 projection is exactly the right move — and where a level prior
  conditioned on the exercise would supply it.

### Arm B — off-the-shelf trackers on the demos (never seen before: they postdate W3)

| tracker | pass /8 | note |
|---|---|---|
| **`librosa_plp`** | **5/8** | raw audio, no knowledge of this project — matches our best own-stream estimator |
| `essentia_re2013` | 3/8 | W3's raw-audio winner; loses here |
| `beat_this` | 2/8 | **returned no usable beats at all on 5 of 8** (frappé, plié, rond-de-jambe, tendu, tendu-warmup) — reported by name, not an install failure |

**`librosa_plp` on raw audio equals the best thing we do on our own event
stream.** That is worth sitting with: a general-purpose music beat tracker,
given the teacher's audio and nothing else, is level with the whole
peakRate-plus-arithmetic path on the material the reset is aimed at.

### Prediction scorecard — 4 hits, 1 partial, 2 falsified, 1 ambiguous

| # | prediction | outcome |
|---|---|---|
| E1 | `all-pairs` beats `median-consec` on plié **and** rond-de-jambe | **PARTIAL** — rond-de-jambe fails→passes (107.7→95.7 vs truth 96); plié fails both ways (108.8→84.3 vs truth 120). 1 of 2 |
| E2 | `comb` beats the control on the 34 by ≥ 3 rows | **HIT, by four times the margin** — +12 (16→28) |
| E3 | `hopf` does not beat the best linear estimator | **HIT** — 19 vs 28, and last on the demo slice. See the caveat below before leaning on it |
| E4 | ≥ 6 of 8 demos carry non-trivial linear energy at the beat (within 6 dB of the peak) | **FALSIFIED — 0 of 8.** The strongest single finding here, and it went the opposite way |
| E5 | ≥ 5 of 8 demos: dominant peak at a non-integer ratio to the beat | **HIT at the threshold — exactly 5 of 8**, and two clips sit on the ±5 % boundary (0.35 vs ⅓). Read as "about half", not as a clean hit (Standing Lesson 7) |
| E6 | no estimator passes both drift clips | **FALSIFIED** — `povel-essens` passes both. Flagged as pre-registered: its frappé reading is 139.0 against a label of 135, and frappé's own taps open at **139** before running to 165, so it is matching the *opening* tempo. Whether that is luck or the right answer is the open truth-side question |
| E7 | ≥ 1 off-the-shelf tracker matches or beats our 4/8 on demos | **HIT** — `librosa_plp` 5/8 |
| E8 | best estimator's half-gap > 0.15 | **AMBIGUOUS, disclosed** — the two winners tie at 28/34: `comb` gaps 0.235 (hit), `all-pairs` 0.118 (miss). The pre-registration did not say how to break a tie on "best". Reported both ways rather than picking the flattering one |
| E9 | containment + pytest | **HIT** — see below |

### Caveats that limit what this can be used for

1. **The Hopf arm is a 60-line reimplementation, not the authors' system.**
   Velasco & Large's published parameters are used unchanged (289
   oscillators, 0.25–16 Hz, α=0, β₁=−1, β₂=−0.25, ε=1), but the readout is
   a global argmax of steady-state amplitude, and the integrator needed two
   numerical fixes to run at all (**disclosed:** sample rate raised 200 →
   2000 Hz and |z| clamped below the 1/√ε singularity, after forward Euler
   overflowed; parameters untouched). It validates on a clean isochronous
   train. **E3 should be read as "a faithful-parameter reimplementation
   found nothing here", not as "nonlinear resonance was refuted."**
2. **`between_levels` and `pass` overlap** at the ±8 % tolerance, as
   recorded on 2026-09-02 — the counts are not disjoint and must not be
   added.
3. **Five estimators against 34 rows.** The 16→28 jump is far too large to
   be noise, but the ordering *among* the top three (28 / 28 / 27) is not
   separable at this n, and the half-gaps range 0.059–0.235.

### What this establishes

- **The median of consecutive gaps is the defect**, and it costs 12 rows
  against the identical event stream. Any adoption increment starts here.
- **Two candidate replacements are indistinguishable at this corpus size**
  (all-pairs 28, comb 28). Choosing between them needs more rows, not more
  analysis.
- **We are not in a missing-pulse regime**, so Review 6 §3's oscillator
  machinery is not the priority, and its §8 ranking is confirmed in the
  order it gave.
- **The demo remains unsolved and is a different problem from the rig.**
  Fixing the arithmetic doubled the rig slice and moved the demos by one
  clip. The demo failure is that the teacher's syllable rate is louder than
  her beat, at a non-integer ratio — which is exactly the case a prior over
  plausible tempos for a known exercise is built to break.

### Recommendation

1. **Adopt nothing yet**, per the commission. But note that unlike SW-1,
   this one has a candidate worth an adoption increment later: replacing
   `calculate_tempo`'s median with an all-pairs or comb period estimate is
   a **logic change under a zero-regression gate**, and it would need its
   own pre-registration and an owner re-bless.
2. **The prior table is now better motivated, not less.** Arm C shows two
   demos whose dominant periodicity is the bar and five whose dominant
   periodicity is the syllable rate at a non-integer ratio. A prior that
   knows a rond de jambe is a waltz near 96 is precisely what turns those
   into the right ×3.
3. **Do not build the oscillator.** Measured, and it is not our disease.
