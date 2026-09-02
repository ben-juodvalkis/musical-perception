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
