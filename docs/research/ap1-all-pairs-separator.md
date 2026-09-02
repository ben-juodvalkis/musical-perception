# AP-1 — the all-pairs separator

**Run 2026-09-02, unattended, on `agent/step-one-blocked-20260902`.
REPORTED-ONLY: nothing here is wired into the pipeline, no eval file or
scorer was touched, and no outcome is pinned.** Pre-registration is the
ledger entry of the same date, committed before the script existed.
Script: `scripts/ap1-all-pairs-separator.py`. Raw results:
`ap1-all-pairs-separator.json`.

---

## 1. What was tested

`pulse-next-step.md` §3 named the defect: `calculate_tempo` medians the
gaps between **consecutive** events, and on the demos peakRate fires 2.6
events per beat at a ratio that is non-integer and varies inside the clip.
Every adjacent gap is corrupted and no ×/÷{2,3} factor recovers the beat.
The beat-to-beat distance nevertheless survives in the set of **all
pairwise** distances, which no code in the repo looks at.

Three arms, same peakRate event stream (media checksum-verified), same
band projection, same 34-row gating set, same ±8 % pass rule. **The
estimator is the only thing that varies.**

| arm | what it does |
|---|---|
| **A** control | median of consecutive IOIs — SW-1's published whole-clip control |
| **B1** primary | all pairwise distances ≤ 3 s → kernel histogram (σ 40 ms) → harmonic sum Σ H(kτ)/k, k ≤ 4 → argmax over τ ∈ [0.20, 1.20] s |
| **B2** secondary | comb/latent-grid score over (period, phase), σ 70 ms, minus a uniform-random null |

## 2. Result

```
arm                                    pass   demo     rig   Acc2  btwn  btwnD    odd   even    gap
A  median-consecutive (CONTROL)     16/34    4/8    12/26     16    21      5   9/17    7/17  0.118
B1 all-pairs harmonic (PRIMARY)     29/34    5/8    24/26     30     6      4  16/17   13/17  0.176
B2 comb null-subtracted             28/34    4/8    24/26     28     6      4  14/17   14/17  0.000
```

**Control reproduces SW-1 exactly** (16/34 · 4/8 · 12/26 · Acc2 16 ·
between-levels 21), which was a pre-registered requirement — the harness
is the same instrument that produced the published control.

Median absolute tempo error, all 34 rows: **8.6 % → 0.9 %**. On the eight
demos: 7.5 % → 5.9 %.

## 3. The headline is the rig half, and the rung is the demos

**Read this before quoting the 29/34.** The win is almost entirely on the
26 rig clips — the owner counting in his own voice — where the pass rate
goes **12/26 → 24/26** with **zero losses**. On the eight barre-6 demos,
the actual target of step one, it is **4/8 → 5/8**, and that single net
clip does not survive scrutiny intact:

| demo | truth | A | B1 | verdict |
|---|---|---|---|---|
| coupé-barre | 108 | 105.5 ✓ | 109.9 ✓ | held |
| dégagé | 110 | 103.8 ✓ | 113.2 ✓ | held |
| tendu | 102 | 105.8 ✓ | 98.0 ✓ | held |
| fondu | 86 | 96.7 ✗ | 92.6 ✓ | **genuine win** |
| rond de jambe | 96 | 107.7 ✗ | 100.0 ✓ | **LUCK-FLAGGED — see below** |
| plié | 120 | 108.8 ✗ | 100.0 ✗ | still wrong, boundary hit |
| frappé | 135 | 119.7 ✗ | 71.9 ✗ | estimator right, projection wrong |
| tendu-warmup | 112 | 110.5 ✓ | 126.1 ✗ | **genuine-trade LOSS** |

**Luck flag, declared.** B1's rond-de-jambe pass is a **search-boundary
artifact**. Exactly two of 34 clips chose a period at the edge of the
search range, and they are the two sparsely-voiced demos: plié and rond
de jambe both pinned τ = 0.20 s, the fast ceiling, i.e. the harmonic sum
never found an interior maximum at all. Both then projected by ⅓ to
exactly 100.0 BPM. Plié's truth is 120 (fails); rond de jambe's is 96, so
100.0 lands 4.2 % out and **passes for a reason that has nothing to do
with finding the period.** One clip of the eight-clip demo half is a
coin landing well.

**The loss is real.** `tendu-warmup` was A's 110.5 against a truth of 112
— a 1.4 % green — and B1 reads 126.1, 12.5 % high and not an octave
relative of anything. Classified **genuine-trade** under ADR-015.

**So the handoff's §4 account survives this experiment.** The two clips
it called "reconstruction" — where the teacher voices only beats 1 and 3
and the beat can fall where nothing sounds — are precisely the two where
all-pairs fails to find any interior period, and the third sparsely-voiced
clip (tendu-warmup, §4's own honest counterexample) is the one this arm
breaks. Period-fitting on the event train does not solve the sparse mode.
Latent-grid inference is still the open question there.

## 4. Two failures that belong to the projection rule, not the estimator

Of B1's five remaining misses, two are the estimator being right and the
**hard band edge** throwing the answer away:

- **frappé**: B1's raw estimate is **143.88 BPM against a truth of 135 —
  6.6 % off, inside tolerance**. It is 2.8 % above the band ceiling of
  140, so factor 1.0 is refused, and the next factor that lands in band is
  ½ → 71.94. A correct reading was halved by the projection.
- **`rig-numbers-4-4-80-triplet` (→ 119.5 vs 80) and
  `adr006-8-counts-triple` (→ 100.7 vs 68)** are both triplet clips where
  B1 locks a level 1.5× from truth. The factor set is {1, 2, ½, 3, ⅓} and
  contains no 3/2 or 2/3, so the projection **cannot** correct a
  three-against-two level confusion by construction.

Both are Standing Lesson 2's territory (a prior applied as a hard fold
rather than at level selection) and neither is fixed by touching the
front end. **Not tuned here** — naming a fix after seeing which fix would
have helped is not a prediction. They are stated as pre-registerable next
tests.

## 5. Confound checked, not assumed

The rig clips are the owner counting **against a metronome**, so a
periodicity method could in principle be finding the click rather than the
voice. It is not: the case notes record the metronome as
*"metronome-locked at 120 in one earbud"* — it was in his ear, never in
the room or the recording. The 12/26 → 24/26 is voice.

## 6. Prediction scorecard — 3 of 7 landed

| # | prediction | outcome |
|---|---|---|
| P1 | B1 demo ≥ 6/8 | **MISS** — 5/8 |
| P2 | B1 rig ≥ 12/26 (no regression) | **HIT** — 24/26 |
| P3 | B1 total ≥ 20/34 | **HIT** — 29/34 |
| P4 | between-levels ≤ 14 total **and** ≤ 2 on demos | **SPLIT** — total 6 (hit), demo 4 (miss) |
| P5 | plié **and** rond de jambe both flip to pass | **MISS** — plié no; rond yes but by boundary luck (§3) |
| P6 | B2 ≤ B1 overall, ≥ B1 on the two sparse demos | **SPLIT** — 28 ≤ 29 (hit); 0/2 vs 1/2 on sparse (miss) |
| P7 | B2 degeneracy guard: < 25 % at the ceiling | **HIT** — 11.8 %, so B2 is interpretable |

Scored strictly: a two-clause prediction with one clause failing is not a
hit. **The direction of the effect was right and its location was wrong** —
the pre-registration bet the fix would show up on the demos and it showed
up on the rig clips instead.

## 7. What this does and does not establish

**Does:** on this corpus, replacing the median of consecutive gaps with an
all-pairs harmonic sum — one function, same input, same band rule — takes
the estimator-level pass rate from 16/34 to 29/34 and cuts median tempo
error from 8.6 % to 0.9 %, with a single classified loss. The mechanism
named in the handoff §3 is real and the remedy Standing Lesson 3 names
works on it.

**Does not:**

- **This is not the shipping path.** Arm A is peakRate + median +
  projection; `analyze()` commits through `normalize_tempo`, the posterior
  and arbitration. The blessed baseline's tempo 0.606 (20/34) is a
  different quantity from A's 16/34, and **no number here may be quoted as
  a baseline delta.** What an adoption would actually move is unmeasured.
- **It does not move the rung's target much.** 4/8 → 5/8 on demos, one of
  which is luck-flagged and one clip lost. n = 8.
- It does not touch meter, structure or style, and it is not the
  exercise-prior work — that remains gated on the owner's blind table,
  untouched by this.

## 8. Cheapest next tests, none of them run here

1. **Band-edge tolerance.** Allow a small slack at 70/140 before refusing
   factor 1.0. frappé alone would flip. Pre-register the slack before
   running it.
2. **The 3/2 level.** Add 3/2 and 2/3 to the projection factors and score
   the two triplet clips as must-move rows.
3. **The sparse mode.** B1 and B2 both fail where the teacher voices beats
   1 and 3 only. That is the latent-grid problem `posterior.py` already
   has the shape for (ADR-017), and it is the honest reading of §4.
4. **Adoption, if the owner wants it**, is its own increment with a
   typed gate on the shipping path — not this artifact.
