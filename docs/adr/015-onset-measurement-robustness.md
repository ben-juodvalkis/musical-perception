# ADR-015: Onset Measurement Robustness — Grid-Fitting the Inter-Onset Intervals

**Date:** 2026-08-09
**Status:** Accepted — the pre-registered kill criterion tripped, and the
owner overrode it; the override and its reasons are on the record below
(2026-08-09)

## Context

[ADR-014](014-tempo-metric-level-ambiguity.md)'s measured-results section split
the 13 tier-1 tempo failures into two kinds and left the larger one open:

> 3 of 13 tempo failures are *selection* failures — the measurement was right
> and the band prior discarded it […]. The other 10 are *measurement*
> failures: the truth is nowhere in the family.

Selection failures are ADR-014's territory (an exercise-type prior over
`alternates`) and are explicitly **not** touched here. This ADR is about the
other ten: cases where `detect_onset_tempo()` (`precision/rhythm.py`) never
measured the right pulse in the first place, so no amount of smarter
level-selection downstream can rescue them.

`detect_onset_tempo()` slides 3-second windows over word onsets, keeps the
windows whose inter-onset intervals (IOIs) have CV < 0.4, and reports
`60 / mean(window IOIs)` per window, aggregated by a duration-weighted median.
**The per-window mean is the defect.** It assumes every IOI spans exactly one
beat and that all of them are drawn from one distribution. Two documented
phenomena break that assumption:

**(a) Agogics.** Expressive bar-boundary lengthening produces isolated long
IOIs inside an otherwise steady window. On `rig-numbers-6-8-100-clean`
(capture-checklist clip 4) the speaker's true word rate was 96.3 — the trace's
own IOIs prove it, nine of them clustered at 0.58–0.68 s — but four
bar-boundary gaps at 0.88–0.94 s pulled the window means down to a reported
84.7. The case's YAML notes already named the fix: "dominant-cluster IOI
instead of global median."

**(b) Sparsity / mixed multiples.** Step-name marking speaks on *some* beats
only, so the IOI set is a mixture of 1×, 2× and 3× the beat period. A single
mean lands between metric levels. `rig-names-3-4-90-clean` (clip 6) is the
clean example: 7 IOIs at 0.66 s (= 1× the true 0.667 s beat) and 4 at ~1.30 s
(= 2×), which the mean renders as 65.1 — neither level.

Both are *measurement* problems inside one metric level's worth of evidence.
They are not the level-ambiguity problem ADR-014 describes, and not the
signal-precedence problem [ADR-013](013-tempo-arbitration.md) solved.

## Pre-registered classification of the 13 tier-1 tempo-wrong rows

Classified from each case's **frozen trace** (`evals/traces/<id>/whisper.json`)
before writing any code, by comparing the word-onset IOIs against the beat
period *P* = 60 / label. "grid support" counts IOIs within ±10% of 1×, 2× or
3× *P*. Baseline is run `20260809T035442Z` (`evals/baseline.json`).

| # | case | clip | truth | reported | classification | evidence |
|---|---|---|---|---|---|---|
| 1 | `rig-numbers-6-8-100-clean` | 4 | 100 | 84.7 | **cluster-fixable** | 9 IOIs at 1×*P*, 4 agogic gaps at ~1.5×*P*; median IOI already 0.623 s → 96.3 |
| 2 | `rig-names-3-4-90-clean` | 6 | 90 | 130.2 | **grid-fixable** | 7 IOIs at 1×*P*, 4 at 2×*P*, 1 at 3×*P*; a competing off-grid cluster at 1.3×*P* |
| 3 | `rig-mixed-4-4-104-quantities` | 22 | 104 | 85.0 | **cluster-fixable (weak)** | dominant cluster at ~0.29 s = ½*P* (13 IOIs) with 9 at 1×*P*; a half-beat reading would re-normalize into band |
| 4 | `frappe` | — | 160 | 74.3 | **selection** (ADR-014) | `truth_in_family=True` (148.6 at ×2) |
| 5 | `rig-names-2-4-160-long` | 13 | 160 | 80.9 | **selection** (ADR-014) | onset path already reads 161.8; only the 70–140 band halves it |
| 6 | `rig-numbers-4-4-60-halftempo` | 12 | 60 | 124.4 | **selection** (ADR-014) | onset path already reads 62.2; band doubles it |
| 7 | `rig-names-4-4-63-adagio` | 14 | 63 | 92.2 | **segmentation-loss** | 19 words for 32 beats; legato delivery, only 12 usable IOIs, section tempos 92/54/94 |
| 8 | `rig-names-4-4-96-allegro` | 21 | 96 | 115.1 | **segmentation-loss / drift** | first section is CV = 0.00 *at* 115 — full-voice marking genuinely rushing the click |
| 9 | `rig-names-4-4-100-quiet` | 23 | 100 | 119.3 | **segmentation-loss** | 12 Whisper tokens for a 16-count phrase; wholesale word loss |
| 10 | `rig-names-2-4-120-clean` | 5 | 120 | 77.8 | **other** | IOIs bimodal at 0.30 s and 0.70 s; exactly **one** IOI within 10% of *P* — the surviving onsets are not on the beat grid |
| 11 | `rig-names-6-8-100-clean` | 8 | 100 | 134.6 | **other** | grid support at 1×/2×/3× *P* is **zero**; only the bar spacing (~3.6 s = 6×*P*) is on-grid, outside any 1–3 fold |
| 12 | `rig-names-4-4-104-coda` | 24 | 104 | 72.2 | **other** | dominant cluster is 0.80 s = 1.39×*P* (10 IOIs) vs 6 at 1×*P* — descriptive step-name phrasing is genuinely slower than the click |
| 13 | `adr006-8-counts-2x` | — | 130 | 100.3 | **other** | all 15 IOIs at 0.60 ± 0.06 s → a spoken word rate of 99.5; nothing near *P* = 0.462 s at any fold |

Totals: 2 cluster-fixable (+1 weak), 1 grid-fixable, 3 selection, 3
segmentation-loss, 4 other.

**Segmentation losses are out of scope and are expected to stay red.** Clips
14, 21 and 23 fail because Whisper did not produce the words — legato vowels
with no boundary, a rushed full-voice delivery, and a quiet mumble that yielded
12 tokens for a 16-count phrase. That is a perception-layer defect. Nothing
`rhythm.py` does to intervals it never received can fix it, and a change that
*did* flip those rows would be measuring luck.

## Decision

Replace the per-window **mean IOI** with a **grid fit**: assume the window's
onsets sit on an integer grid of one beat period, and estimate that period by
counting how many beats the window's elapsed time spans.

```
m  = median(window IOIs)              # anchor; robust to one long gap
k  = clip(round(x / m), 1, 3)         # beats each IOI spans
ok = |x - k·m| <= tol · k · m         # a gap that fits no k is an outlier
period = sum(x[ok]) / sum(k[ok])      # elapsed time / beats spanned
```
refined twice, with `tol = 0.20`.

Parameters, fixed on principle before measuring — not tuned:

- **`tol = 0.20`.** The widest tolerance that keeps the 1× acceptance band
  ([0.8*m*, 1.2*m*]) disjoint from the 2× band ([1.6*m*, 2.4*m*]) with margin,
  and about the size of real expressive timing deviation. An IOI in the dead
  zone between them is an agogic gap and is dropped, not averaged in.
- **folds 1–3.** The same integer set ADR-014's metric-level family already
  uses. A teacher who speaks on every 4th beat is out of scope.
- **median anchor.** The mean is the quantity being corrected, so it cannot
  also be the reference the correction is measured against.

Confidence stops being a pure regularity score and gains a **support** term —
the fraction of a window's IOIs the fitted grid actually explains — so a sparse
reading that happens to be internally tidy is not spuriously confident.

**Deliberately unchanged:** `calculate_tempo()`, `normalize_tempo()` and the
70–140 band, `interpret_meter()` and ADR-013's arbitration thresholds,
ADR-014's `tempo_family()`, the window/step/CV-gate parameters (so which
stretches count as rhythmic, and `rhythmic_coverage`, are untouched), and every
`evals/cases/*.yaml` label.

## Pre-registered expectations

**Expected to flip to correct:**

| case | why |
|---|---|
| `rig-numbers-6-8-100-clean` | agogic gaps become outliers; the 1× cluster carries the estimate (high confidence) |
| `rig-names-3-4-90-clean` | the 2× IOIs fold into the 0.667 s base instead of averaging against it (medium confidence) |
| `rig-mixed-4-4-104-quantities` | the half-beat cluster dominates and re-normalizes into band (low confidence — its onsets are free-form, and this row could equally stay red) |

**Expected to stay red, and why:** clips 14 / 21 / 23 (segmentation loss — no
usable intervals to fit); clips 5, 8, 24 and `adr006-8-counts-2x` (the truth is
not present in the onsets at any fold of the grid — see the evidence column);
`frappe`, clip 12 and clip 13 (selection failures — the measurement is already
right, so a measurement change is the wrong instrument).

**Kill criteria — any one of these stops the change:**

1. **Zero outcome regressions.** Any case, any field, any tier going
   `correct → wrong` or `correct → abstained` kills it. Improvements are
   expected and get re-blessed.
2. **Tier-0 tempo stays 25/25** and its known-failure set stays exactly
   `{t0-3-4-half}`.
3. Any flip not on the list above must be individually explained in the
   measured-results section from its trace, not waved through.

`frappe` is the row to watch for a benign-but-reportable move: its onset
reading sits at a subdivision level, so grid folding could shift it to a
different metric level. If that happens it is a **measured outcome to report**,
never something to tune toward.

## Consequences

- `rhythm.py` stops assuming one onset per beat — the assumption that
  silently produced both documented failure modes.
- `RhythmicSection.cv` changes meaning from "how equal were the IOIs" to
  "how well do they fit the grid". Same field, same scale, strictly more
  informative; `types.py` is untouched.
- The tier-0 synthetic suite's `drop_rate` knob already generates the sparse
  case; combos pinning the new behaviour at sweep level are added only if the
  existing ones do not exercise it.

## Measured results (2026-08-09) — **kill criterion tripped, not blessed**

**What shipped in the working tree.** `_grid_period()` and
`_refit_sections()` in `precision/rhythm.py`, plus a grid-support term in
`_compute_confidence()`. The window sweep and the merge are byte-identical
to before: they decide *where* speech is rhythmic, which was never the
defect. What changed is that each merged section's tempo is now measured
once, over every onset the section contains, instead of being inherited from
whichever 3-second window inside it won the lowest-CV merge. That
restructuring was not in the original design — it landed because a
window-local fit over 4–6 intervals proved to be the dominant source of
variance, and it is what made the estimator's behaviour reproducible enough
to reason about at all.

`pytest -q`: 173 passed, 3 skipped; the only failure is the tier-1 baseline
gate, which is the change asking to be blessed.

### Aggregate

| suite · field | before | after |
|---|---|---|
| tier0 tempo | 25 / 0 / 0 | **25 / 0 / 0** ✓ |
| tier0 meter_triple | 24 / 1 / 0 | 24 / 1 / 0 (known-failure set still `{t0-3-4-half}`) ✓ |
| tier1 tempo | 15 / 13 / 1 | **16 / 12 / 1** |
| tier1 meter_triple | 10 / 18 / 1 | 10 / 18 / 1 |
| tier1 counts | 11 / 9 / 8 | 12 / 9 / 7 |
| tier1 truth_in_family | 3 / 13 | 3 / 12 |
| tier1 ECE | 0.401 | **0.291** |

(correct / wrong / abstained.) The ECE move is the confidence change paying
for itself: the same answers, better calibrated.

### Per-case flips

| case | field | before | after | pre-registered? |
|---|---|---|---|---|
| `rig-numbers-6-8-100-clean` | tempo | wrong 84.7 | **correct 95.2** | ✅ yes, high confidence |
| `rig-numbers-6-8-100-clean` | meter_triple | wrong | **correct** 6/8 @95.2 | follows the tempo |
| `rig-names-4-4-100-quiet` | tempo | wrong 119.3 | **correct 95.8** | ❌ pre-registered to stay red |
| `rig-names-4-4-96-allegro` | tempo | wrong 115.1 | **correct 99.1** | ❌ pre-registered to stay red |
| `rig-names-4-4-96-allegro` | counts | abstained | correct 32 | — |
| `rig-names-3-4-88-waltz` | counts | wrong 24 | abstained | — (abstention is not a regression, ADR-009) |
| `rig-numbers-4-4-104-explained` | counts | abstained | **wrong 24** | ❌ |
| `adr007-plies-demo` | tempo | **correct 127.4** | **wrong 88.0** | ❌ **regression** |
| `adr007-plies-demo` | meter_triple | **correct** | **wrong** | ❌ follows the tempo |
| `rig-names-3-4-88-waltz` | tempo | **correct 91.6** | **wrong 102.6** | ❌ **regression** |

Two pre-registered flips did not happen. `rig-names-3-4-90-clean` stays wrong
because the stretch that dominates its duration genuinely speaks at ~0.9 s
intervals (0.86 / 0.98 / 0.90 / 0.88 / 0.92 / 1.02 / 1.00) — that is 1.3× the
labelled beat, not a fold of it, so the grid has nothing to fold.
`rig-mixed-4-4-104-quantities` flipped correct in intermediate variants and
does not in the shipped one; its onsets are free-form and it was registered
at low confidence for exactly that reason.

### The two regressions, explained

**`adr007-plies-demo`** — this clip has no coherent onset tempo to measure.
Its eight rhythmic sections read 127.4, 262.8, 59.5, 255.3, 176.1, 64.6,
207.1 and 70.1 BPM, over 46.8% coverage: a real class video where marking,
demonstration and continuous explanation alternate. The old answer (127.4)
was the first section's; the new one (88.0) is the duration-weighted median
across all of them. Neither measures anything — the previous `correct` was
the arbitrary draw landing inside an 8% window, and the change reshuffled
the draw. Its confidence (0.52 → 0.56) does not warn about this, which is
its own finding.

**`rig-names-3-4-88-waltz`** — a genuine disagreement, and the grid loses.
Its second section's intervals are `[0.76, 0.34, 1.14, 0.54, 0.46, 1.10,
0.56, 0.58]`. The grid reads a 0.55 s beat with two two-beat gaps (1.14 and
1.10 fit at ±3%) and discards 0.76 and 0.34 as fitting nothing → 109.6 BPM.
The mean reads 0.685 s → 87.6, which is the label. On a swung waltz, where
"ONE-and-ah" syllables sit between beats, treating a long interval as *one
stretched beat* happens to be right and treating it as *two beats* is wrong —
and nothing in the interval pattern distinguishes those two stories. This is
the same evidence-does-not-decide situation ADR-014 documented for metric
level, one level down.

### Verdict

The pre-registered kill criterion was **zero outcome regressions anywhere**.
Two tempo rows went `correct → wrong`, so by the rule written before the
code, this change is not blessed and the baseline is not moved. The net
arithmetic is favourable (+1 tempo, +1 counts, ECE 0.40 → 0.29, tier-0
intact) and both regressions are explained rather than mysterious — but
"favourable net" was explicitly not the bar, precisely so that a change
cannot be argued through on aggregates.

What the measurement establishes regardless of the verdict:

1. **The agogic half of the diagnosis is confirmed and fixable.**
   `rig-numbers-6-8-100-clean` flipped exactly as pre-registered, for exactly
   the pre-registered reason, and took its meter triple with it.
2. **The mixed-multiples half is not.** Neither of the two rows classified
   grid-fixable or cluster-fixable-weak flipped in the shipped variant, and
   the folding that would flip them is the same mechanism that breaks the
   waltz. On this corpus, folding an interval as two beats is as often wrong
   as right.
3. **Several tier-1 tempo rows are decided by noise, not by measurement.**
   `adr007-plies-demo` (8.0% error before), `rig-names-3-4-88-waltz` (4.1%),
   `rig-names-4-4-104-clean` (7.6%) and `adr006-exercise-1-demo` (5.3%) all
   sit within a few points of the 8% tolerance, and every variant tried
   during this work reshuffled which of them were green. Any future onset
   change will keep tripping a zero-regression gate on these rows. That is a
   fact about the corpus, not about this change, and it belongs in the
   record: the honest instruments are the aggregate and the slice, and the
   corpus is too small (n = 29) for either to move meaningfully.
4. **Two of the three segmentation-loss rows flipped green anyway** — clips
   21 and 23, pre-registered to stay red. That is not a refutation of the
   segmentation diagnosis; both clips still lose words. It means the
   surviving intervals happened to fit a grid near the label. Counting them
   as wins would be counting luck.

Left for Ben to decide: whether the trade is worth taking, or whether the
onset path needs the phase-aware grid inference (fit onset *positions* to a
grid with a penalty for unused beats, rather than fitting intervals) that
clip 5's notes have been asking for since the capture programme started.

## Resolution — override, on the record

Ben accepted the trade (2026-08-09), after the diagnoses were verified
independently of the implementing session (suites re-run; the plies-demo
section spread re-derived from its frozen trace; one additional adverse
flip found that the summary totals had absorbed). The kill criterion
above stands exactly as written and as tripped — this section is an
owner override, not a retroactive edit of the gate.

**The verified ledger.** Gains: clip 4 tempo *and* its 6/8 meter triple
(the pre-registered flip, for the pre-registered reason); allegro tempo
+ counts; quiet tempo (both flagged luck, below). Losses: plies-demo
tempo + meter triple — a **fake green lost**: its eight "rhythmic"
sections read 59.5 / 64.6 / 70.1 / 127.4 / 176.1 / 207.1 / 255.3 /
262.8 BPM at 47% coverage, so no measurement ever existed and the old
correct was a dart landing in the 8% window; waltz-lilt tempo — a
**genuine trade**: its gap evidence is truly mixed (some gaps ≈2.0× the
base period, some ≈1.4×), and interval data alone cannot distinguish a
stretched beat from two beats. Also named here because the totals hid
it: `rig-numbers-4-4-104-explained.counts` went abstained→wrong (the
grid-shifted BPM moved a span vote and the estimator now commits where
it held back) — an adverse flip; and waltz-lilt counts went
wrong→abstained, which the abstention policy counts as an improvement.
Calibration improved substantially: tier-1 ECE 0.401 → 0.291.

**Prediction scorecard, honestly:** 1 of 3 pre-registered flips landed
(clips 6 and 22 did not move); 2 unpredicted greens were disclaimed as
luck by the implementing session itself.

**Luck annotations:** clips 21 and 23 are marked in their case notes.
When improved word recovery someday flips them back to red, that flip
is the luck draining out — it must not be treated as a regression.

**Gate amendment for future work (typed gates).** Zero-regression
remains the gate for *logic* changes (ADR-013-class, fully analyzable).
*Measurement* changes on this small corpus get the diagnosed-regression
gate instead: (a) net improvement on the primary metric AND on
calibration (ECE); (b) zero **undiagnosed** regressions; (c) every
regression classified fake-green-lost / genuine-trade / knife-edge;
(d) rows within a few points of the 8% tolerance cannot gate anything
until the corpus grows — at n = 29 they are decided by noise (finding
3 above), which is rule 8 of ADR-009 applied to acceptance gates, the
same lesson ADR-011 recorded for prompt-change gates.

**Follow-ons logged, deliberately not bundled:** phase-aware grid
inference (fit onset positions with an unused-beat penalty — clip 5's
standing request); a marker-anchored grid using word identity to split
lilt from sparsity (the waltz discriminator — beat words and "and"s are
already distinguished by the marker path); the counts estimator's
sensitivity to bpm-vote drift (the explained flip).
