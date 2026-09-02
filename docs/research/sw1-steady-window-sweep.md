# SW-1 — the steady-window sweep

**Commissioned** 2026-09-01 (owner, late evening), search space frozen in
that ledger entry. **REPORTED-ONLY: the winner is not adopted.** Nothing
under `src/musical_perception/` changes; no file under `evals/` is created
or modified. Run 2026-09-02 on the Air, branch `agent/sw1-pr1-air`.

---

## Part 1 — PRE-REGISTRATION

**This section was committed before the sweep script existed.** `git log`
on this branch shows the order.

### The question

Step one asks for the pulse's tempo at the metric level inside 70–140 BPM.
The shipping pipeline reads the whole clip. The owner's idea is that a
human does not: he finds a stretch where the teacher is *steady* — marking
in time rather than explaining — reads the tempo there, and ignores the
rest. This sweep measures whether picking one steady window beats reading
the whole clip, across two pulse sources and three window lengths.

### Search space (frozen at commissioning; nothing added or removed after the first scoring run)

- **Pulse sources (2).**
  - `peakrate-media` — the rung-2 peakRate extractor run on the clip's
    own audio (`PeakRateParams` defaults, voiced-gated). Media is
    checksum-verified against the trace's `media_sha256`; a missing or
    mismatched file is **skipped by name** and per-source coverage is
    stated.
  - `whisper-trace` — word onsets from the frozen trace's
    `whisper.json`. No media needed.
- **Window lengths (3).** L ∈ {3 s, 5 s, 8 s}, slide step 0.5 s.
- **Window pick (1 rule).** The window with the **minimum within-window
  IOI coefficient of variation**, requiring **≥ 6 events** inside it. If
  no window qualifies, fall back to the whole clip and **report the
  fallback by name**.
- **Tempo in window (1 rule).** `60 / median IOI`, then projected into
  [70, 140] by ×/÷{2, 3}. The chosen factor is **reported per clip** —
  never a silent fold (Standing Lesson 2). If no factor lands in band,
  the row abstains.
- **Controls.** The same tempo rule over the whole clip, per source.
- **Ceiling (reported, never a candidate).** Oracle windows from the
  eight demo cases' "Intended-tempo span" notes — the span in which the
  owner says the tempo was knowable. For rig clips the oracle window is
  the whole clip, so the oracle differs from the control on the demo
  slice only.

### Population

The **34-row step-one gating set** (26 rig/counting + 8 owner-tapped
barre-6 demos), read from `evals/cases/` as every non-`reference` case
with `maturity: verified`. `barre6-ballonne-demo` is deferred and absent
by construction.

Coverage facts established before the run, so they cannot be discovered
conveniently afterwards:

- **Media: 34 of 34 present** on this machine. Checksums are still
  verified per clip and any failure is named.
- **`rig-vocables-4-4-100-clean` carries 1 Whisper word.** The
  `whisper-trace` source has essentially no events on it and will fall
  back and then abstain. Named now.
- **`adr006-8-counts-triple`'s truth is 68.38 BPM — below the band.** The
  projection rule cannot emit a value under 70, so this row **cannot pass
  under any variant**. It is a structural zero, stated now, not a finding.

### Metrics per variant

- **Step-one pass** — committed pulse within **±8 %** of the in-band truth
  (`expected_bpm`), the charter's pre-registered criterion.
- **Acc2@8%** — within ±8 % of any {⅓, ½, 1, 2, 3}× the truth, using the
  repo's own `aggregate.acc2` definition.
- **Between-levels count** — the repo's definition: |OE2| in (0.08, 0.585].
- **Split-half stability** — the split is **FIXED NOW as odd/even rows of
  the case ids sorted lexically** (rows 1, 3, 5 … vs 2, 4, 6 …). A winner
  must win on both halves.

### Selection rule

Rank by **stability first**, then **demo-slice pass count**, then **total
pass count**. **The winner is not adopted.** The deliverable is this
comparison table plus the scorecard below.

### Pre-registered predictions

| # | prediction | reason |
|---|---|---|
| **S1** | No variant beats the blessed baseline's step-one pass rate of 0.606 (20 of 34) on the full set. | The shipping path selects the metric level by MAP under a log-normal prior (W9). A bare ×/÷{2,3} projection has no such arbitration; a better *window* cannot buy back a worse *level rule*. |
| **S2** | On the 8-demo slice, at least one window variant beats the whole-clip control **of the same source** by ≥ 2 clips. | The demo is interrupted speech — explanation, then marking. Reading the whole clip averages the explanation in; a steady window is exactly the fix. This is the sweep's reason to exist. |
| **S3** | `peakrate-media` beats `whisper-trace` on the demo slice at **every** window length. | Rung 2 and the 2026-09-01 stage1 finding: word starts collapse on demo material (per-demo pulse F 0.09–0.48). |
| **S4** | `whisper-trace` beats **or ties** `peakrate-media` on the 26-row rig slice. | Rig clips are clean counted speech at the beat; peakRate additionally fires on sub-beat syllables, which median-IOI cannot un-mix. |
| **S5** | Source matters more than window length: the spread of total pass counts across L within a source is **smaller** than the spread across sources at fixed L. | Where the events come from is a bigger lever than how much of the clip you look at. |
| **S6** | The demo-slice oracle ceiling exceeds the best measured window variant by ≥ 2 clips. | If minimum-IOI-CV found the owner's window reliably there would be nothing left to research; predict it does not. |
| **S7** | Split-half instability is real at this n: the best variant's pass rate differs between the odd and even halves by **> 0.15** (≈ 3 of 17 rows). | 34 rows split 17/17. This prediction exists to make the stability number interpretable rather than decorative. |
| **S8** | For `peakrate-media`, the projection factor is **not 1** on more than a third of clips. | peakRate fires at the syllable rate, which sits above the tactus on most of this corpus. |
| **S9** | `adr006-8-counts-triple` fails under every variant. | Structural, per the coverage note above. |
| **S10** | Containment: `git diff --stat origin/main` shows only `docs/research/`, `scripts/`, and the ledger; pytest green. | — |

Late-added measurements, if any, are disclosed in Part 2 in the
W2-reopen style: what was added, when, and whether it was added after
seeing a number point the wrong way.

---

## Part 2 — RESULTS (run 2026-09-02, Air, `agent/sw1-pr1-air`)

**Headline: the steady window is not the win, but one of its variants is —
and not for the reason the idea was proposed. Reading a 5-second stretch of
peakRate events instead of the whole clip gets the tempo right on 23 of the
34 gating rows against the shipping pipeline's 20, but it does that entirely
on the rig clips (21 of 26 vs 12), and on the eight demos — the reset's whole
target — every window variant is level with or worse than just reading the
whole clip. The owner's own "I knew the tempo by here" spans are a *worse*
window than the algorithm's: the oracle that was supposed to be a ceiling
scores below the thing it was ceiling.**

Artifacts: `scripts/sw1-steady-window-sweep.py` (read-only, ~4 min),
`docs/research/sw1-steady-window-sweep.json` (per-clip, per-variant windows,
raw and projected BPM, factor, OE1/OE2, pass, halves).

**Coverage: 34 of 34 rows on both sources, 0 skipped, 0 checksum
mismatches.** Every media file was hashed against its trace's
`media_sha256` before peakRate read it.

### The comparison table

| variant | pass /34 | demo /8 | rig /26 | Acc2@8% | between-lvl | abstain | fallbacks | odd /17 | even /17 | half-gap |
|---|---|---|---|---|---|---|---|---|---|---|
| peakrate-media · window 3 s | 19 | **4** | 15 | 19 | 19 | 0 | 0 | 9 | 10 | **0.059** |
| peakrate-media · window 5 s | **23** | 2 | **21** | 23 | 17 | 0 | 0 | 12 | 11 | **0.059** |
| peakrate-media · window 8 s | 18 | 3 | 15 | 18 | 18 | 0 | 1 | 11 | 7 | 0.235 |
| peakrate-media · whole-clip CONTROL | 16 | **4** | 12 | 16 | 21 | 0 | — | 9 | 7 | 0.118 |
| peakrate-media · ORACLE CEILING | 14 | 2 | 12 | 14 | 22 | 0 | — | 8 | 6 | 0.118 |
| whisper-trace · window 3 s | 18 | **4** | 14 | 19 | 18 | 1 | 7 | 12 | 6 | 0.353 |
| whisper-trace · window 5 s | 18 | 1 | 17 | 18 | 20 | 1 | 1 | 11 | 7 | 0.235 |
| whisper-trace · window 8 s | 15 | 3 | 12 | 16 | 19 | 1 | 1 | 8 | 7 | **0.059** |
| whisper-trace · whole-clip CONTROL | 17 | **4** | 13 | 17 | 18 | 1 | — | 11 | 6 | 0.294 |
| whisper-trace · ORACLE CEILING | 16 | 3 | 13 | 16 | 20 | 1 | — | 12 | 4 | 0.471 |

Blessed baseline for reference (shipping pipeline, same 34 rows):
**20 pass, Acc2@8% 0.697, between-levels 10 of 33 committed.**

### Selection, applied as pre-registered

Rank by stability, then demo passes, then total. Three variants tie on the
stability gap at 0.059, so the tie-breaks decide:

1. **`peakrate-media · window 3 s`** — gap 0.059, demo 4/8, total 19/34 ← **winner**
2. `whisper-trace · window 8 s` — gap 0.059, demo 3/8, total 15/34
3. `peakrate-media · window 5 s` — gap 0.059, demo 2/8, total 23/34

**The winner is NOT adopted.** And the rule's own output argues against
adopting it: the variant that wins the pre-registered ranking is *not* the
variant with the most correct answers. The 5-second variant gets 4 more rows
right and loses the tie-break on the demo slice, which is 8 rows wide. That
is a fact about the selection rule at n = 34, and it is exactly the kind of
thing a REPORTED-ONLY increment exists to surface before anything is wired in.

### Prediction scorecard — 2 hits, 6 falsified, 2 structural

| # | prediction | outcome |
|---|---|---|
| S1 | no variant beats the baseline's 0.606 (20/34) | **FALSIFIED** — `peakrate-media · 5 s` gets 23/34 (0.676), +3 rows. A bare median-IOI rule in a well-chosen window beats the shipping path's whole-clip read on this set |
| S2 | some window beats its own whole-clip control on demos by ≥ 2 | **FALSIFIED, and this is the one that mattered** — best demo window is 4/8, identical to both whole-clip controls. No window variant beats its control on the demo slice by even one clip |
| S3 | peakRate beats Whisper on demos at every L | **FALSIFIED** — 4 vs 4 at 3 s (tie), 2 vs 1 at 5 s (beats), 3 vs 3 at 8 s (tie) |
| S4 | Whisper ≥ peakRate on the rig slice | **FALSIFIED** — peakRate wins at every window length (15/14, 21/17, 15/12); only the whole-clip control matches the prediction (13 vs 12) |
| S5 | source matters more than window length | **FALSIFIED** — mean spread across L within a source is 4.0 rows; across sources at fixed L, 3.0. Small numbers, stated as such |
| S6 | oracle ceiling exceeds the best measured window on demos by ≥ 2 | **FALSIFIED in the opposite direction** — oracle 2/8 (peakRate) and 3/8 (Whisper) vs 4/8 measured. See F3 |
| S7 | winner's half-gap > 0.15 | **FALSIFIED** — winner's gap is 0.059 (9 vs 10 of 17). But three variants tie at that value, so the stability criterion barely discriminates |
| S8 | peakRate projection factor ≠ 1 on > ⅓ of clips | **HIT** — 17/34 (3 s), 12/34 (5 s), 13/34 (8 s); the 5 s figure clears ⅓ by one row, so it is a narrow hit |
| S9 | `adr006-8-counts-triple` fails everywhere | **HIT** — 10 of 10 variants fail it; every one projects to 96–124 BPM against a 68.38 truth |
| S10 | containment | **HIT** — diff and pytest below |

Six of ten falsified, and the two hits are the two that predicted a
structural certainty. The pre-registration was wrong about nearly
everything it was uncertain about, which is what it is for.

### F1 — the win is real and it is entirely on the rig clips

`peakrate-media · 5 s` scores 21 of 26 rig rows against the whole-clip
control's 12. On the owner's own counted-to-a-metronome clips, choosing the
most regular 5 seconds instead of averaging the whole recording is worth
**9 rows**. That is the largest single effect in this table.

It buys **nothing on the demo slice** (2/8, worse than the control's 4/8).
The rig clips are one steady thing throughout; the "steady window" there is
really a *noise filter*, removing prep counts, codas and explanation. The
demo is a different problem: the teacher changes tempo, talks, marks, and
the steady stretch that exists is not necessarily the stretch that carries
the tempo she will be played at.

### F2 — half the corpus lands between metric levels, and that is unchanged

Every variant leaves 17–22 of 34 rows with |OE2| above the between-levels
floor. The projection rule moves a number into 70–140; it does not decide
*which* level it is. This is Standing Lesson 3 arriving on schedule: median
IOI over mixed 1×/2×/3× events lands between levels, and no choice of window
fixes that.

**Disclosed, found after the run:** `pass` and `between_levels` are **not
disjoint** as the repo defines them. Pass is ±8 % as a *ratio* (|OE1| up to
0.111 octaves); between-levels starts at |OE2| > 0.08 *octaves* (≈ 5.7 %).
Rows 5.7–8 % off count as both. In the winner's 5 s sibling, 6 of the 17
between-levels rows are also passes; the honest "landed between levels and
got it wrong" count is **11**, not 17. This applies to the blessed
baseline's "10 of 33" too and is worth the owner's eye — it was not added to
make a number look better, it was found while checking why 23 + 17 exceeded
34.

### F3 — *CONCLUSION WITHDRAWN 2026-09-02 (see the ledger correction of that date): the oracle arm ran the owner's span through the same contaminated event stream and median rule as every other arm, so it never tested the span. Read with his own taps, his spans recover the label within 3% on three of the five clips the probe can read. The measurements below stand; the conclusion does not.*

### F3 (as originally written) — the owner's own window is not the most regular window

The oracle was pre-registered as a ceiling. It came in **below** the
measured variants on the demo slice (2/8 and 3/8 vs 4/8) and below the
whole-clip control. Per clip, the spans where the owner reports the tempo
was knowable produce readings like frappé 83.7 against a truth of 135, and
tendu-warmup 92.5 against 112.

Read plainly: **the stretch where a musician knows the tempo is not the
stretch where the audio is most metrically regular.** He is reading the
demonstration — the shape of the movement, the way she marks the first
count — not the evenness of her syllables. This is the sharpest thing in
the sweep, and it is an argument for W13's information-timing line and
against expecting audio regularity alone to find the window. It also
retires, on evidence, the assumption behind the whole idea: there was no
"steady window" the owner was reading. The movement half of the original
idea, deferred at commissioning per W7/W10, is where this points.

### F4 — window length is not a free parameter here

Within peakRate the three lengths span 18–23 passes, a 5-row swing on 34
rows, and the ordering is not monotone (5 s ≫ 3 s > 8 s). With one corpus
and no held-out split, picking 5 s because it scored best would be fitting
to 34 rows. Named here so that a future adoption increment has to defend the
length, not inherit it.

### Fallbacks and abstentions, by name

- `whisper-trace · 3 s` fell back to the whole clip on **7 rows**:
  `rig-names-3-4-90-clean`, `rig-names-4-4-100-quiet`,
  `rig-names-4-4-63-adagio`, `rig-names-6-8-100-clean`,
  `rig-numbers-4-4-104-clean`, `rig-numbers-4-4-60-halftempo`,
  `rig-vocables-4-4-100-clean` — word onsets are too sparse to put 6 events
  in 3 seconds.
- `whisper-trace · 5 s` and `· 8 s`, and `peakrate-media · 8 s`, fell back
  on one row each (`rig-vocables-4-4-100-clean`, and
  `adr006-8-counts-triple` respectively).
- The single abstention on every `whisper-trace` variant is
  `rig-vocables-4-4-100-clean` — 1 Whisper word, named in the
  pre-registration.

### What this does and does not establish

**Establishes:** picking the most regular window beats reading the whole
clip on clean counted speech, by a large margin, with peakRate events (21/26
vs 12/26). The pre-registered selection rule, applied honestly, does not
select that variant. The owner's stated knowing-window is not recoverable
from audio regularity.

**Does not establish:** anything about the demo, which is step one's target
and where every variant is flat. Nor that 5 s is the right length. Nor that
any of this survives an owner-verified corpus larger than 34 rows — the
half-gaps range 0.06–0.47, and a 17/17 split at this n is a coarse
instrument.

### Recommendation to the owner

1. **Adopt nothing from this sweep yet.** The variant that wins the rule
   and the variant that wins the corpus are different variants; that
   disagreement is the finding, not a tie to be broken by an agent.
2. If any part is worth an adoption increment, it is the **rig-side noise
   filtering**, which is a real 9-row effect — but it is worth least where
   step one is aimed.
3. **F3 is the result to act on.** The idea's own oracle says audio
   regularity is not what he reads. That is a vote for the movement half
   (deferred here) and for W13's line, not for tuning window lengths.
