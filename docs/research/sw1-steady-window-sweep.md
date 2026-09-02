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
