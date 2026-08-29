# The factored meter slice

*Added by W12 (2026-08-29), commissioned by owner ruling 2026-08-28.
EVAL-CHANGE. **REPORTED-ONLY — it gates nothing** until a separate owner
ruling says otherwise.*

`meter_triple` scores meter, BPM and subdivision as **one conjunction**: a
row is correct only if all three are. That was always a deliberate
choice, but it hides which axis actually failed. ADR-017 replaced the
meter state variable with a factored representation, so the eval can now
report the axes separately.

## The two rows

| field | what it scores | credit rule |
|---|---|---|
| `meter_division` | duple / triplet / none, as measured | exact match |
| `meter_grouping` | the bar rung | **duple-family credit** |

Truth is **derived** from the existing `meter` + `subdivision` labels.
Nothing was relabelled and no case file was touched.

### Division truth

| truth meter | division truth |
|---|---|
| 6/8 | **`none`** |
| anything else | `= subdivision`, verbatim |

The 6/8 rule is owner ruling **R-6/8**, made by ear on his own
recordings: *"each of the 8ths is at 100 BPM, and there's an accent every
3 pulses."* The pulse **is** the counted eighth, so nothing divides below
it. The accent every 3 is grouping rung 3; the bar is rung 6. `6/4` does
not inherit this — the override is keyed on beat_unit 8.

### Grouping truth and duple-family credit

| truth meter | bar rung | accepted | note |
|---|---|---|---|
| 2/4 | 2 | **{2, 4}** | duple family |
| 4/4 | 4 | **{2, 4}** | duple family |
| 3/4 | 3 | {3} | no family credit |
| 6/8 | 6 | {6} | accent rung 3 reported |

Owner ruling **R-bar-scoring**. W2 measured 2/4 against 4/4 at r = 0.90
on salience-clock templates — "which duple bar?" is ill-posed on this
corpus, so both count. The **exact** bar is still reported in the row's
`detail` (`exact=y|n`), informationally.

This flips some rows by construction. That was disclosed and accepted at
commissioning, and is safe precisely because the slice gates nothing.

## How "gates nothing" is enforced

One tuple, `scorers.REPORTED_ONLY_FIELDS`, and three exclusions built on
it:

- `runner.outcomes_map` drops these fields, so `compare_outcomes` and the
  tier-1 pytest gate never see them;
- `aggregate._summarize_cases` computes `fields`, `ece`, `risk_coverage`
  and every tag slice from the **gating rows only**;
- the slice lands in its own `factored_meter` block, `None` when absent —
  so a pre-W12 corpus produces byte-identical output.

Verified 2026-08-29 against the blessed baseline: `fields`, `outcomes`,
`ece`, `risk_coverage`, `slices`, `tempo_metrics`, `quality_spearman` and
`provisional` are all **identical** on tier0 and tier1.

## The caveat that must travel with these numbers

**Grouping is not yet an independent bar estimate.** It reads
`normalized.meter.beats_per_measure` — the same derived label
`meter_triple` uses. The ADR-017 `grouping_levels` ladder cannot supply
the bar today: measured on tier-1, it is empty on 20 of 30 clips, and of
the 29 scored grouping rows exactly **one** carries a bar-candidate rung
({2,3,4,6}). What the ladder mostly reports is the **count phrase**
(rung 8, on seven clips) plus gaps artifacts at 10, 14 and 15.

So a rise in `meter_grouping` over `meter_triple` measures **axis
separation and family credit**, not new bar evidence. Anyone quoting it
as the factored representation "working" is quoting the wrong thing. The
ladder becoming a real bar estimator is future W5 work.
