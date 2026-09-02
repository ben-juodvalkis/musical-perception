# Eval Baseline

Generated 2026-09-02T02:16:55+00:00 at git `145948f` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

**truth in family** (ADR-014) counts wrong answers whose reported
metric-level family still contained the expected tempo — a selection
failure rather than a measurement failure. It is informational and
gates nothing; outcomes above are unaffected by it.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | truth in family | failure modes |
|---|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | — | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — | — |

tempo n=25: Acc1 1.0@4% 1.0@8% · Acc2 1.0@4% 1.0@8% · OE1 median 0.0 · |OE2| median 0.0 (max 0.0478) · between-levels rows 0

## tier1 (34 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | truth in family | failure modes |
|---|---|---|---|---|---|---|---|---|
| counts | 34 | 12 | 13 | 9 | 0.48 | [0.3, 0.665] | — | counts_wrong×13 |
| meter_triple | 34 | 14 | 19 | 1 | 0.424 | [0.272, 0.592] | — | tempo_wrong×10, meter_wrong×8, equivalent_reading×1 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — | — |
| tempo | 34 | 20 | 13 | 1 | 0.606 | [0.437, 0.753] | 5/13 | tempo_error×10, metric_level_x2×2, metric_level_div2×1 |

tempo n=33: Acc1 0.515@4% 0.606@8% · Acc2 0.576@4% 0.697@8% · OE1 median -0.0043 · |OE2| median 0.0534 (max 0.4739) · between-levels rows 10

### tier1 — reference slice (18 cases)

Rows demoted from the benchmark by owner ruling (reset
2026-09-01): piano takes — the demo is the case; a take is one
valid realization, kept as reference — and step-one deferrals
(fast triple meters, waiting on the meter step). Verified truth,
gates nothing, pooled into none of the numbers above.

Cases: barre6-ballonne-demo, barre6-ballonne-take1, barre6-ballonne-take2, barre6-coupe-barre-take1, barre6-degage-take1, barre6-degage-take2, barre6-fondu-take1, barre6-fondu-take2, barre6-frappe-take1, barre6-frappe-take2, barre6-plie-take1, barre6-plie-take2, barre6-releve-finish-take1, barre6-rond-de-jambe-take1, barre6-rond-de-jambe-take2, barre6-tendu-take1, barre6-tendu-take2, barre6-tendu-warmup-take1

| field | n | correct | wrong | abstained | accuracy | wilson 95% | truth in family | failure modes |
|---|---|---|---|---|---|---|---|---|
| counts | 18 | 2 | 12 | 4 | 0.143 | [0.04, 0.399] | — | counts_wrong×12 |
| meter_triple | 18 | 4 | 14 | 0 | 0.222 | [0.09, 0.452] | — | meter_wrong×8, tempo_wrong×6 |
| tempo | 18 | 6 | 12 | 0 | 0.333 | [0.163, 0.563] | 3/12 | metric_level_div3×1, tempo_error×11 |

## stage1 (pulse vs beat grids — PROVISIONAL, gates nothing)

whisper-word-starts ±0.07s: P 0.64 R 0.355 F 0.457 over 19 clips; asynchrony mean 2.5 ms
