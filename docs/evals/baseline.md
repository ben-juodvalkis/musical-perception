# Eval Baseline

Generated 2026-08-29T05:18:47+00:00 at git `310a5f8` by `python -m musical_perception.evals bless`. Do not edit by hand.

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

## tier1 (30 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | truth in family | failure modes |
|---|---|---|---|---|---|---|---|---|
| counts | 28 | 13 | 9 | 6 | 0.591 | [0.387, 0.767] | — | counts_wrong×9 |
| meter_triple | 29 | 13 | 15 | 1 | 0.464 | [0.295, 0.642] | — | tempo_wrong×8, meter_wrong×6, equivalent_reading×1 |
| sides | 2 | 2 | 0 | 0 | 1.0 | [0.342, 1.0] | — | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — | — |
| tempo | 30 | 20 | 9 | 1 | 0.69 | [0.508, 0.827] | 5/9 | tempo_error×6, metric_level_div2×1, metric_level_x2×2 |

tempo n=29: Acc1 0.483@4% 0.69@8% · Acc2 0.586@4% 0.793@8% · OE1 median -0.0033 · |OE2| median 0.0467 (max 0.4224) · between-levels rows 6

## stage1 (pulse vs beat grids — PROVISIONAL, gates nothing)

whisper-word-starts ±0.07s: P 0.597 R 0.66 F 0.627 over 2 clips; asynchrony mean 2.2 ms
