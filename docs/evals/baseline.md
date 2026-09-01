# Eval Baseline

Generated 2026-09-01T23:14:11+00:00 at git `8b52169` by `python -m musical_perception.evals bless`. Do not edit by hand.

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

## tier1 (56 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | truth in family | failure modes |
|---|---|---|---|---|---|---|---|---|
| counts | 54 | 16 | 25 | 13 | 0.39 | [0.257, 0.543] | — | counts_wrong×25 |
| meter_triple | 55 | 19 | 35 | 1 | 0.352 | [0.238, 0.485] | — | tempo_wrong×18, meter_wrong×16, equivalent_reading×1 |
| sides | 2 | 2 | 0 | 0 | 1.0 | [0.342, 1.0] | — | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — | — |
| tempo | 56 | 29 | 26 | 1 | 0.527 | [0.398, 0.653] | 9/26 | tempo_error×22, metric_level_div3×1, metric_level_div2×1, metric_level_x2×2 |

tempo n=55: Acc1 0.345@4% 0.527@8% · Acc2 0.4@4% 0.6@8% · OE1 median 0.0154 · |OE2| median 0.0619 (max 0.4838) · between-levels rows 23

## stage1 (pulse vs beat grids — PROVISIONAL, gates nothing)

whisper-word-starts ±0.07s: P 0.628 R 0.441 F 0.518 over 28 clips; asynchrony mean 0.3 ms
