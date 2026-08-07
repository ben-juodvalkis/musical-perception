# Eval Baseline

Generated 2026-08-07T05:30:07+00:00 at git `e324676` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (9 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 7 | 6 | 0 | 1 | 1.0 | [0.61, 1.0] | — |
| meter_triple | 8 | 3 | 5 | 0 | 0.375 | [0.137, 0.694] | tempo_wrong×2, subdivision_wrong×1, meter_wrong×1, equivalent_reading×1 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 8 | 5 | 3 | 0 | 0.625 | [0.306, 0.863] | tempo_error×2, metric_level_div2×1 |
