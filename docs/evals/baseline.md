# Eval Baseline

Generated 2026-08-07T16:59:51+00:00 at git `af841ba` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (12 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 10 | 7 | 2 | 1 | 0.778 | [0.453, 0.937] | counts_wrong×2 |
| meter_triple | 11 | 4 | 7 | 0 | 0.364 | [0.152, 0.646] | tempo_wrong×2, subdivision_wrong×1, meter_wrong×3, equivalent_reading×1 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 11 | 6 | 5 | 0 | 0.545 | [0.28, 0.787] | tempo_error×4, metric_level_div2×1 |
