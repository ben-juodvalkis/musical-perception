# Eval Baseline

Generated 2026-08-08T18:16:57+00:00 at git `9e6c95d` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (22 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 20 | 11 | 5 | 4 | 0.688 | [0.444, 0.858] | counts_wrong×5 |
| meter_triple | 21 | 8 | 13 | 0 | 0.381 | [0.208, 0.591] | tempo_wrong×4, subdivision_wrong×3, meter_wrong×4, equivalent_reading×2 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 21 | 12 | 9 | 0 | 0.571 | [0.365, 0.755] | tempo_error×6, metric_level_div2×2, metric_level_x2×1 |
