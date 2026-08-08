# Eval Baseline

Generated 2026-08-08T18:30:16+00:00 at git `e5d8ac5` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 23 | 12 | 7 | 4 | 0.632 | [0.41, 0.809] | counts_wrong×7 |
| meter_triple | 24 | 11 | 13 | 0 | 0.458 | [0.279, 0.649] | tempo_wrong×4, subdivision_wrong×3, meter_wrong×4, equivalent_reading×2 |
| sides | 2 | 2 | 0 | 0 | 1.0 | [0.342, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 24 | 15 | 9 | 0 | 0.625 | [0.427, 0.788] | tempo_error×6, metric_level_div2×2, metric_level_x2×1 |
