# Eval Baseline

Generated 2026-08-09T03:22:50+00:00 at git `fb90591` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (29 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 27 | 11 | 9 | 7 | 0.55 | [0.342, 0.742] | counts_wrong×9 |
| meter_triple | 28 | 10 | 17 | 1 | 0.37 | [0.215, 0.558] | tempo_wrong×7, subdivision_wrong×3, meter_wrong×5, equivalent_reading×2 |
| sides | 2 | 2 | 0 | 0 | 1.0 | [0.342, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 28 | 15 | 12 | 1 | 0.556 | [0.373, 0.724] | tempo_error×9, metric_level_div2×2, metric_level_x2×1 |
