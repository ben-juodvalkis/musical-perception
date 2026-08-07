# Eval Baseline

Generated 2026-08-07T17:20:46+00:00 at git `882bccc` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (14 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 12 | 7 | 3 | 2 | 0.7 | [0.397, 0.892] | counts_wrong×3 |
| meter_triple | 13 | 4 | 9 | 0 | 0.308 | [0.127, 0.576] | tempo_wrong×2, subdivision_wrong×2, meter_wrong×4, equivalent_reading×1 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 13 | 7 | 6 | 0 | 0.538 | [0.291, 0.768] | tempo_error×5, metric_level_div2×1 |
