# Eval Baseline

Generated 2026-08-07T17:33:43+00:00 at git `25c727c` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (16 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 14 | 7 | 4 | 3 | 0.636 | [0.354, 0.848] | counts_wrong×4 |
| meter_triple | 15 | 5 | 10 | 0 | 0.333 | [0.152, 0.583] | tempo_wrong×2, subdivision_wrong×3, meter_wrong×4, equivalent_reading×1 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 15 | 9 | 6 | 0 | 0.6 | [0.357, 0.802] | tempo_error×5, metric_level_div2×1 |
