# Eval Baseline

Generated 2026-08-07T17:43:23+00:00 at git `84b930b` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (17 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 15 | 8 | 4 | 3 | 0.667 | [0.391, 0.862] | counts_wrong×4 |
| meter_triple | 16 | 6 | 10 | 0 | 0.375 | [0.185, 0.614] | tempo_wrong×2, subdivision_wrong×3, meter_wrong×4, equivalent_reading×1 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 16 | 10 | 6 | 0 | 0.625 | [0.386, 0.815] | tempo_error×5, metric_level_div2×1 |
