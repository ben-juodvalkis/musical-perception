# Eval Baseline

Generated 2026-08-07T17:07:58+00:00 at git `3dee947` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (13 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 11 | 7 | 3 | 1 | 0.7 | [0.397, 0.892] | counts_wrong×3 |
| meter_triple | 12 | 4 | 8 | 0 | 0.333 | [0.138, 0.609] | tempo_wrong×2, subdivision_wrong×2, meter_wrong×3, equivalent_reading×1 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 12 | 7 | 5 | 0 | 0.583 | [0.32, 0.807] | tempo_error×4, metric_level_div2×1 |
