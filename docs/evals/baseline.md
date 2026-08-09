# Eval Baseline

Generated 2026-08-09T03:03:39+00:00 at git `a6cfd68` by `python -m musical_perception.evals bless`. Do not edit by hand.

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
| counts | 23 | 11 | 7 | 5 | 0.611 | [0.386, 0.797] | counts_wrong×7 |
| meter_triple | 24 | 10 | 13 | 1 | 0.435 | [0.256, 0.632] | tempo_wrong×4, subdivision_wrong×3, meter_wrong×4, equivalent_reading×2 |
| sides | 2 | 2 | 0 | 0 | 1.0 | [0.342, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 24 | 14 | 9 | 1 | 0.609 | [0.408, 0.778] | tempo_error×6, metric_level_div2×2, metric_level_x2×1 |
