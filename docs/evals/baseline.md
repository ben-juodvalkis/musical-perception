# Eval Baseline

Generated 2026-08-07T05:35:15+00:00 at git `3428d3d` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (10 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 8 | 7 | 0 | 1 | 1.0 | [0.646, 1.0] | — |
| meter_triple | 9 | 4 | 5 | 0 | 0.444 | [0.189, 0.733] | tempo_wrong×2, subdivision_wrong×1, meter_wrong×1, equivalent_reading×1 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 9 | 6 | 3 | 0 | 0.667 | [0.354, 0.879] | tempo_error×2, metric_level_div2×1 |
