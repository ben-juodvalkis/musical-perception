# Eval Baseline

Generated 2026-08-07T05:17:33+00:00 at git `82e207c` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (8 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 6 | 5 | 0 | 1 | 1.0 | [0.566, 1.0] | — |
| meter_triple | 7 | 3 | 4 | 0 | 0.429 | [0.158, 0.75] | tempo_wrong×1, subdivision_wrong×1, meter_wrong×1, equivalent_reading×1 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 7 | 5 | 2 | 0 | 0.714 | [0.359, 0.918] | tempo_error×1, metric_level_div2×1 |
