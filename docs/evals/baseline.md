# Eval Baseline

Generated 2026-08-02T21:47:52+00:00 at git `6fc5f49` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (24 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 24 | 22 | 2 | 0 | 0.917 | [0.742, 0.977] | tempo_wrong×1, meter_wrong×1 |
| tempo | 24 | 23 | 1 | 0 | 0.958 | [0.798, 0.993] | tempo_error×1 |

## tier1 (6 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 4 | 3 | 0 | 1 | 1.0 | [0.438, 1.0] | — |
| meter_triple | 5 | 3 | 2 | 0 | 0.6 | [0.231, 0.882] | tempo_wrong×1, subdivision_wrong×1 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 5 | 3 | 2 | 0 | 0.6 | [0.231, 0.882] | tempo_error×1, metric_level_div2×1 |
