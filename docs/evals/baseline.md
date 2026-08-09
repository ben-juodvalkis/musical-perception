# Eval Baseline

Generated 2026-08-09T03:19:27+00:00 at git `bce348b` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (28 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 26 | 11 | 8 | 7 | 0.579 | [0.363, 0.769] | counts_wrong×8 |
| meter_triple | 27 | 10 | 16 | 1 | 0.385 | [0.224, 0.575] | tempo_wrong×6, subdivision_wrong×3, meter_wrong×5, equivalent_reading×2 |
| sides | 2 | 2 | 0 | 0 | 1.0 | [0.342, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 27 | 15 | 11 | 1 | 0.577 | [0.389, 0.745] | tempo_error×8, metric_level_div2×2, metric_level_x2×1 |
