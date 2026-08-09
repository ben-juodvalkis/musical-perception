# Eval Baseline

Generated 2026-08-09T03:10:02+00:00 at git `40ed2e3` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (26 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 24 | 11 | 8 | 5 | 0.579 | [0.363, 0.769] | counts_wrong×8 |
| meter_triple | 25 | 10 | 14 | 1 | 0.417 | [0.245, 0.612] | tempo_wrong×4, subdivision_wrong×3, meter_wrong×5, equivalent_reading×2 |
| sides | 2 | 2 | 0 | 0 | 1.0 | [0.342, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 25 | 15 | 9 | 1 | 0.625 | [0.427, 0.788] | tempo_error×6, metric_level_div2×2, metric_level_x2×1 |
