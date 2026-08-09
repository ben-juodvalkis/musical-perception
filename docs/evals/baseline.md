# Eval Baseline

Generated 2026-08-09T03:15:30+00:00 at git `0ac7b34` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (27 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 25 | 11 | 8 | 6 | 0.579 | [0.363, 0.769] | counts_wrong×8 |
| meter_triple | 26 | 10 | 15 | 1 | 0.4 | [0.234, 0.593] | tempo_wrong×5, subdivision_wrong×3, meter_wrong×5, equivalent_reading×2 |
| sides | 2 | 2 | 0 | 0 | 1.0 | [0.342, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 26 | 15 | 10 | 1 | 0.6 | [0.407, 0.766] | tempo_error×7, metric_level_div2×2, metric_level_x2×1 |
