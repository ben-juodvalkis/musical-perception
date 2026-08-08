# Eval Baseline

Generated 2026-08-08T18:04:24+00:00 at git `0b73de4` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (19 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 17 | 9 | 5 | 3 | 0.643 | [0.388, 0.837] | counts_wrong×5 |
| meter_triple | 18 | 6 | 12 | 0 | 0.333 | [0.163, 0.563] | tempo_wrong×3, subdivision_wrong×3, meter_wrong×4, equivalent_reading×2 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 18 | 10 | 8 | 0 | 0.556 | [0.337, 0.754] | tempo_error×5, metric_level_div2×2, metric_level_x2×1 |
