# Eval Baseline

Generated 2026-08-07T16:26:14+00:00 at git `6e15247` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (11 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 9 | 7 | 1 | 1 | 0.875 | [0.529, 0.978] | counts_wrong×1 |
| meter_triple | 10 | 4 | 6 | 0 | 0.4 | [0.168, 0.687] | tempo_wrong×2, subdivision_wrong×1, meter_wrong×2, equivalent_reading×1 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 10 | 6 | 4 | 0 | 0.6 | [0.313, 0.832] | tempo_error×3, metric_level_div2×1 |
