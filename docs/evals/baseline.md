# Eval Baseline

Generated 2026-08-08T17:46:32+00:00 at git `6dc02e5` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — |

## tier1 (18 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |
|---|---|---|---|---|---|---|---|
| counts | 16 | 9 | 4 | 3 | 0.692 | [0.424, 0.873] | counts_wrong×4 |
| meter_triple | 17 | 6 | 11 | 0 | 0.353 | [0.173, 0.587] | tempo_wrong×3, subdivision_wrong×3, meter_wrong×4, equivalent_reading×1 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — |
| tempo | 17 | 10 | 7 | 0 | 0.588 | [0.36, 0.784] | tempo_error×5, metric_level_div2×1, metric_level_x2×1 |
