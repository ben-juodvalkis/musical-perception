# Eval Baseline

Generated 2026-08-09T04:52:35+00:00 at git `15b8164` by `python -m musical_perception.evals bless`. Do not edit by hand.

Outcomes are **correct / wrong / abstained** — abstention is never
counted as wrong (ADR-009). n is small; intervals are the honest part.

**truth in family** (ADR-014) counts wrong answers whose reported
metric-level family still contained the expected tempo — a selection
failure rather than a measurement failure. It is informational and
gates nothing; outcomes above are unaffected by it.

## tier0 (25 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | truth in family | failure modes |
|---|---|---|---|---|---|---|---|---|
| meter_triple | 25 | 24 | 1 | 0 | 0.96 | [0.805, 0.993] | — | meter_wrong×1 |
| tempo | 25 | 25 | 0 | 0 | 1.0 | [0.867, 1.0] | — | — |

## tier1 (30 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | truth in family | failure modes |
|---|---|---|---|---|---|---|---|---|
| counts | 28 | 12 | 9 | 7 | 0.571 | [0.365, 0.755] | — | counts_wrong×9 |
| meter_triple | 29 | 10 | 18 | 1 | 0.357 | [0.207, 0.542] | — | tempo_wrong×6, subdivision_wrong×5, meter_wrong×5, equivalent_reading×2 |
| sides | 2 | 2 | 0 | 0 | 1.0 | [0.342, 1.0] | — | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — | — |
| tempo | 29 | 16 | 12 | 1 | 0.571 | [0.391, 0.735] | 3/12 | tempo_error×9, metric_level_div2×2, metric_level_x2×1 |
