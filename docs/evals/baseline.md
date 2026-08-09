# Eval Baseline

Generated 2026-08-09T03:50:41+00:00 at git `076b379` by `python -m musical_perception.evals bless`. Do not edit by hand.

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
| counts | 28 | 11 | 9 | 8 | 0.55 | [0.342, 0.742] | — | counts_wrong×9 |
| meter_triple | 29 | 10 | 18 | 1 | 0.357 | [0.207, 0.542] | — | tempo_wrong×8, subdivision_wrong×3, meter_wrong×5, equivalent_reading×2 |
| sides | 2 | 2 | 0 | 0 | 1.0 | [0.342, 1.0] | — | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — | — |
| tempo | 29 | 15 | 13 | 1 | 0.536 | [0.358, 0.705] | 3/13 | tempo_error×10, metric_level_div2×2, metric_level_x2×1 |
