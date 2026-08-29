# Eval Baseline

Generated 2026-08-29T03:23:45+00:00 at git `21570ed` by `python -m musical_perception.evals bless`. Do not edit by hand.

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

tempo n=25: Acc1 1.0@4% 1.0@8% · Acc2 1.0@4% 1.0@8% · OE1 median 0.0 · |OE2| median 0.0 (max 0.0478) · between-levels rows 0

## tier1 (30 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | truth in family | failure modes |
|---|---|---|---|---|---|---|---|---|
| counts | 28 | 13 | 8 | 7 | 0.619 | [0.409, 0.792] | — | counts_wrong×8 |
| meter_triple | 29 | 12 | 16 | 1 | 0.429 | [0.265, 0.609] | — | tempo_wrong×5, meter_wrong×6, subdivision_wrong×4, equivalent_reading×1 |
| sides | 2 | 2 | 0 | 0 | 1.0 | [0.342, 1.0] | — | — |
| slot | 4 | 4 | 0 | 0 | 1.0 | [0.51, 1.0] | — | — |
| tempo | 30 | 20 | 9 | 1 | 0.69 | [0.508, 0.827] | 0/9 | tempo_error×9 |

tempo n=29: Acc1 0.483@4% 0.69@8% · Acc2 0.483@4% 0.69@8% · OE1 median 0.0055 · |OE2| median 0.0604 (max 0.491) · between-levels rows 11

## stage1 (pulse vs beat grids — PROVISIONAL, gates nothing)

whisper-word-starts ±0.07s: P 0.597 R 0.66 F 0.627 over 2 clips; asynchrony mean 2.2 ms
