# Eval Baseline

Generated 2026-09-01T23:48:38+00:00 at git `0b29864` by `python -m musical_perception.evals bless`. Do not edit by hand.

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

## tier1 (52 cases)

| field | n | correct | wrong | abstained | accuracy | wilson 95% | truth in family | failure modes |
|---|---|---|---|---|---|---|---|---|
| counts | 52 | 14 | 25 | 13 | 0.359 | [0.227, 0.516] | — | counts_wrong×25 |
| meter_triple | 52 | 17 | 34 | 1 | 0.333 | [0.22, 0.47] | — | meter_wrong×16, tempo_wrong×17, equivalent_reading×1 |
| sides | 1 | 1 | 0 | 0 | 1.0 | [0.207, 1.0] | — | — |
| tempo | 52 | 26 | 25 | 1 | 0.51 | [0.377, 0.641] | 9/25 | metric_level_div3×1, tempo_error×21, metric_level_div2×1, metric_level_x2×2 |

tempo n=51: Acc1 0.373@4% 0.51@8% · Acc2 0.431@4% 0.588@8% · OE1 median 0.0 · |OE2| median 0.0618 (max 0.4838) · between-levels rows 22

## stage1 (pulse vs beat grids — PROVISIONAL, gates nothing)

whisper-word-starts ±0.07s: P 0.63 R 0.43 F 0.511 over 27 clips; asynchrony mean -0.0 ms
