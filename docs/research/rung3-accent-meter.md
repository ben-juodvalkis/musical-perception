# Rung 3 (W2) — accent-periodicity meter votes: the full result tables

Companion to the 2026-08-20 RESEARCH-LOG entry, which carries the
pre-registration, the prediction scorecard, the two findings, and the
recommendation. This file is the raw output, committed so the numbers
quoted there can be checked without re-running anything.

Regenerate with:

```
python scripts/rung3-accent-meter-report.py --markdown
python scripts/rung3-accent-evidence-audit.py
```

Both read only committed files (`evals/grids/`, `evals/cases/`,
`docs/research/rung2-extractor-events.json`) and write nothing.

## Grouping diagnostic

```
| clip                                   | truth | pred   |  ph |  score | margin |      |
|----------------------------------------|-------|--------|-----|--------|--------|------|

--- non-4/4 slice (the diagnostic's primary set) ---
| adr006-exercise-1-demo                 | 3/4   | None   |   - |  0.508 |  0.036 | ABST |
| rig-names-2-4-120-clean                | 2/4   | 4/4    |   1 |  0.257 |  0.051 | FAIL |
| rig-names-2-4-160-long                 | 2/4   | None   |   - |  0.273 |  0.009 | ABST |
| rig-names-3-4-88-waltz                 | 3/4   | None   |   - |  0.469 |  0.020 | ABST |
| rig-names-3-4-90-clean                 | 3/4   | 4/4    |   0 |  0.388 |  0.103 | FAIL |
| rig-names-6-8-100-clean                | 6/8   | 4/4    |   0 |  0.407 |  0.063 | FAIL |
| rig-numbers-2-4-120-clean              | 2/4   | 4/4    |   0 |  0.416 |  0.194 | FAIL |
| rig-numbers-6-8-100-clean              | 6/8   | 6/8    |   0 |  0.247 |  0.060 | PASS |

--- 4/4 slice ---
| adr006-8-counts-2x                     | 4/4   | 4/4    |   0 |  0.667 |  0.234 | PASS |
| adr006-8-counts-triple                 | 4/4   | 6/8    |   4 |  0.629 |  0.200 | FAIL |
| adr010-grande-battement                | 4/4   | 6/8    |   3 |  0.353 |  0.094 | FAIL |
| rig-names-4-4-100-quiet                | 4/4   | None   |   - |  0.300 |  0.013 | ABST |
| rig-names-4-4-104-clean                | 4/4   | 4/4    |   0 |  0.479 |  0.148 | PASS |
| rig-names-4-4-104-coda                 | 4/4   | None   |   - |  0.176 |  0.010 | ABST |
| rig-names-4-4-104-explained            | 4/4   | None   |   - |  0.373 |  0.026 | ABST |
| rig-names-4-4-63-adagio                | 4/4   | None   |   - |  0.532 |  0.032 | ABST |
| rig-names-4-4-96-allegro               | 4/4   | None   |   - |  0.198 |  0.001 | ABST |
| rig-numbers-4-4-104-bothsides          | 4/4   | None   |   - |  0.211 |  0.047 | ABST |
| rig-numbers-4-4-104-clean              | 4/4   | None   |   - |  0.449 |  0.015 | ABST |
| rig-numbers-4-4-104-duple              | 4/4   | 6/8    |   0 |  0.417 |  0.176 | FAIL |
| rig-numbers-4-4-104-explained          | 4/4   | 6/8    |   4 |  0.374 |  0.054 | FAIL |
| rig-numbers-4-4-104-fourx8             | 4/4   | None   |   - |  0.457 |  0.010 | ABST |
| rig-numbers-4-4-104-prep               | 4/4   | None   |   - |  0.711 |  0.043 | ABST |
| rig-numbers-4-4-60-halftempo           | 4/4   | None   |   - |  0.540 |  0.016 | ABST |
| rig-numbers-4-4-80-triplet             | 4/4   | 6/8    |   1 |  0.300 |  0.093 | FAIL |
| rig-vocables-4-4-100-clean             | 4/4   | 4/4    |   0 |  0.412 |  0.113 | PASS |

--- excluded (provisional / degenerate / no truth meter) ---
  adr007-plies-demo                      truth=4/4   pred=None    [provisional grid]
  frappe                                 truth=None  pred=None    [no truth meter]
  rig-mixed-4-4-104-quantities           truth=4/4   pred=4/4     [provisional grid]
  rig-numbers-3-4-90-clean               truth=3/4   pred=6/8     [grid at the number level; 3/4 lives below the tactus]

=== summary ===
non-4/4 grouping: 1/8 correct, 3 abstained
4/4 grouping: 3/18 correct, 10 abstained
all scoreable: 4/26 correct, 13 abstained
  2/4: 0/3 correct, 1 abstained
  3/4: 0/3 correct, 2 abstained
  6/8: 1/2 correct, 0 abstained

family (duple vs triple/compound), committed rows only: 6/13 correct
family, non-4/4 slice: 3/5 correct

confusions (scoreable, wrong, not abstained):
  rig-names-2-4-120-clean                2/4 -> 4/4  (margin 0.051)
  rig-names-3-4-90-clean                 3/4 -> 4/4  (margin 0.103)
  rig-names-6-8-100-clean                6/8 -> 4/4  (margin 0.063)
  rig-numbers-2-4-120-clean              2/4 -> 4/4  (margin 0.194)
  adr006-8-counts-triple                 4/4 -> 6/8  (margin 0.200)
  adr010-grande-battement                4/4 -> 6/8  (margin 0.094)
  rig-numbers-4-4-104-duple              4/4 -> 6/8  (margin 0.176)
  rig-numbers-4-4-104-explained          4/4 -> 6/8  (margin 0.054)
  rig-numbers-4-4-80-triplet             4/4 -> 6/8  (margin 0.093)

abstentions:
  adr006-exercise-1-demo                 truth=3/4  margin 0.036 < 0.05 (3/4 vs 6/8)
  rig-names-2-4-160-long                 truth=2/4  margin 0.009 < 0.05 (2/4 vs 4/4)
  rig-names-3-4-88-waltz                 truth=3/4  margin 0.020 < 0.05 (6/8 vs 3/4)
  rig-names-4-4-100-quiet                truth=4/4  margin 0.013 < 0.05 (6/8 vs 3/4)
  rig-names-4-4-104-coda                 truth=4/4  margin 0.010 < 0.05 (2/4 vs 4/4)
  rig-names-4-4-104-explained            truth=4/4  margin 0.026 < 0.05 (6/8 vs 4/4)
  rig-names-4-4-63-adagio                truth=4/4  margin 0.032 < 0.05 (4/4 vs 2/4)
  rig-names-4-4-96-allegro               truth=4/4  margin 0.001 < 0.05 (6/8 vs 2/4)
  rig-numbers-4-4-104-bothsides          truth=4/4  margin 0.047 < 0.05 (6/8 vs 3/4)
  rig-numbers-4-4-104-clean              truth=4/4  margin 0.015 < 0.05 (2/4 vs 4/4)
  rig-numbers-4-4-104-fourx8             truth=4/4  margin 0.010 < 0.05 (4/4 vs 2/4)
  rig-numbers-4-4-104-prep               truth=4/4  margin 0.043 < 0.05 (2/4 vs 4/4)
  rig-numbers-4-4-60-halftempo           truth=4/4  margin 0.016 < 0.05 (4/4 vs 2/4)
```

## Evidence audit — is bar-level accent periodicity present at all?

```
clip                                   truth lag2     lag3     lag4     lag6     lag8      winner
-------------------------------------------------------------------------------------------------
adr006-8-counts-2x                     4/4    0.29     0.05     0.60*    0.25     0.82     4
adr006-8-counts-triple                 4/4    0.27     0.34     0.42      n/a      n/a     -
adr006-exercise-1-demo                 3/4    0.09     0.04     0.22     0.24     0.35     -
adr010-grande-battement                4/4    0.20     0.36     0.49     0.71     0.72     -
frappe                                 None   0.01     0.12     0.07     0.22     0.10     -
rig-names-2-4-120-clean                2/4    0.21     0.04     0.30     0.37     0.46     -
rig-names-2-4-160-long                 2/4    0.34*    0.18     0.39     0.37     0.54*    8
rig-names-3-4-88-waltz                 3/4    0.00     0.49*    0.09     0.55     0.45     3
rig-names-3-4-90-clean                 3/4    0.29     0.12     0.47*    0.28     0.51     4
rig-names-4-4-100-quiet                4/4    0.07     0.33     0.49     0.46     0.56     -
rig-names-4-4-104-clean                4/4    0.30     0.38     0.77*    0.40     1.19*    8
rig-names-4-4-104-coda                 4/4    0.41*    0.13     0.35     0.33     0.35     2
rig-names-4-4-104-explained            4/4    0.29     0.34     0.64*    0.83     1.22     4
rig-names-4-4-63-adagio                4/4    0.64*    0.11     0.66*    0.69     0.59     4
rig-names-4-4-96-allegro               4/4    0.20     0.20     0.26     0.25     0.38     -
rig-names-6-8-100-clean                6/8    0.26     0.32     0.62*    0.55     0.66*    8
rig-numbers-2-4-120-clean              2/4    0.15     0.03     0.42*    0.23     0.55*    8
rig-numbers-3-4-90-clean               3/4    0.08     0.12     0.09     0.28     0.57     -
rig-numbers-4-4-104-bothsides          4/4    0.10     0.21     0.23     0.35     0.45     -
rig-numbers-4-4-104-clean              4/4    0.36*    0.21     0.29     0.41     0.74*    8
rig-numbers-4-4-104-duple              4/4    0.08     0.23     0.08     0.64*    0.50     6
rig-numbers-4-4-104-explained          4/4    0.26     0.35     0.58     0.82     1.09     -
rig-numbers-4-4-104-fourx8             4/4    0.41*    0.11     0.38     0.37     0.43     2
rig-numbers-4-4-104-prep               4/4    0.61*    0.02     0.45     0.60     0.77*    8
rig-numbers-4-4-60-halftempo           4/4    0.47*    0.22     0.45     0.52     0.93     2
rig-numbers-4-4-80-triplet             4/4    0.01     0.20     0.23     0.45     0.47     -
rig-numbers-6-8-100-clean              6/8    0.03     0.19     0.15     0.35     0.38     -
rig-vocables-4-4-100-clean             4/4    0.37     0.26     0.64     0.67     0.99     -

* = beats a 400-draw phase-shuffle null at p<0.05
winning lag (strongest significant), counted over verified grids:
  lag 2: 3 clips
  lag 3: 1 clips
  lag 4: 4 clips
  lag 6: 1 clips
  lag 8: 6 clips
  no significant lag: 13 clips

template confusability (max |corr| over relative phase, 24 beats):
          2/4    3/4    4/4    6/8
   2/4   1.00   0.00   0.90   0.22
   3/4   0.00   1.00   0.00   0.93
   4/4   0.90   0.00   1.00   0.20
   6/8   0.22   0.93   0.20   1.00
```
