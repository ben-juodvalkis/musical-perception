# Rung-2 kill-test — blessed-gate results

Tolerance ±70 ms; 28 owner-verified grids; declined and excluded by name: adr007-plies-demo, rig-mixed-4-4-104-quantities.

P0: §2.2 baseline table reproduced exactly (all 12 numbers).

### Baseline (whisper-word-starts)

| slice | n | R@tac | P_lc | F_lc |
|---|---|---|---|---|
| ALL | 28 | 0.449 | 0.506 | 0.452 |
| numbers | 14 | 0.568 | 0.604 | 0.577 |
| step_names | 13 | 0.349 | 0.363 | 0.343 |
| vocables | 1 | 0.062 | 1.000 | 0.118 |

### Extractor (acoustic-pulse/1)

| slice | n | R@tac | P_lc | F_lc |
|---|---|---|---|---|
| ALL | 28 | 0.828 | 0.867 | 0.839 |
| numbers | 14 | 0.926 | 0.931 | 0.926 |
| step_names | 13 | 0.719 | 0.798 | 0.742 |
| vocables | 1 | 0.875 | 0.875 | 0.875 |

### step_names per clip (gate condition 2)

| clip | n_ref | baseline R@tac | extractor R@tac | improved |
|---|---|---|---|---|
| adr006-exercise-1-demo | 41 | 0.488 | 0.561 | YES |
| frappe | 55 | 0.673 | 0.691 | YES |
| rig-names-2-4-120-clean | 27 | 0.296 | 0.926 | YES |
| rig-names-2-4-160-long | 54 | 0.519 | 0.815 | YES |
| rig-names-3-4-88-waltz | 24 | 0.500 | 0.792 | YES |
| rig-names-3-4-90-clean | 43 | 0.279 | 0.860 | YES |
| rig-names-4-4-100-quiet | 16 | 0.312 | 0.312 | no |
| rig-names-4-4-104-clean | 24 | 0.125 | 0.708 | YES |
| rig-names-4-4-104-coda | 41 | 0.366 | 0.537 | YES |
| rig-names-4-4-104-explained | 26 | 0.192 | 0.885 | YES |
| rig-names-4-4-63-adagio | 26 | 0.192 | 0.385 | YES |
| rig-names-4-4-96-allegro | 27 | 0.370 | 0.963 | YES |
| rig-names-6-8-100-clean | 22 | 0.227 | 0.909 | YES |

Improved on 12 of 13 step_names clips.

### Gate

```json
{
 "1_step_names_r_tac": {
  "value": 0.719,
  "threshold": 0.499,
  "pass": true
 },
 "2_step_names_improved": {
  "value": 12,
  "threshold": 9,
  "of": 13,
  "pass": true
 },
 "3_vocables_n1": {
  "r_tac": 0.875,
  "p_lc": 0.875,
  "thresholds": [
   0.6,
   0.5
  ],
  "pass": true
 },
 "4_numbers_f_lc": {
  "value": 0.926,
  "threshold": 0.527,
  "pass": true
 }
}
```

## VERDICT: PASS
