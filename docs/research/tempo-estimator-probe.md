# Tempo estimator probe (owner, 2026-08-26)

Same tick streams, two estimators. Truth = the metronome BPM in each
rig clip's filename; "correct" = within 4%. Not a blessed
metric, not tier-1 tempo accuracy — see RESEARCH-LOG 2026-08-26.

| estimator | V0 (first-per-nucleus) | V1 (all-in-nucleus) |
|---|---|---|
| median-of-consecutive-gaps (ships today) | 11/23 | 8/23 |
| pairwise-IOI histogram (probe) | 20/23 | 20/23 |

## Where V1's extra ticks land

- on a beat V0 already found: **0**
- on a beat V0 missed (the recoveries): **32**
- between beats (the clutter): **132**

Phase of the between-beat extras within their beat interval:

```
  0.000-0.125  ### 3
  0.125-0.250  ############################################## 46
  0.250-0.375  ############################### 31
  0.375-0.500  ################## 18
  0.500-0.625  ################## 18
  0.625-0.750  ######## 8
  0.750-0.875  ##### 5
  0.875-1.000   0
```

## Per clip

| clip | true | med V0 | med V1 | hist V0 | hist V1 |
|---|---|---|---|---|---|
| rig-names-2-4-120-clean | 120 | 117.3 | 119.6 | 120.5 | 120.5 |
| rig-names-2-4-160-long | 160 | 75.9 | 77.6 | 80.0 | 80.0 |
| rig-names-3-4-88-waltz | 88 | 83.9 | 95.7 | 88.2 | 88.8 |
| rig-names-3-4-90-clean | 90 | 90.4 | 109.9 | 90.6 | 90.6 |
| rig-names-4-4-100-quiet | 100 | 80.0 | 84.6 | 99.0 | 99.3 |
| rig-names-4-4-104-clean | 104 | 96.9 | 103.9 | 104.2 | 104.5 |
| rig-names-4-4-104-coda | 104 | 92.1 | 106.4 | 103.1 | 103.8 |
| rig-names-4-4-104-explained | 104 | 104.4 | 131.5 | 104.5 | 104.5 |
| rig-names-4-4-63-adagio | 63 | 77.8 | 96.2 | 133.9 | 71.8 |
| rig-names-4-4-96-allegro | 96 | 93.8 | 101.5 | 95.8 | 95.5 |
| rig-names-6-8-100-clean | 100 | 97.0 | 100.6 | 99.7 | 100.0 |
| rig-numbers-2-4-120-clean | 120 | 119.7 | 120.2 | 120.0 | 120.0 |
| rig-numbers-3-4-90-clean | 90 | 105.2 | 109.0 | 90.4 | 90.4 |
| rig-numbers-4-4-104-bothsides | 104 | 101.9 | 113.3 | 103.8 | 103.8 |
| rig-numbers-4-4-104-clean | 104 | 105.8 | 105.8 | 103.8 | 103.8 |
| rig-numbers-4-4-104-duple | 104 | 102.0 | 102.0 | 104.5 | 104.5 |
| rig-numbers-4-4-104-explained | 104 | 108.2 | 118.4 | 103.4 | 103.4 |
| rig-numbers-4-4-104-fourx8 | 104 | 105.6 | 112.7 | 104.5 | 104.5 |
| rig-numbers-4-4-104-prep | 104 | 109.0 | 115.3 | 104.2 | 104.5 |
| rig-numbers-4-4-60-halftempo | 60 | 121.4 | 72.1 | 121.7 | 122.2 |
| rig-numbers-4-4-80-triplet | 80 | 102.2 | 108.0 | 80.9 | 80.9 |
| rig-numbers-6-8-100-clean | 100 | 98.8 | 100.9 | 100.3 | 100.7 |
| rig-vocables-4-4-100-clean | 100 | 92.8 | 92.8 | 100.7 | 100.7 |
