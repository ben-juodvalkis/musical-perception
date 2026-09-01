# W13(b) — the machine's time-to-commitment curve

Generated 2026-09-01T01:08:39+00:00 by `scripts/w13b-prefix-replay.py` (read-only over frozen traces; no media, no models, no API key). Pre-registered in the ledger, 2026-08-31.

Convergence time t\* = the earliest prefix from which a field's answer never leaves its final value again; normalized by the clip's voiced span (last word end). Numeric fields match within 4% (Standing Lesson 7).

**Condition A (granted) times are a LOWER bound**: the frozen trace holds one whole-clip Gemini answer, so the semantic fields are granted at t=0. Condition B suppresses them entirely.


## Condition granted


### verified slice (n=30 clips)

| field | n scored | median t\*/span | converged before 30% of span | median answer moves | excluded (final None) |
|---|---|---|---|---|---|
| `exercise` | 30 | 0.0 | 1.0 | 1.0 | 0 |
| `meter` | 30 | 0.0 | 0.933 | 1.0 | 0 |
| `grouping` | 29 | 0.1955 | 0.793 | 1.0 | 1 |
| `division` | 29 | 0.3495 | 0.379 | 2.0 | 1 |
| `tempo_bpm` | 29 | 0.6035 | 0.069 | 5.0 | 1 |
| `counts` | 23 | 0.5728 | 0.13 | 6.5 | 7 |
| `onset_bpm` | 29 | 0.5638 | 0.069 | 4.0 | 1 |
| `marker_bpm` | 26 | 0.6619 | 0.269 | 3.0 | 4 |

### provisional slice (n=22 clips)

| field | n scored | median t\*/span | converged before 30% of span | median answer moves | excluded (final None) |
|---|---|---|---|---|---|
| `exercise` | 21 | 0.0 | 1.0 | 1.0 | 1 |
| `meter` | 21 | 0.0 | 0.857 | 1.0 | 1 |
| `grouping` | 16 | 0.1743 | 0.75 | 1.0 | 6 |
| `division` | 16 | 0.3626 | 0.375 | 2.0 | 6 |
| `tempo_bpm` | 16 | 0.8834 | 0.062 | 8.0 | 6 |
| `counts` | 14 | 0.491 | 0.214 | 6.5 | 8 |
| `onset_bpm` | 16 | 0.8782 | 0.062 | 5.5 | 6 |
| `marker_bpm` | 14 | 0.788 | 0.071 | 9.0 | 8 |

### verified slice, clips whose FINAL answer is correct

| field | n right | median t\*/span | dropped (final wrong/unlabelled) |
|---|---|---|---|
| `exercise` | — | — | no truth label in any case file |
| `meter` | 22 | 0.0 | 8 |
| `grouping` | 21 | 0.1975 | 8 |
| `division` | 21 | 0.3207 | 8 |
| `tempo_bpm` | 20 | 0.5934 | 9 |
| `counts` | 13 | 0.5708 | 10 |

## Condition withheld


### verified slice (n=30 clips)

| field | n scored | median t\*/span | converged before 30% of span | median answer moves | excluded (final None) |
|---|---|---|---|---|---|
| `exercise` | 0 | None | None | 0.0 | 30 |
| `meter` | 29 | 0.1955 | 0.793 | 1.0 | 1 |
| `grouping` | 29 | 0.1955 | 0.793 | 1.0 | 1 |
| `division` | 29 | 0.3696 | 0.379 | 2.0 | 1 |
| `tempo_bpm` | 29 | 0.6035 | 0.069 | 5.0 | 1 |
| `counts` | 15 | 0.5244 | 0.133 | 4.5 | 15 |
| `onset_bpm` | 29 | 0.5638 | 0.069 | 4.0 | 1 |
| `marker_bpm` | 26 | 0.6619 | 0.269 | 3.0 | 4 |

### provisional slice (n=22 clips)

| field | n scored | median t\*/span | converged before 30% of span | median answer moves | excluded (final None) |
|---|---|---|---|---|---|
| `exercise` | 0 | None | None | 0.0 | 22 |
| `meter` | 16 | 0.2197 | 0.688 | 1.0 | 6 |
| `grouping` | 16 | 0.2197 | 0.688 | 1.0 | 6 |
| `division` | 16 | 0.3626 | 0.375 | 2.0 | 6 |
| `tempo_bpm` | 16 | 0.8834 | 0.062 | 8.0 | 6 |
| `counts` | 2 | 0.6844 | 0.0 | 2.0 | 20 |
| `onset_bpm` | 16 | 0.8782 | 0.062 | 5.5 | 6 |
| `marker_bpm` | 14 | 0.788 | 0.071 | 9.0 | 8 |

### verified slice, clips whose FINAL answer is correct

| field | n right | median t\*/span | dropped (final wrong/unlabelled) |
|---|---|---|---|
| `exercise` | — | — | no truth label in any case file |
| `meter` | 19 | 0.2028 | 10 |
| `grouping` | 19 | 0.2028 | 10 |
| `division` | 21 | 0.3207 | 8 |
| `tempo_bpm` | 20 | 0.5934 | 9 |
| `counts` | 11 | 0.5244 | 4 |

## Granted vs withheld: does Gemini's clip-level meter change the answer?

Per field, over all 52 clips: how many end at a DIFFERENT final value when Gemini's clip-level fields are suppressed, and how the median convergence time moves. This is the P5 probe.

| field | clips with both finals non-None | different final | median t\*/span granted | withheld |
|---|---|---|---|---|
| `exercise` | 0 | 0 | 0.0 | None |
| `meter` | 45 | 7 | 0.0 | 0.1955 |
| `grouping` | 45 | 7 | 0.1881 | 0.1955 |
| `division` | 45 | 0 | 0.3618 | 0.3634 |
| `tempo_bpm` | 45 | 0 | 0.7649 | 0.7649 |
| `counts` | 16 | 0 | 0.5244 | 0.5244 |
| `onset_bpm` | 45 | 0 | 0.5936 | 0.5936 |
| `marker_bpm` | 40 | 0 | 0.7332 | 0.7332 |

## Absolute seconds, teacher-demo material (the W13(a) comparison)

W13(a)'s clip was a 37.8s demo video. The owner committed at: exercise ~3s, meter ~3s, tempo ~9-12s, quality ~9-12s, structure ~30-33s.

| subset | field | median t\* (s) | median span (s) | n |
|---|---|---|---|---|
| verified demo videos | `exercise` | 0.0 | 49.8155 | 4/4 |
| verified demo videos | `meter` | 0.0 | 49.8155 | 4/4 |
| verified demo videos | `grouping` | 4.961 | 49.8155 | 4/4 |
| verified demo videos | `division` | 14.1005 | 49.8155 | 4/4 |
| verified demo videos | `tempo_bpm` | 31.301 | 49.8155 | 4/4 |
| verified demo videos | `counts` | 26.628 | 49.8155 | 3/4 |
| Barre-1 demo takes (provisional) | `exercise` | 0.0 | 50.008 | 7/7 |
| Barre-1 demo takes (provisional) | `meter` | 0.0 | 50.008 | 7/7 |
| Barre-1 demo takes (provisional) | `grouping` | 6.064 | 50.008 | 7/7 |
| Barre-1 demo takes (provisional) | `division` | 9.384 | 50.008 | 7/7 |
| Barre-1 demo takes (provisional) | `tempo_bpm` | 41.745 | 50.008 | 7/7 |
| Barre-1 demo takes (provisional) | `counts` | 29.0845 | 50.008 | 6/7 |

## Per-clip convergence (condition granted, seconds)

| clip | span | `exercise` | `meter` | `grouping` | `division` | `tempo_bpm` | `counts` | `onset_bpm` | `marker_bpm` | maturity |
|---|---|---|---|---|---|---|---|---|---|---|
| 8-counts-2x | 9.7 | 0.0 | 0.0 | 1.0 | 1.0 | 5.8 | 5.1 | 2.8 | 2.2 | verified |
| 8-counts-triple | 8.4 | 0.0 | 0.0 | 1.8 | 3.6 | 5.5 | 7.2 | 6.9 | 3.6 | verified |
| barre1-A-s | 103.5 | 0.0 | 0.0 | 5.6 | 62.6 | 95.8 | 52.2 | 95.8 | 57.7 | provisional |
| barre1-B-d | 45.7 | 0.0 | 0.0 | 6.1 | 6.1 | 17.1 | 8.5 | 17.1 | 20.4 | provisional |
| barre1-B-el | 23.9 | 0.0 | 0.0 | — | — | — | 0.0 | — | — | provisional |
| barre1-B-er | 33.3 | 0.0 | 8.9 | 8.9 | 7.7 | 13.0 | — | 17.0 | 29.3 | provisional |
| barre1-C-d | 46.3 | 0.0 | 38.5 | 38.5 | 14.3 | 45.7 | 36.8 | 42.8 | 43.8 | provisional |
| barre1-C-el | 0.0 | 0.0 | 0.0 | — | — | — | — | — | — | provisional |
| barre1-C-er | 61.7 | 0.0 | 16.1 | 16.1 | 55.0 | 55.0 | 23.4 | 55.0 | 52.2 | provisional |
| barre1-D-d | 75.9 | 0.0 | 66.4 | 66.4 | 67.7 | 67.7 | 56.8 | 67.7 | 66.8 | provisional |
| barre1-D-el | 84.2 | 0.0 | 0.0 | 55.5 | 55.6 | 76.5 | — | 76.5 | — | provisional |
| barre1-D-er | 145.7 | 0.0 | 14.4 | 14.4 | 18.8 | 143.1 | 71.5 | 143.1 | 102.5 | provisional |
| barre1-E-d | 55.3 | 0.0 | 0.0 | 5.6 | 5.6 | 9.2 | — | 9.2 | 50.9 | provisional |
| barre1-E-el | 6.5 | 0.0 | 0.0 | — | — | — | — | — | — | provisional |
| barre1-E-er | 81.8 | 0.0 | 24.2 | 24.2 | 68.2 | 68.2 | 34.3 | 68.2 | 68.7 | provisional |
| barre1-F-d | 29.8 | 0.0 | 0.0 | 3.8 | 3.8 | 24.8 | 9.4 | 26.2 | 5.3 | provisional |
| barre1-F-el | 5.7 | 0.0 | 0.0 | — | — | — | — | — | — | provisional |
| barre1-F-er | 40.3 | 0.0 | 27.8 | 27.8 | 14.6 | 28.6 | 31.5 | 28.6 | 29.6 | provisional |
| barre1-G-d | 50.3 | 0.0 | 0.0 | 8.5 | 18.3 | 41.7 | 33.6 | 25.9 | 19.3 | provisional |
| barre1-G-el | 21.7 | 0.0 | 0.0 | — | — | — | — | — | — | provisional |
| barre1-G-er | 55.8 | 0.0 | 0.0 | 10.0 | 54.0 | 54.0 | 0.0 | 54.0 | — | provisional |
| barre1-H-d | 50.0 | 0.0 | 0.0 | 4.1 | 9.4 | 47.8 | 24.6 | 21.4 | 24.6 | provisional |
| barre1-H-el | 10.6 | 0.0 | 0.0 | — | — | — | — | — | — | provisional |
| barre1-H-er | 70.6 | 0.0 | 0.0 | 7.3 | 54.7 | 61.8 | 64.6 | 61.8 | 64.6 | provisional |
| exercise-1-demo | 60.1 | 0.0 | 0.0 | 3.0 | 3.0 | 32.4 | — | 32.4 | — | verified |
| frappe | 39.5 | 0.0 | 0.0 | 4.4 | 22.7 | 30.2 | 26.6 | 14.4 | 23.5 | verified |
| grande-battement | 39.5 | 0.0 | 0.0 | 5.5 | 5.5 | 16.5 | 17.0 | 16.5 | 15.5 | verified |
| plies-demo | 73.3 | 0.0 | 39.8 | 39.8 | 48.2 | 51.0 | 57.1 | 51.0 | 65.5 | verified |
| rig-mixed-4-4-104-quantities | 20.8 | 0.0 | 0.0 | 7.3 | 7.3 | 18.2 | — | 18.2 | — | verified |
| rig-names-2-4-120-clean | 16.5 | 0.0 | 0.0 | 3.6 | 5.4 | 16.5 | 12.4 | 6.8 | 14.8 | verified |
| rig-names-2-4-160-long | 26.9 | 0.0 | 0.0 | 4.1 | 8.6 | 25.4 | 11.6 | 9.7 | 22.9 | verified |
| rig-names-3-4-88-waltz | 16.8 | 0.0 | 0.0 | 1.7 | 1.7 | 16.1 | 16.1 | 12.6 | 16.1 | verified |
| rig-names-3-4-90-clean | 38.5 | 0.0 | 11.3 | 11.3 | 14.5 | 37.1 | — | 21.2 | 35.8 | verified |
| rig-names-4-4-100-quiet | 12.0 | 0.0 | 0.0 | 5.8 | 6.7 | 7.1 | 0.0 | 7.1 | 9.0 | verified |
| rig-names-4-4-104-clean | 19.6 | 0.0 | 0.0 | 4.0 | 11.9 | 19.6 | 16.7 | 16.7 | 19.6 | verified |
| rig-names-4-4-104-coda | 34.6 | 0.0 | 0.0 | 4.1 | 17.9 | 26.2 | 21.0 | 13.6 | 4.1 | verified |
| rig-names-4-4-104-explained | 22.7 | 0.0 | 0.0 | 3.9 | 21.2 | 20.1 | — | 17.2 | 16.6 | verified |
| rig-names-4-4-63-adagio | 34.3 | 0.0 | 0.0 | 18.8 | 18.8 | 19.9 | — | 19.9 | — | verified |
| rig-names-4-4-96-allegro | 22.5 | 0.0 | 0.0 | 4.2 | 21.4 | 10.9 | 20.4 | 10.9 | 19.7 | verified |
| rig-names-6-8-100-clean | 16.0 | 0.0 | 5.5 | 5.5 | 8.4 | 12.4 | — | 12.4 | 13.6 | verified |
| rig-numbers-2-4-120-clean | 18.1 | 0.0 | 2.3 | 2.3 | 7.4 | 10.8 | 4.1 | 7.4 | 18.1 | verified |
| rig-numbers-3-4-90-clean | 24.5 | 0.0 | 0.0 | 3.7 | 4.4 | 5.7 | 7.7 | 8.3 | 5.7 | verified |
| rig-numbers-4-4-104-bothsides | 39.0 | 0.0 | 0.0 | 3.3 | 3.3 | 7.4 | 6.8 | 33.4 | 4.4 | verified |
| rig-numbers-4-4-104-clean | 16.0 | 0.0 | 0.0 | 3.1 | 3.1 | 5.5 | 11.2 | 16.0 | 12.4 | verified |
| rig-numbers-4-4-104-duple | 11.6 | 0.0 | 0.0 | 3.1 | 4.3 | 3.7 | 6.6 | 6.9 | 3.7 | verified |
| rig-numbers-4-4-104-explained | 22.2 | 0.0 | 0.0 | 3.2 | 3.2 | 18.2 | 19.3 | 16.3 | 6.2 | verified |
| rig-numbers-4-4-104-fourx8 | 20.8 | 0.0 | 0.0 | 3.3 | 3.3 | 12.6 | 7.1 | 5.7 | 5.7 | verified |
| rig-numbers-4-4-104-prep | 13.9 | 0.0 | 0.0 | 3.3 | 3.3 | 6.0 | 4.3 | 6.0 | 6.0 | verified |
| rig-numbers-4-4-60-halftempo | 23.7 | 0.0 | 0.0 | 9.3 | 9.3 | 23.7 | 15.3 | 13.4 | 20.5 | verified |
| rig-numbers-4-4-80-triplet | 15.0 | 0.0 | 0.0 | 4.1 | 4.9 | 6.4 | 8.6 | 15.0 | 7.1 | verified |
| rig-numbers-6-8-100-clean | 16.7 | 0.0 | 0.0 | 3.3 | 3.3 | 8.8 | 5.8 | 5.8 | 4.6 | verified |
| rig-vocables-4-4-100-clean | 12.0 | 0.0 | 0.0 | — | — | — | — | — | — | verified |

## Identity check (P1)

Clips where the full prefix reproduced the untruncated replay exactly: 104 / 104.
