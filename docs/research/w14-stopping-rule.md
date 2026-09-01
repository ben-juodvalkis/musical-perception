# W14 — the commitment stopping rule

Generated 2026-09-01T01:10:02+00:00 by `scripts/w14-stopping-rule.py`, read-only over the W13(b) prefix replay (no media, no models, no API key). Pre-registered in the ledger, 2026-08-31.

**REPORTED-ONLY.** Nothing in `src/` changes, no eval suite gains a metric, no outcome is pinned. Two families are scored:

- **F1 k-stable-prefixes** — commit once the answer has held for `k` consecutive grid points (k = 1..8).
- **F2 confidence ≥ θ** — commit once the confidence the pipeline already computes first reaches θ (θ = 0.10..0.90).

**Operating point** = smallest median commit time among settings whose premature-commit rate is ≤ **0.10** on the slice. The ceiling was fixed before any number was seen; a family with no qualifying setting has **no operating point**, and that is the result.

Premature-commit rate is computed over the clips where the rule fired; `no-commit` is reported separately. Clips whose final answer for a field is `None` are excluded, the same exclusion W13(b) used. Commit is only permitted on a non-`None` answer: a rule that "commits" to *no answer yet* is not a stopping rule.

## The confidence map (F2's hard limit, read off the types)

| field | confidence stream the pipeline computes |
|---|---|
| `exercise` | exercise |
| `meter` | normalized_tempo |
| `grouping` | normalized_tempo |
| `division` | normalized_tempo |
| `tempo_bpm` | normalized_tempo |
| `counts` | **none — F2 cannot score this field** |
| `onset_bpm` | onset_tempo |
| `marker_bpm` | marker_tempo |

Four committed fields share one number (`NormalizedTempo.confidence`, the posterior mass of the committed ±8% neighbourhood, ADR-017), so F2 is a *metric-block* rule, not a per-field one. `counts` has no confidence at all.


## Condition granted · verified slice (n=30 clips)

| field | eligible n | F1 best k | F1 premature | F1 median t/span | F1 no-commit | F2 best θ | F2 premature | F2 median t/span | F2 no-commit |
|---|---|---|---|---|---|---|---|---|---|
| `exercise` | 30 | k=1 | 0.000 | 0.000 | 0.000 | θ=0.1 | 0.000 | 0.000 | 0.200 |
| `meter` | 30 | k=1 | 0.000 | 0.000 | 0.000 | **none** | — | — | — |
| `grouping` | 29 | k=2 | 0.000 | 0.213 | 0.000 | **none** | — | — | — |
| `division` | 29 | **none** | — | — | — | **none** | — | — | — |
| `tempo_bpm` | 29 | **none** | — | — | — | **none** | — | — | — |
| `counts` | 23 | **none** | — | — | — | **n/a — no confidence** | — | — | — |
| `onset_bpm` | 29 | **none** | — | — | — | **none** | — | — | — |
| `marker_bpm` | 26 | **none** | — | — | — | **none** | — | — | — |

## Condition granted · provisional slice (n=22 clips)

| field | eligible n | F1 best k | F1 premature | F1 median t/span | F1 no-commit | F2 best θ | F2 premature | F2 median t/span | F2 no-commit |
|---|---|---|---|---|---|---|---|---|---|
| `exercise` | 22 | k=1 | 0.000 | 0.000 | 0.000 | θ=0.1 | 0.000 | 0.000 | 0.182 |
| `meter` | 22 | k=1 | 0.000 | 0.000 | 0.000 | **none** | — | — | — |
| `grouping` | 16 | k=3 | 0.062 | 0.163 | 0.000 | **none** | — | — | — |
| `division` | 16 | **none** | — | — | — | **none** | — | — | — |
| `tempo_bpm` | 16 | **none** | — | — | — | **none** | — | — | — |
| `counts` | 14 | **none** | — | — | — | **n/a — no confidence** | — | — | — |
| `onset_bpm` | 16 | **none** | — | — | — | **none** | — | — | — |
| `marker_bpm` | 14 | **none** | — | — | — | **none** | — | — | — |

## Condition withheld · verified slice (n=30 clips)

| field | eligible n | F1 best k | F1 premature | F1 median t/span | F1 no-commit | F2 best θ | F2 premature | F2 median t/span | F2 no-commit |
|---|---|---|---|---|---|---|---|---|---|
| `exercise` | 0 | no eligible clip | — | — | — | no eligible clip | — | — | — |
| `meter` | 29 | k=2 | 0.000 | 0.213 | 0.000 | **none** | — | — | — |
| `grouping` | 29 | k=2 | 0.000 | 0.213 | 0.000 | **none** | — | — | — |
| `division` | 29 | **none** | — | — | — | **none** | — | — | — |
| `tempo_bpm` | 29 | **none** | — | — | — | **none** | — | — | — |
| `counts` | 15 | k=7 | 0.000 | 0.567 | 0.133 | **n/a — no confidence** | — | — | — |
| `onset_bpm` | 29 | **none** | — | — | — | **none** | — | — | — |
| `marker_bpm` | 26 | **none** | — | — | — | **none** | — | — | — |

## Condition withheld · provisional slice (n=22 clips)

| field | eligible n | F1 best k | F1 premature | F1 median t/span | F1 no-commit | F2 best θ | F2 premature | F2 median t/span | F2 no-commit |
|---|---|---|---|---|---|---|---|---|---|
| `exercise` | 0 | no eligible clip | — | — | — | no eligible clip | — | — | — |
| `meter` | 16 | k=3 | 0.062 | 0.163 | 0.000 | **none** | — | — | — |
| `grouping` | 16 | k=3 | 0.062 | 0.163 | 0.000 | **none** | — | — | — |
| `division` | 16 | **none** | — | — | — | **none** | — | — | — |
| `tempo_bpm` | 16 | **none** | — | — | — | **none** | — | — | — |
| `counts` | 2 | **none** | — | — | — | **n/a — no confidence** | — | — | — |
| `onset_bpm` | 16 | **none** | — | — | — | **none** | — | — | — |
| `marker_bpm` | 14 | **none** | — | — | — | **none** | — | — | — |

## Full F1 sweep — condition granted, verified slice

| field | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
|---|---|---|---|---|---|---|---|---|
| `exercise` | 0.000<br>@0.000 | 0.000<br>@0.126 | 0.000<br>@0.145 | 0.000<br>@0.172 | 0.000<br>@0.220 | 0.000<br>@0.255 | 0.000<br>@0.283 | 0.000<br>@0.322 |
| `meter` | 0.000<br>@0.000 | 0.000<br>@0.126 | 0.000<br>@0.158 | 0.000<br>@0.185 | 0.000<br>@0.220 | 0.000<br>@0.255 | 0.000<br>@0.283 | 0.000<br>@0.322 |
| `grouping` | 0.103<br>@0.188 | 0.000<br>@0.213 | 0.000<br>@0.255 | 0.000<br>@0.292 | 0.000<br>@0.326 | 0.000<br>@0.370 | 0.000<br>@0.404 | 0.000<br>@0.438 |
| `division` | 0.345<br>@0.188 | 0.310<br>@0.213 | 0.345<br>@0.263 | 0.276<br>@0.293 | 0.172<br>@0.351 | 0.172<br>@0.393 | 0.138<br>@0.450 | 0.103<br>@0.473 |
| `tempo_bpm` | 0.966<br>@0.188 | 0.724<br>@0.259 | 0.655<br>@0.355 | 0.517<br>@0.397 | 0.448<br>@0.459 | 0.444<br>@0.473 | 0.370<br>@0.571 | 0.400<br>@0.515 |
| `counts` | 0.609<br>@0.000 | 0.609<br>@0.153 | 0.304<br>@0.298 | 0.348<br>@0.378 | 0.182<br>@0.495 | 0.182<br>@0.544 | 0.191<br>@0.567 | 0.150<br>@0.600 |
| `onset_bpm` | 0.931<br>@0.293 | 0.655<br>@0.349 | 0.552<br>@0.404 | 0.552<br>@0.445 | 0.448<br>@0.465 | 0.407<br>@0.501 | 0.407<br>@0.537 | 0.360<br>@0.550 |
| `marker_bpm` | 0.923<br>@0.180 | 0.769<br>@0.231 | 0.577<br>@0.324 | 0.462<br>@0.395 | 0.348<br>@0.422 | 0.273<br>@0.449 | 0.238<br>@0.521 | 0.263<br>@0.526 |

Each cell: premature-commit rate over the top, median commit time as a fraction of span underneath.


## Full F2 sweep — condition granted, verified slice

| field | θ=0.1 | θ=0.2 | θ=0.3 | θ=0.4 | θ=0.5 | θ=0.6 | θ=0.7 | θ=0.8 | θ=0.9 |
|---|---|---|---|---|---|---|---|---|---|
| `exercise` | 0.000<br>@0.000 | 0.000<br>@0.000 | 0.000<br>@0.000 | 0.000<br>@0.000 | 0.000<br>@0.000 | 0.000<br>@0.000 | 0.000<br>@0.000 | 0.000<br>@0.000 | 0.000<br>@0.000 |
| `meter` | 0.103<br>@0.188 | 0.103<br>@0.188 | 0.103<br>@0.188 | 0.103<br>@0.188 | 0.103<br>@0.196 | 0.107<br>@0.192 | 0.107<br>@0.192 | 0.115<br>@0.192 | 0.125<br>@0.192 |
| `grouping` | 0.103<br>@0.188 | 0.103<br>@0.188 | 0.103<br>@0.188 | 0.103<br>@0.188 | 0.103<br>@0.196 | 0.107<br>@0.192 | 0.107<br>@0.192 | 0.115<br>@0.192 | 0.125<br>@0.192 |
| `division` | 0.345<br>@0.188 | 0.345<br>@0.188 | 0.345<br>@0.188 | 0.345<br>@0.188 | 0.345<br>@0.196 | 0.357<br>@0.192 | 0.321<br>@0.192 | 0.346<br>@0.192 | 0.333<br>@0.192 |
| `tempo_bpm` | 0.966<br>@0.188 | 0.966<br>@0.188 | 0.966<br>@0.188 | 0.966<br>@0.188 | 0.931<br>@0.196 | 0.929<br>@0.192 | 0.929<br>@0.192 | 0.962<br>@0.192 | 0.958<br>@0.192 |
| `counts` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `onset_bpm` | 0.931<br>@0.293 | 0.931<br>@0.293 | 0.931<br>@0.293 | 0.931<br>@0.293 | 0.897<br>@0.293 | 0.893<br>@0.295 | 0.846<br>@0.295 | 0.556<br>@0.392 | 0.750<br>@0.545 |
| `marker_bpm` | 0.923<br>@0.180 | 0.923<br>@0.180 | 0.923<br>@0.180 | 0.923<br>@0.180 | 0.923<br>@0.180 | 0.923<br>@0.180 | 0.923<br>@0.180 | 0.923<br>@0.180 | 0.923<br>@0.180 |

(θ shown every 0.10; the scored sweep steps by 0.05 and is complete in `w14-stopping-rule.json`.)


## Why F2 fails: the confidence runs backwards

F2 is nearly flat in θ, which a threshold sweep alone would not explain. The reason is in the confidence stream itself, read straight off the recorded prefixes (condition granted, verified slice):

| stream | median at the FIRST prefix that has one | median on the FULL clip | clips already ≥0.90 at that first prefix | clips never reaching 0.50 |
|---|---|---|---|---|
| `normalized_tempo` | 1.000 | 0.780 | 23/29 | 0/29 |
| `onset_tempo` | 0.740 | 0.770 | 0/29 | 0/29 |
| `marker_tempo` | 1.000 | 0.525 | 26/26 | 0/26 |
| `exercise` | 0.800 | 0.800 | 14/30 | 8/30 |

The metric block's confidence is **at its maximum when the pipeline knows the least** and *falls* as evidence arrives. That is not a miscalibrated threshold; it is a signal pointing the wrong way — consistency over two or three intervals is trivially perfect, and only starts paying for its mistakes once there are enough intervals to disagree. No θ can fix it, which is why the sweep is flat rather than merely badly placed.


## The owner's curve, laid against the best operating points

W13(a): on a 37.8s demo the owner committed exercise ~3s, meter ~3s, tempo ~9–12s, structure ~30–33s.

| field | owner t/span | F1 best (granted, verified) | F2 best | verdict |
|---|---|---|---|---|
| `exercise` | 0.079 | 0.000 | 0.000 | **earlier than the owner** |
| `meter` | 0.079 | 0.000 | — | **earlier than the owner** |
| `tempo_bpm` | 0.278 | — | — | no operating point |
| `counts` | 0.833 | — | n/a | no operating point |
