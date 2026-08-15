# Beat grids, the tap-assist annotator, and stage-1 pulse scoring

**Added at agent-charter rung 1 (EVAL-CHANGE, 2026-08-11), branch
`agent/rung-1-stage-scoring`.** Ground truth for stage-level scoring is a
per-clip **beat grid**: beat times anchored at **vowel onsets**
(P-centers), never word starts (Standing Lesson 1;
[review-1 §2.9](../research/review-1-onsets-pcenters.md)).

## Grid format — `evals/grids/<case-id>.yaml`

```yaml
format: 2                         # 1 still loads unchanged — see below
clip: rig-names-4-4-104-clean     # == case id == trace dir name
provisional: true                 # flipped false ONLY by the owner (rung 1.5)
media: audio/rig/rig-names-4-4-104-clean.mp3
media_sha256: fc9ddd09…           # provenance; matches the trace meta
annotator: peakrate-tap-assist/1
annotation_method: anchored       # anchored | from_scratch | null (rung 2.5)
created_at: "2026-08-11T22:36:00+00:00"
params: {sr: 16000, lowpass_hz: 10.0, …}   # frozen detector constants
beats:  [2.3244, 3.0053, …]       # EDITABLE annotation (owner corrects)
onsets: [2.3244, 3.0053, …]       # FROZEN peakRate evidence (never edited)
regions:                          # optional tags, PARALLEL to beats (rung 2.5)
  - {start: 25.91, end: 34.0, kind: free_time, note: "owner: that end is out of time"}
notes: ""
```

Two lists, two contracts: `beats` is the annotation the owner corrects
(delete non-beats, nudge times); `onsets` is the immutable acoustic
evidence — it feeds the hallucination guard and rung-2 debugging even
after `beats` is decimated to true beat times. **Provisional grids never
gate anything and always report as a separate slice**; only owner-verified
grids (`provisional: false`) will participate in typed-gate decisions.

Grids are *new* files under `evals/` (the charter's add-only ingestion
carve-out). Existing `evals/cases/`, `evals/traces/`, and
`evals/baseline.json` are untouched by grid work.

### Format 2 — tagged regions (rung 2.5, EVAL-CHANGE)

Format 2 is **additive**. `beats` is still a flat sorted time list with
identical meaning, `load_grid` accepts formats 1 and 2, and every grid
verified at rung 1.5 stays valid **with no edit** — stage-1 output over
the 28 verified grids is byte-identical across the change.

`regions` lifts the C6 limitation
([convention (d′)](annotation-convention.md)): three kinds of hole that a
flat time list cannot tell apart, tagged in a parallel structure rather
than interleaved with beats.

| kind | means |
|---|---|
| `silent_beat` | the pulse continued here but was not voiced (ruling (b)) |
| `free_time` | no metric beat exists here (rubato coda, out-of-time demo) |
| `excluded_explanation` | material deliberately outside the annotation |

Regions must be sorted, non-overlapping, and carry a known kind; anything
else is a load error. `annotation_method` records the rung-1.5
anchored-vs-from-scratch cohort ([convention §2.4](annotation-convention.md))
per grid, because the two cohorts sit ~20 ms apart and must never be
pooled silently. `null` means "not recorded", never "anchored".

In the Audacity round trip, regions are ordinary **region labels**
(`start<TAB>end<TAB>kind`, dragged rather than clicked) and beats stay
point labels. A dragged label whose text is not a known kind is a loud
error, never a silently absorbed beat time.

## Tap-assist annotator (needs `[prosody,eval]` extras)

```bash
python -m musical_perception.annotation generate            # all cases with media
python -m musical_perception.annotation generate --only ID --force
python -m musical_perception.annotation to-labels ID        # Audacity label track
python -m musical_perception.annotation from-labels ID FILE [--verified]
python -m musical_perception.annotation qc [ID ...]         # convention §4 checks
python -m musical_perception.annotation set-method ID anchored   # owner act
```

peakRate recipe (frozen from review-1 "steal this first" #1 — Oganian &
Chang 2019): bandpass 300–3000 Hz → rectify → 4th-order zero-phase
Butterworth low-pass **10 Hz** → derivative → half-wave rectify →
`find_peaks` with prominence ≥ 3·MAD and ≥120 ms spacing → keep peaks
Praat calls voiced within ±30 ms. A conditional degenerate-input guard
(silence-dominated signals only, MAD < 0.05% of max slope) is verified to
leave every DEV clip's output bit-identical to the pure recipe.

**Owner correction loop (rung 1.5):** `to-labels` → correct beats by ear
in Audacity → `from-labels ID FILE --verified` (flips `provisional`
off — an owner act; agent sessions never pass `--verified`).

## QC checks (`annotation qc`) — convention §4, amended 2026-08-14

Three checks per grid, implemented at rung 2.5 and required before a grid
is trusted. Thresholds are frozen constants in `annotation/qc.py`,
pre-registered before the checks were ever run:

| check | flags | threshold |
|---|---|---|
| `bpm_vs_label` | grid-implied BPM vs the case's `marking_bpm` | > 4% |
| `min_ioi` | an interval far shorter than the clip's own beat | < 0.5 × median IOI |
| `ioi_spread` | IOI variance *within* a phrase | CV > 15% |

Phrases are maximal runs split at intervals over 1.75 × the clip median;
CV is population sd / mean over ≥ 3 intervals. The BPM check is the
weakest of the three — it *false-passed* at +3.51% on a grid carrying
three spurious labels and a missing beat, which is why the other two were
ratified.

**Both amendment checks are suppressed inside tagged regions.** A gap the
annotator explained is not evidence of an error, and before format 2
there was no way to say so in the file. On `rig-names-4-4-104-coda`,
tagging the out-of-time coda as `free_time` moves within-phrase BPM from
108.93 to the 106.44 the owner recorded by hand and clears all seven
false positives; on `rig-names-4-4-63-adagio`, tagging the six unvoiced
beats reproduces the owner's 15.0% within-phrase CV — two of those six
gaps compress to 1.67–1.72 × the median under rubato, i.e. *below* the
break ratio, so only the tag can exclude them.

## stage1 suite

```bash
python -m musical_perception.evals run --suite tier0,tier1,stage1
```

Scores a predicted pulse stream per clip against `grid.beats` with
mir_eval-style one-to-one matching at **±70 ms**: precision / recall / F
plus **signed asynchrony** (predicted − reference, ms; negative = early).
The rung-1 pulse source is `whisper-word-starts` — the pipeline's only
timing channel today and the baseline the rung-2 acoustic extractor must
beat on these same grids. Aggregates are split
`aggregate_provisional` / `aggregate_verified` and sliced by
`count_style`; missing grids are listed loudly. stage1 pins no outcomes
and is absent from `evals/baseline.json` — the tier-1 gate is unaffected.

Caveat while grids are provisional: reference times are themselves
peakRate suggestions, so stage1 currently measures words-vs-peakRate,
not words-vs-human-truth. Interpret F and asynchrony as
instrument-calibration numbers until rung 1.5 verification.

## Tier-1 tempo metrics (Review 2 §4.2, verbatim)

`aggregate.tempo_metrics` adds to every tier-0/tier-1 summary, over
committed tempo rows: **Acc1/Acc2** at the field-standard ±4% (and house
±8%) with Acc2's fixed family {⅓, ½, 1, 2, 3} — the literature's name for
`truth_in_family` — plus **OE1** = log₂(est/ref) and **OE2** = OE1 after
removing the best family factor, reported as distributions with a
`between_levels` count (|OE2| ∈ (0.08, 0.585]) — the "landed between
metric levels" failure binary tallies cannot see. Informational only:
never enters outcomes, credit, or any gate.

## Onset-vs-token guard (ADR-016 clip-17)

`replay_bundle` now cross-checks each trace's transcript token count
against the grid's frozen `onsets` count on load and warns when
tokens > 1.5 × onsets + 8 (or when onsets are zero): a fluent transcript
without acoustic support is the hallucination signature that once scored
all-green. No grid → nothing to check. First real catch, same day it
landed: `rig-numbers-3-4-90-clean` (94 tokens vs 52 voiced onsets).
