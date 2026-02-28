# ADR-007: Coherent Metric Interpretation

**Date:** 2026-02-28
**Status:** Accepted

## Context

After ADR-006 introduced BPM normalization, `MusicalParameters` had four
separate fields for tempo-related information: `normalized_bpm`,
`tempo_multiplier`, `meter`, and `subdivision`. These were determined
independently and could contradict each other.

The concrete problem (GitHub issue #10): Gemini reports 4/4 with triplet
subdivision for a tendu exercise that is actually in 3/4. The normalization
step already implicitly knew the answer — when it tripled the BPM to reach
the 70–140 range, that meant triple meter — but this information wasn't
propagated to meter or subdivision.

Musically, 3/4 at 120 BPM and 4/4 at 40 BPM with triplet subdivision produce
the same rhythmic surface. From the accompanist's perspective, the distinction
is irrelevant — the music played is identical. But the user needs a coherent
triple of (BPM, meter, subdivision) to communicate clearly, not a set of
fields that can say contradictory things.

## Decision

### 1. `NormalizedTempo` type (`types.py`)

A single dataclass that jointly commits to BPM + meter + subdivision as one
coherent answer:

```python
@dataclass
class NormalizedTempo:
    bpm: float              # Beat-level BPM in 70-140 range
    meter: Meter            # Derived from how raw BPM was scaled
    subdivision: str        # "none", "duple", "triplet"
    confidence: float       # 0-1
    raw_bpm: float          # Original BPM before normalization
    tempo_multiplier: int   # How raw was scaled (x2, x3, /2, etc.)
```

### 2. `interpret_meter()` function (`precision/tempo.py`)

A KEEP precision function that replaces the ad-hoc normalization block in
`analyze()`. Takes raw tempo signals from both onset detection and Gemini,
normalizes the BPM, and derives meter and subdivision from the multiplier:

- `multiplier=1`: BPM already at beat level — trust Gemini meter/subdivision
- `multiplier=2`: raw was at measure level, doubled — 4/4, no subdivision
- `multiplier=3`: raw was at measure level, tripled — **3/4, no subdivision**
- `multiplier=-2`: raw was at subdivision level, halved — duple subdivision
- `multiplier=-3`: raw was at subdivision level, divided by 3 — triplet subdivision

**Cross-signal check:** When onset BPM is in range (multiplier=1) but the
onset/Gemini BPM ratio is approximately 3, the function overrides to triple
meter. This catches the case where the onset detector finds the beat-level
pulse (~115 BPM) while Gemini locks onto the measure level (~40 BPM).

### 3. Backward compatibility

`MusicalParameters` gains `normalized_tempo: NormalizedTempo | None`. The old
fields (`normalized_bpm`, `tempo_multiplier`, `meter`, `subdivision`) are
preserved but marked deprecated — they are populated from `normalized_tempo`
for backward compatibility.

## Results on real files

| File | BPM | Meter | Subdivision | Correct? |
|------|-----|-------|-------------|----------|
| Exercise 1 Demo.m4v | 110.8 | 4/4 | duple | No — Gemini BPM now close to onset (ADR-006 prompt fix), cross-signal ratio ~1.09 |
| plies demo.m4v | 118.0 | 4/4 | duple | Yes |
| 8-counts-2x.aif | 129.8 | 4/4 | none | Yes |
| 8-counts-triple.aif | 82.7 | 4/4 | duple | No — normalize_tempo prefers /2 over /3 |

## What this does NOT solve

**Noisy upstream signals.** The interpret_meter() logic is correct when the
onset and Gemini BPM signals diverge by clean ratios (2x, 3x). In practice,
real-world videos contain mixed speech (explanations + counting), causing the
onset detector to pick up conversational speech as rhythmic sections. The
cross-signal check requires a ratio of ~3 between onset and Gemini BPM, which
doesn't occur when both detectors are confused by the same noisy input.

**Recommendation:** Until AI models improve at separating rhythmic counting
from conversational speech, teachers should be instructed to:
- Count with numbers (1, 2, 3...) rather than step names alone
- Separate explanation from counting — explain first, then count cleanly
- Be rhythmically precise with steady pulse and clear articulation

## Consequences

- `MusicalParameters` gains `normalized_tempo: NormalizedTempo | None`
- Downstream consumers should prefer `normalized_tempo` over the deprecated
  individual fields
- The `interpret_meter()` function is pure math (KEEP, no AI dependencies)
- 10 new unit tests cover all multiplier paths and cross-signal detection
- The framework is ready for improved upstream signals — when onset and Gemini
  BPM diverge cleanly, meter detection will be correct automatically
