# ADR-004: Quality Model Redesign + Pose Estimation (Phase 3b)

**Date:** 2026-02-25
**Status:** Accepted

## Context

Phase 3a (ADR-003) introduced a six-dimension `QualityProfile` with numeric
floats. While functional, review revealed two problems:

1. **Redundant dimensions.** `smoothness` (staccato→legato) and `attack`
   (soft→percussive) measure opposite ends of the same spectrum. `weight`,
   `groundedness`, and `energy` overlap significantly. Six dimensions gave
   the illusion of independence when only ~3 axes were truly independent.

2. **Piano-specific framing.** The dimensions were described in terms of
   pianistic decisions (articulation, pedaling, register/voicing). The system
   needs to serve any music generator — pianist, DJ, generative AI — so the
   quality model should use generic musical vocabulary.

A review of Maestro Modern (a dance music selection app by the same author)
showed a clean three-axis approach: melodic↔rhythmic, sparse↔dense,
safe↔weird. That system describes *what kind of music to play*. Our system
describes *what kind of movement is happening* — complementary but similarly
concise.

## Decisions

### 1. Reduce QualityProfile from six dimensions to three

```python
@dataclass
class QualityProfile:
    articulation: float  # 0 = staccato, 1 = legato
    weight: float        # 0 = light, 1 = heavy
    energy: float        # 0 = calm, 1 = energetic
```

Each dimension is:
- A single spectrum (not two related dimensions)
- A standard musical term any generator understands
- Independent of the others (tested: exercises exist that separate all axes)

**Dimension collapse:**
- `smoothness` + `attack` → **articulation** (how movements connect)
- `groundedness` + `weight` → **weight** (gravitational quality)
- `energy` stays, reframed as **calm ↔ energetic** (activation level)
- `sustain` dropped — covered by articulation + tempo (already measured)

**Calibration points:**

| Dimension | ~0.1 | ~0.5 | ~0.9 |
|-----------|------|------|------|
| articulation | frappé | tendu | port de bras |
| weight | petit allegro | tendu | grand plié |
| energy | adagio | tendu | grand allegro |

**Independence check — exercises that separate axes:**
- Legato + light + calm: slow port de bras
- Legato + heavy + calm: adagio développé
- Staccato + light + energetic: petit allegro
- Staccato + heavy + energetic: grand allegro
- Legato + light + energetic: quick waltz balancé

### 2. Three-tier conceptual model for MusicalParameters

The output has a natural hierarchy, though the code stays flat:

| Tier | Fields | Who needs it |
|------|--------|-------------|
| **1. Rhythmic Foundation** (universal) | tempo, meter, subdivision | Every music generator |
| **2. Movement Character** (general-purpose) | quality, structure | Most adaptive generators |
| **3. Domain Detail** (consumer-specific) | exercise, words, markers, stress_labels, counting_signature | Domain-specific apps |

This framing keeps `MusicalParameters` as one dataclass (no over-engineering)
while making it clear which fields are universal and which are optional
enrichment. Different consumers ignore what they don't need.

### 3. Add pose estimation as second quality source

**`perception/pose.py`** (DISPOSABLE) — MediaPipe BlazePose wrapper.
Extracts 33-point landmarks from video frames into `LandmarkTimeSeries`.

**`precision/dynamics.py`** (KEEP) — Pure math module. Derives
`QualityProfile` from landmark time series:
- `articulation` ← jerk magnitude (high jerk = staccato)
- `weight` ← hip vertical position (low hips = heavy)
- `energy` ← velocity magnitude (high velocity = energetic)

### 4. Gemini-privileged synthesis

When both Gemini and pose quality are available, use weighted average:
- Gemini: 0.7 weight
- Pose: 0.3 weight

Rationale: Gemini has semantic understanding of the exercise (knows what a
fondu *should* feel like), while pose measures what *actually happened*.
Gemini's judgment is more reliable with limited pose calibration data.

When only one source is available, use it directly.

**This is provisional.** The synthesis strategy needs refinement:
- Per-dimension weights (pose may be better at energy, Gemini at articulation)
- Bayesian approach (Gemini as prior, pose as evidence)
- Pose override when confidence is high
- Validation across exercise types (currently calibrated on one video)

### 5. New optional dependency group

```toml
pose = ["mediapipe>=0.10", "opencv-python>=4.8"]
```

Heavy dependencies, so optional. The `[all]` group includes `[pose]`.

## Consequences

- **Breaking change** to `QualityProfile` — six fields replaced by three.
  No external consumers exist yet, so impact is zero.
- **Generic musical vocabulary** — any music generator can interpret
  articulation/weight/energy without domain-specific mapping.
- **Two quality sources** — Gemini (semantic) and pose (measured) can
  now contribute to the same output.
- **Scaling constants are provisional** — `dynamics.py` mappings are
  calibrated against a single 18-second video. Need more samples.
- **Pose requires video** — audio-only input uses Gemini quality as-is.

## Open Questions

1. **Synthesis weights.** 0.7/0.3 is a starting point. Need cross-exercise
   validation to determine if per-dimension weights are better.

2. **Signal-to-dimension mapping validation.** Jerk→articulation and
   velocity→energy are straightforward. Hip Y→weight works for exercises
   with vertical movement (plié, fondu) but may not generalize to
   traveling exercises (chaîné turns, grand allegro across the floor).

3. **More video samples needed.** All calibration is from one clip.

4. **Detection rate handling.** Currently, frames with failed pose
   detection are filled with NaN and excluded from computation. If
   detection rate is very low, quality estimates will be noisy.

## Files Changed

- `types.py` — `QualityProfile` reduced to 3 fields, added `LandmarkTimeSeries`
- `perception/gemini.py` — updated schema, prompt, and parser for 3 dimensions
- `perception/pose.py` — new module (DISPOSABLE)
- `precision/dynamics.py` — new module (KEEP)
- `analyze.py` — added `use_pose` parameter, pose+dynamics wiring
- `__main__.py` — updated quality display, added `--pose` flag
- `pyproject.toml` — added `[pose]` dependency group
- `tests/test_gemini_merge.py` — updated for 3-field QualityProfile
- `tests/test_dynamics.py` — new tests for dynamics computation and synthesis
