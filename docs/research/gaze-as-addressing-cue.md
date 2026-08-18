# Eye contact as an addressing cue — feasibility note

**Date:** 2026-08-18
**Status:** PROPOSED design note. No pipeline change; nothing here is
measured yet. Owner question: *"A teacher will often make eye contact
with me to show me they want to give me tempo. Could we detect this via
camera?"*
**Feeds:** W7 (pose/gesture channel prototyping),
[Vision 07 §7.2](../vision/07-interaction-design.md) cue detection,
[Vision 05 §5.6](../vision/05-perception-strategy.md) calibration profile.

---

## 1. The short answer

Yes — the perception is tractable with off-the-shelf parts we already
depend on, and it is a good fit for
[Reframe 1](../vision/02-reframes.md) because eye contact is *already
part of the room's protocol*: nobody is being asked to learn a new
gesture. Two things decide whether it is worth building, and neither is
a computer-vision problem:

1. **Capture geometry.** "Eye contact with me" is a ray to the
   accompanist's head. A camera anywhere else measures a different ray.
   This constrains the rig, and it means **no clip in the current corpus
   can validate the idea** unless it happened to be shot from the piano.
2. **Lead time.** The signal is only useful if the eye contact *precedes*
   the audible cue ("aaand—") by enough to arm on. If the two are
   simultaneous, gaze adds little the audio channel does not already
   carry; if it leads by 0.5–3 s, it is a genuine early warning and a
   clean solution to the addressee problem.

Both are answered by measurement on new capture, not by argument. §5
pre-registers what to measure.

## 2. What the signal actually is

The teacher's eye contact does not mean "start playing." It means
*"attend to me — what comes next is for you."* Placed in the
[architecture](../vision/04-system-architecture.md), that is an
**addressing signal**: it selects the accompanist as the addressee of the
next few seconds of speech and gesture. The cue proper ("aaand—", the
arm lift) still arrives afterward.

That distinction sets its role, and the
[asymmetric error policy](../vision/07-interaction-design.md) §7.4 makes
it non-negotiable: **gaze may permit, never trigger.** It raises the
prior on a following cue, it can move the system from ATTENDING to
ARMED, and it can protect the grace rule for counting-over-music. It
must never by itself produce a `start`. A camera artefact that starts
the music is precisely the trust-destroying failure the whole policy is
built to avoid.

The corollary is a cheaper problem than it first appears: as a gate, the
detector is allowed to miss. Recall can be sacrificed freely; precision
cannot.

## 3. Feasibility, rung by rung

A three-rung ladder, each rung usable on its own, each needing no
dependency we do not already declare (`mediapipe`, `opencv-python` under
the `pose` extra):

**Rung A — head orientation from existing pose output (zero new deps).**
BlazePose already returns nose (0), eyes (1–6), ears (7–8) in the
33-point topology `pose.py` extracts. Ear-visibility asymmetry plus the
nose's horizontal offset between the ears gives a coarse yaw; shoulder
line gives torso reference, so "head turned relative to body toward the
piano" is computable today from `LandmarkTimeSeries`. Crude — roughly
10–15° — but it is the component that survives distance, and at
conversational eccentricities people turn the head, not just the eyes.

**Rung B — head pose from MediaPipe Face Landmarker.** Same package, no
new dependency: 478 landmarks and an optional 4×4 facial transformation
matrix giving proper yaw/pitch/roll, real-time on CPU. This is the
workhorse rung.

**Rung C — iris/gaze refinement.** The Face Landmarker's iris points
(and, if needed, an appearance-based gaze model) push angular error into
the ~4–6° range under favorable conditions. Only worth it if rung B's
precision proves insufficient.

**What the geometry demands.** Angular error θ at distance *d* is a
lateral error of *d*·tan θ: at 4 m, 5° ≈ 0.35 m and 12° ≈ 0.85 m. Piano
to nearest dancer is typically well over 1.5 m, so even rung A plausibly
separates "looking at the pianist" from "looking at the room" — but
nothing separates the pianist from someone standing beside the piano.

**What the optics demand.** Iris landmarks want an inter-ocular distance
of roughly 40–50 px. At 4 m through a ~70° horizontal FOV lens the frame
spans ~5.6 m, so a 16 cm face is ~55 px wide at 1080p — inter-ocular
~30 px, marginal. The same lens at 4K gives ~110 px face width and
~60 px inter-ocular, which works. So: **4K, or a narrower lens, if rung C
is wanted.** Rung A/B tolerate 1080p.

Temporally nothing is demanding: 15–30 fps is ample for events lasting
0.5–2 s.

## 4. The four things that will actually break it

1. **Mirrors.** A studio has a mirrored wall. A teacher facing the mirror
   — i.e. facing *away* from the pianist — produces a reflected face that
   can read as looking straight at the camera. This is the most likely
   source of false positives and it correlates with exactly the moment
   the teacher is *not* addressing us. Mitigations: mask the mirror
   polygon during install calibration; exploit that the reflected face is
   at ~2× the path length and therefore smaller; treat two simultaneous
   detections of the same identity as a mirror signature.
2. **Who is looking.** A dancer looking at the pianist is not the
   teacher. Needs teacher identification — enrolled face embedding at
   calibration, or fusion with the teacher-mic stream (who is speaking),
   or both.
3. **Camera-vs-pianist offset.** A camera on the music desk still sits
   maybe a metre from the accompanist's head; at 4 m that is ~14°, larger
   than the detector's own error. **Do not hard-code "looking at the
   camera."** Learn the target direction per teacher and per room from
   shadow-mode samples — the same mechanism as `cue_signature` in
   [Vision 05 §5.6](../vision/05-perception-strategy.md), stored as a
   `gaze_signature` in the calibration profile. This turns the rig's
   geometry from a constraint into a calibrated parameter, and it absorbs
   per-teacher differences in how squarely they address the piano.
4. **Dwell, not frames.** Incidental scanning glances are short.
   Intentional address dwells. Threshold on sustained fixation (order
   300–500 ms) with hysteresis, never on a single frame.

## 5. What to measure first (pre-registration)

The cheap experiment, on new shadow-mode capture with the camera on the
piano:

- **Q1 (the decider): lead-time distribution.** Time from eye-contact
  onset to cue onset, per exercise. Prediction: median lead 0.5–3 s and
  positive in ≥ 70% of exercises. If the median is ≤ 0.2 s or the sign is
  inconsistent, the channel is redundant with audio and W7 should not
  spend on it.
- **Q2: precision at usable recall.** Report precision at the operating
  point, plus **false-address events per class** as the headline (§7.4
  asymmetry). Recall is reported, never optimized.
- **Q3: rung A vs rung B.** Does the free head-orientation rung already
  separate the classes? Prediction: rung A reaches usable precision for
  the *gate* role; rung C is unnecessary.
- **Q4: mirror false-positive rate**, measured explicitly rather than
  assumed away.

Ground truth is human, per the standing convention: the owner labels
eye-contact intervals and marks which ones preceded a tempo-giving
event. Agent-proposed labels ship `provisional` and gate nothing.

## 6. Privacy

Gaze work processes faces, which is a step beyond pose landmarks and
must not widen scope by accident
([Vision 07 §7.8](../vision/07-interaction-design.md)): teacher only;
landmarks and derived angles retained, not frames; students explicitly
out of analytic scope; face embeddings for teacher identification are
calibration-profile data on-device, never uploaded. Frames retained only
in shadow/benchmark mode under the existing consent regime.

## 7. Recommendation

Fold this into **W7** as its first increment rather than opening a new
workstream, and do it in this order: capture geometry decided → Q1
measured on a handful of exercises → only then any detector work beyond
rung A. Rung A is a few hours on data that already exists in the right
form; rungs B–C are only worth their cost if Q1 says the channel leads
the audio.

**BLOCKED on owner (queue item):** was the Ballet Barre 1 material shot
from the accompanist's position, or from the room? If from the room, the
answer is not "unusable" — head-orientation *relative to the teacher's
own torso* and gaze toward a fixed room location are still measurable,
and the corpus can answer a weaker version of Q1 (does the teacher orient
toward one consistent off-camera location shortly before cueing?). But
Q2 needs capture from the piano.
