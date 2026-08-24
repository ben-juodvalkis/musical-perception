# Review 5 · Gaze, eye contact, and cueing gestures: off-the-shelf tools

*Companion to [gaze-as-addressing-cue.md](gaze-as-addressing-cue.md)
(2026-08-18), feeding W7. Scope assumptions carried through: a studio
camera 3–5 m from the teacher, mirrored wall behind, dancers in frame,
teacher-mic audio available, accuracy first, and — because this is
product-bound, not paper-bound — **commercial licensability is a
first-class column, not a footnote.***

> **Sourcing disclosure.** arxiv.org, pubmed.ncbi.nlm.nih.gov, and
> journals.sagepub.com are blocked by this environment's egress proxy.
> Repository and docs pages were fetched directly and are verified;
> the paper-level numbers in §(c) rest on abstract-level summaries and
> are marked *(abstract-level)*. Verify those before quoting externally.

---

## (a) Tool-by-tool table

| Tool (2026 state) | Entry points | Outputs | Failure mode on *this* signal | License reality | Cost |
|---|---|---|---|---|---|
| **MediaPipe Face Landmarker** ([Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)) — already an implicit dep via the `pose` extra | `FaceLandmarker` with `output_facial_transformation_matrixes=True`, `output_face_blendshapes=True` | 478 landmarks (incl. 10 iris), 52 blendshapes (incl. `eyeLookIn/Out/Up/DownLeft/Right`), 4×4 rigid head-pose matrix | Head pose is solid; the blendshape eye-look coefficients are an expression model, not a calibrated gaze vector — usable as a *relative* eye-offset feature, not as degrees. Degrades as face pixels shrink; iris points want ~40–50 px inter-ocular | **[Apache 2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE) — code and weights. The only genuinely ship-clean option in this table.** | Already installed; real-time CPU |
| **eye-contact-cnn** (Chong et al., [*Nat. Commun.* 2020](https://www.nature.com/articles/s41467-020-19712-x); [repo](https://github.com/rehg-lab/eye-contact-cnn)) | CLI over video/frames; ResNet-based binary classifier | Per-frame **eye-contact-with-the-camera** probability | Trained on egocentric video from glasses-mounted PoV cameras — i.e. it *assumes the camera sits at the addressee's head*. That is our exact question, but also our exact geometric constraint. No mirror handling; a reflected face is a face | **[GTRC academic license](https://github.com/rehg-lab/eye-contact-cnn/blob/master/LICENSE): "noncommercial internal research purposes" only, and derivatives are owned by GTRC.** Prototype-only | Small model; real-time CPU |
| **Gaze-LLE** ([CVPR 2025 highlight](https://openaccess.thecvf.com/content/CVPR2025/html/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.html); [repo](https://github.com/fkryan/gazelle)) | PyTorch Hub; `(448×448 image, head bbox)` → heatmap; 6 checkpoints (ViT-B/L × GazeFollow / +VideoAttentionTarget / +ChildPlay) | 64×64 **gaze-target heatmap** over the image; `_inout` variants add an in/out-of-frame score | Answers "*where in the room* is the teacher looking", not "is she looking at the camera". For a room camera that sees teacher **and** piano, that is the better question. For a camera *at* the piano the target is out of frame, so only the `inout` score carries signal | **Code MIT**; weights trained on GazeFollow / VideoAttentionTarget / ChildPlay — verify each dataset's terms before shipping | ViT-B/L + DINOv2; GPU for real-time. ONNX/DINOv3 port: [PINTO0309/gazelle-dinov3](https://github.com/PINTO0309/gazelle-dinov3) |
| **L2CS-Net** ([arXiv 2203.03339](https://arxiv.org/abs/2203.03339); [repo](https://github.com/ahmednull/l2cs-net)) | `from l2cs import Pipeline, render`; `Pipeline(weights='L2CSNet_gaze360.pkl', arch='ResNet50')`; `.step(frame)` | 3D gaze yaw/pitch per detected face | **3.92° on MPIIGaze but 10.41° on Gaze360** — and Gaze360 is the benchmark that matches our geometry (unconstrained, at distance). Budget ~10°, not ~4°. Needs a separate face detector | **Code MIT, weights are the trap:** the [Gaze360 license](https://github.com/erkil1452/gaze360/blob/master/LICENSE.md) bars commercial use of "models trained on dataset". Research-only in practice | ResNet-50; real-time GPU. On-device port exists ([Luxonis](https://models.luxonis.com/luxonis/l2cs-net/7051c9d2-78a4-420b-91a8-2d40ecf958dd)) |
| **OpenFace 3.0** (CMU MultiComp Lab, 2025; [repo](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0), arXiv 2506.02891) | `pip install openface-test`; `openface download`; CLI `openface detect`, or `FaceDetector` / `LandmarkDetector` / `MultitaskPredictor` | 68 landmarks, gaze (yaw/pitch), head pose, action units, 8-way emotion — one multitask model | The convenient "everything at once" baseline, and AUs are a bonus channel (brow raise on the cue). Not specialized for eye-contact-with-camera; still needs the target-direction calibration | **Commercial use requires a CMU licence** (innovation@cmu.edu / CMU Flintbox). Research-clean, product-encumbered | Lightweight, real-time |
| **Onfocus / ECIIN** ([Sci. China Inf. Sci.](https://link.springer.com/article/10.1007/s11432-020-3181-9); [repo](https://github.com/wintercho/focus)) | Research code | Binary individual-camera eye contact from **unconstrained** stills | The in-the-wild framing (OFDIW, 20,623 images) is closer to a room camera than the egocentric-glasses framing. Single-image, so no dwell modelling — we'd add temporal smoothing ourselves | Research code; terms unstated — assume non-commercial | Prototype-only |
| **Mirror segmentation** — MirrorNet ([ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Where_Is_My_Mirror_ICCV_2019_paper.pdf)), PMD, SANet, HetNet, SATNet (~85% IoU on MSD), [MirrorSAM2](https://arxiv.org/pdf/2509.17220) | Research code | Mirror region mask | Solves our named worst failure mode — but note it is a **one-time install-calibration** problem, not a per-frame one. A studio's mirror wall does not move | Research licences, and irrelevant if used offline once | Run once per room, or just draw the polygon by hand |
| **InsightFace** (teacher identification) | `pip install insightface` | Face embeddings for enrolment | The obvious pick for "which face is the teacher" | **Code MIT, [pretrained models non-commercial](https://github.com/deepinsight/insightface/issues/2022).** Same trap again | Real-time |

## (b) The structural finding: two geometries, two different tools

The tools do not sort by accuracy. They sort by **where the camera is**,
and each geometry makes a different tool correct:

**Geometry 1 — camera at the accompanist's position.** The question is
"is she looking at *me*", a binary. This is exactly what eye-contact-CNN
was built for (its training data is a PoV camera worn by the interaction
partner), and its reported numbers are the strongest evidence in this
review that the task is solvable at all. Cost: the rig is pinned to the
piano, and the model is licence-locked to prototyping.

**Geometry 2 — room camera seeing teacher and piano together.** The
question becomes "is her gaze target the piano region", which is
gaze-*target* estimation — Gaze-LLE's problem, not eye-contact-CNN's.
This geometry is more forgiving (no rig constraint, existing corpus may
partly serve, the piano is a fixed in-frame region to calibrate once) and
it degrades more gracefully: a heatmap that lands *near* the piano is
still evidence, whereas a binary that flips is just wrong.

That was not obvious before the survey, and it changes the
[design note's](gaze-as-addressing-cue.md) framing: **the capture
decision is not "at the piano or nothing" — it is a choice between two
tractable setups with different tool stacks.** Geometry 2 deserves the
first prototype precisely because it does not require the rig decision to
be made first.

## (c) Closest-analog work

**Ensemble cueing gestures — the domain's own literature, and the best
find in this review.** Bishop & Goebl, ["Beating time: How ensemble
musicians' cueing gestures communicate beat position and
tempo"](https://journals.sagepub.com/doi/10.1177/0305735617702971)
(*Psychology of Music*, 2018) studies precisely our event: one musician
cueing another into a piece. Findings *(abstract-level)*: **gesture
acceleration patterns indicate beat position** — specifically peak
acceleration, and the deceleration period following acceleration peaks,
in leaders' head-nodding gestures — while **gesture periodicity,
duration, and peak velocity indicate tempo**. Viewers synchronised best
with gestures low in jerk and large in magnitude. And, directly relevant:
**visual cues at re-entry points after long pauses are especially
salient — pianists synchronise more precisely with those than with cues
where timing is already predictable.** A ballet class is re-entry after
talk, every single time.

The consequence is larger than gaze. The owner's question was "eye
contact to show they want to *give me tempo*" — and this literature says
the giving is real and measurable: the head nod carries beat position and
tempo in kinematics we already compute. `precision/dynamics.py` derives
velocity from `LandmarkTimeSeries`; peak acceleration and periodicity are
a short step from there. **Eye contact selects the addressee; the nod
carries the beat.** W7 should scope both, and the nod may be the more
valuable half.

See also [Move like everyone is watching: social context affects head
motion and gaze in string quartet
performance](https://www.tandfonline.com/doi/full/10.1080/09298215.2021.1977338)
(2021) for how ensemble gaze behaviour shifts with audience presence — a
caution that rehearsal-room calibration may not transfer to a watched
class.

**Addressee detection — the tempering counter-evidence.** The
human-machine addressee-detection literature ([Tsai et al.
2015](https://www.slaney.org/malcolm/Google/Tsai2015MultimodalAddresseeDetectionHumanHumanComputerInteraction.pdf);
[Sensors 2020](https://www.mdpi.com/1424-8220/20/9/2740)) is the closest
work to our gate role, and it does **not** flatter the gaze channel:
energy-based acoustic features dominated, and **head pose "yields little
nonredundant information due to the system acting as a situational
attractor"** — people look at the device anyway, so gaze stops
discriminating. Two reasons this may not transfer: their device sat in
front of a seated user, whereas our accompanist is one of several
competing targets in a large room; and their task was per-utterance
classification, not the *lead-time* question §(e) makes primary. But it
is the strongest published reason to expect Q1 to come back negative, and
it should be cited in the pre-registration rather than discovered
afterwards.

**Eye contact as a clinical measure.** The Chong et al. line of work
(autism assessment) is why a well-validated eye-contact detector exists
at all, and it supplies the only human-parity benchmark: precision 0.936
/ recall 0.943 against 10 trained human coders at 0.918 / 0.946
*(abstract-level)*, from 4.3 M annotated frames over 103 subjects. Read
as a ceiling under favourable conditions (close-range PoV camera, dyadic
interaction), not as a forecast for a 4 m studio shot.

## (d) The licensing trap (read before building anything)

The same pattern recurs across every capable tool: **permissive code,
research-only weights.** L2CS-Net is MIT but its Gaze360 weights are
barred from commercial use *by the dataset licence, which explicitly
names "models trained on dataset"*. InsightFace is MIT with
non-commercial models. OpenFace 3.0 needs a CMU licence. eye-contact-CNN
is non-commercial and assigns derivative ownership to GTRC.

Practical consequence: **prototype freely, but do not let a research
checkpoint become load-bearing.** The honest options for a shipping
system are (1) MediaPipe-only features, Apache 2.0 throughout, coarser
but ours; (2) train our own head/gaze head on permissively-licensed or
self-collected calibration data — plausible precisely because the gate
role tolerates low recall and needs only a per-teacher decision boundary;
(3) buy a commercial licence. Deciding this late is how a working
prototype turns into an unshippable one.

## (e) What to actually run, in order

1. **MediaPipe Face Landmarker on existing Barre 1 video** — head-pose
   yaw/pitch time series per frame, plus the `eyeLookIn/Out` blendshapes.
   Apache 2.0, already installed, no rig decision required. Deliverable:
   does the teacher's head orientation cluster around one consistent
   off-camera direction in the seconds before a cue? That answers the
   weak form of Q1 from the design note **on data we already have.**
2. **Nod kinematics from `dynamics.py`** — peak acceleration and
   periodicity of head motion in the 3 s before each cue, per Bishop &
   Goebl. Cheapest high-value experiment in the whole plan, needs no new
   model at all, and tests the "gives me tempo" claim literally.
3. **Gaze-LLE `_inout` ViT-B on the same clips** — gaze-target heatmaps;
   does the mass concentrate on one room location before cues? Validates
   geometry 2 before any rig is bought.
4. **eye-contact-CNN** — prototype only, and only once a piano-position
   recording exists. Treat its output as the *upper bound* the
   licence-clean path is aiming at, not as a component.
5. **Mirror mask** — hand-drawn polygon at install. Do not install a
   segmentation network for a wall that never moves.

Scoring follows the design note's §5 and Vision 07 §7.4: precision at a
fixed operating point, **false-address events per class as the headline**,
recall reported but never optimised, and Q1's lead-time distribution as
the decider.

## (f) Datasets (for transfer/sanity, not for training a shipped model)

GazeFollow and VideoAttentionTarget (gaze targets in scenes; Gaze-LLE's
training data) · ChildPlay (gaze targets, non-dyadic, in-the-wild) ·
Gaze360 (238 subjects, 360° gaze, **non-commercial**) · ETH-XGaze and
MPIIFaceGaze (close-range, high-precision, research-only) · OFDIW (20,623
in-the-wild camera-eye-contact images) · MSD (4,018 mirror-segmentation
images). None contains a ballet studio; all are transfer material.

## (g) Rot and caveats as of Aug 2026

- **Angular-error numbers do not transfer between benchmarks.** Quote
  Gaze360-class figures (~10°) for our setting, never MPIIGaze-class
  figures (~4°). A 6° difference is 0.4 m of lateral error at 4 m.
- **MediaPipe blendshapes are not calibrated gaze.** `eyeLookIn/Out` are
  expression coefficients; treat as monotonic features, fit the mapping
  per teacher, never read as degrees.
- **Gaze-LLE needs head bounding boxes**, so a face/person detector sits
  upstream and its licence matters too.
- **eye-contact-CNN's derivative clause** (GTRC owns modifications) makes
  even fine-tuning it a legal question, not just a technical one.
- Everything here processes faces of everyone in frame. The privacy
  posture in [Vision 07 §7.8](../vision/07-interaction-design.md) —
  teacher only, landmarks retained not frames, students out of analytic
  scope — must be enforced in the prototype, not retrofitted.
