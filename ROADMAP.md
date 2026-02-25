# Roadmap: Multimodal Perception

Where musical-perception is headed — from audio-only analysis of a teacher's
voice to combined audio+video understanding of timing, structure, and style.

## What We Analyze

A dance teacher counting aloud while demonstrating exercises. No music plays.
The input is the teacher's **voice** and **body**, not a musical track.

| Signal    | From Audio (voice)                        | From Video (movement)                        |
|-----------|-------------------------------------------|----------------------------------------------|
| Timing    | Word timestamps, prosodic emphasis, tempo  | Movement tempo, visual rhythm of repetitions |
| Structure | Counting patterns, spoken exercise names   | Exercise type from movement shape, reps      |
| Style     | Vocal dynamics, emphasis weight, energy    | Movement quality, energy, character          |

## Current Pipeline (audio-only)

```
audio file
  → Whisper transcription (word-level timestamps)
  → Praat prosody extraction (pitch + intensity contours)
  → Scaffolding: word classification (beat/and/ah), exercise detection
  → Precision math: tempo, subdivision, counting signature
  → MusicalParameters output
```

This works. The architecture labels (KEEP / DISPOSABLE / SCAFFOLDING) reflect
how each layer will evolve.

---

## Phase 1: Upgrade Audio Perception

Low-risk improvements to the existing audio pipeline. No architecture changes.

### Replace Whisper with WhisperX

- WhisperX adds wav2vec2-based forced alignment on top of Whisper
- Produces tighter word boundaries than Whisper's native timestamps
- Better word boundaries → better per-word prosody slicing in `prosody.py`
- Drop-in replacement for `perception/whisper.py`
- Open source, free: https://github.com/m-bain/whisperX

### Add WhiStress for Stress Detection

- Interspeech 2025 paper: extends Whisper with a stress detection head
- Outputs per-token stress labels alongside transcription
- Could replace or validate `CountingSignature.weight_placement`
- Complements Praat's numeric measurements with learned emphasis labels
- Open source: https://github.com/slp-rl/WhiStress

### Keep Praat/Parselmouth

No model currently outputs numeric pitch (Hz) and intensity (dB) per word.
Multimodal LLMs are qualitative ("this word sounds emphasized"), not
quantitative. The numbers:

- GPT-4o scores **53-59% on stress detection** (near random; humans: 92.6%)
- Gemini cannot return pitch contours
- Praat gives exact F0 and intensity at 10ms resolution, for free

Praat stays until models catch up.

### Fix: Clean Up Temp Files in prosody.py

The temp WAV file created with `delete=False` is never removed. Add
`os.unlink()` cleanup.

---

## Phase 2: Replace Scaffolding with Gemini

The SCAFFOLDING layer (`markers.py`, `exercise.py`) is hand-built heuristics.
Replace with a multimodal model that understands context.

### Why Gemini

| Requirement                    | Gemini 2.5/3 Flash | GPT-4o         | Claude        |
|--------------------------------|---------------------|----------------|---------------|
| Audio input (native)           | Yes                 | Yes            | Limited       |
| Video input (native)           | Yes                 | Frames only    | No            |
| Audio+video together           | Yes, one request    | Separate calls | No            |
| Structured JSON output         | Yes, schema-enforced| Partial        | N/A           |
| Cost (4-min clip, 1 FPS)       | ~$0.03              | ~$0.96         | N/A           |
| Cost (4-min clip, 4 FPS)       | ~$0.09              | ~$1.71         | N/A           |

Gemini is the only major API that processes video with its audio track in a
single request, returns schema-enforced JSON, and costs pennies per clip.

### What Gemini Replaces

| Current Module              | Gemini Replaces? | Notes                                    |
|-----------------------------|------------------|------------------------------------------|
| `scaffolding/markers.py`    | Yes              | Classifies words from audio context      |
| `scaffolding/exercise.py`   | Yes              | Identifies exercises from audio+video    |
| `perception/whisper.py`     | Partially        | Transcribes but lacks word-level timestamps |
| `perception/prosody.py`     | No               | Cannot extract numeric F0/intensity      |
| `precision/tempo.py`        | No               | Needs precise word timestamps as input   |
| `precision/subdivision.py`  | No               | Depends on precise timestamp intervals   |
| `precision/signature.py`    | No               | Depends on numeric prosodic measurements |

### New Module: perception/gemini.py

Thin DISPOSABLE wrapper around the Gemini API. Sends audio+video, receives
structured JSON for exercise type, word classifications, and qualitative
observations about emphasis and style. Does NOT replace the precision layer.

### Open-Source Alternatives

If API dependency is unacceptable:

- **Qwen2.5-Omni-7B** — true omnimodal (audio+video+text), Apache 2.0,
  runs on 24GB GPU (~12GB quantized). Best open-source option.
- **MiniCPM-o 2.6** — 8B params, outperforms GPT-4o-realtime on audio,
  supports real-time streaming, runs on consumer hardware.

---

## Phase 3: Add Video Perception

Extract timing, structure, and style from the teacher's visible movement.

### Important Caveat: Video LLMs Don't Truly Understand Motion

Apple's NeurIPS 2025 research found that when video frames are **temporally
shuffled**, GPT-4o and Gemini scores barely change. Current video LLMs do
"bag-of-frames" reasoning — they recognize what's present but don't track
motion dynamics. This means:

| Task                       | Use Video LLM (Gemini)     | Use Pose Estimation + Math |
|----------------------------|----------------------------|----------------------------|
| Exercise identification    | Yes (semantic, works well) | Also works                 |
| Visual tempo estimation    | No (unreliable)            | Yes                        |
| Repetition counting        | No (unreliable)            | Yes                        |
| Movement quality/style     | Yes (qualitative)          | Partial                    |
| Temporal grounding (when)  | Partial                    | Yes                        |

### Pose Estimation: MediaPipe BlazePose

- 33 body landmarks at 30 FPS, real-time on CPU
- Validated on dance datasets specifically
- Outputs joint positions → compute velocity/acceleration time series
- Open source, free, runs on-device

### The Key Insight: Precision Math Is Input-Agnostic

`precision/tempo.py` computes BPM from timestamped events. It doesn't care
whether those timestamps are:

- Whisper word onsets ("1", "2", "3", "4")
- MediaPipe knee-bend nadirs (bottom of each plié)
- Joint velocity peaks (maximum speed of arm movement)

Same algorithm, different input source. The KEEP label was right.

### New Modules

```
perception/
  pose.py          # DISPOSABLE — MediaPipe wrapper, extracts landmarks
  video_timing.py  # DISPOSABLE — converts pose sequences to timestamped events
```

These feed into the existing `precision/` layer unchanged.

### Specialized Models Worth Watching

For temporal grounding (pinpointing when actions happen in video):

- **TRACE** (ICLR 2025) — temporal grounding via causal event modeling
- **TimeLoc** (2025) — handles 30k+ frame videos, SOTA on multiple benchmarks
- **Grounded-VideoLLM** (EMNLP 2025) — discrete temporal tokens

For repetition counting:

- **UniCount** (2025) — universal rep counting, no predefined action categories

These are research models. Evaluate when video perception becomes a priority.

---

## Target Architecture

```
DISPOSABLE (perception — thin wrappers, will be swapped):
  audio/whisperx.py     Transcription + forced alignment → TimestampedWord[]
  audio/prosody.py      Praat pitch/intensity extraction → contours
  audio/whistress.py    Whisper + stress detection head → stress labels
  video/pose.py         MediaPipe pose extraction → landmarks
  video/timing.py       Pose sequences → timestamped events
  video/gemini.py       Gemini API → exercise ID, qualitative analysis

KEEP (precision math — pure functions, tested, input-agnostic):
  precision/tempo.py         BPM from any timestamped events
  precision/subdivision.py   Duple/triplet from interval ratios
  precision/signature.py     Prosodic weight profile from measurements

SCAFFOLDING → DELETED (replaced by Gemini / multimodal model):
  scaffolding/markers.py     → gemini.py or whistress.py
  scaffolding/exercise.py    → gemini.py (audio+video context)
```

---

## What We Ruled Out

| Model/Tool                         | Why Not                                            |
|------------------------------------|----------------------------------------------------|
| Music Flamingo, MERT, beat trackers | Trained on music, not speech                      |
| U-SAM                              | No weights released, no timestamps, stalled repo  |
| GPT-4o for prosody                 | $100/M audio tokens, 53% stress accuracy          |
| GPT-4o for video                   | No native video API, frame extraction required     |
| Claude for audio/video             | No audio input, no video input                     |
| openSMILE                          | Overkill — Praat gives exactly what we need        |
| wav2vec2/HuBERT for prosody        | Outputs embeddings, not interpretable Hz/dB values |

---

## Open Questions

- **How reliable is Gemini for ballet exercise vocabulary?** No published
  benchmarks exist for identifying pliés, tendus, battements from video.
  Needs empirical testing.
- **WhiStress on counting speech?** Trained on standard speech, not rhythmic
  counting. May need fine-tuning or at least evaluation on our domain.
- **Optimal video FPS for dance?** Gemini defaults to 1 FPS. Dance exercises
  at 120 BPM have beats every 0.5s. Need at least 4 FPS, probably 10 FPS
  (Gemini 3 Pro) for fast exercises like petite allegro.
- **MusicalParameters schema evolution?** The `meter`, `quality`, and
  `structure` fields are stubbed as `dict | None`. As video perception
  fills these in, they'll need proper types.

---

## Research Sources

Audio/speech models:
- WhisperX: https://github.com/m-bain/whisperX
- WhiStress (Interspeech 2025): https://github.com/slp-rl/WhiStress
- Fine-tuning Whisper for prosodic stress (ACL 2025): https://arxiv.org/abs/2503.02907
- StressTest benchmark: https://arxiv.org/html/2505.22765v2

Multimodal models:
- Gemini video understanding: https://ai.google.dev/gemini-api/docs/video-understanding
- Gemini audio understanding: https://ai.google.dev/gemini-api/docs/audio
- Gemini structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Qwen2.5-Omni: https://github.com/QwenLM/Qwen2.5-Omni
- MiniCPM-o: https://github.com/OpenBMB/MiniCPM-o

Video temporal understanding:
- Apple "Breaking Down Video LLM Benchmarks" (NeurIPS 2025): https://machinelearning.apple.com/research/breaking-down
- TRACE (ICLR 2025): https://github.com/gyxxyg/TRACE
- TimeLoc: https://github.com/sming256/timeloc
- Gemini 3 Pro vision: https://blog.google/technology/developers/gemini-3-pro-vision/
- UniCount (2025): repetitive action counting without predefined categories
- MediaPipe BlazePose: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

Prosody and speech rhythm:
- Parselmouth (Praat in Python): https://github.com/YannickJadworski/Parselmouth
- ProsodyLM (COLM 2025): https://github.com/auspicious3000/ProsodyLM
- Comparative toolkit evaluation (2025): https://arxiv.org/html/2506.01129v1
