# 03 · Landscape (July 2026)

What exists in the world — products, research, and component technology — as of
mid-2026. Sourced from a structured scan (2026-07-17); update this document
when the ground shifts. Bottom line: **the niche is empty while every
component has matured.**

---

## 3.1 Products ballet teachers actually use

The market converged on one category: **curated recordings + manual
parameterization**.

| Product | What it does | What it proves |
|---|---|---|
| **Ballegro Player** ("Your Virtual Ballet Pianist") | Curated class library, search by exercise/length/rhythm/genre, pitch-preserving tempo change incl. mid-exercise | Teachers want *parameterized* class music and will pay |
| **Ballet Class** (balletclass.eu) | Piano pieces in traditional class order; teacher sets **speed and number of bars** per exercise | The market's UI *is* `MusicalParameters` entered by hand |
| **BalletBox** (iOS, subscription) | Music by step; tempo/structure changes mid-playback; per-class playlists remember settings | Per-class memory matters; settings recur |
| **Cadence Ballet Music Player** (iOS) | 0.5–1.5× pitch-preserving stretch, A–B loops | Time-stretch is table stakes |
| **Cadance — Dance Tempo Control** | −50%/+100% tempo, playlists per class, iPad split-screen | Same |

Two conclusions. First, **willingness to pay is proven** for exactly this
problem, solved manually. Second, every one of these products still charges
the teacher the *interaction tax* — walk to the device, find the track, set
the tempo, set the bars, press play, twenty times per class. That tax is the
product opportunity: **delete the form** ([02 · Reframes](02-reframes.md)).

**Direct competitors for the automated version: none found.** No product,
startup, or funding announcement for a system that perceives a teacher's
marking and plays automatically.

## 3.2 The nearest AI systems solve a different problem

AI accompanists exist for **instrumentalists**: Metronaut (Antescofo SAS — the
IRCAM score-following spinoff), Cadenza Live Accompanist, MyPianist, Yamaha's
AI Music Ensemble (which drove a Disklavier beside human players in its 2016
demo). All of them **follow a performer through a known score** — 25 years of
mature research (Raphael's *Music Plus One*: HMM listening + Bayesian onset
prediction; Cont's *Antescofo*, used by major orchestras).

The ballet-class problem is structurally inverted: **perception, then
metronomic performance**. Infer the parameters up front from a spoken/gestural
demonstration; then hold steady (at barre, holding steady *is* the
musicianship). Score-following research supplies useful components (onset
prediction, tempo models) but nobody has built the perception half. That is
simultaneously the opportunity and the caution: there is no prior art to lean
on for the hard part.

## 3.3 Component stack status

| Component | State of the art (mid-2026) | Implication |
|---|---|---|
| Streaming ASR + word timestamps | AssemblyAI Universal-3 Pro ~150–240 ms; Deepgram Nova-3 ~450 ms median; local Whisper-lineage (SimulStreaming, WhisperFlow) ~1 s on laptop | Real-time transcription is commodity; timestamps survive streaming |
| Live multimodal APIs | Gemini Live: continuous audio, built-in VAD/interruption; **video at 1 FPS, 2-min uncompressed A/V session cap** | Enough to watch a 30–60 s *marking*; useless for fine movement timing → timing stays local (validates the repo's Whisper-owns-timestamps split) |
| On-device pose | MediaPipe/MoveNet class: 30+ FPS on phone-grade hardware | Solved for this use case |
| Full-duplex voice / turn-taking | Moshi lineage; NVIDIA PersonaPlex; ByteDance Seeduplex in production (−50% false response/interruption); OpenAI GPT-Live now the consumer default; FLEXI benchmark formalizes ~200 ms turn arbitration | Design patterns exist; the persistent failure mode is **premature interruption during pauses** — exactly why we adopt a silence-biased policy instead of trusting general turn-taking |
| "Is this speech for me?" | Device-directed speech detection: Apple (false-trigger mitigation, multimodal DDSD), Amazon (wake-word-free Conversation Mode via audiovisual directedness) — deployed only in constrained settings | The closest named research problem to "when do I start playing?" — and it is *not* solved in noisy open rooms. Our variant is inverted (act on domain-recognizable content not addressed to us). Mitigations: teacher mic, cue detection, silence bias |
| Always-on classroom audio | Merlyn Mind (far-field teacher assistant), TeachFX (whole-lesson recording analytics, district contracts) | Privacy/acceptance path has commercial precedent |

## 3.4 Music generation and playback reality

**Generative audio cannot honor structural contracts.** Lyria RealTime streams
BPM-conditioned music but has no bar-count, phrase, or ending control (tempo
changes require a context reset); Magenta RealTime 2 adds local deployment and
MIDI/text controls but no structural guarantees; Suno/Udio hold tempo to
±2–5 BPM and ignore their own time-signature pickers. None can promise
"exactly 32 counts of 3/4 at 138 with a cadence on 32." An accompanist that is
only *usually* right about structure is unusable.

**Symbolic MIDI enforces structure by construction.** REMI-style tokenizations
carry explicit bar/position/tempo tokens; controllable symbolic models
(MIDI-GPT, text2midi, NotaGen) exist for the day generation is wanted.
Pianoteq-class physical modeling renders MIDI at any tempo with commodity
realism.

**Pitch-preserving time-stretch of recordings** (zplane élastique /
SoundTouch — the engines inside the ballet apps above) is artifact-free in
roughly the **0.8–1.3×** band: a legitimate stopgap for tempo, but recordings
can never re-phrase to a different count structure. Hence
[06 · Performance Engine](06-performance-engine.md): symbolic-first, stretch
as fallback.

## 3.5 Academic whitespace — twice over

- **Machine comprehension of ballet marking: nothing found.** The closest
  work is Kirsh's cognitive ethnography of marking (establishes that marking
  is a structured, *recoverable* representation — the signal is real), and
  ballet CV limited to posture correction.
- **Beat tracking from spoken counting: nothing found.** Beat-tracking
  research assumes musical periodicity; deriving tempo from the timing of
  counted speech — this repo's onset-tempo method — appears academically
  unclaimed.
- Supporting precedent: conducting-gesture research extracts real-time BPM
  from movement at >86% recognition (the vision-side analogue); the human
  accompanist practice literature instructs teachers to "mark the first 8
  counts in the exact tempo you want" — i.e., the signal is *designed* to be
  readable, by profession-wide convention.

Consequence: the [benchmark corpus](08-benchmark-and-shadow-mode.md) would be
first-of-its-kind, and both the dataset and the onset-tempo method are
publishable ([10 · Pivots](10-pivots.md), P2).

## 3.6 What to watch

Re-scan triggers — any of these would materially change decisions here:

- A live multimodal API with **>1 FPS video + long-session A/V** (would let
  cloud models see movement timing; today it stays local).
- A generative music API with **hard bar-count and cadence contracts** (would
  reopen the generation-vs-library decision in
  [06](06-performance-engine.md)).
- Open-weight full-duplex models with room-robust directedness detection
  (would raise the ceiling of ladder rung 1 in
  [07 · Interaction Design](07-interaction-design.md)).
- Any entrant productizing marking perception (the empty niche will not stay
  empty forever; the [dataset](08-benchmark-and-shadow-mode.md) is the moat).

## Sources

Products: ballegroplayer.com · balletclass.eu · BalletBox, Cadence, Cadance
(iOS) · metronautapp.com · metamusic.ai · mypianist.app · Yamaha AI Music
Ensemble (yamaha.com/en/tech-design/research/technologies/muens)

Stack: AssemblyAI / Deepgram streaming docs · Gemini Live API docs ·
SimulStreaming; WhisperFlow (arXiv 2412.11272) · NVIDIA PersonaPlex · ByteDance
Seeduplex · FLEXI (arXiv 2509.22243) · Apple DDSD (arXiv 2312.03632) · Amazon
Alexa Conversation Mode · merlyn.org · teachfx.com

Music: Lyria RealTime docs · Magenta RealTime 2 (arXiv 2508.04651) · MIDI-GPT
(arXiv 2501.17011) · text2midi (AAAI 2025) · modartt.com (Pianoteq) · zplane
élastique / SoundTouch

Academic: Kirsh, *How Marking in Dance Constitutes Thinking with the Body* ·
Raphael, *Music Plus One* (ISMIR 2004) · Cont, *Antescofo* (2007) · conducting
gesture BPM (MDPI Appl. Sci. 2019; arXiv 2604.27957) · dance–music co-creation
(arXiv 2506.12008) · beat tracking (arXiv 2510.14391)
