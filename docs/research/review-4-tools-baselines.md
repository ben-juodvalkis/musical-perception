# Review 4 · Tools and baselines: off-the-shelf systems, analog tasks, datasets

*Part of the [voice-as-drum literature review](voice-as-drum-review.md)
(2026-08-09, companion to [ADR-016](../adr/016-rhythm-core-reset.md)).
Produced by a web-research agent. Scope assumptions carried through:
10–30 s close-mic clips, sparse accented syllabic onsets, 40–200 BPM,
interleaved plain talk, ~30 clips, accuracy first.*

## (a) Benchmark plan

### Tool-by-tool table

| Tool (2026 state) | Exact entry points | Outputs | Expected failure mode on this signal | Cost to run (30 clips) |
|---|---|---|---|---|
| **librosa 0.11.0** (healthy; Mar 2025 release) | `librosa.onset.onset_strength` (use `aggregate=np.median`), `librosa.onset.onset_detect(..., backtrack, delta, wait, pre_max, post_max, ...)`, `librosa.beat.beat_track(y, sr, start_bpm=120, tightness=100, trim, units='time')` (Ellis DP tracker), `librosa.beat.plp(tempo_min=30, tempo_max=300, win_length=384, prior=<scipy dist>)`, `librosa.feature.tempogram`, `librosa.feature.fourier_tempogram`, `librosa.feature.tempo` (moved from `librosa.beat.tempo`; see [0.11 docs](https://librosa.org/doc/latest/generated/librosa.feature.tempo.html), [plp docs](https://librosa.org/doc/main/generated/librosa.beat.plp.html)) | onsets; beats + global tempo (beat_track); pulse curve (PLP); no downbeats/meter | Spectral-flux envelope fires on every consonant burst and fricative, not on perceived beats (P-centre offset: vocal beats align to vowel onsets, so acoustic-onset beats run systematically early). DP tracker assumes quasi-continuous periodic onsets → invents beats through silence and talk; `start_bpm=120` default biases to double-time at 40–60 BPM markings. PLP better for gaps/tempo drift but returns a pulse, not a committed beat list. | `pip install librosa`; <1 s/clip CPU; zero friction |
| **madmom 0.16.1 / git main** (abandoned-ish; see §d) | `madmom.features.beats.RNNBeatProcessor` → `DBNBeatTrackingProcessor(min_bpm=55, max_bpm=215, transition_lambda=100, observation_lambda=16, correct=True, fps=100)`; `madmom.features.downbeats.RNNDownBeatProcessor` → `DBNDownBeatTrackingProcessor(beats_per_bar=[2,3,4], num_tempi=60)` ([docs](https://madmom.readthedocs.io/en/v0.16.1/modules/features/beats.html)) | beats; beats + downbeats (meter hypothesis via `beats_per_bar`) | **Default `min_bpm=55` silently octave-doubles 40–54 BPM clips — must set `min_bpm=40`.** RNN trained on music; on speech the activation is weak/noisy, and the DBN has no "no beat" state, so it hallucinates a grid through talk unless gated with VAD. `transition_lambda=100` enforces smooth tempo — good within a marking, bad across talk gaps. | Dedicated venv (Python 3.9 + PyPI, or py3.10–3.12 + git install, numpy<2); ~2–5 s/clip CPU |
| **Beat This!** (ISMIR 2024, Foscarin/Schlüter/Widmer; healthy) — [repo](https://github.com/CPJKU/beat_this), [paper](https://arxiv.org/html/2407.21658v1) | CLI `beat_this audio -o out.beats`; Python `from beat_this.inference import File2Beats; File2Beats(checkpoint_path="final0", device="cuda", dbn=False)` → `(beats, downbeats)`; checkpoints `final0-2` (78 MB), `small0-2` (8 MB) | beats + downbeats (no meter label; no tempo — derive from IBIs) | Best published beat/downbeat F1 on music (GTZAN), transformer, no DBN — but trained on 16 **music** datasets, zero speech. No tempo-prior knob exists (that's the point of dropping the DBN), so the 40–200 constraint cannot be injected; peak-picking will emit confident beats during talk. Optional DBN mode re-imports madmom (its [issue #9](https://github.com/CPJKU/beat_this/issues/9) documents that friction). | `pip install pytorch` + repo install; ~1–2 s/clip GPU, ~5–15 s CPU; 30 min setup |
| **Essentia 2.1-beta6-dev** | `RhythmExtractor2013(method='multifeature'\|'degara', minTempo=40, maxTempo=208)` → (bpm, ticks, confidence, estimates, bpmIntervals) — confidence only meaningful for `multifeature` ([ref](https://essentia.upf.edu/reference/std_RhythmExtractor2013.html)); `TempoCNN` via `essentia-tensorflow` with models `deepsquare-k16`, `deeptemp-k4/k16` from [essentia.upf.edu/models](https://essentia.upf.edu/models/) ([original tempo-cnn](https://github.com/hendriks73/tempo-cnn)) | beats + BPM + confidence; TempoCNN: global/local BPM only, no beats | RhythmExtractor2013's default 40–208 range actually matches the task best of any tool. Beat trackers inside are 2011–2013-era onset-driven → same syllable/silence problems. TempoCNN (Schreiber & Müller, ISMIR 2018) classifies 11.9 s mel windows into 30–286 BPM classes; trained on music (ballroom/EDM-heavy) → documented octave-error tendency, and any window overlapping talk is contaminated. | `pip install essentia essentia-tensorflow` (Linux/macOS wheels only); <1 s/clip |
| **aubio 0.4.9** (inactive; see §d) | `aubio.tempo`, `aubio.onset` (methods `hfc`, `specflux`, `complex`) | online beats + BPM | Oldest algorithmics of the set; nothing it does that librosa doesn't, plus install rot. | Skip |
| **BeatNet** (ISMIR 2021, Heydari & Duan) — [repo](https://github.com/mjhydri/BeatNet) | `BeatNet(1, mode='offline'\|'online'\|'realtime'\|'stream', inference_model='DBN'\|'PF')` | beats + downbeats (+ tempo, meter implied by downbeat spacing) — the only off-the-shelf joint meter-ish output | CRNN trained on music; cascade particle filter assumes music-density onset activations — sparse vocal onsets risk filter degeneracy/drift. `setup.py` hard-depends on `madmom>=0.16.1` → inherits all madmom install pain, no `python_requires` declared. Successor **BeatNet+** ([TISMIR 2024](https://transactions.ismir.net/articles/10.5334/tismir.198)) explicitly targets "diverse audio" incl. singing voice but code release lags. | Only worth it once the madmom venv exists; ~real-time per clip |
| **Speech-native hybrid (build, don't install)** | de Jong–Wempe syllable nuclei (Praat/parselmouth, [python port](https://github.com/drfeinberg/PraatScripts/blob/master/syllable_nuclei.py)) → nucleus times → (i) the existing `precision/tempo.py` math; (ii) as a sparse custom `onset_envelope` into `librosa.beat.beat_track`/PLP | syllable-nucleus onsets → tempo/beats | This is the task-appropriate front end: intensity-peak nuclei approximate P-centres far better than spectral flux. Failure mode: merges unstressed syllables at fast tempi, catches filled pauses/talk syllables (gate with the Gemini/Whisper marking-vs-talk classification first). | parselmouth already in-project; <1 s/clip |

### Recommended baselines to actually run, in order

1. **librosa control suite** — `onset_detect` + `beat_track` (sweep
   `start_bpm` ∈ {60, 120} and `tightness` ∈ {100, 400}) + `plp` with a
   log-normal `prior` over 40–200. Cheapest, fully transparent, and its
   onset envelope is the diagnostic plot for every other tool's behavior.
2. **Beat This! `final0`, `dbn=False`** — the 2026 SOTA music reference
   point. If a music-trained transformer already nails marked accents, that
   ceiling matters; where it fails during talk sections is the headline
   failure-mode figure.
3. **madmom RNN+DBN with `min_bpm=40, max_bpm=210`** — the only strong
   tracker with an explicit tempo-prior knob covering this range, plus
   `DBNDownBeatTrackingProcessor(beats_per_bar=[2,3,4])` for a meter
   hypothesis. Run both defaults and widened settings to quantify the
   octave-error cost of defaults.
4. **Essentia RhythmExtractor2013 (multifeature)** — for its confidence
   output (a usable "is there a beat at all?" signal) — plus **TempoCNN** as
   the global-BPM-only cross-check that will demonstrate octave-error
   structure.
5. **Syllable-nuclei hybrid** (de Jong–Wempe nuclei → `precision/tempo.py`,
   and nuclei → librosa DP) — the domain-native baseline the paper/ADR
   comparisons actually need, and the fairest comparator to the
   Whisper→Gemini→precision pipeline.
6. *(Optional)* **BeatNet offline** — only if the madmom venv already
   exists; its particle-filter behavior on sparse input and its meter output
   are informative but not worth standalone setup pain.

Run order rationale: 1 before 2–4 so the onset-envelope diagnostics exist to
interpret failures; 3 and 6 share a venv; 5 reuses the existing stack.
Everything is local and free; total compute for 30 × 10–30 s clips is
minutes. The real cost is ground truth: tap-annotate beats in Sonic
Visualiser then correct (~2–4 h for 30 clips) — annotate to the *perceived*
beat (vowel onset/P-centre), not the consonant burst, or every acoustic
tracker gets a spurious ~30–50 ms systematic penalty.

**Uniform harness protocol:** one CLI wrapper per tool → `.beats` text file
(Beat This's Sonic-Visualiser-compatible format) → score with mir_eval; gate
each tracker with/without Silero-VAD-restricted marking regions as an
ablation (VAD alone can't separate marking from talk — both are speech — so
use the existing Gemini/Whisper segmentation for the "marking-only"
condition).

### Evaluation tooling (exact names)

- **`mir_eval.beat`** ([source](https://github.com/craffel/mir_eval/blob/master/mir_eval/beat.py)):
  `f_measure` (`f_measure_threshold=0.07` s), `cemgil` (`cemgil_sigma=0.04`),
  `goto`, `p_score` (tolerance 0.2 × median reference IBI), `continuity` →
  returns **CMLc, CMLt, AMLc, AMLt** (`continuity_phase_threshold=0.175`,
  `continuity_period_threshold=0.175`; AML admits ×2, ×½, off-beat
  variants), `information_gain` (`bins=41`), `evaluate` runs all. Convention
  alert: `mir_eval.beat.trim_beats(min_beat_time=5.0)` discards the first
  5 s — on 10–30 s clips that eats up to half the clip; either disable and
  say so, or keep it and report clip lengths. Standard reporting (per the
  [Davies/Böck metric study](https://archives.ismir.net/ismir2014/paper/000238.pdf)
  and Beat This): **F-measure@70 ms + CMLt + AMLt**, Cemgil optional; with
  N=30 report per-clip distributions, not just means.
- **`mir_eval.tempo`**: single function `detection(reference_tempi,
  reference_weight, estimated_tempi, tol=0.08)` → `(p_score, one_correct,
  both_correct)` — the MIREX two-tempo protocol, awkward for single-tempo
  GT. For this case use **Accuracy1 (±4 %) and Accuracy2 (±4 % allowing ×2,
  ×3, ½, ⅓)** plus OE1/OE2 octave error, as implemented in
  [tempo_eval](https://github.com/tempoeval/tempo_eval) (Schreiber, Urbano &
  Müller, ["Music Tempo Estimation: Are We Done Yet?", TISMIR 2020](https://transactions.ismir.net/articles/10.5334/tismir.43)).
  Given the ADR-014 metric-level-family design, report Accuracy2 and the
  identity of the chosen level, not just Accuracy1.

## (b) Closest-analog work (rare — this is the relevant literature)

**Singing-voice beat tracking** (nearest published task; documents exactly
the expected failure — music trackers degrade on isolated vocals lacking
percussive/harmonic profile):

1. Heydari & Duan, "Singing Beat Tracking with Self-supervised Front-end and
   Linear Transformers," **ISMIR 2022** —
   [arXiv:2208.14578](https://arxiv.org/abs/2208.14578). Built 741
   separated-vocal tracks from GTZAN with beat annotations (source
   separation + tracking on full mix + manual correction);
   WavLM/DistilHuBERT **speech** SSL features beat music-trained baselines
   on vocals — direct evidence that speech representations help vocal
   rhythm. Project page:
   [U. Rochester AIR lab](https://labsites.rochester.edu/air/projects/MusicRhythmAnalysis.htm).
2. "SingNet: A Real-time Singing Voice Beat and Downbeat Tracking System,"
   **ICASSP 2023** — [arXiv:2306.02372](https://arxiv.org/abs/2306.02372).
3. Deng, Ju, Yang, Lui & Liu, "Efficient Adapter Tuning for Joint Singing
   Voice Beat and Downbeat Tracking with SSL Features," **ISMIR 2024** —
   [arXiv:2503.10086](https://www.arxiv.org/abs/2503.10086),
   [poster](https://ismir2024program.ismir.net/poster_48.html).
4. Heydari et al., "BeatNet+: Real-Time Rhythm Analysis for Diverse Music
   Audio," **TISMIR 2024** —
   [DOI 10.5334/tismir.198](https://transactions.ismir.net/articles/10.5334/tismir.198).

**Beatboxing / vocal percussion** (voice-as-drum onset detection +
classification):

5. Stowell & Plumbley, "Delayed Decision-making in Real-time Beatbox
   Percussion Classification," **JNMR 39(3), 2010** —
   [PDF](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/2581/STOWELLDelayedDecision2010POST.pdf)
   — plus the **beatboxset1** dataset
   ([archive.org](https://archive.org/details/beatboxset1)).
6. Delgado et al., "A New Dataset for Amateur Vocal Percussion Analysis,"
   **Audio Mostly 2019** —
   [arXiv:2009.11737](https://arxiv.org/pdf/2009.11737); follow-ups on
   query-by-vocal-percussion
   [arXiv:2110.09223](https://arxiv.org/pdf/2110.09223) and embeddings
   [arXiv:2204.04646](https://arxiv.org/pdf/2204.04646).
7. Mehta, Maheshwari, Joshi & Chakraborty, "BaDumTss: Multi-task Learning
   for Beatbox Transcription," **PAKDD 2022** —
   [Springer](https://link.springer.com/chapter/10.1007/978-3-031-05981-0_14),
   [code+data](https://github.com/LCS2-IIITD/BaDumTss-PAKDD22).

**Tapping-based tempo/beat ground truth:**

8. McKinney, Moelants, Davies & Klapuri, "Evaluation of Audio Beat Tracking
   and Music Tempo Extraction Algorithms," **JNMR 36(1), 2007** —
   [T&F](https://www.tandfonline.com/doi/abs/10.1080/09298210701653252); the
   [MIREX 2006 beat data](https://music-ir.org/mirex/wiki/2006:Audio_Beat_Tracking)
   is literally 40 human tappers per excerpt — the canonical treatment of
   metrical-level ambiguity that the ADR-014 alternates encode.

**Rap flow** (vocal rhythm against a grid):

9. Condit-Schultz, "MCFlow: A Digital Corpus of Rap Transcriptions,"
   Empirical Musicology Review 2016 —
   [paper](https://www.researchgate.net/publication/312406564_MCFlow_A_Digital_Corpus_of_Rap_Transcriptions);
   Ohriner, *Flow: The Rhythmic Voice in Rap Music* (OUP 2019, with corpus)
   — [chapter](https://academic.oup.com/book/37383/chapter/331379660).

**Speech-rhythm signal processing** (methods, not beat trackers):

10. de Jong & Wempe, "Praat script to detect syllable nuclei and measure
    speech rate automatically," Behavior Research Methods 41(2), 2009 —
    [Springer](https://link.springer.com/article/10.3758/BRM.41.2.385); v3 +
    filled pauses: de Jong, Pacilly & Heeren 2021,
    [DOI 10.1080/0969594X.2021.1951162](https://www.tandfonline.com/doi/abs/10.1080/0969594X.2021.1951162),
    scripts at [uhm-o-meter](https://sites.google.com/view/uhm-o-meter).
11. Gibbon, "The Rhythms of Rhythm" (rhythm-formant analysis: FFT of the AM
    envelope below 20 Hz — effectively a tempogram for speech) —
    [paper](http://wwwhomes.uni-bielefeld.de/gibbon/Dafydd_Gibbon_Publication_PDFs/2021_Gibbon_the-rhythms-of-rhythm.pdf),
    code under [github.com/dafyddg](https://github.com/dafyddg). Cheap DIY
    analog: run `librosa.feature.tempogram` on a speech-band intensity
    envelope instead of spectral flux.

**Count-in detection:** genuine gap — no dedicated paper or repo found
(searches across ISMIR/ICASSP/GitHub). Closest machinery: BeatNet streaming
mode and Heydari's online tracker
["Don't Look Back" (arXiv:2011.02619)](https://arxiv.org/pdf/2011.02619).
Treat "count-in" as this project's own contribution territory; the
Whisper+Gemini word classification is already ahead of anything published.

## (c) Datasets (transfer/sanity material with beat-level or onset-level vocal annotations)

| Dataset | Content | Annotations | Where |
|---|---|---|---|
| GTZAN separated vocals + beats (Heydari & Duan 2022) | 741 isolated singing tracks | beats (manually corrected) | via [AIR lab](https://labsites.rochester.edu/air/projects/MusicRhythmAnalysis.htm) / paper authors |
| Dagstuhl ChoirSet | a cappella choir multitrack, close mics (incl. larynx) | **beat positions**, F0, aligned scores | [Zenodo 4618287](https://zenodo.org/records/4618287), [TISMIR paper](https://transactions.ismir.net/articles/10.5334/tismir.48) |
| beatboxset1 (Stowell) | 14 beatboxers, 12–95 s clips | onsets + event classes, 2 annotators | [archive.org](https://archive.org/details/beatboxset1) |
| AVP / AVP-LVT | 9,867 vocal-percussion utterances, 28 subjects | onsets + kick/snare/hh classes (+ IPA syllables in LVT) | [Zenodo 3245959](https://zenodo.org/records/3245959), [Zenodo 5578744](https://zenodo.org/records/5578744) |
| BaDumTss | monophonic beatbox sequences | MIDI-aligned onsets/labels | [GitHub](https://github.com/LCS2-IIITD/BaDumTss-PAKDD22) |
| SMC_MIREX | 217 deliberately hard 30 s excerpts (quiet, non-percussive, slow, expressive) | beats | mirdata loader; [background](https://repositorio.inesctec.pt/server/api/core/bitstreams/4c744b20-9085-4aa7-9739-d05e19033d84/content) — best proxy for "sparse, tracker-hostile" audio |
| MIREX 2006 tempo/beat set | 160 × 30 s excerpts | 40 tappers/excerpt (perceptual tempo distributions) | [MIREX wiki](https://music-ir.org/mirex/wiki/2006:Audio_Beat_Tracking); access historically restricted |
| VocalSketch / Vocal Imitation Set / VimSketch | vocal imitations incl. 30 drum samples × 14 musicians | reference-imitation pairs (no beat grids) | [Interactive Audio Lab](https://interactiveaudiolab.github.io/resources/datasets.html), [figshare](https://figshare.com/articles/dataset/VocalSketch_Data_Set_v1_0_4/6372658/1) |
| BonnTempo | 5 languages read at 5 speeds | C/V intervals, syllables (speech-rate GT, no beats) | [ISCA 2004](https://www.isca-archive.org/interspeech_2004/dellwo04_interspeech.html) |
| Ballroom / GTZAN-Rhythm | music with beats | beats/tempo | via mirdata — use purely as harness smoke-test that each tool is wired correctly |

Note the structural gap: **no public corpus has beat annotations on rhythmic
*speech*** — 30 annotated marking clips would be a first-of-kind eval set;
SMC + Dagstuhl + beatboxset1 are the sanity triangle around it
(sparse-music / a-cappella-pitch / vocal-percussion).

## (d) Tool-rot warnings (as of Aug 2026)

- **madmom: effectively unmaintained.** Last release 0.16.1 was **Nov 14,
  2017** ([releases](https://github.com/CPJKU/madmom/releases)); ~60 open
  issues; PyPI wheel breaks on Python ≥3.10 (`collections.MutableSequence`
  import, [issue #535](https://github.com/CPJKU/madmom/issues/535)) and pins
  ancient numpy. Working recipe:
  `pip install git+https://github.com/CPJKU/madmom.git` on py3.10–3.12 with
  `numpy<2` and Cython present, in an isolated venv. Downstream contagion:
  **BeatNet requires madmom>=0.16.1** in setup.py, and Beat This's optional
  DBN mode does too ([beat_this #9](https://github.com/CPJKU/beat_this/issues/9),
  opened Jan 2025, still the documented state).
- **aubio: inactive.** PyPI 0.4.9 (2019), Snyk flags it discontinued-grade;
  repeated modern build failures
  ([#247](https://github.com/aubio/aubio/issues/247),
  [#328](https://github.com/aubio/aubio/issues/328)); 0.5.0-alpha never
  shipped. Skip.
- **Essentia: alive but permanently "2.1-beta6-dev".** Linux/macOS wheels
  only, no Windows pip; py3.12 build reports
  ([#1415](https://github.com/MTG/essentia/issues/1415)); TempoCNN requires
  the separate `essentia-tensorflow` package with its own TF pinning —
  containerize it (`mtgupf/essentia-tensorflow` Docker exists).
- **torchaudio: entered maintenance phase at 2.8 (Aug 2025)**; large API
  removals landed in 2.9, though `forced_align` was **reprieved after
  community pushback**
  ([pytorch/audio #3902](https://github.com/pytorch/audio/issues/3902)). If
  `MMS_FA`/CTC alignment is used for accent-onset refinement, pin torchaudio
  and record the version in the eval traces.
- **Healthy:** librosa (0.11.0, Mar 2025), Beat This (active CPJKU repo,
  C++/Rust ports appearing), parselmouth (0.4.7 Nov 2025, 0.5.0 in dev),
  mir_eval (now under the mir-evaluation org), Silero VAD, MFA (3.x,
  conda-forge only — don't fight pip). WhisperX remains popular but its
  wav2vec2 word timestamps are documented as less accurate than MFA
  ([whisperX #1247](https://github.com/m-bain/whisperX/issues/1247)) —
  relevant since the pipeline trusts Whisper timestamps: for the eval,
  spot-check accented-word onsets against MFA or Praat intensity peaks
  before treating transcript-derived intervals as ground truth.

One cross-cutting warning: every neural baseline above was trained
exclusively on music, and the singing-beat literature (items 1–4 in §b)
exists precisely because those models degrade on isolated voice; expect the
benchmark's most informative output to be *where* they break (talk
boundaries, sub-55 BPM, fricative-heavy syllables), not whether.
