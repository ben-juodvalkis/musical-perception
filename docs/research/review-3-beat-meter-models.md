# Review 3 · Probabilistic beat/downbeat/meter tracking and meter induction

*Part of the [voice-as-drum literature review](voice-as-drum-review.md)
(2026-08-09, companion to [ADR-016](../adr/016-rhythm-core-reset.md)).
Prior art for a joint posterior over (period, phase, meter, subdivision,
phrase length). Produced by a web-research agent.*

*Method note: most primary PDF hosts (ISMIR archives, JKU, Zenodo, S2 API,
PLOS/PMC, arXiv abs) were egress-blocked from the research environment;
details below are cross-verified via search-result extracts of the papers,
GitHub implementations (BeatNet, PIPPET READMEs fetched in full), and
publisher metadata. Everything load-bearing was verified against at least
one source.*

## 1. The bar-pointer model family (the direct ancestor)

**1.1 Whiteley, Cemgil & Godsill (2006), "Bayesian Modelling of Temporal Structure in Musical Audio," ISMIR 2006.**
The origin of the bar-pointer model. Latent state per frame k: bar position
φ_k ∈ [0, M_θ) (a "pointer" sweeping one bar), tempo/velocity φ̇_k (position
increment per frame), meter indicator θ_k (e.g., 3/4 vs 4/4 — different bar
lengths), and rhythmic-pattern indicator r_k. Dynamics: φ advances by φ̇ each
frame (mod bar length); φ̇ follows a (truncated) random walk — this is how
tempo drift is native, not bolted on; θ and r may switch only at bar
crossings with small probability. Two observation models: an inhomogeneous
**Poisson point-process** for symbolic/MIDI onsets whose intensity is a
function of bar position (an accent template per pattern/meter), and a
Gaussian-process model for raw audio frames. Inference: discretize (φ, φ̇) →
exact HMM forward–backward / Viterbi; jointly yields tempo + meter + phase
(downbeat = pointer zero-crossing). Cost was the known weakness: naive
discretization gives 10^5–10^6 states.
**Transferability: very high — the Poisson-points observation model was
designed for sparse symbolic onsets, which is exactly the marker stream; the
meter indicator gives 2/4 vs 3/4 vs 4/4 vs 6/8 as a first-class latent
variable.** ([ISMIR 2006 PDF](https://archives.ismir.net/ismir2006/paper/000044.pdf),
[Semantic Scholar](https://www.semanticscholar.org/paper/d55e5e4ee40ff78aa80302590c236fe7217d6414))

**1.2 Krebs, Böck & Widmer (2013), "Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical Audio," ISMIR 2013.**
First serious audio instantiation of the bar pointer. Same latent state;
observation model = **GMMs over 2-band spectral-flux features, one GMM per
(rhythmic pattern, 16th-note bar-position cell)** — i.e., the accent profile
of each meter/style is learned as an emission template indexed by bar
position. Inference: Viterbi over the discretized HMM; 8 ballroom patterns,
meters 3/4 and 4/4. Meter/pattern selection falls out of decoding.
**Transferability: the architecture transfers; the learned ballroom GMM
templates do not. Relearn "emission given bar position" from marking
salience features.** ([dblp](https://dblp.org/rec/conf/ismir/KrebsBW13.html),
[madmom source](https://madmom.readthedocs.io/en/v0.16/_modules/madmom/features/downbeats.html))

**1.3 Krebs, Böck & Widmer (2015), "An Efficient State-Space Model for Joint Tempo and Meter Tracking," ISMIR 2015.**
The engineering breakthrough that makes exact joint inference cheap.
Reparameterize tempo as an **integer number of frames per beat**; give each
tempo its own bar-position discretization with exactly that many position
states per beat, so the pointer advances deterministically one state per
frame ("left-to-right" structure), and **tempo transitions happen only at
beat boundaries** with p(T'|T) ∝ exp(−λ|T'/T − 1|) (transition_lambda),
optionally with log-spaced tempo states. Result: orders-of-magnitude fewer
states than the 2006/2013 discretization with *higher* beat/downbeat
accuracy. This is what madmom's `BeatStateSpace`/`BarStateSpace` +
`BarTransitionModel` implement.
**Transferability: this is the state-space to copy. At 50 fps, 30 s, tempo
40–200 BPM (≈15–75 frames/beat), meters {2,3,4,6} beats/bar: |states| ≈
Σ_meters Σ_T (beats/bar × T) ≈ 2×10^4 — exact Viterbi/forward–backward in
milliseconds.** ([JKU PDF](https://www.cp.jku.at/research/papers/Krebs_etal_ISMIR_2015.pdf),
[IR Anthology](https://ir.webis.de/anthology/2015.ismir_conference-2015.9/),
[Zenodo](https://zenodo.org/records/1414966))

**1.4 Holzapfel, Krebs & Srinivasamurthy (2014), "Tracking the 'Odd': Meter Inference in a Culturally Diverse Music Corpus," ISMIR 2014; and Srinivasamurthy et al. (2015), "Particle Filters for Efficient Meter Tracking with Dynamic Bayesian Networks," ISMIR 2015.**
Bar-pointer applied to Turkish makam (9/8 usul), Cretan, and Carnatic
corpora: meter inference among genuinely odd meters works when patterns are
learned per style; exact HMM inference is the bottleneck, so
auxiliary/mixture particle filters approximate it with large speedups at
small accuracy cost.
**Transferability: evidence the family handles non-4/4 meter discrimination
robustly; also evidence particle filters are NOT needed at these clip
lengths — PF only pays off when the state space explodes (long cycles, many
patterns).** ([ISMIR 2014 PDF](https://archives.ismir.net/ismir2014/paper/000265.pdf),
[Zenodo](https://zenodo.org/records/1415000),
[CompMusic](https://compmusic.upf.edu/ismir-2015-pf))

**1.5 Cemgil & Kappen line: Cemgil, Kappen, Desain & Honing (2001), "On Tempo Tracking: Tempogram Representation and Kalman Filtering," J. New Music Research 29(4):259–273; Cemgil & Kappen (2003), "Monte Carlo Methods for Tempo Tracking and Rhythm Quantization," JAIR 18:45–81.**
Input is a **sparse list of performed onset times** (MIDI piano), not audio —
the closest classical setting to the marker stream. Model (2003): a
**switching state-space model**: discrete switch variables = quantized score
locations (which grid position each onset belongs to, i.e., rhythm
quantization), continuous latent = (beat time, tempo period) with
linear-Gaussian dynamics (tempo random walk → drift handled by the Kalman
prior). Tempogram (2001) = a localized wavelet-like transform of the onset
train scoring (period, phase) evidence, used to initialize/track via Kalman
filtering. Inference (2003): exact posterior intractable
(discrete×continuous) → MCMC/simulated annealing offline, Rao-Blackwellized
particle filters online (sample switches, Kalman-marginalize tempo).
**Transferability: very high conceptually — joint (tempo, phase,
onset-to-grid assignment) posterior from sparse noisy onsets with expressive
deviation is literally this project's problem minus meter; add a
meter/subdivision switch and the model is complete. The RBPF machinery is
overkill for 10–30 s offline clips, where a discretized HMM does the same
exactly.** ([JAIR](https://jair.org/index.php/jair/article/view/10322),
[arXiv:1106.4863](https://arxiv.org/abs/1106.4863),
[JNMR paper](https://www.mcg.uva.nl/mcg-2023/papers/mmm-27.pdf),
[NeurIPS 1999 precursor](http://papers.neurips.cc/paper/1999-tempo-tracking-and-rhythm-quantization-by-sequential-monte-carlo.pdf))

**1.6 Hainsworth & Macleod (2004), "Particle Filtering Applied to Musical Tempo Tracking," EURASIP JASP 2004:15, 2385–2395.**
Audio → onset detection → sequential MC over (tempo period, phase, quantized
metrical location of each onset). Two algorithms: a Rao-Blackwellised,
near-deterministic jump formulation (best-performing) and a Brownian-motion
tempo model. Observation: onset times modeled with Gaussian error around
predicted metrical positions; onset amplitude/salience enters the weighting.
**Transferability: moderate — same event-based likelihood idea as Cemgil;
historically important for showing PF beat tracking on real audio, but again
PF is unnecessary at this scale.**
([EURASIP](https://asp-eurasipjournals.springeropen.com/articles/10.1155/S1110865704408099))

## 2. Meter induction from accent/timing patterns

**2.1 Povel & Essens (1985), "Perception of Temporal Patterns," Music Perception 2(4):411–440.**
The clock model. Stage 1: rule-based **accent assignment** to event onsets
(accents on: temporally isolated tones; the second of a pair; first and last
of runs ≥3). Stage 2: enumerate candidate internal clocks (period, phase) on
the tatum grid; score each by **counter-evidence** C = W·(#ticks on
unaccented events) + (#ticks on silence); the clock minimizing C is induced;
pattern complexity ≈ C. Deterministic template scoring, no inference
machinery.
**Transferability: high as a likelihood design, not as a system — "−C" is a
log-likelihood of a (period, phase) hypothesis given
accented/unaccented/silent tick positions. Salience-weighted markers slot
directly into a soft version (weighted counts). The negative-evidence term
(clock ticks landing on silence are penalized) is exactly how ABSENCE of a
marker at a hypothesized strong beat becomes informative.**
([UC Press](https://online.ucpress.edu/mp/article/2/4/411/62235/Perception-of-Temporal-Patterns),
[Yale Virtual Lab summary](https://rhythmcoglab.coursepress.yale.edu/wiki/bibliography/experimental-studies/povel-dirk-jan-and-peter-essens-1985-perception-of-temporal-patterns-music-perception-2-4-411-440/))

**2.2 Parncutt (1994), "A Perceptual Model of Pulse Salience and Metrical Accent in Musical Rhythms," Music Perception 11(4):409–464.**
For every candidate pulse (period, phase): pulse-match salience = sum of
**durational accents** (long IOI after an event → stronger accent) of
coinciding events, multiplied by a **Gaussian tempo-preference window over
log-period centered near ~700 ms** (moderate-tempo preference, validated by
tapping data). Metrical accent of an event = summed salience of all pulses
through it. No dynamics, no inference; a static salience map.
**Transferability: high as the PRIOR — Parncutt's log-normal tempo window is
the canonical soft prior over beat period (the 40–200 BPM range needs
exactly this to break octave/metrical-level ambiguity), and durational
accent (IOI-based) should be one of the salience features since ballet
markers before long gaps are perceptually accented.**
([UC Press](https://online.ucpress.edu/mp/article-abstract/11/4/409/46407/A-Perceptual-Model-of-Pulse-Salience-and-Metrical))

**2.3 Longuet-Higgins & Lee (1982), "The Perception of Musical Rhythms," Perception 11:115–128 (and 1984 on syncopation).**
Incremental symbolic parser: initial metrical unit hypothesized from the
first IOIs, then deterministic rules (conflate/stretch/update/longnote) grow
or revise a binary/ternary metrical tree; static tolerance window decides
"on the beat" vs "subdividing." 1984 defines syncopation as a note-vs-rest
metrical weight violation.
**Transferability: low as a system (deterministic, greedy, brittle to
expressive timing and interleaved talk), but historically defines the
hypothesis space (nested duple/triple divisions) that all later models use.
Read about it; don't implement it.**
([Sage](https://journals.sagepub.com/doi/10.1068/p110115))

**2.4 Klapuri, Eronen & Astola (2006), "Analysis of the Meter of Acoustic Musical Signals," IEEE TASLP 14(1):342–355.**
Joint three-level meter analysis: **tatum, tactus, measure simultaneously**.
Front end: a novel **degree-of-musical-accent signal in 4 registral
frequency channels** (subband envelopes, mu-law compression,
differentiation, half-wave rectification); periodicity via a bank of
comb-filter resonators; then a **hand-designed HMM ties the three periods
together** with priors favoring integer ratios between levels and lognormal
priors on absolute periods; phase estimated per level in a second stage.
Both causal (filtering) and non-causal (Viterbi) variants. Robust across
genres without pattern learning.
**Transferability: medium-high structurally — the classic demonstration that
jointly constraining subdivision/beat/bar levels (integer-ratio soft priors)
resolves the metrical-level ambiguity that kills single-level trackers at
40–200 BPM. Its audio front end is irrelevant here, but its inter-level
coupling priors are exactly the (subdivision, beat, meter) coupling.**
([IEEE/ACM DL](https://dl.acm.org/doi/10.1109/TSA.2005.854090),
[PDF mirror](https://www.iro.umontreal.ca/~pift6080/H09/documents/papers/klapuri_meter.pdf),
[Tampere portal](https://researchportal.tuni.fi/en/publications/analysis-of-the-meter-of-acoustic-musical-signals))

**2.5 Temperley (2007), *Music and Probability*, MIT Press, chs. 2–3; and Temperley (2009), "A Unified Probabilistic Model for Polyphonic Music Analysis," JNMR 38(1):3–18.**
The closest published thing to the committed design: a **fully generative
joint model with soft priors, inferred exactly**. Generative story
(monophonic rhythm model): time is a 50 ms "pip" lattice; (1) generate
tactus beats: each tactus interval drawn from a distribution centered on the
previous interval (tempo-continuity prior → drift is a random walk), initial
interval from a broad prior; (2) generate upper level: duple or triple
grouping of tactus beats plus phase (which beats are strong) — this IS the
meter variable; (3) generate lower level: each tactus interval split into 2
or 3 (duple/triplet subdivision), sub-beats anchored proportionally between
tactus beats; (4) generate note onsets: at each pip, P(onset) depends only
on the metrical level of that pip (strong beat ≫ weak beat ≫ off-beat) —
learned from corpora. Inference: dynamic programming/Viterbi over
(previous-beat time, interval, upper-level phase, meter type) — an exact
joint posterior over (period, phase, duple/triple × duple/triple) given
sparse onsets. The 2009 JNMR paper extends it to polyphony jointly with
harmony and streaming.
**Transferability: highest of the classical models. It is literally "one
joint posterior over (beat period, phase, meter, subdivision) where sparse
onsets contribute likelihoods and priors are soft." Its two gaps: onsets are
unweighted (no salience — condition P(onset)·f(salience) on metrical level),
and there is no 6/8 as distinct from 3/4-with-duple-subdivision unless
compound-meter structure is added (6/8 = duple grouping × triple
subdivision — which the model's factorization actually expresses
natively).** ([MIT Press](https://mitpress.mit.edu/9780262201667/music-and-probability/),
[JNMR 2009 PDF](https://davidtemperley.com/wp-content/uploads/2015/11/temperley-jnmr09.pdf),
[Melisma v2](https://davidtemperley.com/melisma-v2/))

**2.6 van der Weij, Pearce & Honing (2017), "A Probabilistic Model of Meter Perception: Simulating Enculturation," Frontiers in Psychology 8:824.**
Bayesian inference over metrical interpretations (meter category + phase) of
a rhythmic surface, where the **prior over meters and the likelihoods are
learned from corpus exposure** (IDyOM-style variable-order Markov
statistical learning); shows culture-specific priors explain cross-cultural
meter perception differences.
**Transferability: medium — main value is the argument + mechanism for
corpus-derived soft meter priors (the domain prior: ballet class music is
overwhelmingly 2/4, 3/4, 4/4, 6/8 with genre-specific frequencies per
exercise type — the exercise label already extracted via Gemini is a
legitimate prior conditioner).**
([Frontiers](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2017.00824/full))

**2.7 (Non-Bayesian alternative, for completeness) Large & Jones (1999), "The Dynamics of Attending," Psych. Review 106:119–159.**
Entrainment: nonlinear oscillators with adaptive period/phase and an
"attentional pulse" whose width narrows as entrainment improves. Handles
drift gracefully, online, but gives no calibrated posterior, no principled
meter variable, and multi-oscillator meter versions are heuristic.
**Verdict: skip for this architecture; PIPPET (below) is the Bayesian
formalization of this intuition.**

## 3. Modern learned systems — and the sparse/vocal-input question

**3.1 Böck, Krebs & Widmer (2016), "Joint Beat and Downbeat Tracking with Recurrent Neural Networks," ISMIR 2016 (= madmom's `RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor`).**
BLSTM over multi-resolution log-spectrograms emits a per-frame softmax over
{beat, downbeat, no-beat}; a **DBN (the Krebs 2015 efficient bar-pointer
HMM) with bars of variable length (beats_per_bar = [3,4] in parallel
sub-state-spaces)** Viterbi-decodes the globally best (tempo, phase, meter)
path. Observation model: within each beat, the first 1/observation_lambda
fraction of position states emit the network's beat/downbeat activation,
the rest emit the no-beat probability — i.e., the NN activation is treated
as a per-frame likelihood indexed by bar position. Meter = which sub-space
the best path lives in.
**Transferability: the DBN half transfers wholesale (it is genre-agnostic
math); the RNN half does not (trained on full music mixes, will fire
erratically on speech). The design lesson: keep the observation an
exchangeable per-frame/per-event likelihood so NN activations can be swapped
for salience-feature likelihoods without touching inference.**
([ISMIR 2016 PDF](https://archives.ismir.net/ismir2016/paper/000186.pdf),
[dblp](https://dblp.org/rec/conf/ismir/BockKW16.html),
[madmom docs](https://madmom.readthedocs.io/en/v0.16/_modules/madmom/features/downbeats.html))

**3.2 Heydari, Cwitkowitz & Duan (2021), "BeatNet: CRNN and Particle Filtering for Online Joint Beat Downbeat and Meter Tracking," ISMIR 2021; plus Heydari & Duan (2021), "A Novel 1D State Space for Efficient Music Rhythmic Analysis" (arXiv:2111.00704); BeatNet+ (TISMIR 2024).**
CRNN emits beat/downbeat/no-beat activations; inference is a **two-stage
cascade of sequential Monte Carlo filters**: stage 1 tracks (tempo, beat
phase); stage 2, conditioned on beats, tracks (downbeat, meter) — meter is
estimated on the fly, no time-signature priming, with an "information gate"
that only runs particle updates near informative frames (large cost
savings). The 1D follow-up collapses the 2D bar-pointer to a single phase
counter with jump-back transitions for cheaper online inference.
**Transferability: the cascade legitimizes a staged factorization (beat
first, meter conditioned on beats) if online operation is ever needed; for
offline 10–30 s clips it is strictly dominated by exact HMM inference. The
"no meter priming, meter as posterior" stance matches the redesign.**
([arXiv:2108.03576](https://arxiv.org/abs/2108.03576),
[GitHub](https://github.com/mjhydri/BeatNet),
[ISMIR PDF](https://archives.ismir.net/ismir2021/paper/000033.pdf))

**3.3 Foscarin, Schlüter & Widmer (2024), "Beat This! Accurate Beat Tracking Without DBN Postprocessing," ISMIR 2024 (arXiv:2407.21658).**
Conv frontend + alternating time/frequency transformer blocks + task heads;
trained on a large multi-dataset compilation with augmentation and a
**shift-tolerant loss**; postprocessing is bare peak-picking — no DBN, no
tempo/meter state at all. SOTA beat/downbeat F-measure; explicitly does NOT
output meter/time signature, and its accuracy depends on in-domain
supervised data.
**Transferability: low-to-none directly (no speech training data, no meter
output, no posterior, no way to inject priors), but it is the strongest
evidence for the counterargument to be able to answer: with enough labeled
data, temporal models become unnecessary. This project does not have
"enough labeled ballet-marking data," which is precisely when the
DBN/bar-pointer machinery earns its keep — the Böck 2016 vs Beat This!
contrast is the justification paragraph.**
([Semantic Scholar](https://www.semanticscholar.org/paper/7c674e285e7a47dba57ddba1526d2286026f2476),
[GitHub CPJKU/beat_this](https://github.com/CPJKU/beat_this))

**3.4 Vocal/sparse input — the closest analogs.**

- **Heydari & Duan (2022), "Singing Beat Tracking with Self-Supervised
  Front-End and Linear Transformers," ISMIR 2022 (arXiv:2208.14578)**: beat
  tracking on *isolated singing voice* (source-separated from beat-annotated
  datasets). Key findings: existing music beat trackers fail on solo voice
  (no percussive/harmonic rhythm scaffolding); **pre-trained speech SSL
  representations (WavLM, DistilHuBERT) beat spectral features as front-ends
  for sparse vocal rhythm** — directly relevant since this input IS speech.
  Follow-up: "Efficient Adapter Tuning for Joint Singing Voice Beat and
  Downbeat Tracking with SSL Features" (arXiv:2503.10086, ICASSP 2025).
  **Verdict: closest published task; confirms (i) sparse vocal beat tracking
  is hard for learned systems, (ii) speech-model features are the right
  front-end, (iii) nobody has done meter/phrase inference on spoken rhythm —
  the gap is real.**
  ([ISMIR 2022](https://ismir2022program.ismir.net/poster_250.html),
  [arXiv:2208.14578](https://arxiv.org/abs/2208.14578),
  [arXiv:2503.10086](https://arxiv.org/pdf/2503.10086))
- **Beat tracking on speech per se: no direct literature found** (multiple
  searches). The adjacent psycholinguistics is actionable, though:
  **P-centers** (Morton/Marcus/Frankish 1976 onward; recent acoustic model
  in JASA 2024; "speech-to-speech synchronization is governed by the
  P-center," Comms. Biology 2025): the perceived rhythmic moment of a
  syllable lags its acoustic/word-boundary onset and approximates the
  **vowel onset**. Whisper word-start timestamps are systematically early
  relative to the perceptual beat, with consonant-cluster-dependent
  offsets — correct markers toward vowel onsets (or model a learned
  per-token offset + inflated variance) before they enter the likelihood.
  ([JASA 2024](https://pubs.aip.org/asa/jasa/article/155/4/2698/3283278/),
  [Comms Bio 2025](https://www.nature.com/articles/s42003-025-07544-8),
  [Haskins review](https://haskinslabs.org/sites/default/files/files/Reprints/HL0262.pdf))
- **Cummins & Port (1998), "Rhythmic Constraints on Stress Timing in
  English," J. Phonetics 26:145–171** (speech cycling): repeated spoken
  phrases lock stressed-syllable onsets to **simple harmonic phases (1/2,
  1/3, 2/3) of the repetition cycle** — rhythmically produced speech
  genuinely has nested metrical structure like music. This is the empirical
  license to apply musical meter models to marking at all, and it predicts
  markers cluster at simple bar-phase fractions.
  ([figure/summary](https://www.researchgate.net/publication/228587484_Speech_timing_in_linguistics))
- **Tapping/QBT**: MIREX Query-by-Tapping matched tapped onset sequences to
  songs but never inferred meter; sensorimotor-synchronization modeling is
  subsumed by PIPPET below.
- **Cannon (2021), "Expectancy-Based Rhythmic Entrainment as Continuous
  Bayesian Inference" (PIPPET/PATIPPET), PLOS Comp. Biol. 17(6):e1009025.**
  Belongs here as the modern sparse-input model. Latent state: phase φ
  (PIPPET) or (phase, tempo) (PATIPPET) evolving as drift-diffusion.
  Generative model: an **inhomogeneous point process whose rate is an
  expectation template τ(φ) = λ_0 + Σ_i λ_i·N(φ; φ_i, v_i)** — Gaussian
  bumps at expected phases with per-position strength λ_i (how confidently
  an event is expected) and precision v_i (timing tolerance), plus
  **background rate λ_0 for events unrelated to the pulse**. Inference:
  continuous-time variational/moment-matching filter (Gaussian posterior
  over phase/tempo): between events the posterior drifts and its variance
  grows; at each event it jumps toward the nearest strong template bump
  (weighted by λ_i and current uncertainty); crucially, **the absence of an
  event where the template strongly expects one also updates the posterior**
  (silence at a hypothesized downbeat is evidence). mPIPPET generalizes to
  multiple event types with their own templates — i.e., salience classes
  (stressed vs unstressed syllables, "one" vs "and"). Reproduces
  failure-to-entrain to weak/isochrony-violating stimuli, swung-rhythm
  tracking, tempo drift.
  **Transferability: essentially a design document for the observation
  model. Meter/phrase inference = maintain one template per (meter,
  subdivision, phrase-length) hypothesis and a discrete posterior over
  hypotheses alongside each continuous (phase, tempo) filter — a small
  switching-filter bank. The 2020 GitHub revision note matters for
  implementation: a variance-calculation fix and rate rescaling so tempo
  changes don't artificially weaken expectations.**
  ([PLOS](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009025),
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/34106918/),
  [GitHub joncannon/PIPPET](https://github.com/joncannon/PIPPET),
  [pyPIPPET](https://github.com/Kappers/pyPIPPET),
  [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.11.05.369603v2.full))

## 4. Phrase length / hypermeter

The blunt finding: **there is no established probabilistic audio system for
hypermetrical (8-count phrase) inference** — this is the least-developed of
the four latents, and it will be composed from three ingredients:

**4.1 Temperley (2001), *The Cognition of Basic Musical Structures*, MIT Press — the "Grouper" module.**
Phrase segmentation of a note stream by dynamic programming over candidate
boundary placements with three preference terms: (1) **gap rule** —
boundaries preferred at large inter-onset + offset-to-onset gaps; (2)
**phrase-length prior** — penalty for deviation from a preferred length
(implemented as deviation from ~8 notes / from the log-mean length); (3)
**metrical parallelism** — prefer segmentations where successive phrases
begin at the same metrical position. Exact DP inference.
**Transferability: high — swap "notes" for counts: a soft prior over phrase
length in beats {4, 8, 16} (heavily 8 for ballet), boundary likelihood from
pauses/breaths/"and-a" pickups, parallelism coupling phrase starts to
downbeats. It's a preference-rule score, but each term exponentiates into a
clean log-linear probabilistic model.**
([MIT Press](https://mitpress.mit.edu/9780262701051/the-cognition-of-basic-musical-structures/);
see also Temperley (2008), "Hypermetrical Transitions," Music Theory
Spectrum 30(2), for evidence that hypermeter is duple-regular with
occasional metrical reinterpretations — supports a strong 8-count prior with
a small switch probability.)

**4.2 Srinivasamurthy, Holzapfel, Cemgil & Serra (2016), "A Generalized Bayesian Model for Tracking Long Metrical Cycles in Acoustic Music Signals," ICASSP 2016, pp. 76–80.**
Extends the bar pointer to **long metrical cycles** (Indian tala cycles of
8–16+ beats with internal sections) — a "cycle pointer" whose observation
model has section-dependent structure, with particle-filter inference
because the state space grows with cycle length.
**Transferability: direct — an 8-count ballet phrase IS a long metrical
cycle; this paper is the proof that the same bar-pointer machinery scales up
one hierarchical level. At these clip lengths it can still be done exactly:
a hypermeasure pointer over 8 beats × meter is only ~8× the bar state space,
or more cheaply a second-pass HMM over downbeats (phase-within-phrase
∈ {1..8}).** ([Wikidata](https://www.wikidata.org/wiki/Q57758398),
[CompMusic](https://compmusic.upf.edu/publications))

**4.3 Boundary-evidence models: Cambouropoulos (2001) LBDM; Pearce, Müllensiefen & Wiggins (2010), "Melodic Grouping in MIR: New Methods and Applications" (IDyOM boundary detection).**
LBDM: boundary strength = weighted local change in IOI/pitch/rest —
trivially portable to salience/IOI change in marking. IDyOM: boundaries at
peaks of information content/unexpectedness under a learned sequence model —
needs a symbolic corpus, marginal for now. Lerdahl & Jackendoff's GTTM
grouping preference rules (1983) are the theoretical basis for both;
Rothstein (1989) *Phrase Rhythm in Tonal Music* for hypermeter-vs-grouping
distinctions. **Verdict: use LBDM-style boundary features as one evidence
channel feeding the phrase posterior; skip IDyOM.**
([Springer chapter](https://link.springer.com/chapter/10.1007/978-3-642-11674-2_16))

---

## (a) Concrete recommendation: what to implement first

**Implement a Krebs-2015-style discretized bar-pointer HMM with a
Temperley/PIPPET-style event-based observation model — one exact joint
posterior, no particle filters, no neural front end.**

- **State space**: s = (φ, T, m, d) with meter m ∈ {2/4, 3/4, 4/4, 6/8} as
  parallel sub-state-spaces (madmom's beats_per_bar pattern, with 6/8 as 2
  beats × triple subdivision — note this factorization means (meter,
  subdivision) need not be independent variables: 6/8 vs 2/4 differ only in
  d, 3/4 vs 4/4 only in bar length), tempo T = integer frames per beat at
  50 fps (T ∈ [15, 75] for 40–200 BPM), position φ with tempo-dependent
  resolution (one state per frame), subdivision d ∈ {duple, triple} gating
  where sub-beat template bumps sit. Tempo transitions only at beat
  boundaries with p ∝ exp(−λ|T′/T − 1|) — this handles drift and expressive
  lengthening exactly as designed. Size: ≈ (2+3+4+2)·Σ_T T ≈ 30k states,
  sparse transitions (≤3 successors/state) → forward–backward over a
  1,500-frame clip ≈ 10^8 multiply-adds worst case, well under a second in
  numpy with the standard tempo-change-only-at-beat-boundary sparsity
  (madmom does exactly this in real time). **Run forward–backward, not just
  Viterbi — the committed design wants a posterior, and metrical-level
  ambiguity (2/4 vs 4/4, 3/4 vs 6/8) should be reported as calibrated
  probability, not silently argmaxed** (consonant with the existing ADR-014
  stance on tempo alternates).
- **Phrase length**: add phase-within-phrase p ∈ {1..8} over downbeats as
  either (i) a hypermeasure extension of the pointer (Srinivasamurthy 2016;
  ~8× states, still fine), or (ii) a second exact HMM over the downbeat
  lattice with phrase-length prior {4:small, 8:large, 16:small} and boundary
  evidence (gap rule, breath, "and" pickups, parallelism) — start with (ii);
  it keeps the first model unchanged and phrase evidence is naturally
  downbeat-synchronous.
- **Why this family and not the alternatives**: Temperley's pip-lattice DP
  is the same math with a less reusable discretization; Cemgil's switching
  Kalman needs RBPF for exact-ish inference (unneeded complexity offline);
  PIPPET's continuous filter is elegant but online-oriented, gives only a
  Gaussian phase posterior (the multi-modal phase ambiguities — is the "one"
  here or there — demand the HMM's full discrete posterior), and its real
  contribution is the observation model, which the HMM can adopt wholesale;
  learned systems (madmom RNN, BeatNet, Beat This!) have no speech-domain
  training data and (Beat This!) no priors, no meter output, no posterior.
  The bar-pointer HMM is also the only family with a decade of published
  evidence on meter *discrimination* (ballroom 3/4-vs-4/4, Turkish 9/8,
  Carnatic long cycles).

## (b) How salience/accent enters the observation model in the best candidates — and should enter here

- **Whiteley 2006**: Poisson intensity as a function of bar position —
  accent pattern = intensity template; salient positions have high λ(φ).
- **Krebs 2013 / madmom**: emissions indexed by (pattern, bar-position
  cell): GMMs over accent features (spectral flux), i.e., P(feature |
  position); madmom's DBN gates NN beat/downbeat activations by bar position
  (observation_lambda), i.e., P(activation | position).
- **Klapuri 2006**: salience is upstream — the "degree of musical accent"
  signal in 4 bands is *the* observation; period likelihoods are comb-filter
  energies of that accent signal.
- **Temperley 2007**: P(onset | metrical level of position) tables — accent
  enters as level-conditioned onset probability, learned from corpora.
- **PIPPET/mPIPPET (the cleanest for this project)**: each event carries a
  type/class; template bump strengths λ_i and variances v_i per class set
  how strongly and how precisely each class is expected at each phase;
  background rate λ_0 absorbs off-template events.
- **The synthesis**: per marker j with salience features x_j (WhiStress
  stress prob, Praat intensity/pitch peak, vowel duration, word class
  "one/and/ah", IOI-derived durational accent per Parncutt), emit under
  state s with bar-phase-dependent likelihood: log P(marker_j | s) =
  log[λ_0 + Σ_bumps λ_b(m,d)·w(x_j)·N(t_j; t_b(s), σ_b² + σ_ASR²)] — a
  log-linear salience weight w(x_j) multiplying position-class-dependent
  bump strengths (downbeat bumps expect high-salience markers; "and" bumps
  expect the token "and" with low stress), with per-marker timing variance
  inflated by Whisper timestamp noise and **P-center correction of
  timestamps toward vowel onsets first**. Between markers, frames emit the
  no-event likelihood — which is where Povel–Essens negative evidence lives:
  a state whose template expected a strong-beat marker that didn't come pays
  exp(−λ_strong·Δ). Interleaved talk is handled by λ_0 plus, if Gemini's
  marker/talk classification is kept, simply excluding talk tokens (λ_0 then
  only covers classifier misses).

## (c) Top-5 papers to read in full, ranked

1. **Krebs, Böck & Widmer (2015), "An Efficient State-Space Model for Joint
   Tempo and Meter Tracking," ISMIR 2015** — the state space, transition
   model, and complexity engineering to actually implement; read alongside
   madmom's `downbeats.py`.
2. **Cannon (2021), "Expectancy-Based Rhythmic Entrainment as Continuous
   Bayesian Inference," PLOS Comp. Biol.** — the observation model for
   sparse salience-weighted events: template bumps (λ_i, v_i), background
   rate, informative absences; reference Python at Kappers/pyPIPPET.
3. **Temperley (2007), *Music and Probability*, chs. 2–3** (with the 2009
   JNMR paper as the freely available companion) — the complete worked
   example of a joint generative (period, phase, duple/triple × duple/triple)
   posterior over sparse onsets with soft priors and exact DP inference.
4. **Whiteley, Cemgil & Godsill (2006), "Bayesian Modelling of Temporal
   Structure in Musical Audio," ISMIR 2006** — the original joint (position,
   tempo, meter, pattern) formulation and the Poisson point-process
   observation model — the correct likelihood *form* for a marker stream.
5. **Klapuri, Eronen & Astola (2006), "Analysis of the Meter of Acoustic
   Musical Signals," IEEE TASLP** — the reference for coupling
   subdivision/beat/measure with integer-ratio and absolute-period priors,
   and for causal vs non-causal variants.

(Next tier, in order: Srinivasamurthy et al. 2016 — the moment phrase
tracking is built; Cemgil & Kappen 2003 JAIR — if continuous-time
event-based inference is ever needed; Böck et al. 2016 — for the
meter-as-parallel-subspaces decoding trick; Heydari & Duan 2022 — before
anyone proposes "just fine-tune a beat tracker"; Povel & Essens 1985 +
Parncutt 1994 — short reads that will directly shape the salience features
and tempo prior.)
