# Review 2 · Tempo estimation, periodicity analysis, and the tempo-octave problem

*Part of the [voice-as-drum literature review](voice-as-drum-review.md)
(2026-08-09, companion to [ADR-016](../adr/016-rhythm-core-reset.md)).
Produced by a web-research agent; metric definitions were verified against
mir_eval 0.8.2 and madmom source, librosa signatures confirmed locally.*

**The project's three documented failures, in the field's vocabulary.**
(1) Mean-IOI landing between metric levels when onsets mix 1×/2×/3× is the
*metrical-level mixing* problem; the field abandoned first-moment IOI
statistics for exactly this reason ~2000–2005 (Gouyon & Dixon 2005, "A review
of automatic rhythm description systems," *CMJ* 29(1):34–54) in favor of
periodicity functions and histogram methods where family members *vote*
rather than *average*. (2) A hard 70–140 fold is a crude form of *octave
correction*; the literature's replacement is a soft unimodal perceptual prior
applied at level *selection*, plus multi-level output with salience, because
ground truth itself is multimodal (McKinney & Moelants 2006). (3) Agogic drag
is *phrase-final lengthening*, which is structured and predictable (Todd;
Repp; and in speech, Wightman et al. 1992) — model or censor it; don't
average it.

A domain caveat that applies to everything below: acoustic-envelope
periodicity of *speech* peaks at the syllable rate ~4–5 Hz, vs ~2 Hz for
music (Ding, Patel, Chen, Butler, Luo & Poeppel 2017, *Neurosci. Biobehav.
Rev.* 81:181–187). Any method run on a raw onset-strength envelope of marking
audio will tend to lock to syllables, not counts. Everything transfers best
when the "novelty curve" is a **synthesized impulse train at the
Whisper/Gemini marker times** (weighted by stress/confidence), which the
pipeline already produces (`TimedMarker` in
`src/musical_perception/types.py`, feeding
`src/musical_perception/precision/tempo.py`).

---

## 1. Periodicity representations

**1.1 Autocorrelation of onset strength.** Ellis, D.P.W. (2007), "Beat
tracking by dynamic programming," *JNMR* 36(1):51–60; librosa
`feature.tempogram` (local ACF, default `win_length=384` frames ≈ 8.9 s). ACF
of the onset envelope over a window; peaks at the beat lag and its *integer
multiples* — i.e., ACF emphasizes tempo **subharmonics** (slow-octave bias).
Ellis picks the global peak after perceptual weighting (below) then decodes
beats by DP with a `tightness` penalty on log-interval deviation.
**Transfer: partially.** Fine on a marker impulse train; on raw audio it
locks to syllable rate. With sparse events (10–30 s, 40 BPM ⇒ ~7–20 beats)
windowed ACF is high-variance, and the subharmonic bias will systematically
prefer the slow member of the family. Use only paired with a Fourier
tempogram (1.4).

**1.2 Comb filterbanks.** Scheirer, E.D. (1998), "Tempo and beat analysis of
acoustic musical signals," *JASA* 103(1):588–601. Six band-pass envelope
channels drive ~150 log-spaced resonating comb filters per channel
(half-energy time 1500–2000 ms); resonator output energy across delays =
tempo salience; the winning resonator's phase predicts beat times, causally.
Industrial descendants: Klapuri, Eronen & Astola (2006), *IEEE TASLP*
14(1):342–355 — accent bands → comb resonators → probabilistic *joint* model
of tatum/tactus/measure with lognormal period priors
p(τᵢ) = 1/(τᵢσᵢ√2π)·exp(−ln²(τᵢ/mᵢ)/2σᵢ²) and coupling on the integer ratios
between levels; Böck, Krebs & Widmer (2015), "Accurate tempo estimation based
on recurrent neural networks and resonating comb filters," *ISMIR*,
pp. 625–631 — RNN beat activation → resonating combs → histogram (madmom
`TempoEstimationProcessor` defaults, from source: `method='comb'`,
`MIN_BPM=40`, `MAX_BPM=250`, `ACT_SMOOTH=0.14`, `HIST_SMOOTH=9`,
`ALPHA=0.79`).
**Transfer: partially.** The multi-band audio front end is wrong for
voice-as-drum, but comb resonators on a synthesized marker activation give
causal salience cheaply, and madmom's histogram+peaks output format (ranked
(bpm, strength) pairs) is worth copying. Klapuri's *coupled-level*
probabilistic model is the right conceptual machinery for 1×/2×/3× mixes: it
never estimates one period in isolation, it estimates a family with integer
constraints. Long integrator time-constants smear drift on 10–30 s clips.

**1.3 Fourier tempogram.** Grosche & Müller (2011, below); Müller, *FMP*
Ch. 6; librosa `feature.fourier_tempogram`. STFT of the novelty curve
evaluated at tempo frequencies: F(t, ω) = Σₙ Δ(n)w(n−t)e^(−iωn). For an
impulse train this is exactly the point-process periodogram (per-window
Rayleigh-type statistic). Key property (FMP; Grosche/Müller/Kurth 2010): the
Fourier tempogram emphasizes tempo **harmonics** (2×, 3× the true tempo) and
suppresses subharmonics — the mirror image of ACF.
**Transfer: strongly.** It operates natively on sparse impulse trains; onsets
at mixed 1×/2×/3× levels all contribute energy at the beat frequency's
harmonic stack, so summing harmonics recovers the beat even when *no single
level dominates* — the direct fix for the between-levels mean. Bias is
fast-octave; fix by pairing with 1.1 or by harmonic weighting (see steal #1).

**1.4 Cyclic tempograms (octave-invariant).** Grosche, Müller & Kurth (2010),
"Cyclic tempogram — a mid-level tempo representation for music signals,"
*ICASSP*, Dallas. Fold the tempo axis by identifying tempi differing by
powers of 2 (scale parameter s ∈ [1,2), like chroma for pitch); defined for
both Fourier- and ACF-based tempograms; robust local tempo features for
segmentation and cross-version analysis. The two underlying tempograms are
explicitly described as complementary: ACF indicates subharmonics while
suppressing harmonics; Fourier the reverse — their combination cancels both
octave biases.
**Transfer: partially.** Excellent for (i) drift visualization on expressive
clips, (ii) *segmenting marking from interleaved talk* (tempo-class
stability is high while marking, chaotic while talking) — a real gap in the
current pipeline, and (iii) octave-robust confidence. Two limits: folding is
by powers of **2 only** — ×3/triplet relations are not identified (a
3-cyclic analogue is definable but not in the literature); and the final BPM
still requires unfolding, i.e., the octave decision is deferred, not solved.

**1.5 Predominant Local Pulse (PLP).** Grosche, P. & Müller, M. (2011),
"Extracting predominant local pulse information from music recordings,"
*IEEE TASLP* 19(6):1688–1701. Per frame, take the maximizing (tempo, phase)
of the complex Fourier tempogram; overlap-add windowed cosine kernels;
half-wave rectify → a locally adaptive pulse curve whose peaks are pulse
positions, with tempogram magnitude as local confidence. librosa:
`beat.plp(onset_envelope=…, tempo_min=30, tempo_max=300,
prior=<any scipy.stats rv>)` (docstring example uses
`scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)`). Known failure mode:
strong rubato — documented on the Chopin Mazurka corpus (Grosche, Müller &
Sapp 2010, "What makes beat tracking difficult? A case study on Chopin
Mazurkas," *ISMIR*, pp. 649–654: errors cluster at non-event beats,
*boundary beats*, ornaments).
**Transfer: partially.** On a marker-impulse novelty curve, PLP gives a
smooth "internal metronome" plus per-frame confidence, locally re-locking
after tempo changes — attractive as the bridge from perception markers to
the precision layer. But each window needs enough events (≥ ~6): at 40 BPM
with beat-level-only marking, librosa's default ~8.9 s window is borderline;
and kernels blur across talk/marking boundaries unless gated by segment
first. The per-frame max also snaps between octaves frame-to-frame on
ambiguous material — constrain the tempo set or apply the prior.

**1.6 IOI histograms/clustering with integer-ratio reinforcement.** Dixon, S.
(2001), "Automatic extraction of tempo and beat from expressive
performances," *JNMR* 30(1):39–58; Dixon (2007), "Evaluation of the audio
beat tracking system BeatRoot," *JNMR* 36(1):39–50. Compute IOIs between
*all* (not just adjacent) onset pairs within a span; cluster with a fixed
tolerance (tens of ms); score clusters by cardinality, onset amplitude, and —
the crucial move — **support from clusters related by simple integer
ratios**; ranked cluster centers seed a multi-agent beat tracker (each agent
= (period, phase) hypothesis with inner/outer tolerance windows, scored by
regularity and salience of matched onsets). Built and validated on
*expressive performances*.
**Transfer: strongly.** This is the classical, near-zero-dependency antidote
to failure #1: levels never get averaged, they reinforce a common tactus
through the ratio graph. Works from ~15 onsets; amplitude weighting maps
directly onto stress/beat-class weights ("beat" markers weigh more than
"and"/"ah"). Multi-agent tolerance absorbs moderate agogics.

**1.7 GCD/grid ("tatum") methods and the point-process view.** Seppänen
(2001), "Tatum grid analysis of musical signals," *IEEE WASPAA* (GCD-level
grid from IOI histogram); Klapuri 2006's tatum level; and — from pulsar
astronomy, the mathematically identical problem of sparse event times at
unknown harmonic mixes — epoch folding / Rayleigh test with harmonic
summing: de Jager, Raubenheimer & Swanepoel (1989), "A powerful test for
weak periodic signals with unknown light curve shape in sparse data," *A&A*
221:180–190 (H-test: Z²ₘ = (2/N)Σₖ₌₁..ₘ|Σⱼe^(ikωtⱼ)|², H = maxₘ(Z²ₘ − 4m + 4),
i.e., *adaptive* harmonic summing).
**Transfer: strongly**, with one warning: exact-GCD arithmetic breaks under
expressive deviation; always score a candidate grid with a tolerance kernel
(Gaussian on the remainder |tⱼ − nearest grid point|), not exact division.
The H-test framing is tailor-made for "10–30 s, few events, onsets at
unknown subdivisions."

---

## 2. The tempo octave / metrical-level ambiguity

**2.1 Human tempo perception is genuinely multimodal.** McKinney, M.F. &
Moelants, D. (2006), "Ambiguity in tempo perception: What draws listeners to
different metrical levels?" *Music Perception* 24(2):155–166. Listeners
tapped the most salient pulse of excerpts; per-excerpt tap distributions are
often bi/multimodal across metrical levels (factors 2 or 3), with the
winning level pulled toward ~120 BPM but overridden by musical content.
Companion: McKinney, Moelants, Davies & Klapuri (2007), "Evaluation of audio
beat tracking and music tempo extraction algorithms," *JNMR* 36(1):1–16 —
MIREX'06 data: 140 excerpts × 40 tappers; ground truth = **two tempi + a
salience weight** derived from the tap histogram; P-score defined on that.
**Transfer: verbatim.** A marking clip's "true tempo" may legitimately exist
at both the count level and the step level. Stop treating one scalar as
truth: annotate (T1, T2, salience), emit ranked candidates. The
`truth_in_family` eval concept is the algorithmic mirror of this
psychophysics.

**2.2 Preferred-tempo resonance — the principled soft prior.** van Noorden,
L. & Moelants, D. (1999), "Resonance in the perception of musical pulse,"
*JNMR* 28(1):43–66: pulse salience behaves like a damped harmonic resonator
peaked near **2 Hz (120 BPM)**; effective resonance curve
W(f) = [(f₀²−f²)² + βf²]^(−1/2) − [f₀⁴ + f⁴]^(−1/2), f₀ ≈ 2 Hz, β fitted ≈ 1.
Moelants (2002), "Preferred tempo reconsidered," *ICMPC7*: spontaneous
motor/preferred tempo 120–130 BPM (revising Fraisse's ~100). Parncutt, R.
(1994), "A perceptual model of pulse salience and metrical accent in musical
rhythms," *Music Perception* 11(4):409–464: salience is a **Gaussian in log
pulse-period**, peak ≈ 600–700 ms (~85–100 BPM), existence region
~200–1800 ms. MIR adoptions: Ellis 2007 weights the ACF with
W(τ) = exp(−½(log₂(τ/τ₀)/σ_τ)²), τ₀ = 0.5 s, σ_τ = **1.4 octaves**; librosa
`feature.rhythm.tempo` implements exactly
`logprior = −0.5·((log2(bpm) − log2(start_bpm))/std_bpm)²` with
`start_bpm=120`, `std_bpm=1.0`, and accepts any `scipy.stats` rv as
replacement; Davies & Plumbley (2007), "Context-dependent beat tracking of
musical audio," *IEEE TASLP* 15(3):1009–1020, use a **Rayleigh** weighting
over lag peaking ~120 BPM.
**Transfer: strongly** — this is the direct, literature-sanctioned
replacement for the 70–140 wall; parameters and application point in
recommendation (b).

**2.3 "Perceptual tempo" estimation — predicting the level from side
information.** Peeters & Flocon-Cholet (2012), "Perceptual tempo estimation
using GMM regression"; Elowsson & Friberg (2015), "Modeling the perception
of tempo," *JASA* 137(6):3163–3177 (perceived speed regressed from onset
density and rhythmic features); Hörschläger, Vogl, Böck & Widmer (2015),
*SMC*: fix octave errors in electronic music by conditioning on
**style-specific tempo priors harvested from Wikipedia**.
**Transfer: partially — but the recipe transfers perfectly.** Their audio
features break on speech; the design pattern "choose the octave using a
conditioning variable + event density" is exactly this situation: Gemini
already outputs the *exercise type* (plié vs tendu vs petit allegro), which
is a stronger tempo-range predictor than any acoustic feature, and marker
density (events/s) is the single best level cue in this literature.

**2.4 Modern output formats: distributions and two tempi with salience.**
Schreiber & Müller (2018), "A single-step approach to musical tempo
estimation using a convolutional neural network," *ISMIR*, pp. 98–105
(TempoCNN): 11.9 s mel input → softmax over **256 tempo classes
(~30–286 BPM)** — a *distribution*, not a scalar; the `tempo-cnn` repo also
ships DT-Maz models trained for local-tempo tempograms of Chopin Mazurkas
(Schreiber, Zalkow & Müller 2020, "Modeling and estimating local tempo: a
case study on Chopin's Mazurkas," *ISMIR*). Böck & Davies (2020),
"Deconstruct, analyse, reconstruct: How to improve tempo, beat, and downbeat
estimation," *ISMIR*: multi-task TCN with a tempo-distribution head.
madmom's `TempoDetector` returns a ranked histogram of (bpm, strength).
MIREX convention: report T1 < T2 plus salience of T1.
**Transfer: schema verbatim, models no.** The CNNs are music-timbre-bound.
But `NormalizedTempo.alternates` (ADR-014) is already halfway to the right
output; add normalized salience weights per candidate and let downstream
consumers see the distribution.

**2.5 Joint multi-level inference.** Klapuri 2006 (above); Whiteley, Cemgil
& Godsill (2006) bar-pointer model → Krebs/Böck DBN (madmom
`DBNBeatTrackingProcessor`, from source: `MIN_BPM=55`, `MAX_BPM=215`
(settable), `TRANSITION_LAMBDA=100` — exponential penalty on tempo *change*
between beats, higher = more constant tempo, `OBSERVATION_LAMBDA=16`,
`CORRECT=True`).
**Transfer: strongly as a concept, moderately as code.** For marking, the
latent state wanted is (beat period, phase, per-onset level ∈ {1, ½, ⅓, 2}) —
which is precisely Cemgil & Kappen's model (3.3). madmom's DBN transfers
today if an activation is synthesized (Gaussian bumps σ ≈ 30–50 ms at marker
times), the BPM range widened, and `transition_lambda` lowered for rubato.

---

## 3. Expressive timing, drift, and robust sparse estimation

**3.1 Repp — lengthening is structural and perceptually *expected*.** Repp,
B.H. (1992), "Diversity and commonality in music performance: An analysis of
timing microstructure in Schumann's Träumerei," *JASA* 92:2546–2568: across
28 pianists, the dominant timing pattern is phrase-final lengthening, nested
across the phrase hierarchy. Repp (1992), "Probing the cognitive
representation of musical time," *Cognition* 44:241–281: listeners' ability
to detect an artificially lengthened interval **dips precisely where
expressive lengthening is expected** (phrase boundaries) — the deviation is
part of the percept, not noise. Repp (2005), "Sensorimotor synchronization:
A review of the tapping literature," *Psychon. Bull. Rev.* 12(6):969–992
(and Repp & Su 2013): synchronization comfortable for IOIs ~200 ms–1.8 s;
trained tappers' CV ~3–5% — the perceptual basis for ±4%/±8% tolerances.
**Transfer: verbatim as design constraints.** (i) Phrase-final IOIs are
predictably inflated — exclude them from tempo, don't average them; (ii) the
eval tolerances have a psychophysical floor; sub-4% disagreements are below
human tapping noise.

**3.2 Parametric lengthening models.** Todd, N.P.M. (1985), "A model of
expressive timing in tonal music," *Music Perception* 3(1):33–58 (+ Todd
1992, *JASA* 91:3540–3550): rubato ≈ nested parabolic/kinematic tempo curves
per phrase unit — local tempo is U-shaped with minima at boundaries,
generated from the phrase hierarchy. Friberg, A. & Sundberg, J. (1999),
"Does music performance allude to locomotion? A model of final ritardandi
derived from measurements of stopping runners," *JASA* 105(3):1469–1484:
final ritards follow constant-braking kinematics,
v(x) = v₀(1 + (w^q − 1)x)^(1/q) with q ≈ 3 fitting performances best
(Honing's "When a good fit is not good enough" cautions against over-literal
kinematics, but the monotone boundary-deceleration shape is robust). Speech
twin: Wightman, Shattuck-Hufnagel, Ostendorf & Price (1992), "Segmental
durations in the vicinity of prosodic phrase boundaries," *JASA*
91:1707–1717 — preboundary lengthening concentrates on the phrase-final
syllable rime and scales with boundary depth (multiple boundary strengths
are duration-distinguishable).
**Transfer: strongly, as a correction/censoring model.** Ballet marking sits
in *both* traditions (it's speech performing music). Concretely: regress
log-IOI on phrase position using Gemini's structure/phrase output; take the
plateau (intercept) as tempo; or simply censor the final 1–2 IOIs before any
pause > ~250 ms or flagged phrase boundary. Wightman licenses using the
existing word-timestamp pauses as boundary-strength detectors.

**3.3 Tempo drift tracking on sparse onsets — the closest published match to
this problem.** Cemgil, A.T., Kappen, B., Desain, P. & Honing, H. (2000),
"On tempo tracking: Tempogram representation and Kalman filtering," *JNMR*
29(4):259–273: tempo as a hidden state (log-period random walk) with a
Bayesian *tempogram* computed directly from **onset lists** (symbolic/MIDI,
i.e., sparse events — no audio needed); Kalman filtering yields smooth tempo
tracks with uncertainty. Cemgil & Kappen (2003), "Monte Carlo methods for
tempo tracking and rhythm quantization," *JAIR* 18:45–81 (arXiv 1106.4863):
switching state-space model where discrete switches are each onset's **score
position** (which grid slot — beat, eighth, triplet…) and the continuous
state is tempo; particle filtering/MCMC does joint MAP quantization + tempo
tracking. Hainsworth & Macleod (2004), "Particle filtering applied to
musical tempo tracking," *EURASIP JASP*.
**Transfer: strongly — this is the project's problem, published.** "Assign
each onset to 1×/2×/3× while tracking a drifting tempo from sparse
expressive onsets" is literally the Cemgil & Kappen generative model; the
Gemini beat/and/ah classification even gives *observed* (noisy) switch
labels, making inference far easier than their blind case. A 1D Kalman
filter over log-period with marker-index observations is an afternoon of
numpy.

**3.4 Robust estimators for few events.** Median/trimmed IOI *within an
assigned level* (never across levels); robust line fit of onset time vs
cumulative grid index — slope = period — via Theil–Sen
(`scipy.stats.theilslopes`) or Huber IRLS, with ~29% breakdown absorbing 1–2
agogic outliers per phrase; Povel, D.-J. & Essens, P. (1985), "Perception of
temporal patterns," *Music Perception* 2(4):411–440 — the internal-clock
model scores candidate (period, phase) clocks by *negative evidence* (clock
ticks falling on silence/unaccented positions), designed for sparse tone
patterns and validated against human induction; and the H-test (1.7) for
period search.
**Transfer: strongly; all trivial in numpy/scipy.** Povel & Essens'
counterevidence idea is a good tiebreaker between family members: the
correct beat level rarely predicts pulses where the teacher voices
*nothing*.

---

## 4. Evaluation science

**4.1 mir_eval.tempo (verified from source, v0.8.2).**
`detection(reference_tempi (2, ascending), reference_weight ∈ [0,1],
estimated_tempi (2,), tol=0.08)`: hitᵢ = minⱼ|estⱼ − refᵢ| ≤ tol·refᵢ;
returns **P-score** = w·hit₁ + (1−w)·hit₂, **One-correct**, **Both-correct**.
This is the MIREX Audio Tempo Estimation protocol, i.e., McKinney et al.
2007's perceptual two-level format operationalized. Note tol = 8% (MIREX),
not 4%.

**4.2 Acc1/Acc2 vs `truth_in_family`.** Gouyon, Klapuri, Dixon, Alonso,
Tzanetakis, Uhle & Cano (2006), "An experimental comparison of audio tempo
induction algorithms," *IEEE TASLP* 14(5):1832–1844 (ISMIR'04 contest):
**Acc1** = estimate within ±4% of the annotated tempo; **Acc2** = within ±4%
of {⅓, ½, 1, 2, 3}× the annotation; headline finding: >80% accuracy "if we
do not insist on finding a specific metrical level." Acc2's family is
*identical* to `truth_in_family` — adopt the standard name and ±4% tolerance
so the numbers are comparable to 20 years of published results.
Modernization: Schreiber, H., Urbano, J. & Müller, M. (2020), "Music tempo
estimation: Are we done yet?" *TISMIR* 3(1):111–125 — binary accuracies hide
error structure; use **OE1 = log₂(est/ref)** and **OE2** = OE1 after
removing the best factor ∈ {1, 2, 3, ½, ⅓} (AOE1/AOE2 = absolute values),
reported as distributions with nonparametric CIs; their `tempo_eval` toolkit
implements all of this with per-dataset reports. The "mean-IOI lands between
levels" failure is *invisible* to Acc1/Acc2 tallies but shows up directly as
|OE2| mass between 0 and log₂(1.5) ≈ 0.585.

**4.3 Beat-level metrics (verified from mir_eval + madmom source).**
mir_eval.beat: `f_measure` window ±**0.07 s**; `cemgil` Gaussian error
σ = **0.04 s** (returns score vs true beats and max over metrical
variations) — from Cemgil et al. 2000; `goto` binary (error threshold 0.35,
mean < 0.2, σ < 0.2 of the inter-beat-normalized error over the longest
continuous region); `p_score` window = **0.2 × median inter-annotation
interval** (McKinney's beat P-score); `continuity` → **CMLc/CMLt/AMLc/AMLt**
with phase *and* period tolerance **0.175** (Davies, Degara & Plumbley 2009,
"Evaluation methods for musical audio beat tracking algorithms," QMUL Tech.
Rep. C4DM-TR-09-06). **Critical implementation gotcha, confirmed in
source:** mir_eval's AML variations are only {true, off-beat, double,
half-odd, half-even} — **no triple/third**; madmom's `evaluation.beats`
generates offbeat + double/half + **triple/third** variations by default at
the same 0.175 tolerance. For triplet-heavy ballet marking, mir_eval's AMLt
will punish legitimate ×3 level choices; use madmom's evaluator or extend
mir_eval's `_get_reference_beat_variations`.

**4.4 Annotation practice.** Two tempi + salience from tap histograms
(McKinney/Moelants MIREX'06: 140 excerpts × 40 tappers); crowdsourced
tap-derived salience for EDM (Schreiber & Müller, ISMIR 2018 crowdsourcing
paper); **difficulty-aware corpus curation**: Holzapfel, Davies, Zapata,
Oliveira & Gouyon (2012), "Selective sampling for beat tracking evaluation,"
*IEEE TASLP* 20(9):2539–2548 — build the eval set from clips where a
committee of trackers *disagrees* (the SMC dataset that resulted — romantic,
rubato-heavy — is the closest public analogue to marking audio); for
expressive material, annotate **local tempo curves** rather than one global
BPM and evaluate windowed, octave-aware (Schreiber, Zalkow & Müller, ISMIR
2020).

---

## (a) Top-5 "steal this first," with implementation notes

1. **Harmonic-summed point-process periodogram (Fourier tempogram on the
   markers) — replaces mean-IOI outright.** With marker times tⱼ and weights
   wⱼ (beat=1.0, and=0.5, ah=0.3, say): for each candidate tempo f on a log
   grid 30–300 BPM, S₁(f) = |Σⱼ wⱼ e^(−2πif tⱼ)|²/Σwⱼ; score
   S(f) = Σ_{h∈{1,2,3}} aₕS₁(hf) with a = (1.0, 0.5, 0.33) (or the H-test's
   adaptive m; de Jager 1989). Peak of S = beat frequency even when onsets
   are an arbitrary 1×/2×/3× mix; the phase of the winning bin gives beat
   phase for free. ~15 lines of numpy; the librosa equivalent is
   `fourier_tempogram` on a synthesized impulse `onset_envelope` (use one
   full-clip window for global tempo, ~6 s windows for drift).
2. **Dual-bias cross-check + family salience via librosa.** Compute the ACF
   tempogram too (`librosa.feature.tempogram`, `ac_size≈8`) and
   geometric-mean it with the Fourier tempogram before peak-picking —
   Fourier suppresses subharmonics, ACF suppresses harmonics, the product
   suppresses both (FMP Ch. 6; Grosche/Müller/Kurth 2010). Then
   `librosa.feature.tempogram_ratio` (verified default factors
   `[4, 8/3, 3, 2, 4/3, 3/2, 1, 2/3, 3/4, 1/2, 1/3, 3/8, 1/4]`, after
   Prockup et al. 2015) reads off the salience of every family member of the
   chosen reference — exactly the numbers `TempoCandidate`/`alternates`
   should carry.
3. **Dixon-style IOI clustering with integer-ratio reinforcement as the
   independent second opinion.** Pure Python, no deps: all pairwise IOIs
   < 2.5 s → greedy clustering with 25–50 ms tolerance → cluster score =
   Σ weights + λ·(support from clusters at ratios 2, 3, 4, 6) → ranked
   period candidates. Cross-check method 1's family pick; disagreement ⇒
   lower confidence in `TempoResult`. (Dixon 2001 built this for expressive
   performances specifically.)
4. **Agogic-aware robust period regression (the precision-layer upgrade).**
   Given a candidate (τ, φ) from step 1, assign each marker its grid index
   nⱼ = round((tⱼ−φ)/τ) — this is the Cemgil & Kappen quantization step made
   trivial by the beat/and/ah labels — then robust-fit tⱼ ≈ φ + τ·nⱼ with
   `scipy.stats.theilslopes` (or Huber IRLS), **down-weighting or excluding
   phrase-final markers** (last word before a pause > ~250 ms, or a Gemini
   phrase boundary) per Repp/Todd/Wightman. Slope = period; MAD of residuals
   = confidence; fit per half-clip (or a 1-D Kalman on log τ, Cemgil 2000)
   to expose drift instead of averaging over it.
5. **Octave decision as explicit prior-weighted classification, with
   two-tempo output.** Candidates {τ/3, τ/2, τ, 2τ, 3τ}: posteriorₖ ∝
   salienceₖ (steps 1–2) × perceptual prior (b) × exercise-conditioned
   likelihood (marker density, `SubdivisionResult`, Gemini exercise label).
   Report top-1 as primary and keep the runner-up with normalized salience
   in `alternates` (MIREX format). This generalizes Percival & Tzanetakis
   (2014, *IEEE/ACM TASLP* 22(12):1765–1776), whose entire final stage is a
   linear SVM choosing a multiplier ∈ {0.5, 1, 2} on the raw period estimate
   (verified in Essentia's `percivalbpmestimator.cpp`) — this project needs
   the {3, ⅓} extension they didn't.

## (b) Replacing the hard 70–140 band with a perceptual prior

**Curve:** log-Gaussian in log₂-BPM — the field's standard (Ellis 2007;
librosa's built-in default), consistent with Parncutt's log-period Gaussian
salience and a close smooth proxy for van Noorden & Moelants' resonance
curve:

  w(T) = exp(−½·((log₂T − log₂T₀)/σ)²)

**Literature parameter anchors:** T₀ = 120 BPM, σ = 1.4 octaves (Ellis 2007);
librosa ships `start_bpm=120, std_bpm=1.0`; Parncutt's peak ≈ 100 BPM
(600–700 ms period); Moelants' preferred tempo 120–130; vN&M resonance
f₀ = 2 Hz with W(f) = [(f₀²−f²)²+βf²]^(−½) − [f₀⁴+f⁴]^(−½), β ≈ 1, for the
literature-exact asymmetric curve.

**For ballet marking specifically:** start with **T₀ = 100–110 BPM,
σ = 1.2–1.4 octaves** (Parncutt-to-Moelants midpoint; wide). Sanity check at
the extremes: with T₀=100, σ=1.3, w(40 BPM) ≈ 0.60 and w(200 BPM) ≈ 0.74 —
out-of-band truths survive, unlike the current band which zeroes them. Three
rules from the literature: (1) apply the prior **only when selecting the
reported level within the family** — multiply candidate salience, never gate
or fold the raw measurement (Ellis applies it to the periodicity function,
not to the output; ADR-014's raw-pulse philosophy is exactly right);
(2) upgrade to **exercise-conditioned priors** à la Hörschläger et al.
2015 — fit T₀(exercise), σ(exercise) by MLE on log₂-tempo of the blessed
eval clips per Gemini exercise label, shrunk toward the global prior while
counts are small (adagio/plié vs petit allegro genuinely occupy different
octaves, and Gemini already hands over the label); (3) when two family
members' posteriors are within ~1.5:1, **emit both with salience** rather
than forcing one — McKinney & Moelants 2006 shows the ambiguity is real in
the ground truth, so a forced scalar is a category error.

## (c) Evaluation metrics to adopt verbatim

1. **Acc1 and Acc2 at ±4%** (Gouyon et al. 2006), with Acc2's family
   {⅓, ½, 1, 2, 3} — rename/alias `truth_in_family` to Acc2; it is the same
   metric, and the shared name buys comparability with the entire
   tempo-induction literature.
2. **OE1/OE2 (and AOE1/AOE2) distributions** (Schreiber, Urbano & Müller
   2020; `tempo_eval` toolkit): OE1 = log₂(est/ref); OE2 = residual after
   removing the best {2, 3, ½, ⅓} factor. Add both to the tier-0/1 harness
   in `src/musical_perception/evals/` — OE2 is the *only* standard metric
   that directly measures the "landed between levels" failure, and as a
   continuous quantity it makes the tier-1 no-regression gate sharper than
   binary outcome flips.
3. **mir_eval.tempo.detection** (P-score, One-correct, Both-correct;
   tol=0.08 for MIREX comparability, additionally at 0.04) — which requires
   upgrading case YAML ground truth to (tempo1, tempo2, salience); default
   salience=1.0 keeps existing single-level cases valid, and the field
   format matches Vision 08 §8.2-style two-level annotation practice.
4. **For marker/beat sequences** (TimedMarker stream vs annotated beats):
   continuity **AMLt/CMLt at 17.5%** tolerance — but via
   **madmom.evaluation.beats** (offbeat + double/half + triple/third
   variations, verified in source), *not* mir_eval's `continuity`, which
   lacks triple/third and would penalize correct triplet-level tracking;
   plus **Cemgil σ=40 ms** for timing sharpness and **F-measure ±70 ms** as
   the blunt headline number.
5. Keep the standard tolerance constants as named constants: **±4% tempo,
   ±70 ms / 17.5% beat windows** — both are perceptually grounded (Repp
   2005's 3–5% tapping CV; MIREX practice), so eval disagreements below them
   are noise by construction.

---

## Sources

- [Scheirer 1998, JASA (PDF)](https://pubs.aip.org/asa/jasa/article-pdf/103/1/588/8083614/588_1_online.pdf) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/9440344/)
- [Grosche, Müller & Kurth 2010, Cyclic tempogram (PDF)](https://resources.mpi-inf.mpg.de/MIR/tempogramtoolbox/2010_GroscheMuellerKurth_TempogramCyclic_ICASSP.pdf) · [Tempogram Toolbox](https://www.audiolabs-erlangen.de/resources/MIR/tempogramtoolbox/)
- [Grosche & Müller 2011, PLP (Semantic Scholar)](https://www.semanticscholar.org/paper/Extracting-Predominant-Local-Pulse-Information-From-Grosche-M%C3%BCller/72e92b29cf36a2f0262f17f82d9d28f545eaadac) · [librosa.beat.plp](https://librosa.org/doc/main/generated/librosa.beat.plp.html)
- [FMP notebooks — autocorrelation tempogram](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S2_TempogramAutocorrelation.html) · [FMP — DP beat tracking](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S3_BeatTracking.html)
- [Dixon 2001 (OFAI TR PDF)](https://ofai.at/papers/oefai-tr-2001-19.pdf) · [BeatRoot eval, JNMR 2007 (PDF)](http://www.eecs.qmul.ac.uk/~simond/pub/2007/jnmr07.pdf)
- [Ellis 2007, Beat tracking by DP (PDF)](https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf)
- [Davies & Plumbley 2007 (Semantic Scholar)](https://www.semanticscholar.org/paper/Context-Dependent-Beat-Tracking-of-Musical-Audio-Davies-Plumbley/10d27c8a99a989bd9947f60ee3349b99374cf2ec)
- [Klapuri, Eronen & Astola 2006 (PDF)](https://www.iro.umontreal.ca/~pift6080/H09/documents/papers/klapuri_meter.pdf)
- [McKinney & Moelants 2006, Music Perception](https://online.ucpress.edu/mp/article-abstract/24/2/155/62298/) · [McKinney et al. 2007, JNMR](https://www.tandfonline.com/doi/abs/10.1080/09298210701653252)
- [van Noorden & Moelants 1999, JNMR](https://www.tandfonline.com/doi/abs/10.1076/jnmr.28.1.43.3122) · [Moelants 2002 (Semantic Scholar)](https://www.semanticscholar.org/paper/Preferred-tempo-reconsidered.-Moelants/b0db06a5a8b2c1942afff5c317c5f6da55a7dcf7)
- [Parncutt 1994, Music Perception](https://online.ucpress.edu/mp/article-abstract/11/4/409/46407/)
- [Cemgil et al. 2000, Tempogram + Kalman (PDF)](https://www.mcg.uva.nl/mcg-2023/papers/mmm-27.pdf) · [Cemgil & Kappen 2003, JAIR (arXiv)](https://arxiv.org/abs/1106.4863)
- [Povel & Essens 1985, Music Perception](https://online.ucpress.edu/mp/article/2/4/411/62235/Perception-of-Temporal-Patterns)
- [Gouyon et al. 2006, tempo induction comparison](https://dl.acm.org/doi/10.1109/TSA.2005.858509)
- [Davies, Degara & Plumbley 2009, beat eval tech report (ResearchGate)](https://www.researchgate.net/publication/268132820_Evaluation_Methods_for_Musical_Audio_Beat_Tracking_Algorithms)
- [Schreiber, Urbano & Müller 2020, TISMIR "Are We Done Yet?" (PDF)](https://www.audiolabs-erlangen.de/content/05_fau/professor/00_mueller/03_publications/2020_SchreiberUM_MusicTempo_TISMIR_ePrint.pdf) · [tempo_eval toolkit](https://github.com/tempoeval/tempo_eval)
- [Percival & Tzanetakis 2014 (PDF)](https://webhome.csc.uvic.ca/~gtzan/output/taslp2014-tempo-gtzan.pdf) · [Essentia PercivalBpmEstimator source](https://github.com/MTG/essentia/blob/master/src/algorithms/rhythm/percivalbpmestimator.cpp)
- [Böck, Krebs & Widmer 2015, ISMIR (PDF)](https://archives.ismir.net/ismir2015/paper/000196.pdf) · [madmom](https://github.com/CPJKU/madmom)
- [Schreiber & Müller 2018, TempoCNN (Zenodo)](https://zenodo.org/records/1492353) · [tempo-cnn repo (DT-Maz local-tempo models)](https://github.com/hendriks73/tempo-cnn)
- [Grosche, Müller & Sapp 2010, Mazurka beat-tracking difficulty (PDF)](https://archives.ismir.net/ismir2010/paper/000110.pdf) · [Schreiber, Zalkow & Müller 2020, local tempo (PDF)](https://www.audiolabs-erlangen.de/content/05_fau/professor/00_mueller/03_publications/2020_SchreiberZM_LocalTempoChopin_ISMIR.pdf)
- [Repp 1992, Cognition (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/001002779290003Z) · [Todd 1985, Music Perception](http://mp.ucpress.edu/content/3/1/33) · [Friberg & Sundberg 1999, JASA](https://pubs.aip.org/asa/jasa/article-abstract/105/3/1469/558501/)
- [Wightman et al. 1992 context (Byrd & Saltzman PDF)](https://sail.usc.edu/~dbyrd/byrd_saltzjphon98AP.pdf)
- [Ding et al. 2017, speech vs music modulation spectra](https://pubmed.ncbi.nlm.nih.gov/28212857/)
- mir_eval 0.8.2 and madmom source (metric definitions verified from source during the review session)
