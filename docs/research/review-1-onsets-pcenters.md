# Review 1 · Onset/event detection for voice + where the beat sits in a syllable

*Part of the [voice-as-drum literature review](voice-as-drum-review.md)
(2026-08-09, companion to [ADR-016](../adr/016-rhythm-core-reset.md)).
Produced by a web-research agent; sourcing caveats are noted at the end.*

**Headline conclusions up front:**

1. The perceptual beat of a syllable is NOT its acoustic onset and NOT an ASR
   word timestamp — it sits at/near the **vowel onset**, operationalized best
   as the **maximum of the amplitude-envelope derivative ("peakRate")**. This
   is now convergently supported by perception (P-center literature),
   production (speech cycling), sensorimotor synchronization (tapping
   studies), and cortical electrophysiology.
2. The right detector family for "voice as drum" is **envelope/derivative-based
   syllable-nucleus detection with a voicing gate**, not MIR spectral-flux
   onset detection (which mis-times and false-fires on voice) and not ASR
   (which the project already falsified).
3. For meter, the strongest transferable idea is **delta-band (~beat) vs
   theta-band (~syllable) envelope decomposition with phase alignment**
   (Leong & Goswami), because ballet marking at 40–200 BPM puts the beat in
   the delta band while syllables stay in theta.

---

## 1. Syllable nuclei / syllable onsets / speech-rate estimation

**1.1 Mermelstein (1975), "Automatic segmentation of speech into syllabic units," JASA 58(4):880–883.**
Computes a "loudness" function from the short-time power spectrum band-limited
to ~500–4000 Hz, low-passed at 40 Hz; then recursively computes the **convex
hull** of the loudness contour over each segment and splits at the point of
maximum hull-minus-loudness difference if it exceeds a dip threshold. On ~400
syllables of continuous text: 6.9% missed, 2.6% inserted. The convex-hull
trick makes dip detection self-normalizing against slow loudness drift.
**TRANSFERS.** The hull-difference criterion is exactly what you want under
expressive crescendo/decrescendo across a marking phrase (a fixed dB dip
threshold fails there). Gives nuclei, not beat instants — pair with a
within-nucleus landmark (see P-centers). Trivial to implement on any envelope.

**1.2 de Jong & Wempe (2009), "Praat script to detect syllable nuclei and measure speech rate automatically," Behavior Research Methods 41(2):385–390; v3: de Jong, Pacilly & Heeren (2021), Assessment in Education 28(4).**
Praat intensity contour (~50 ms analysis frames); threshold speech vs silence
at **−25 dB relative to the 99th-percentile intensity maximum**; candidate
nuclei = intensity peaks that are preceded/followed by dips of at least
**2 dB** (min dip), with minimum pause ~0.3–0.4 s; then **discard unvoiced
peaks** using Praat autocorrelation pitch (30–450 Hz, 20 ms steps). Validated
against human syllable counts on two Dutch corpora with high correlations. v3
adds filled-pause ("uhm") detection and fluency measures; a Python path exists
via Parselmouth (Jadoul, Thompson & de Boer 2018, J. Phonetics 71:1–15).
**TRANSFERS with retuning.** This is the workhorse baseline for our signal:
close-mic, one speaker. The voicedness gate kills breaths, clicks, consonant
bursts — the main false-positive sources. Retune: raise min dip to ~4–8 dB
(marked speech is hyper-modulated; 2 dB over-detects), shorten min pause
(~0.15–0.25 s; marking has short inter-beat gaps), and keep the
quantile-relative silence threshold (robust to level changes). Failure mode:
whispered/devoiced pickup syllables ("t" in "tendu" is fine — the nucleus is
voiced — but fully whispered vocables get dropped).

**1.3 Wang & Narayanan (2007), "Robust speech rate estimation for spontaneous speech," IEEE TASLP 15(8):2190–2201.**
Direct (transcription-free) syllable-rate estimation: 19 subband envelopes;
select the **prominent subbands** per frame; combine **temporal correlation**
(envelope self-similarity) with **selected cross-subband correlation**;
smooth; count peaks between pauses. Standard reference method on Switchboard;
the basis of most later "speaking-rate without ASR" work.
**PARTIALLY transfers.** Built to output a rate (syll/s) over utterances, not
event times, and tuned for conversational (not metrically produced) speech.
Steal the two robustness tricks — prominent-subband selection and temporal
correlation — if the simple intensity method proves noisy; otherwise skip.

**1.4 Tilsen & Johnson (2008), "Low-frequency Fourier analysis of speech rhythm," JASA-EL 124(2):EL34–EL39; Tilsen & Arvaniti (2013), JASA 134(1):628–639.**
The "rhythm spectrum": bandpass the waveform in a vocalic band (~700–1300 Hz),
rectify, low-pass (~10 Hz), window, FFT; peaks of this modulation spectrum
characterize rhythmicity (2013 version uses Hilbert envelope + empirical mode
decomposition to handle non-stationarity).
**PARTIALLY transfers — as a prior, not an estimator.** Zhang, Zou & Ding
(2023, Neurosci Biobehav Rev 147:105111) showed the modulation-spectrum peak
tracks syllable rate reliably **only when pooled over minutes**; on 10–30 s
clips with interleaved talk it is unstable, and phrase-level prosody injects
delta-band peaks that masquerade as beat peaks. Use the rhythm spectrum only
to sanity-bound tempo hypotheses (e.g., reject a beat-rate hypothesis with
zero envelope modulation energy near it), never as the tempo estimate.

**1.5 Theta-band speech rhythm: Ding, Patel, Chen, Butler, Luo & Poeppel (2017), "Temporal modulations in speech and music," Neurosci Biobehav Rev 81:181–187.**
Across 9 languages / 25+ h of speech, the envelope modulation spectrum peaks
at ~4–5 Hz (2–8 Hz band = syllable rate); music peaks near ~2 Hz.
Cortical-tracking work (Giraud & Poeppel 2012 and successors) treats 4–8 Hz
as the syllabic carrier band.
**TRANSFERS as a design constraint.** Ballet marking beats at 40–200 BPM =
0.67–3.3 Hz — mostly BELOW the syllabic band. Interpretation: in marking, the
*beat* is a delta-rate structure carried by *stressed* syllable nuclei, while
unstressed syllables ("-du" of "tendu," "and-a") fill theta. So (a) don't
equate detected syllable rate with tempo — expect a small integer ratio;
(b) meter lives in the delta/theta relationship (see 3.4).

**1.6 Räsänen, Doyle & Frank (2018), "Pre-linguistic segmentation of speech into syllable-like units," Cognition 171:130–150 — the `thetaOscillator`.**
Gammatone filterbank → envelopes (downsampled to 1 kHz) → sonority estimate
drives a bank of damped **harmonic oscillators at theta rate**; oscillator
peaks = syllable nuclei, troughs = boundaries. Language-independent; MATLAB
reference implementation public (orasanen/thetaOscillator; unofficial Python
notebook port exists).
**TRANSFERS.** Conceptually ideal (entrainment-based, tempo-flexible), and it
outputs both nuclei and boundaries. Cost: MATLAB port, more machinery than
the Praat method for likely similar accuracy on clean close-mic solo speech.
Try after the simple method plateaus.

**1.7 Modern neural syllable detectors: SylNet — Seshadri & Räsänen (2019), IEEE Signal Processing Letters 26(9):1359–1363; Sylber — Cho et al. (ICLR 2025, arXiv:2410.07168); SD-HuBERT — Cho et al. (ICASSP 2024, arXiv:2310.10803).**
SylNet: WaveNet-style conv net + LSTM on log-mel features, trained end-to-end
to output syllable **counts** (code public). Sylber/SD-HuBERT: self-supervised
speech transformers that spontaneously organize frame representations into
syllabic segments; Sylber does clean syllable **boundary detection**, F1 > 71
at standard tolerance across English/Spanish/Mandarin without fine-tuning.
**PARTIALLY transfers.** SylNet gives counts, not times — only useful for
cross-checking. Sylber gives boundaries and is genuinely promising, but
(a) boundary ≠ beat instant (still needs a P-center landmark within the
segment), (b) heavyweight dependency, (c) untested on shouted/chanted teacher
speech. Rank below envelope methods for v1.

**1.8 Oganian & Chang (2019), "A speech envelope landmark for syllable encoding in human superior temporal gyrus," Science Advances 5(11):eaay6279 — "peakRate."**
ECoG: mid-STG neural populations fire discretely at **local maxima of the
envelope's rate of change (peakRate events)** — not at envelope peaks, not
continuously. In English, peakRate closely aligns with **vowel onsets** (the
onset-to-nucleus transition). MATLAB code for peakRate extraction is public.
**TRANSFERS — this is the drum-trigger definition.** The brain itself
apparently discretizes speech with an onset-strength function: derivative of
a smoothed envelope, peak-picked. Directly implementable in ~15 lines of
scipy (recipe below).

**1.9 MacIntyre, Cai & Scott (2022), "Pushing the envelope: Evaluating speech rhythm with different envelope extraction techniques," JASA 151(3):2002–2026.**
Systematically compares envelope extraction pipelines (Hilbert vs
rectify-and-smooth, different cutoffs, audition-informed filterbanks) ×
candidate landmarks against manual phonetic annotation across speech styles.
Best proxy for annotated **vowel onsets: peaks in the first derivative of a
human-audition-informed envelope**; both the extraction method and the speech
style materially change landmark timing. Companion MATLAB toolbox on MATLAB
File Exchange ("Speech Envelope and Acoustic Landmarks").
**TRANSFERS.** This is the parameter-tuning manual for the onset front-end,
and independent validation of 1.8. Note their warning: pipeline choices shift
landmarks by tens of ms — calibrate once against a hand-labeled marking clip.

**1.10 Zhang, Zou & Ding (2023), Neurosci Biobehav Rev 147:105111 (see 1.4).**
1000+ h, multiple languages, seq2seq modeling: syllable onsets explain ~24%
of envelope variance via a speaker-independent kernel; **local envelope
features are phase-locked to syllable onsets and beat the modulation spectrum
as a syllable-rate correlate on short timescales.**
**TRANSFERS** — the quantitative justification for "discrete local onsets,
then fit a grid" over "Fourier the envelope."

---

## 2. P-centers: where the perceptual beat sits in a syllable

**2.1 Morton, Marcus & Frankish (1976), "Perceptual centers (P-centers)," Psychological Review 83(5):405–408.**
Founding demonstration: digit sequences whose *acoustic onsets* are
isochronous sound irregular; sequences adjusted until they *sound* regular
define each word's P-center — its "psychological moment of occurrence."
P-centers are what is regular in perceptually regular speech.
**TRANSFERS as the framing result:** a beat grid fit to acoustic onsets is
fit to the wrong events by construction.

**2.2 Marcus (1981), "Acoustic determinants of perceptual center (P-center) location," Perception & Psychophysics 30(3):247–256.**
Rhythm-adjustment experiments on digits "one"–"nine." P-center location
(measured from acoustic onset) increases with **initial consonant(cluster)
duration** (primary) and with **rhyme (vowel+coda) duration** (secondary,
smaller weight); a two-parameter linear model fits well. The widely
reproduced regression (via secondary sources — Villing 2010 thesis; Patel
2008 — the paywalled primary PDF was not reachable from this environment) is
approximately **P ≈ 0.65·(onset consonant duration) + 0.25·(rhyme duration)**
after stimulus onset. Sibilant-initial digits ("six," "seven") deviate from
the simple consonant-duration correlation.
**TRANSFERS directly as a correction formula** whenever word identity + rough
segmentation are known (Whisper gives both): the beat leads the vowel
slightly and lags the acoustic onset by roughly two-thirds of the
onset-cluster duration. Caveat: coefficients are a fit to isolated English
digits; treat as prior, not law — and note expressive rhyme lengthening at
phrase ends *moves the P-center later* via the 0.25 term, partially
explaining "expressive lengthening" tempo wobble.

**2.3 Howell (1988), "Prediction of P-center location from the distribution of energy in the amplitude envelope," Perception & Psychophysics 43:90–93.**
P-center predicted from the amplitude envelope's energy distribution (a
center-of-gravity-like account) rather than segmental durations — first
fully signal-based model.
**PARTIALLY transfers:** center-of-gravity blurs under long sustained rhymes
("o-o-one a-a-nd"); derivative-peak models beat it, but it motivates
envelope-only estimation when no transcript exists.

**2.4 Pompino-Marschall (1989), "On the psychoacoustic nature of the P-centre phenomenon," Journal of Phonetics 17:175–192.**
Psychoacoustic model: P-center computed from the timing of **loudness
onsets/offsets within auditory filters** (critical-band excitation),
combining band-wise rise events into one perceptual moment.
**PARTIALLY transfers:** the multi-band-rise idea survives in modern form
(Harsin, peakRate on subband envelopes); full loudness-model machinery is
overkill for this signal.

**2.5 Scott (1993), PhD thesis, UCL, "P-centres in speech: an acoustic analysis"; Scott (1998), "The point of P-centres," Psychological Research 61:4–11.**
Manipulating onset ramps and offsets: P-center location is driven mainly by
the **envelope rise around the syllable-nucleus onset**; longer onset rise
times push the P-center later (reported slope ~0.24 of ramp duration in the
1998 experiments); offsets/duration have secondary effects. Scott proposed
the perceptual moment tracks the rapid loudness increase into the vowel.
**TRANSFERS:** the "rise into the vowel" is the thing to detect; also
predicts that soft glide onsets ("one" [w]) have later, fuzzier P-centers
than plosive onsets ("two") — expect word-dependent jitter if anchoring to
acoustic onsets.

**2.6 Harsin (1997), "Perceptual-center modeling is affected by including acoustic rate-of-change modulations," Perception & Psychophysics 59(2):243–251.**
Six-band filterbank; models P-centers as the weighted center of
**low-frequency loudness rate-of-change** components. Including
rate-of-change (velocity) information significantly improves P-center
prediction for CV and VC syllables over static energy models.
**TRANSFERS:** the direct ancestor of peakRate — quantitative evidence that
*derivative* features, not energy features, predict the perceptual beat.

**2.7 Villing (2010), "Hearing the Moment: Measures and Models of the Perceptual Centre," PhD thesis, NUI Maynooth; Villing, Ward & Timoney (2003), ISSC ("P-Centre extraction from speech: the need for a more reliable measure"); Villing, Repp, Ward & Timoney (2011), "Measuring perceptual centers using the phase correction response," Attention, Perception, & Psychophysics 73:1614–1629.**
The definitive review across speech, music, and articulatory P-center
research. Key conclusions: rhythm-adjustment vs tap-based measures broadly
agree; the phase-correction-response method gives efficient P-center
measurement; **no existing acoustic model is fully reliable across stimulus
types** — duration models (Marcus) fail on non-digit material, envelope
models (Howell, Pompino-Marschall, Harsin) each fail somewhere; P-center
offsets from acoustic onset range from ~0 ms (vowel-initial) to well over
100 ms (long onset clusters).
**TRANSFERS as calibration literature.** Practical read: use a *simple,
consistent* landmark (envelope-derivative peak) and absorb residual per-word
bias into the grid-fitting layer rather than chasing a perfect P-center model.

**2.8 Rathcke, Smit, Lin & Kubozono (2024), "Testing an acoustic model of the P-center in English and Japanese," JASA 155(4):2698–2706.**
Tests the model "P-center = moment of **fastest energy change in the syllabic
amplitude envelope**" in *natural connected speech*. English: P-center effect
confirmed; acoustically derived P-centers were statistically
indistinguishable from manually labeled **vowel onsets**. Japanese: effect
exists but the acoustic model fits less well (mora timing). P-center behaves
like perceptual anticipation of the vowel.
**TRANSFERS — the validation needed:** max-envelope-derivative ≈ vowel onset
≈ perceptual beat, in running English speech, with a documented language
caveat.

**2.9 Tap/beat alignment evidence: Lidji, Palmer, Peretz & Morningstar (2011), "Listeners feel the beat: Entrainment to English and French speech rhythms," Psychonomic Bulletin & Review 18:1035–1041; Rathcke, Lin, Falk & Dalla Bella (2021), "Tapping into linguistic rhythm," Laboratory Phonology 12(1); Franich lab (2023), "An acoustic study of rhythmic synchronization with natural English speech," J. Phonetics (S0095447023000529); London et al. (2019), "A comparison of methods for investigating the perceptual center of musical sounds," APP 81:2088–2101; Matters Arising + Assaneo reply (2025), "The timing of speech-to-speech synchronization is governed by the P-center," Communications Biology 8.**
Converging behavioral results: when people tap or speak along with speech,
**taps/utterances anchor to vowel onsets** (Rathcke 2021: consistent
tap-vowel-onset anchoring in looped-sentence synchronization; Lidji 2011:
reliable entrainment to English at stressed-syllable level). The 2025 Comm
Biol exchange shows even the popular Speech-to-Speech Synchronization test's
high/low-synchronizer split is partly a P-center measurement artifact —
mis-specifying the beat location corrupts downstream synchrony metrics.
**TRANSFERS as ground-truth protocol:** when hand-labeling beats in marking
clips for evals, label **vowel onsets of counted syllables** (or collect tap
data), not word starts; and treat "which landmark did the annotator use" as
a versioned decision, or the eval baseline bakes in the same error ASR did.

---

## 3. Rhythmic / entrained speech specifically

**3.1 Cummins & Port (1998), "Rhythmic constraints on stress timing in English," Journal of Phonetics 26(2):145–171 (+ Cummins & Port, ICSLP 1996; Port 2003, "Meter and speech," J. Phonetics 31:599–611).**
**Speech cycling:** subjects repeat "X for a Y" phrases to a metronome; the
phase of each stressed-**vowel onset** within the cycle is measured (beats
located with an envelope-based "beat extractor" over a vocalic band — the
lineage behind Tilsen & Johnson's 700–1300 Hz band). Produced phases are
strongly multimodal at **~1/3, 1/2, 2/3 of the cycle** (empirically
0.36 / 0.5 / 0.6): stressed vowel onsets snap to harmonic fractions of the
repetition cycle — a "harmonic timing" attractor landscape.
**TRANSFERS STRONGLY.** Ballet marking *is* quasi-speech-cycling
(self-entrained, metrically intended speech). Two steals: (a) measure
everything at stressed vowel onsets; (b) fit the beat grid assuming onsets
fall at small-integer phase fractions (1/2, 1/3, 2/3, 1/4) — the
duple/triple discriminator operating on exactly the attractors real speakers
use.

**3.2 Cummins (2009), "Rhythm as entrainment: The case of synchronous speech," Journal of Phonetics 37(1):16–28.**
Two speakers reading in synchrony stay locked within a few tens of ms without
a metronome — speech affords stable mutual entrainment; timing anchors are
again vocalic.
**Transfers as supporting evidence** that entrained speech has a recoverable,
low-jitter temporal skeleton (the tempo signal is really there, unlike in
conversational prosody).

**3.3 Chant/rap corpora: Condit-Schultz (2016), "MCFlow: A Digital Corpus of Rap Transcriptions," Empirical Musicology Review 11(2):124–147; Ohriner (2019), *Flow: The Rhythmic Voice in Rap Music*, Oxford UP.**
Manual corpora aligning rap syllables to a 16th-note metric grid;
methodological conventions: syllable position = vowel onset; accent from
stress+rhyme+duration; systematic treatment of syncopation and non-alignment
(Ohriner's MTO 2019 analysis of non-aligned flow).
**PARTIALLY transfers:** no detectors to steal, but the *annotation schema*
(syllable→grid position, accent tier, allowed syncopation categories) is a
ready-made design for the `TimedMarker`→grid layer and for eval file
formats. Rap flow also proves voice-to-grid alignment tolerates sparse/silent
beats.

**3.4 Leong & Goswami (2014/2015): "Impaired extraction of speech rhythm from temporal modulation patterns in speech in developmental dyslexia," Front Hum Neurosci 8:96; "Acoustic-Emergent Phonology in the Amplitude Envelope of Child-Directed Speech," PLoS ONE 10(12):e0144411 — the S-AMPH model.**
On 44 spoken English **nursery rhymes** (the closest published analog to
ballet marking): envelope energy concentrates in three AM bands — **~2 Hz
(delta, stress feet), ~5 Hz (theta, syllables), ~20 Hz (beta, onset-rime)**.
The **phase relation of delta and theta AMs encodes meter**: which theta
cycles ride delta peaks determines trochaic vs iambic vs dactylic patterns.
Automatic identification: 72% of stressed syllables, 82% of syllables, 78%
of onset-rime units.
**TRANSFERS STRONGLY for meter.** Concrete recipe: bandpass the envelope
~0.9–2.5 Hz and ~2.5–12 Hz, Hilbert-phase both, and read strong/weak
alternation from the delta phase at each detected nucleus. This yields
duple/triple and downbeat candidates *without* needing every syllable
detected — robust to sparse onsets.

**3.5 Singing-voice onset detection (the cautionary literature):**

- **Toh, Zhang & Wang (2008), "Multiple-feature fusion based onset detection
  for solo singing voice," ISMIR 2008.** Supervised GMMs classify onset vs
  non-onset frames over MFCC-family features (81 mel bands), with feature-
  and decision-level fusion; motivated precisely because energy/flux methods
  fail on soft onsets, portamento, vibrato. Best configurations reach only
  ~0.7 F on solo singing — versus >0.9 routinely for percussive onsets.
- **Böck & Widmer (2013), "Maximum filter vibrato suppression for onset
  detection," DAFx-13 — SuperFlux.** Spectral flux computed on a
  log-magnitude, 24-bands/octave filterbank spectrogram (defaults from
  reference implementation: 200 fps, frame 2048, fmin 30 Hz, fmax 17 kHz),
  with a **maximum filter over 3 frequency bands** applied to the reference
  frame before the positive difference (typical lag ≈ 2 frames / half the
  46 ms window), then online peak picking (threshold 1.1; pre_avg 0.15 s;
  pre_max 0.01 s; post_max 0.05 s; combine 30 ms). On operatic solo voice it
  cuts false positives by up to ~60% vs plain spectral flux by ignoring
  FM/vibrato trajectories.
- **Schlüter & Böck (2014), "Improved musical onset detection with CNNs,"
  ICASSP 2014** (+ madmom's `CNNOnsetProcessor`): CNN on mel spectrograms,
  F ≈ 0.88–0.90 on the standard Böck dataset — state of the art — yet
  singing/soft-onset material remains the weakest class in every MIREX-style
  evaluation.
- **Gong, Pons & Serra (2017), "Score-informed syllable segmentation for a
  cappella singing voice with CNNs," arXiv:1707.03544; Gong & Serra (2018),
  Interspeech, arXiv:1806.01665.** For jingju (Beijing opera) singing: CNN
  onset functions + **duration-informed HMM decoding** (a coarse duration
  prior per syllable) beat both plain peak-picking and HSMM forced alignment
  by large margins.

**Verdict:** MOSTLY NEGATIVE TRANSFER as detectors, but three lessons
transfer: (i) pure energy/flux is known-bad on voice; (ii) suppress
pitch-trajectory artifacts (max-filter) before differencing; (iii) the single
biggest win on vocal onsets comes from **decoding with duration/grid priors**
rather than from a better frame-level detector — which here means: let the
tempo-grid hypothesis explain sparse detections, don't demand a perfect onset
stream.

---

## 4. MIR onset-detection functions applied to voice

**4.1 Bello, Daudet, Abdallah, Duxbury, Davies & Sandler (2005), "A tutorial on onset detection in music signals," IEEE Trans. Speech & Audio Processing 13(5):1035–1047.**
Canonical taxonomy: energy derivative, high-frequency content, spectral flux,
phase deviation, complex-domain deviation, plus peak-picking practice
(adaptive median threshold). Energy-based ODFs excel only on percussive
attacks; tonal/soft onsets need spectral or complex-domain functions.

**4.2 Duxbury, Bello, Davies & Sandler (2003), DAFx — complex-domain onset detection; Dixon (2006), DAFx, "Onset detection revisited."**
Half-wave-rectified spectral flux is the best simple ODF; complex domain
helps tonal onsets but is phase-noise-sensitive. **On voice:** phase-based
functions degrade badly with breathiness/aspiration (noisy phase), and all
flux variants fire on formant transitions and pitch glides *within* a single
syllable.

**4.3 SuperFlux (see 3.5) is the best classical ODF for vocal material;** its
max-filter is specifically a *vocal-artifact* (vibrato/glide) suppressor.

**4.4 librosa implementation** (documented "Superflux onsets" example mirrors
Böck's settings: hop = 1/200 s, n_mels = 138 ≈ 24 bands/octave, fmin 27.5 Hz,
fmax 16 kHz, `lag=2`, `max_size=3`):
`librosa.onset.onset_strength(S=power_to_db(melspec), lag=2, max_size=3)` →
`onset_detect`.

**Overall verdict: PARTIAL.** For vocables with plosive onsets ("TWO," "da,"
"DUM") plain spectral flux works; it breaks on vowel-/glide-/nasal-initial
beat words — and "**one**" [wʌn] and "**and**" [ænd], the two most frequent
counting tokens, are exactly the pathological gradual-rise cases.
Complex-domain adds nothing on breathy close-mic speech. Use SuperFlux only
as a *secondary* voter alongside the envelope-derivative (peakRate) detector;
never as the primary beat anchor, because flux peaks time-lock to consonant
attack, i.e., they inherit the full P-center bias.

**Related tools worth knowing, with verdicts:** Chronset (Roux, Armstrong &
Carreiras 2016, Behav Res Methods — multi-feature *utterance*-onset detector
for RT experiments: wrong granularity, skip); Praat Vocal Toolkit port of
Syllable Nuclei v3 (convenient); vowel-onset-point (VOP) detection from
excitation source (Prasanna et al., IEEE — LP-residual/glottal-excitation
energy marks vowel onsets; a good third voter that is robust to fricative
noise since it tracks voicing excitation, not spectral energy).

---

## STEAL THIS FIRST (ranked)

**1. peakRate detector = smoothed envelope → first derivative → voiced-gated
peak picking.** (Oganian & Chang 2019; validated as ≈vowel onset ≈ P-center
in running English by Rathcke et al. 2024; best-landmark result in MacIntyre
et al. 2022.)
Recipe (scipy/librosa, ~20 lines): resample 16 kHz → envelope: either
`|hilbert(x)|` or bandpass 300–3000 Hz + full-wave rectify → 4th-order
Butterworth low-pass **10 Hz** (zero-phase `filtfilt`; MacIntyre: cutoff
choice shifts timing, so freeze it and calibrate once) → `np.diff` →
half-wave rectify → `scipy.signal.find_peaks` with prominence ≥
k·MAD(derivative) (start k≈3) and min distance ~120 ms → keep peaks where
Parselmouth/pYIN says voiced within ±30 ms. Output = beat-candidate times.
This *is* the drum-transient stream.

**2. de Jong & Wempe syllable-nuclei gate via Parselmouth.** Praat
`To Intensity: 50, 0`; silence threshold −25 dB re 99th-percentile max;
**min dip raised to 4–8 dB** for marked speech (default 2 dB); min pause
~0.2 s; voicing check with AC pitch 30–450 Hz (raise floor to ~75 Hz for
adult close-mic to skip creak octave errors). Use nuclei as *regions*; take
the peakRate event inside each region as the event time. This pairing
(region proposal + landmark) removes breaths/clicks/plosive-burst doubles
almost entirely.

**3. Harmonic-phase grid fitting on stressed-vowel onsets (Cummins & Port).**
When scoring a tempo/meter hypothesis (period T, phase φ, subdivision
d∈{2,3}), score each detected event by circular distance to the nearest grid
position in {0, 1/2} (duple) or {0, 1/3, 2/3} (triple) of the beat cycle,
weighting events by local envelope rise size (accent proxy). Sparse onsets
are fine: missing beats cost nothing; only *misplaced* events count. This
encodes the empirically observed attractors (0.36/0.5/0.6) of entrained
English speech rather than assuming every syllable is a beat.

**4. Marcus correction for any word-timestamp source kept in the pipeline.**
If Whisper stays as a redundant channel: beat_time ≈ word_start +
0.65·(onset-consonant duration) + 0.25·(rhyme duration) — or, cheaper and
better: beat_time = first peakRate event inside the word span. Precompute
onset-cluster classes for the closed ballet vocabulary (vowel-initial ≈ 0 ms;
glide/nasal "one" ≈ 40–80 ms; stop "two/tendu" ≈ 10–40 ms after burst;
/s/-clusters "six/step" ≈ 80–150 ms) so even without acoustics the ASR
timestamps can be de-biased per token. Treat coefficients as English-digit
priors, not constants (Villing 2010; Rathcke 2024's Japanese caveat).

**5. S-AMPH-lite meter reader (Leong & Goswami).** Envelope (as in #1,
before differentiation) → two zero-phase bandpasses: **0.9–2.5 Hz
(beat/foot)** and **2.5–12 Hz (syllable)** → Hilbert instantaneous phase of
the slow band sampled at each event from #1/#2 → events clustering at
slow-band phase ≈ 0 are strong beats; count events per slow cycle and test
2:1 vs 3:1 theta:delta ratio for duple/triple; downbeat = phase of
largest-amplitude delta cycles. Works on 10–30 s, tolerates missing onsets,
and directly feeds the existing `SubdivisionResult`/`CountingSignature`
types.

---

## KNOWN TRAPS

1. **Anchoring the grid to acoustic onsets (incl. ASR word starts) injects
   word-dependent lead of 0–150 ms.** The error is *systematic per word
   type*, so it aliases into tempo drift and subdivision misclassification,
   not just noise (Morton 1976; Marcus 1981; Villing 2010).
   "Six"/"seven"-like sibilant words are the worst outliers.
2. **Envelope peak ≠ beat.** The perceptual moment is on the *rise* (max
   derivative), not at max amplitude — envelope-peak picking lags the beat
   by half the vowel.
3. **Envelope-pipeline sensitivity:** Hilbert-vs-rectify, filter order, and
   cutoff move landmarks by tens of ms (MacIntyre et al. 2022). Freeze one
   pipeline, calibrate against one hand-labeled clip, version it (fits the
   eval-harness/bless workflow).
4. **Modulation-spectrum tempo on short clips is a mirage** (Zhang et al.
   2023): the spectral peak needs minutes to stabilize, and phrase-level
   delta energy (not the beat) dominates 0.5–2 Hz. Use local events + grid
   fit; use spectra only as weak priors.
5. **The most frequent counting words have the hardest onsets:** "one"
   (glide), "and" (vowel), "a" (schwa pickup). Any detector benchmarked on
   plosive vocables ("da-da-DUM") will silently over-perform relative to
   real counting speech. Build the eval set around glide/vowel-initial
   tokens.
6. **Close-mic artifacts:** breaths, lip smacks, plosive pops, chair/floor
   thumps all produce beautiful envelope rises. The voicing gate (#2) is
   non-negotiable; consider a high-pass at 80–100 Hz before envelope
   extraction to kill pops and thumps.
7. **Vibrato/glides on sustained vocables** ("uuup," "aaand") make
   spectral-flux ODFs fire mid-syllable; if any flux ODF is used, use
   SuperFlux's max-filter variant, never plain flux (Böck & Widmer 2013).
8. **Expressive lengthening at phrase boundaries** both stretches the rhyme
   (shifting P-centers later via Marcus's 0.25 term) and violates isochrony;
   fit tempo with outlier-robust methods on inter-P-center intervals (the
   precision layer's median/confidence machinery is the right shape) and let
   phrase-final intervals carry low weight.
9. **Silent beats are missing data, not detector failures.** Tune the
   front-end for precision; recall is the grid's job (duration/grid-prior
   decoding is what fixed singing segmentation — Gong & Serra 2018).
10. **Interleaved non-rhythmic talk:** don't try to classify "rhythmic vs
    talk" at the onset level; fit local grids over sliding windows and keep
    spans where grid residuals are low — entrained speech is dramatically
    more regular than talk (Cummins 2009), so the residual gap is large.
11. **Language/style caveat on the vowel-onset ≈ P-center identity:** solid
    for English stress-timed material; weaker for mora-timed Japanese
    (Rathcke et al. 2024) — relevant if teachers count in other languages;
    the peakRate landmark remains the best available anchor regardless, but
    re-calibrate offsets.
12. **MIR evaluation tolerances hide the bias:** onset F-measures use
    ±25–50 ms windows, so a detector can score 0.9 while carrying a
    consistent +40 ms consonant-attack bias that corrupts beat phase and
    swing/subdivision estimates. Evaluate *signed* asynchrony, not just hit
    rate.

---

## Sources

- [de Jong & Wempe 2009, Behavior Research Methods (Springer)](https://link.springer.com/article/10.3758/BRM.41.2.385) | [PDF](https://www.fon.hum.uva.nl/archive/2009/2009-brm-JongWempe.pdf) | [script site](https://sites.google.com/site/speechrate) | [v2 script source (GitHub)](https://github.com/FieldDB/Praat-Scripts/blob/main/praat-script-syllable-nuclei-v2file.praat) | [v3 / Vocal Toolkit](https://www.praatvocaltoolkit.com/syllable-nuclei-v3.html) | [de Jong, Pacilly & Heeren 2021](https://www.tandfonline.com/doi/abs/10.1080/0969594X.2021.1951162)
- [Mermelstein 1975, JASA](https://pubs.aip.org/asa/jasa/article/58/4/880/677696/Automatic-segmentation-of-speech-into-syllabic) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/1194547/)
- [Wang & Narayanan 2007, IEEE TASLP](https://dl.acm.org/doi/abs/10.1109/TASL.2007.905178) | [SAIL page](https://sail.usc.edu/publications/html/b2hd-Wang2007Robustspeechrateestimation.html)
- [Tilsen & Johnson 2008, JASA-EL](https://pubs.aip.org/asa/jasa/article/124/2/EL34/841615/Low-frequency-Fourier-analysis-of-speech-rhythm) | [Tilsen & Arvaniti 2013, JASA](https://pubs.aip.org/asa/jasa/article/134/1/628/614239/Speech-rhythm-analysis-with-decomposition-of-the)
- [Ding et al. 2017, Neurosci Biobehav Rev](https://www.sciencedirect.com/science/article/abs/pii/S0149763416305668) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/28212857/)
- [Räsänen, Doyle & Frank 2018, Cognition](https://www.sciencedirect.com/science/article/abs/pii/S0010027717302901) | [thetaOscillator code](https://github.com/orasanen/thetaOscillator)
- [SylNet, Seshadri & Räsänen 2019, IEEE SPL](https://researchportal.tuni.fi/en/publications/sylnet-an-adaptable-end-to-end-syllable-count-estimator-for-speec/) | [code](https://github.com/shreyas253/SylNet) | [Sylber, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/37e0287d95305b355043250262fb2f92-Paper-Conference.pdf) | [SD-HuBERT](https://arxiv.org/html/2310.10803)
- [Oganian & Chang 2019, Science Advances (peakRate)](https://www.science.org/doi/10.1126/sciadv.aay6279) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/31976369/)
- [MacIntyre, Cai & Scott 2022, JASA "Pushing the envelope"](https://pubs.aip.org/asa/jasa/article/151/3/2002/2838371/Pushing-the-envelope-Evaluating-speech-rhythm-with) | [UCL Discovery PDF](https://discovery.ucl.ac.uk/id/eprint/10147582/1/10.0009844.pdf) | [MATLAB toolbox](https://www.mathworks.com/matlabcentral/fileexchange/95628-speech-envelope-and-acoustic-landmarks)
- [Zhang, Zou & Ding 2023, Neurosci Biobehav Rev](https://www.sciencedirect.com/science/article/pii/S0149763423000805) | [arXiv](https://arxiv.org/abs/2301.05898)
- [Morton, Marcus & Frankish 1976, Psychological Review](https://philpapers.org/rec/MORPC-6) | [Semantic Scholar](https://www.semanticscholar.org/paper/Perceptual-centers-(P-centers).-Morton-Marcus/0394a204fabbc362736534db149d7adfc71a9d9c)
- [Marcus 1981, Perception & Psychophysics](https://link.springer.com/article/10.3758/BF03214280) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/7322800/)
- [Scott 1998, "The point of P-centres," Psychological Research](https://link.springer.com/article/10.1007/PL00008162)
- [Harsin 1997, Perception & Psychophysics](https://link.springer.com/article/10.3758/BF03211892)
- [Pompino-Marschall 1989 / P-center reviews (Villing, "Perceptual centers (P-centers)")](https://www.researchgate.net/publication/232497075_Perceptual_centers_P-centers)
- [Villing 2010 PhD thesis, MURAL](https://mural.maynoothuniversity.ie/2284) | [Villing et al. 2011, phase correction response](https://link.springer.com/article/10.3758/s13414-011-0110-1)
- [Rathcke, Smit, Lin & Kubozono 2024, JASA](https://pubs.aip.org/asa/jasa/article/155/4/2698/3283278/Testing-an-acoustic-model-of-the-P-center-in) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38639561/)
- [Lidji et al. 2011, Psychonomic Bulletin & Review](https://link.springer.com/article/10.3758/s13423-011-0163-0) | [Rathcke et al. 2021, Laboratory Phonology (PDF)](https://dallabella-lab.ca/wp-content/uploads/2022/12/Rathcke-et-al_2021a_Tapping-into-linguistic-rhythm.pdf) | [Franich-lab 2023, J. Phonetics](https://www.sciencedirect.com/science/article/abs/pii/S0095447023000529) | [Comm Biol 2025 Matters Arising](https://www.nature.com/articles/s42003-025-07544-8) | [Assaneo reply](https://www.nature.com/articles/s42003-025-07546-6)
- [Cummins & Port 1996 ICSLP (PDF)](https://www.isca-archive.org/icslp_1996/cummins96_icslp.pdf) | [Cummins & Port 1998 overview](https://www.semanticscholar.org/paper/Rhythmic-constraints-on-stress-timing-in-Cummins-Port/4eac7ce125631594c45210a5762fb295b5b9952e) | [Cummins 2009 synchronous speech](https://www.researchgate.net/publication/222817318_Rhythm_as_entrainment_The_case_of_synchronous_speech)
- [Leong & Goswami 2014, Front Hum Neurosci](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2014.00096/full) | [Leong & Goswami 2015, PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0144411)
- [Condit-Schultz, MCFlow](https://www.researchgate.net/publication/312406564_MCFlow_A_Digital_Corpus_of_Rap_Transcriptions) | [Ohriner, MTO 2019](https://mtosmt.org/issues/mto.19.25.1/mto.19.25.1.ohriner.html)
- [Toh, Zhang & Wang, ISMIR 2008](https://www.researchgate.net/publication/220722898_Multiple-Feature_Fusion_Based_Onset_Detection_for_Solo_Singing_Voice) | [Böck & Widmer 2013, SuperFlux (DAFx PDF)](https://www.dafx.de/paper-archive/2013/papers/09.dafx2013_submission_12.pdf) | [SuperFlux reference code](https://github.com/CPJKU/SuperFlux) | [librosa Superflux example](https://librosa.org/doc/main/auto_examples/plot_superflux.html) | [Schlüter & Böck 2014, ICASSP](https://www.semanticscholar.org/paper/Improved-musical-onset-detection-with-Convolutional-Schl%C3%BCter-B%C3%B6ck/d19bb0ba1c4cf6f9e0ab904eee371d04201c9657) | [Gong, Pons & Serra 2017](https://arxiv.org/pdf/1707.03544) | [Gong & Serra 2018, Interspeech](https://www.isca-archive.org/interspeech_2018/gong18_interspeech.html)
- [Bello et al. 2005 onset tutorial (via FMP notebooks)](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S1_OnsetDetection.html) | [MIREX Audio Onset Detection](https://www.music-ir.org/mirex/wiki/2018:Audio_Onset_Detection)
- [Chronset, Behav Res Methods 2016](https://link.springer.com/article/10.3758/s13428-016-0830-1)

*Sourcing caveat: the research environment's egress proxy blocked most
publisher PDFs (Springer, AIP, arXiv, PMC, ISCA), so per-paper details come
from search-layer summaries of those pages plus directly fetched sources
(GitHub code for SuperFlux defaults and the Praat v2 script defaults). The
Marcus 0.65/0.25 coefficients are flagged in-text as reproduced from
secondary literature rather than verified against the 1981 primary PDF; all
other parameter values (Praat −25 dB / 2 dB / 30–450 Hz; SuperFlux 200 fps /
24 bands / max 3 bins / threshold 1.1 / pre_avg 0.15 s / combine 30 ms) were
read from the actual script/code.*
