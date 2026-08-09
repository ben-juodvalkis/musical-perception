# Voice-as-Drum: what the literature already knows

**Date:** 2026-08-09
**Status:** Research note — companion to [ADR-016](../adr/016-rhythm-core-reset.md).
Four-track survey of the beat-tracking, meter-induction, speech-rhythm, and
evaluation literature, run the day ADR-016 was accepted, to check the reset's
six commitments against prior art before any implementation starts.

The four full reviews (dense, cited, with implementation recipes):

1. [Onset/event detection for voice + P-centers](review-1-onsets-pcenters.md)
   — where the perceptual beat sits in a syllable, and how to detect it.
2. [Tempo, periodicity, and the octave problem](review-2-tempo-periodicity.md)
   — periodicity representations, the perceptual tempo prior, expressive
   timing, and the evaluation metrics to adopt.
3. [Probabilistic beat/meter models](review-3-beat-meter-models.md)
   — the bar-pointer family, meter induction, PIPPET, and the concrete
   model recommendation for the joint posterior.
4. [Tools, baselines, analogs, datasets](review-4-tools-baselines.md)
   — what to run on the 30 clips as baselines, closest published tasks,
   transfer datasets, tool-rot warnings.

## The headline

**The reset's architecture has 20 years of prior art, and the niche is
empty.** Three independent traditions converge on ADR-016's design: MIR's
bar-pointer models (2006–2015) track exactly the committed latent state
(position-in-bar, tempo, meter) with exact sub-second inference at our clip
lengths and a decade of published evidence on meter *discrimination*; music
psychology's clock models (Povel & Essens 1985) score candidate meters by
accent fit — including treating silence at an expected strong beat as
evidence; and computational neuroscience's PIPPET (2021) is essentially a
design document for the observation layer (salience-weighted sparse events,
a background rate for interleaved talk, informative absences). Temperley's
*Music and Probability* (2007) is a complete worked example of the whole
thing. Meanwhile the closest published task is beat tracking on isolated
*singing* voice (2022–2024): it confirms music-trained trackers fail on bare
voice, that speech-model features work better — and that **meter inference
on rhythmic speech has never been published, and no public corpus has beat
annotations on rhythmic speech.** The 30 annotated clips would be a
first-of-kind evaluation set; the P2 (dataset-as-contribution) posture in
[Vision 10](../vision/10-pivots.md) is strengthened accordingly.

## Five findings that change the plan immediately

**1. The "drum hit" of a voice has a precise, validated definition
(peakRate).** Perception (P-centers, 1976→2024), production (speech
cycling), tapping studies, and cortical electrophysiology (Oganian & Chang
2019) converge: the perceived beat of a syllable is the moment of fastest
loudness rise into the vowel — not the word's acoustic start. Detection is
~20 lines of scipy (envelope → derivative → voiced-gated peaks; full recipe
in [Review 1](review-1-onsets-pcenters.md)). Corollary trap: ASR word
timestamps run 0–150 ms early with *word-dependent* bias — "one" and "and,"
the two most frequent counting tokens, are the worst cases — so the bias
aliases into tempo drift and subdivision error rather than averaging out.
Ground-truth annotation must target vowel onsets too.

**2. Mean-IOI failed for a reason the field named ~2000–2005, and the
replacements are tiny.** Mixed 1×/2×/3× onsets must *vote, not average*:
harmonic-summed point-process periodogram on the marker times (~15 lines of
numpy; the same math astronomers use for pulsars in sparse photon data),
ACF × Fourier tempogram product to cancel both octave biases, Dixon's IOI
clustering with integer-ratio reinforcement. Recipes in
[Review 2](review-2-tempo-periodicity.md) §(a).

**3. The 70–140 band has a principled, parameterized replacement.** A
log-Gaussian prior over log-tempo (T₀ ≈ 100–110 BPM, σ ≈ 1.2–1.4 octaves per
the resonance literature), applied **only when selecting the reported level
within the family — never folding the raw measurement**. At 40 BPM the
prior weight is ≈0.6 instead of the current 0. Exercise-conditioned priors
are a published pattern (style priors fixing octave errors), and human tempo
ground truth is *genuinely multimodal* (40 tappers disagree on the level) —
two-tempi-plus-salience output is the field's standard, validating ADR-014.

**4. Expressive lengthening is structure, not noise.** Phrase-final
lengthening is the dominant timing pattern in performance (Repp), is
perceptually *expected* (listeners can't hear it as deviation), and speech
does the same (Wightman: pre-boundary rime lengthening scales with boundary
depth). Fix: censor/down-weight the final interval(s) before a pause;
robust Theil–Sen grid regression. This is clips 4 and 24, explained.

**5. The evaluation upgrade is off-the-shelf and sharpens the gates.**
`truth_in_family` is the field's **Accuracy-2** metric (±4%, family
{⅓, ½, 1, 2, 3}) — adopt the standard name for comparability. **OE2**
(octave error after removing the best family factor) is the only standard
metric that directly measures "landed between levels," and as a continuous
quantity it sharpens the tier-1 gate beyond binary flips. Two source-code
traps: mir_eval's allowed-metrical-levels beat scoring *omits triple/third*
(use madmom's evaluator for waltz-heavy material), and its `trim_beats`
default discards the first 5 s of each clip. And a closure: trained tappers'
variability is 3–5%, so eval disagreements below ~4% are noise by
construction — ADR-015's knife-edge finding, rediscovered as psychophysics.

## Two further findings

**Run the baselines on the right signal.** Speech's envelope pulses at
syllable rate (~4–5 Hz) while marking beats live at 0.7–3.3 Hz, so music
tools on raw audio lock to syllables. The benchmark plan
([Review 4](review-4-tools-baselines.md)) runs six tools in a fixed order —
librosa suite, Beat This!, madmom (with `min_bpm=40`, since the default 55
silently octave-doubles slow clips), Essentia, the syllable-nuclei hybrid,
optionally BeatNet — on raw audio *and* on marker streams. Compute is
minutes; the cost is 2–4 h of tap annotation, which is reset step 1 anyway.

**The two tempo pipelines fuse naturally in one published model.** Cemgil &
Kappen (2003) solved "jointly infer a drifting tempo and each onset's grid
slot from sparse expressive onsets" — this project's problem — and the hard
part in their blind setting was the unknown grid assignment. Gemini's
beat/and/ah labels are noisy *observations* of exactly that assignment: the
marker path and the acoustic path become two evidence types in one model
rather than rivals to arbitrate.

## Map to the reset plan

| ADR-016 step | What the literature hands us | Where |
|---|---|---|
| 1 · Annotate + stage scoring | Annotate vowel onsets (P-centers), not word starts; Acc2/OE2/AMLt-with-triples metrics; signed asynchrony; two-tempi+salience case format | Reviews 1 §2.9, 2 §4 |
| 2 · Acoustic extractor | peakRate recipe + syllable-nuclei gate + known traps (12 of them) | Review 1 §Steal/Traps |
| 3 · Accent-periodicity meter | S-AMPH delta/theta phase reader; Povel–Essens/Parncutt salience clocks; harmonic phase fractions (Cummins & Port) | Reviews 1 §3, 3 §2 |
| 4 · Joint posterior | Krebs-2015 state space + Whiteley/PIPPET observation model; ~30k states, exact forward–backward; ranked 5-paper reading list | Review 3 §(a)–(c) |
| 5 · Ensembled semantics | Marker labels as observed grid-assignment switches (Cemgil & Kappen) | Reviews 2 §3.3, 3 §1.5 |
| 6 · Corpus | First-of-kind dataset claim; SMC/ChoirSet/beatboxset1 as transfer sanity; difficulty-aware curation practice | Review 4 §(b)–(c) |
| Baselines (step-2 kill-test context) | Six-tool benchmark plan with exact entry points, params, and rot warnings | Review 4 §(a), (d) |

## The meta-lesson

The project independently reinvented at least four published results: the
metric-level family (ADR-014 = McKinney & Moelants' multimodal tempo
perception), Accuracy-2 scoring (`truth_in_family`), grid-fitting (ADR-015 ≈
tatum methods, with the same documented failure modes), and the knife-edge
noise floor (ADR-015 finding 3 = tapping-variability psychophysics).
Reassuring — the trail was real — and the sharpest possible argument for
*survey before building*. This review would have cost the same in February
and saved three ADRs.
