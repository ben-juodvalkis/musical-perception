# Review 6 · How a pulse is recovered when the events don't sit on it

**Written 2026-09-02**, at the owner's request, after he described a second
mode of hearing: *"sometimes I hear a rate, but sometimes it's more like
some sporadic beats, and I have to reconstruct the underlying pulse, like
if it's really syncopated or something"*
([pulse-next-step.md](pulse-next-step.md) §4).

Companion to [Review 3](review-3-beat-meter-models.md), which already
covers Povel–Essens, Longuet-Higgins & Lee, Large & Jones, the bar-pointer
family and the modern learned trackers. **This review does not repeat
those.** It covers the two things Review 3 does not: the perceptual
evidence that humans recover a pulse with *no acoustic energy at the pulse
frequency*, and which existing algorithms actually do that.

---

## 1. First, a distinction this project must not blur

The literature word "syncopation" means **events land off the metrical
grid** — a note on a weak position, silence on a strong one
(Longuet-Higgins & Lee 1984's definition). Our failing demos are mostly
**not** that, and importing syncopation machinery for them would be
solving the wrong problem. Three different failures, currently
indistinguishable in our metrics:

| failure | what it looks like | our clips | is it syncopation? |
|---|---|---|---|
| **clutter** | 2–4 events per beat, all roughly on or around the grid, count varying per beat | all 8 demos (1.9×–4.1×) | **no** |
| **sparse sampling** | she voices only *some* beats (1 and 3 of a 3/4 bar) — events are ON the grid, just not on every beat | plié, rond-de-jambe, tendu-warmup | **no** |
| **true syncopation** | events consistently *between* beats; strong beats silent | not established on this corpus | yes |
| **non-stationarity** | the tempo genuinely moves | frappé (139→132→**165**), fondu | orthogonal |

The arithmetic matters. A 1-and-3-of-3/4 pattern has events at 0, 2, 3, 5,
6… beat-units, i.e. a period-3 pattern; its spectrum carries the beat
frequency as a harmonic. **There is energy at the beat frequency** — a
median of consecutive gaps just cannot see it, because the gaps alternate
2 and 1. That is a *different* problem from the missing-pulse case in §2,
and it is cheaper to fix.

**Recommendation, stated up front: our known failures are clutter and
sparse sampling, not syncopation. The syncopation literature is the right
place to look for the *shape* of the machine (§3, §6), and the wrong place
to look for a specific fix to the clips we currently fail.**

## 2. The perceptual evidence: the pulse is not in the signal

**2.1 Tal, Large, Rabinovitch, Wei, Schroeder, Poeppel & Zion Golumbic
(2017), "Neural Entrainment to the Beat: The 'Missing-Pulse' Phenomenon,"
J. Neuroscience 37(26):6331–6341.** MEG while listeners heard 30-second
syncopated drum sequences *constructed to contain no net energy at the
pulse frequency under linear analysis*. Auditory cortex nonetheless shows
enhanced response **at the pulse frequency**. The beat is manufactured,
not transduced.

**2.2 Nozaradan, Peretz, Missal & Mouraux (2011/2012), "Selective Neuronal
Entrainment to the Beat and Meter Embedded in a Musical Rhythm,"
J. Neuroscience 32(49):17572–17581** (and the 2011 tagging paper).
Frequency-tagged EEG: steady-state responses are selectively enhanced at
beat and meter frequencies **even where acoustic energy at those
frequencies is not predominant**, for syncopated as well as unsyncopated
sequences. Caveat the field itself raised — **Rajendran & Schnupp / Novembre
& Iannetti's PNAS exchange (2019)** argues some of the tagging signal is
better read as overlapping event-related responses than as entrainment.
The phenomenon survives; the mechanism label is contested. Cite it as
"the beat frequency appears in the brain and not in the stimulus," not as
"an oscillator was observed."

**Consequence for this project, and it is a real one.** Standing Lesson 6
("silence is evidence — a hypothesis predicting a strong beat where the
teacher voiced nothing pays for it") is, read literally, the opposite of
what §2.1–2.2 describe. The lesson is right that unexplained silence is a
*cost*; it is wrong if implemented as a veto. **Proposed amendment for the
owner (not made): a hypothesis predicting a beat where nothing sounded
pays a cost that stronger explanation elsewhere can outweigh.**

## 3. Why nonlinear machinery is needed, and the numbers behind it

**3.1 Velasco & Large (2011), "Pulse Detection in Syncopated Rhythms Using
Neural Oscillators," ISMIR 2011.** The cleanest demonstration of the
mechanism, and it is small enough to reimplement. A network of **289
oscillators, log-spaced 0.25–16 Hz**, each a canonical Hopf-normal-form
oscillator with nonlinear stimulus coupling, run in a **critical regime
(α = 0, β₁ = −1, β₂ = −0.25, ε = 1)** poised between damped and
spontaneous oscillation. Stimuli: 16 rhythms (1 isochronous, 2 metrical,
3 clave, 10 purpose-built "missing pulse" patterns balancing four events
on strong and four on weak beats), all at 120 BPM so the pulse frequency
is 2 Hz.

Result: Fourier analysis confirms the pulse frequency is weak or absent in
the syncopated stimuli. A **linear** filter bank and the nonlinear network
*both* separate duple from triple meter — but **only the nonlinear network
resonates at the pulse frequency for the hardest rhythms.** The mechanism
is higher-order resonance: a nonlinear oscillator at frequency *f*
responds to harmonics, subharmonics and integer ratios of *f*, and to
combination frequencies when several are present. It **adds frequency
information** rather than merely transducing it — pattern completion, in
the frequency domain.

A second property worth noting for our non-stationary clips: in a
double-limit-cycle regime the oscillator **maintains oscillation after the
stimulus ceases** — a memory of the rate. That is exactly what should
carry a tempo across an explanation break, which is a thing our demos are
full of.

**Transfers: as a mechanism, strongly; as an adoption, not yet.** It is
the correct shape for the "reconstruction" mode, and it is the same family
Review 3 §2.7 (Large & Jones) already flagged. But it is a *far* bigger
change than the corpus can currently justify (8 demos), and §1 says our
present failures do not need it.

**3.2 Implementations that exist.** MATLAB **GrFNN Toolbox**
(MusicDynamicsLab) and a Python port, **pyGrFNN** (`jorgehatccrma/pyGrFNN`,
developed with Large's lab, with stated differences from the MATLAB
version). Both are research code, unmaintained-looking, and neither is a
beat tracker out of the box — they are oscillator-network simulators. **A
cheap first probe does not need either:** run our event train through a
bank of resonators and look at where energy appears, which is ~30 lines of
numpy and directly answers "is the beat frequency recoverable from this
stream at all."

## 4. What listeners do when it gets too syncopated: they re-hear it

**4.1 Fitch & Rosenfeld (2007), "Perception and Production of Syncopated
Rhythms," Music Perception 25(1):43–58.** Listeners tapped to, reproduced,
and later recognised syncopated patterns. The finding that matters here:
with highly syncopated rhythms participants **reset the phase of their
internally generated pulse — reinterpreting the rhythm as** ***less***
**syncopated.** Complexity also degraded memory: simpler patterns were
more robustly encoded after 24 hours.

**Read for this project:** the human solution to an unresolvable stream is
not a better estimate, it is **a switch to a different hypothesis**. Any
model of the owner's second mode needs *competing hypotheses with
switching*, not a single tracked estimate — which is an argument for the
posterior/lattice machinery already in `posterior.py` (ADR-017) over any
point estimator, and an argument that "the pipeline re-decides tempo a
median of 5 times per clip" (W13b) may be partly *correct behaviour*
rather than pure instability.

**4.2 Witek, Clarke, Wallentin, Kringelbach & Vuust (2014), "Syncopation,
Body-Movement and Pleasure in Groove Music," PLOS ONE 9(4):e94446;** and
**Vuust & Witek (2014), "Rhythmic complexity and predictive coding,"
Frontiers in Psychology 5:1111.** Inverted-U: *medium* syncopation
maximises the urge to move and pleasure, framed as a predictive-coding
balance between predictability and surprise. **Transfers as framing only,
not as method** — but it does say the interesting regime is the middle
one, and a teacher marking an exercise is squarely in it.

## 5. Which cues actually carry the pulse when the surface doesn't

**5.1 Snyder & Krumhansl (2001), "Tapping to Ragtime: Cues to Pulse
Finding," Music Perception 18(4):455–489.** Participants tapped a
comfortable pulse to ragtime (metronomically rendered — no expressive
timing). Two manipulations:

- **Flattening all pitches to a single note barely mattered.** Melodic and
  harmonic cues are largely dispensable for pulse finding.
- **Removing the left-hand part was severely damaging.** The regular,
  alternating bass — the part that *is* on the beat — carried it.

Their analysis names the operative cues: a predictable alternating bass
pattern, and a majority of onsets on metrically strong positions.

**This is the most directly transferable finding in the review, and it
argues against a fashionable move.** Pulse finding in a syncopated texture
worked because *some stream in the texture was regular*. Our demo has no
left hand. The teacher is the whole texture, and when she talks, the
regular stream disappears. **The implication is not a cleverer estimator
over one cluttered channel — it is to find or supply a second, regular
stream.** Candidates already in this project: her movement (deferred at
SW-1's commissioning, W7/W10 negative on head-nod but limb arrival
untested), and the count words themselves, which *label* which onsets are
beats when she is counting rather than describing.

## 6. Existing algorithms, with verdicts

| approach | does it recover a pulse with no energy at the pulse frequency? | verdict here |
|---|---|---|
| **Nonlinear resonance / GrFNN** (Velasco & Large 2011; Large 2009) | **Yes — this is the point of it** | right mechanism, too big a step for n=8; probe it before adopting |
| **Povel–Essens clock induction** (Review 3 §2.1) | Yes, by construction — scores candidate clocks by *counterevidence*, including silent strong beats | already in this repo's vocabulary (`accent_meter`); cheap to score as a candidate-ranker |
| **Bar-pointer HMM / DBN** (Krebs et al.; `posterior.py`, ADR-017) | Yes in principle — latent grid, events are observations, empty beats allowed | **already built here and not used for tempo**; the closest reachable thing |
| **Inner Metric Analysis** (Volk; Utrecht `monochord/ima` code) | Partly — finds coinciding periodicities across *all* onsets and yields metric-weight profiles; local-vs-global comparison quantifies syncopation | onsets in, weights out; a genuinely cheap all-pairs alternative to median-IOI |
| **Autocorrelation / comb / tempogram** (Klapuri; librosa) | **No** — linear analysis cannot invent absent frequency content (Velasco & Large's own control) | fine for the *clutter* case, useless for missing-pulse |
| **Median of consecutive IOIs** (what we ship) | No, and it also fails the clutter and sparse cases | the current defect |
| **Learned trackers** (madmom, BeatNet, Beat This!) | Empirically yes on music — trained through syncopation | Review 3 §3 and W3 already benchmarked them; note the 2025 failure-mode work below |

**6.1 Failure modes of the state of the art, freshly documented.** A 2025
analysis, *"The SMC Blind Spot: A Failure Mode Analysis of State-of-the-Art
Beat Tracking"* (arXiv 2605.12287), reports the standard failures we are
also seeing — offbeat patterns produce strong sub-beat autocorrelation that
**doubles the detected tempo**, and one documented case where Beat This!
"fires at double tempo with offset phase, placing peaks between GT beats."
It also notes perceptual weighting (a log-Gaussian prior centred near 120
BPM) as the standard octave-ambiguity fix — **which is exactly what W9
already installed here.** Worth reading as confirmation that our failure
class is the field's, not a local bug.

## 7. Syncopation measures — diagnostics, not trackers

Seven formal measures are in circulation: Longuet-Higgins & Lee (metric
weight of note-vs-rest violations), Pressing, Toussaint's metric complexity
and off-beatness, Keith, Sioros & Guedes, and Gómez et al.'s **Weighted
Note-to-Beat Distance** (syncopation inversely related to a note's distance
from the nearest beat, weighted up when it crosses the following beat).
Comparative work reports that **hierarchically dependent measures (the
Longuet-Higgins & Lee family, metric complexity) align best with
perceptual syncopation ratings**; **Fram (2023), "Syncopation as
Probabilistic Expectation," Cognitive Science 47(11):e13390**, recasts the
whole family as expectation violation rather than grid violation.

**All of these presuppose a known meter and grid** — they measure
syncopation *given* the beat. They cannot find the beat. **Their use here
is diagnostic only:** applied to the owner's tapped grids they would tell
us, per clip, whether we are actually in the syncopated regime at all —
which §1 says we have never established. That is a genuinely cheap
measurement and the honest first step before any of §3 is contemplated.

## 8. What to take from this, ranked

1. **Establish which regime we are in before importing machinery** (§1, §7).
   Our four failures look like clutter, sparse sampling and drift. Nothing
   in the corpus has been shown to be syncopated. One measurement settles it.
2. **All-pairs, not adjacent-pairs** (§6, Inner Metric Analysis). The
   single cheapest change consistent with everything above, and it fixes
   the sparse-sampling case (plié, rond-de-jambe) without any oscillator.
3. **Find a second, regular stream** (§5). Snyder & Krumhansl is the
   strongest transferable result in this review: syncopated pulse finding
   worked because the left hand was regular. Our demo has no left hand.
   This is an argument for the movement channel and for the count words —
   both already in the project, both currently unused for tempo.
4. **Competing hypotheses with switching, not a tracked point estimate**
   (§4.1). Supports using `posterior.py` for tempo rather than
   `calculate_tempo`, and reframes W13b's "re-decides 5 times" partly as
   correct behaviour.
5. **Amend Standing Lesson 6** (§2). Silence is a cost, not a veto.
6. **Nonlinear resonance last** (§3). Right mechanism, wrong moment: it is
   a large change, and at n = 8 demos it could not be honestly evaluated
   even if it worked.

**None of this displaces the standing next step**, which is the owner's
blind exercise→tempo prior table ([pulse-next-step.md](pulse-next-step.md)
§6). Nothing in this literature is cheaper than that or better aimed at the
9–12 % errors we actually have.

## 9. Sources

- [Tal et al. 2017, "Neural Entrainment to the Beat: The 'Missing-Pulse' Phenomenon," J. Neurosci](https://www.jneurosci.org/content/37/26/6331) · [PDF](https://musicdynamicslab.uconn.edu/wp-content/uploads/sites/433/2017/06/Tal_etal_2017.pdf) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/28559379/)
- [Nozaradan et al., "Selective Neuronal Entrainment to the Beat and Meter," J. Neurosci 32(49)](https://www.jneurosci.org/content/32/49/17572) · [the tagging-vs-ERP critique, PNAS](https://www.pnas.org/doi/10.1073/pnas.1815311115)
- [Velasco & Large 2011, "Pulse Detection in Syncopated Rhythms Using Neural Oscillators," ISMIR (PDF)](https://musicdynamicslab.uconn.edu/wp-content/uploads/sites/433/2016/03/VelascoLarge2011ISMIRPubsAHEdits.pdf)
- [GrFNN Toolbox (MATLAB)](https://musicdynamicslab.uconn.edu/home/multimedia/grfnn-toolbox/) · [pyGrFNN (Python)](https://github.com/jorgehatccrma/pyGrFNN)
- [Fitch & Rosenfeld 2007, "Perception and Production of Syncopated Rhythms," Music Perception (PDF)](https://web.uvic.ca/~aschloss/course_mat/MU320/Global%20Rhtyhm%20and%20Human%20Consciousness/ARTICLES%20AND%20REFS%20FOR%20320/FitchRosenfeld20071.pdf) · [journal](https://online.ucpress.edu/mp/article-abstract/25/1/43/95281/Perception-and-Production-of-Syncopated-Rhythms)
- [Snyder & Krumhansl 2001, "Tapping to Ragtime: Cues to Pulse Finding," Music Perception](https://online.ucpress.edu/mp/article-abstract/18/4/455/62091/Tapping-to-Ragtime-Cues-to-Pulse-Finding)
- [Witek et al. 2014, "Syncopation, Body-Movement and Pleasure in Groove Music," PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0094446) · [Vuust & Witek 2014, Frontiers in Psychology](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2014.01111/full)
- [Volk, "The Study of Syncopation using Inner Metric Analysis," JNMR (PDF)](https://webspace.science.uu.nl/~veltk101/publications/art/JNMR08.pdf) · [IMA code, Utrecht](https://www.projects.science.uu.nl/monochord/ima/)
- [Gómez et al., "Mathematical Measures of Syncopation"](https://www.semanticscholar.org/paper/Mathematical-Measures-of-Syncopation-G%C3%B3mez-Melvin/9ad4ac3c73bcf0264a39ffd97ed7794dcb4ab4b6) · [Fram 2023, "Syncopation as Probabilistic Expectation," Cognitive Science](https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.13390)
- [Large 2009, "Pulse and Meter as Neural Resonance," Ann. NY Acad. Sci](https://nyaspubs.onlinelibrary.wiley.com/doi/abs/10.1111/j.1749-6632.2009.04550.x)
- ["The SMC Blind Spot: A Failure Mode Analysis of State-of-the-Art Beat Tracking" (2025)](https://arxiv.org/pdf/2605.12287)
