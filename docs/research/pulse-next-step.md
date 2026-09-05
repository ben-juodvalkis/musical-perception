# Step one, the pulse — where it stands and what happens next

**Written 2026-09-02 in an owner-attended session, after the SW-1 sweep.**
This is the handoff document: a fresh agent thread should read this file,
the charter's CURRENT RUNG block, and the Standing Lessons, and it will
have the picture without re-reading the whole ledger.

**The next step is an owner action, described in §6.**

**Updated 2026-09-02 after EB-1 and Review 6** — §3 and §5 changed
materially: the beat/syllable diagnosis is now *measured* rather than
inferred, a concrete estimator fix is on the table, and the syncopation
hypothesis is closed. §8's do-not list grew.

---

## 1. The rung, restated

Step one is the **pulse only**: the tempo of the beat you would tap, at
the metric level inside 70–140 BPM, on the **demo** — the teacher
demonstrating, before anyone plays. Meter, structure and style are steps
two to four and gate nothing here.

Scored on the 34-row gating set: 26 rig/counting clips in the owner's own
voice, plus 8 barre-6 demos with owner-tapped grids. Pass = committed
pulse within ±8 % of the in-band truth. Blessed baseline: **20 of 34**.

**The accent-periodicity line is HELD until step two** (charter amendment
proposed 2026-09-02). Do not tap grids for it, re-run it, or build the
chunked WhiStress adapter PR-1 parked.

## 2. What SW-1 established, and what it did not

The sweep asked whether reading one steady window beats reading the whole
clip. Full results: [sw1-steady-window-sweep.md](sw1-steady-window-sweep.md).

- **On the owner's own recordings it works, and well.** peakRate events
  in the steadiest 5-second window: **21 of 26**, against **12 of 26**
  for the whole clip. Nine clips. This is the largest single effect
  measured.
- **On the demos it does nothing.** Every window variant ties or loses to
  simply reading the whole clip (4/8 either way). The demos are step
  one's entire target.
- **F3's conclusion is WITHDRAWN.** The sweep reported that the owner's
  "Intended-tempo span" notes scored worse than the algorithm's windows
  and concluded he was not reading audio regularity. That arm ran his
  span through the same broken event stream and median rule as everything
  else, so it never tested his spans. Read with **his own taps**, his
  spans recover the label within 3 % on three of the five clips a naive
  probe can read. See the ledger correction of 2026-09-02.

### EB-1: the estimator, and it is the biggest lever found so far

[eb1-estimator-bakeoff.md](eb1-estimator-bakeoff.md). Five estimators on
one fixed peakRate event stream, so only the arithmetic varies:

| estimator | pass /34 | demo /8 | rig /26 | between-levels |
|---|---|---|---|---|
| `median-consec` *(ships today)* | 16 | 4 | 12 | 21 |
| **`all-pairs`** | **28** | 4 | **24** | **7** |
| **`comb`** | **28** | **5** | 23 | 8 |
| `povel-essens` | 27 | 5 | 22 | 8 |
| `hopf` (nonlinear resonance) | 19 | 2 | 17 | 20 |

Blessed pipeline on the same rows: 20. **Changing only the arithmetic is
worth 12 rows**, and nearly all of it is rig-side (12 → 24 of 26); the
demos move 4 → 5. `all-pairs` and `comb` are not separable at n = 34.

Also measured: `librosa_plp`, off the shelf on raw audio, scores **5/8 on
the demos** — equal to the best thing we do on our own event stream.
`beat_this` returned no usable beats on 5 of 8.

## 3. The diagnosis, now measured rather than inferred

This is the finding the session converged on, and it is not subtle.

`calculate_tempo` takes the gaps between **consecutive** events and
returns the median. `normalize_tempo` then multiplies or divides that by
2 or 3 and picks a level under a log-normal prior. **The second step is a
level chooser, not a separator** — it assumes the beat is already an exact
integer multiple of what was measured.

It is not. Across the 8 demos, peakRate produces these ratios of events
to owner-tapped beats:

`1.9 · 2.2 · 2.5 · 2.5 · 2.5 · 2.7 · 3.0 · 4.1`

Non-integer, and not constant inside a clip — the teacher puts one
syllable on some beats and three on others. There is no factor that
recovers the beat. That is exactly why **all four demo misses land 9–12 %
off and not one is a clean double or half**.

Specific own-goal worth naming: **only adjacent gaps are examined.** When
syllables sit between beats every adjacent gap is corrupted, but the
beat-to-beat distance survives in the set of *all pairwise* distances.
The methods Standing Lesson 3 already names — harmonic summing,
ratio-reinforced IOI clustering, comb/grid scoring — all work on all-pairs
or on testing candidate periods against the whole event train, and none
of them care how many syllables sit between two beats.

**EB-1 Arm C settled which regime we are in** — the measurement Review 6
§1 said had never been run. Dominant periodicity in the event stream,
divided by the true beat:

| demo | ratio | reading |
|---|---|---|
| coupé-barre · dégagé · frappé · plié · tendu | 2.00 · 2.13 · 2.10 · 2.50 · 2.85 | **clutter** — the syllable rate wins, at a *non-integer* multiple, so no ×/÷{2,3} recovers the beat |
| fondu | 0.54 | ≈ a 2-beat group |
| rond-de-jambe · tendu-warmup | 0.35 · 0.35 | **the 3/4 bar** — voiced 1-and-3, so the bar is the strongest rhythm and ×3 is exactly right |

**On 0 of 8 demos is the beat the dominant periodicity** (it sits 7.6–29.4
dB below the peak). **And this is NOT a missing-pulse corpus:** if it were,
the nonlinear oscillator would recover what linear methods cannot. It came
last. The syncopation hypothesis is closed — see §8.

Supporting measurements (all on the 8 demos, owner taps as truth):

- **1,100 machine events against 419 taps — 2.6×.** On the owner's own
  recordings it is **1.3×**, because when he counts, one syllable is one
  beat. The detector is not worse on her; there is more to hear.
- **peakRate fires a median −69 ms early**, per-clip range −7 to −95 ms.
  **This cannot move a tempo number** — a constant offset cancels out of
  every interval (shifting the taps 0/69/150 ms returns identical BPM).
  Do not build an offset correction for tempo's sake. It will matter when
  phase matters — where to place a note — and not before.
- The beat-to-beat scatter (MAD 40–109 ms) is **not** evidence about the
  owner's tapping: at 2.6 events per beat the "nearest event" to a tap is
  often a different syllable. Do not cite it as annotation quality.

## 4. The owner's introspection: there are two modes, not one

Asked whether he hears a rate and locks to it, or feels a few beats and
derives the rate, he rejected the dichotomy:

> *"sometimes I hear a rate, but sometimes it's more like some sporadic
> beats, and I have to reconstruct the underlying pulse, like if it's
> really syncopated or something"*

These are two different machines:

- **Entrainment.** Dense events, mostly on the beat; find the period that
  best predicts them. Autocorrelation / comb scoring is the right tool.
- **Reconstruction.** Sparse or syncopated events, off the grid; **the
  beat can be at a moment where nothing sounds.** No period-fitting method
  can place a beat where there is no event. This needs latent-grid
  inference — what best *explains* these events — which is precisely the
  shape of the bar-pointer posterior already in the repo (`posterior.py`,
  ADR-017), and not the shape of a median of gaps.

The four failing demos split along exactly those two modes:

| clip | why it fails | mode |
|---|---|---|
| frappé | ~~tempo genuinely moves: 139 → 132 → **165** across the clip~~ **CORRECTED 2026-09-05 (W17)** — measured against the owner's tap grid the clip sits at **132.3 BPM throughout** (beat-cluster gaps, n=24, constant by region). There is **one 2.3s faster run at 20.1–22.3s** during the *petit battement* passage — 7 gaps, all 0.363s — at **exactly 5:4** to the surrounding tempo (0.0% off 5:4; 2:1 off by 37%, 3:2 by 17%). Owner's ruling: the teacher speeds up there. Caveat logged: 25% is larger than the "a little bit" he described, so "those taps track the movement, not the pulse" is not excluded. **Not a drift, and not across the clip.** | brief non-stationarity, not entrainment |
| fondu | moves 91 → 83 → 85 | entrainment, non-stationary |
| plié | she voices only beats 1 and 3 of a 3/4 bar | reconstruction |
| rond-de-jambe | same sparse voicing | reconstruction |

Honest counterexample: **tendu-warmup is also voiced-1-and-3 and passes
anyway.** Four clips is four clips.

**Tension flagged for the owner (not resolved):** Standing Lesson 6 says a
hypothesis predicting a strong beat where nothing was voiced *pays for
it*. Read literally that penalises the syncopated reading above, where the
empty strong beat is the point. It needs to be a cost that better
explanation can outweigh, not a veto.

**Consequence for frappé and fondu:** if the tempo moves 26 BPM across a
clip, that clip does not have "a tempo," and its single truth label is a
summary of a moving target. **Only the owner can rule** what an
accompanist should commit to — the tempo she starts at, the one she
settles into, or the one at the moment he would have to start playing.
This is an open question, not an agent's to decide.

## 5. The lever nobody has pulled: the domain knowledge is already in the traces

Asked how he resolves the ambiguity, the owner said he uses domain
knowledge to narrow it. **That channel already exists in every frozen
trace and is wired to nothing.** Scored on the 8 demos:

| what Gemini is asked | how it does |
|---|---|
| which exercise is this | **6 of 8**, at 0.9–1.0 confidence |
| what meter | **6 of 8** |
| what tempo | **2 of 8** |

Its two exercise misses are near-neighbours (coupé-barre called a jeté,
dégagé called a tendu). Its tempo guesses are useless — 200 BPM against a
truth of 108, 69 against 102, 68 against 96.

So: **a channel that reliably knows what the exercise is, a separate
channel that measures rate but cannot tell beats from syllables, and
nothing connecting them.** The move is not to ask the model for the
tempo. It is to let the exercise label pick the *prior* and let the
acoustic measurement do the measuring inside it. The misses are 9–12 %
off; a prior saying a rond de jambe is a waltz in the 90s resolves most of
them without touching the front end.

The charter's rung 4 already anticipated this — "exercise-conditioned
priors at level selection only" — and it was never built.

**EB-1 strengthened this, and named the clips it would fix.**
Rond-de-jambe and tendu-warmup are the two demos whose strongest rhythm is
the bar at ⅓ the beat rate: on those, ×3 is exactly the right projection
and only knowledge of the exercise supplies it. On the other five the
dominant rate is a *non-integer* multiple of the beat, which no projection
can fix and only a prior over plausible tempos can break.

## 6. THE OWNER'S PRIOR TABLE — filled 2026-09-05 (attended session)

**Status: ANSWERED by the owner, 2026-09-05.** Written from professional
knowledge in an owner-attended session, dictated by the owner and typed by
the agent. The original ask and its blind condition are preserved below
the table.

**Blindness caveat, disclosed rather than assumed away (rule 7).** The
blind condition was **partially broken before the table was written**, by
the agent and not by the owner. Earlier in the same session, while
presenting PP-1 for blessing, the agent displayed the per-clip truth
tempos for all eight gating demos — `coupe-barre 108, degage 110,
fondu 86, frappe 135, plie 120, rond-de-jambe 96, tendu 102,
tendu-warmup 112`. Eight of the twelve rows below therefore had one
labelled example each visible to the owner beforehand. Four rows were
never exposed and are marked **(blind)**: rond de jambe en l'air, petit
battement, adage/développé, grand battement.

**Evidence the exposure did not drive the answers, measured after the
fact.** Two of the six callable exposed rows have ranges that **exclude**
the number the owner had seen — `degage` was given 85–105 against a shown
110, and `fondu` "around 100" against a shown 86. A table fitted to the
displayed key would contain them. The exposure is recorded as a real
weakening of the blind condition, not as one that visibly bit.

| exercise | typical tempo range (BPM, at the tapped beat) | usual meter | notes |
|---|---|---|---|
| plié | 85–125 | 3/4 | |
| battement tendu | 73–120 | 4/4 or 3/4 | |
| dégagé / jeté / glissé | 85–105 | 3/4 or 4/4 | |
| rond de jambe à terre | 85–120 | 3/4 | |
| fondu | ~100 | 4/4 (usually) | given as a point, not a range |
| frappé | 120–140 | 4/4 usually, "but not always" | |
| rond de jambe en l'air **(blind)** | — | — | **"varies too much"** — declined, and that is the answer |
| petit battement **(blind)** | 130–150 | 4/4 | |
| adage / développé **(blind)** | 85–125 | 3/4 | |
| grand battement **(blind)** | 120–130 | 3/4 or 4/4 | |
| coupé-barre | — | — | **"I don't know"** — declined |
| tendu warm-up | — | — | **"I don't know"** — declined |

Three of twelve rows are declined: one as genuinely too variable to call
(rond de jambe en l'air), two as outside the owner's confident knowledge
(coupé-barre, tendu warm-up). Declines are data and are not to be filled
in by an agent later.

### Q1 — what counts as "the tempo" when the teacher's tempo moves

Owner's answer, verbatim:

> "usually the first clear tempo is the right one, so once the teacher
> does say 4 counts with full steady consistent speed, especially if they
> are actually doing the movement fullout and not just marking."

This is **not** "the tempo they start at." It is an operational rule with
three conjunctive conditions, and each is separately detectable:

1. **first** — earliest qualifying window wins; later drift does not
   revise it;
2. **clear and steady** — roughly four counts held at consistent speed,
   not one or two isolated counts;
3. **full-out, not marking** — the tempo is read from the demonstration
   proper, and marking is explicitly excluded.

**A fourth channel the rule does not mention, found by the W17 frappé pass
(2026-09-05): the spoken count-in.** On `barre6-frappe-demo` the teacher
*states* the tempo — "seven… eight", 0.94s per count = **127.7 BPM** read as
counting in 2s, against a danced 132.3 — **before any dancing**, and every
estimator that commits early is reading it (all-pairs returns 141.2 BPM from
the six acoustic onsets before 2.0s). A count-in is neither full-out nor
marking, so condition 3 does not classify it. The annotation vocabulary gained
`countin`; **whether a count-in is an admissible source of truth under Q1 is
an open owner question.**

**Condition 3 does not bear on which field is truth.** This session first
read it as a collision with `marking_bpm` and was wrong; "marking" means
the teacher setting the combination there, and sketching-rather-than-
dancing here. Q1 governs *how a tempo is read off a clip*, not *which
recorded quantity the benchmark grades against* — see §6.1.

### Q2 — does the exercise name alone carry the prior

Owner's answer, verbatim:

> "exercise name is never enough. it always needs to be corraborated with
> the observation"

**This is a negative result on the shape of §7's ablation as originally
written, and it is the owner's own ruling, not an agent's inference.** The
exercise label is not a sufficient prior at any strength; it is admissible
only as one input to be corroborated against what is actually heard or
seen in the clip. An exercise-keyed prior applied on the name alone is
therefore ruled out as a design, independent of what it might score.

The table's own numbers corroborate the ruling arithmetically (measured,
not eyeballed — an earlier draft of this paragraph overstated it and is
corrected here per rule 7). Across the **nine callable rows**:

- **26 of the 36 exercise pairs (72%) have overlapping ranges** — for most
  pairs the table cannot separate the two exercises at all;
- **100 BPM falls inside 6 of the 9 ranges** (plié, tendu, dégagé, rond de
  jambe à terre, fondu, adage) — a single tempo consistent with two thirds
  of the barre;
- **only `petit battement` (130–150) fails to touch the crowded 85–125
  mass**, making it the one row that is genuinely discriminative on its
  own.

As a discriminator between exercises the table is close to uninformative;
as a measurement of how little the exercise name carries by itself, it is
decisive and independently confirms Q2.

### §6.1 — CLOSED: the demo scoring is correct; nine case notes are stale

**Opened and closed in the same session (2026-09-05). Recorded in full
because the agent's first reading was wrong and the wrong reading was
briefly committed.**

The agent measured that all nine barre-6 demo cases carry `marking_bpm`
inside `expect:` and no `performance_bpm` there, while each case's `notes:`
block asserts *"expected_bpm prefers performance_bpm, so this row grades
against what was played, not against the marking."* The agent wrote this up
as three open readings requiring an owner ruling. **That was wrong: the
ruling already existed and the agent had not read it.**

**The standing ruling (owner, 2026-09-01 ledger entry), which governs:**

> "The target is not 'what the pianist played' — the accompanist has
> latitude, and a different valid realization is not an error. The target
> is what the marking specifies."

and, restated by the owner on 2026-09-05 when the question was wrongly
re-raised:

> "we are not supposed to use the pianist tempo … i have already gone
> through each demo clip and recorded the tempo that should be played"

So: `marking_bpm` holds **the owner's own per-clip determination of the
tempo that should be played**, arrived at by watching each demo. It is
correctly named, correctly populated, and correctly scored.
`performance_bpm` was **deliberately removed** from the demo cases' graded
block as the implementation of that ruling; `played_bpm` survives as an
unread tag precisely because it is "what he played, not what was
required," and the `answer_key` tag was renamed `pianist_take` in the same
pass to stop the take reading as correct.

**The terminology hazard that caused the error, recorded so it does not
recur.** "Marking" carries two senses in this project and they collided:

- **the teacher's marking of the combination** — setting/demonstrating it,
  which is what `marking_bpm` refers to and what the benchmark grades
  against;
- **marking in the dancer's sense** — sketching a movement rather than
  dancing it full-out, which is what the owner's Q1 answer excludes.

Q1 is guidance on **how to read a tempo off a clip** (prefer the first
clear, steady, full-out window). It is not a statement about **which field
is truth**. Reading the second sense into the first manufactures a conflict
that does not exist. Agents: do not re-open this.

**The one real defect, and it is owner-gated only because of rule 2.** The
`notes:` prose in all nine demo case files still states the pre-ruling
behaviour and therefore contradicts both the ruling and the code. It is
false text sitting in the corpus where the next reader will trust it — this
session did. Correcting it means editing files under `evals/cases/`, which
rule 2 forbids to agents even for a comment, so it needs the owner's word.
**No truth value, tag, or scored field would change** — the edit is
confined to the `notes:` block of nine files, replacing the false sentence
with the ruling above.

---

*The original ask, preserved:*

**This was an owner action. No agent may author this table, and no agent
may derive it from the corpus.** Deriving "rond de jambe ≈ 96" from the
single rond de jambe clip in the gating set is memorising the answer key,
not building a prior. The table had to come from professional knowledge,
**written before looking at what our clips are labelled.**

## 7. The ablation, RE-SPECIFIED for name-plus-observation (2026-09-05)

**Why this section was rewritten.** The original §7 keyed a tempo prior on
the **exercise name alone** and asked whether it helped. The owner's Q2
ruling closes that design before it runs: *"exercise name is never enough.
it always needs to be corraborated with the observation"*. The old arms are
preserved at the bottom for the record; they are **not** the increment to
run.

### 7.1 The reframe: the name does not supply a tempo, it picks a multiple

Measured this session on the eight gating demos, from PP-1's own per-clip
table. Where the acoustic pulse produced an estimate at all, compare it to
the truth as a *ratio* rather than a difference:

| demo | truth | pulse | truth/pulse | nearest integer |
|---|---|---|---|---|
| coupé-barre | 108 | 108.3 | 1.00 | ×1 (0.3% off) |
| dégagé | 110 | 113.9 | 0.97 | ×1 (3.4%) |
| frappé | 135 | 144.0 | 0.94 | ×1 (6.2%) |
| fondu | 86 | 45.5 | 1.89 | ×2 (5.5%) |
| plié | 120 | 42.2 | 2.84 | ×3 (5.2%) |
| rond de jambe | 96 | 31.9 | 3.01 | ×3 (0.3%) |
| tendu warm-up | 112 | 42.2 | 2.65 | ×3 (11.5%) ✗ |
| tendu | 102 | *refused* | — | — |

**On six of the seven clips where the pulse answered, it lands within 8% of
an exact integer subdivision of the truth.** The pulse is not wrong on the
five demos PP-1 described as "the wrong metric level" — it is *right about a
periodicity* and wrong only about **which multiple of it is the beat**. That
is a different and much more tractable problem, and it is the one the
exercise name is actually equipped to help with: a name cannot tell you
120.0, but it can tell you that 126 is a plausible plié beat and 42 is not.

This is the mechanical content of "corroborated with the observation":
**observation proposes the periodicity, the name selects among its
multiples, and neither may overrule the other.**

### 7.2 The hard-filter design is already falsified — do not build it

The obvious implementation is to keep only those multiples that fall inside
the owner's §6 band. **Measured, that fails**, and the reason matters:

| demo | pulse | §6 band | multiples inside band | result |
|---|---|---|---|---|
| plié | 42.2 | 85–125 | none (×2=84.4, ×3=126.6 — **both miss by <2 BPM**) | MISS |
| frappé | 144.0 | 120–140 | none (×1=144, misses by 4) | MISS |
| dégagé | 113.9 | 85–105 | none | MISS |
| fondu | 45.5 | ~100 ±8% | none (×2=91, misses by 1) | MISS |
| rond de jambe | 31.9 | 85–120 | ×3 = 95.7 | **HIT** (truth 96) |

**1 hit, 4 misses**, and three of the four misses are boundary misses of
under 4 BPM. A hard band is a fold, and **Standing Lesson 2 forbids folds**
— the same doctrine PP-1 obeyed by capping its prior so it could never zero
a hypothesis. A band used as a filter is exactly the failure mode that
doctrine exists to prevent, now demonstrated on this corpus rather than
argued.

**Worse, and this bounds the whole idea:** the owner's own bands do not
always contain the owner's own labels. Of the six demos with both, **the
truth falls outside the band on two** — `degage` (truth 110, band 85–105)
and `fondu` (truth 86, band ~100). A hard band would **veto the correct
answer on a third of the callable demos**. This is not an error in the
table; it is a measurement of how much real classes spread around what a
dance musician would predict, and it caps what any name-keyed prior can
deliver.

### 7.3 The increment to run — REPORTED-ONLY, pre-registered before code

**Mechanism.** Extend PP-1's existing bounded-prior seam; do not build a new
path. PP-1 already multiplies a log-normal bump into the lattice's tempo
marginal with a mixture floor that can tilt but never veto. The change is
**what the bump is centred on**:

- **PP-1 today:** one bump at the raw all-pairs pulse period.
- **This increment:** a bump at **each integer multiple** ×1…×4 (and ÷2, ÷3)
  of that period, with each multiple's weight scaled by how well it agrees
  with the §6 band for the exercise the pipeline *believes* it is seeing.
- **Agreement is soft and must stay soft:** a multiple outside the band is
  **down-weighted, never removed**. `W` stays capped as PP-1 capped it; the
  mixture floor is non-negotiable. Any implementation that can drive a
  hypothesis to zero is out of spec by §7.2.
- **Declined rows are declined.** `rond de jambe en l'air`, `coupé-barre` and
  `tendu warm-up` have no band. Those clips get the **uniform** treatment
  over multiples — the increment must not invent a band, and an agent may
  not fill the table in.

**Arms.**

- **Arm A — control:** PP-1 as blessed today. (Known: tier1 tempo 21/34,
  Acc2@8% 0.727, demo slice 4/8.)
- **Arm B — the real question:** multiples weighted by the §6 band keyed on
  **Gemini's own exercise guess, errors included, no oracle.** Its naming
  accuracy is part of the measured result, not assumed away.
- **Arm C — the honest control:** the same, keyed on the **true** exercise,
  separating "the prior helps" from "the labelling is good enough."
- **Arm D — the observation-only control, new and required:** multiples
  weighted **uniformly**, no band at all. **Without Arm D the increment
  cannot tell "the name helped" from "considering multiples at all
  helped"** — and §7.1 suggests multiples alone may carry most of it. If D
  ≈ B, the name is decoration and Q2's ruling has been satisfied in the
  most deflationary way available. That is a publishable answer.

**Gate.** REPORTED-ONLY; nothing adopted in this increment. The comparison
of interest is **B vs D first**, then B vs A, then C vs B.

**A bias to pre-register a prediction about (W17, 2026-09-05).** On the three
demos where all-pairs lands at the right metric level it reads **high every
time**: coupé-barre +0.3%, dégagé +3.5%, frappé +6.6% (mean +3.5%, n=3, all
positive). n=3 is far too small to call it a calibrated offset and **nothing
should be corrected for it** — but a one-sided error of that size against a
±8% tolerance means an adopted all-pairs prior could pass on this corpus while
sitting a couple of points from failing. The running session must state in
advance what it expects the signed error to do, and report signed error and
margin-to-threshold, not only pass/fail. Note this is a property of the
*standalone estimator*, not of the shipping pipeline, which predicts frappé at
134.9 against a truth of 135.

**What the running session must pre-register before writing code** —
predictions for each arm on tier1 committed tempo, Acc2@8%, between-levels,
the 8-demo slice, and ECE; plus the refusal behaviour on the three declined
exercises and on `tendu`, where the pulse refuses outright and no amount of
prior can help.

**Disclosure, so the pre-registration is honest.** §7.1 and §7.2 were
computed **before** this specification was written and are known to whoever
pre-registers. Predictions must be made with these numbers on the table and
must not be read off them: the integer-multiple finding (6 of 7) and the
hard-filter failure (1 of 5) are facts about the *inputs*, not scores for
any proposed mechanism, and nothing here has been tuned.

### 7.4 Honest limits, unchanged and worth restating

**n is brutal.** Eight demos, roughly one clip per exercise, three of them
with no band at all and one where the pulse refuses — so Arm B has **four
clips** on which the name can do anything. A result at this size is
indicative and never settled. The binding constraint on this whole benchmark
remains owner-verified corpus growth, not cleverness in the estimator.

---

*Superseded, kept for the record — the original name-only §7:*

> **Arm A:** the current estimator alone. **Arm B:** the same estimator with
> the tempo prior conditioned on the exercise label, using Gemini's own
> guess. **Arm C:** the same prior keyed off the true exercise. Gate: if B
> beats A the knowledge is doing work; if B ties A but C beats both, the
> prior is right and the labelling is the bottleneck.

**Closed by the owner's Q2 ruling, 2026-09-05.** The name alone is not a
sufficient prior at any strength, so Arm B as originally written measures a
design that has already been ruled out.

## 8. Things a fresh thread should NOT do

- Do not chase the −69 ms offset. It cannot move a tempo number (§3).
- Do not run another window sweep. SW-1 answered it: helps on rig, does
  nothing on demos (§2).
- Do not take the accent-periodicity line, tap grids for it, or build the
  WhiStress adapter — held until step two (§1).
- Do not cite SW-1's F3 for "the owner does not read audio regularity."
  That conclusion is withdrawn (§2).
- Do not author the prior table or infer it from the corpus (§6).
- **Do not build the nonlinear oscillator / GrFNN.** EB-1 measured it: this
  is not a missing-pulse corpus and the Hopf bank came last (19/34, 2/8 on
  demos). Review 6 §3 explains the mechanism; §8 ranks it last for a reason.
- **Do not re-run the estimator bake-off.** EB-1 answered it. `all-pairs`
  and `comb` tie at 28/34 and are not separable at this n — that needs more
  verified rows, not more analysis.

## 9. Artifacts from this session

- [sw1-steady-window-sweep.md](sw1-steady-window-sweep.md) + `.json` — the
  sweep, pre-registration and results.
- [w2-reopen-prominence-audit-barre6.json](w2-reopen-prominence-audit-barre6.json)
  (+ `-whistress.json`) — PR-1, complete, held.
- `machine-hearing/` — per-demo Audacity label tracks: every peakRate
  event tagged as landing on an owner beat or as an extra, the word
  starts, the owner's beats the machine missed, and the windows the
  algorithm chose with what tempo it read in each. Import the audio, then
  File → Import → Labels.
- [eb1-estimator-bakeoff.md](eb1-estimator-bakeoff.md) + `.json`, and
  `eb1-arm-b-trackers.json` — the estimator bake-off, the regime
  diagnostic, and the off-the-shelf trackers on the demos.
- [review-6-syncopation-and-pulse-reconstruction.md](review-6-syncopation-and-pulse-reconstruction.md)
  — how a pulse is recovered when events don't sit on it, with transfer
  verdicts per algorithm.
- `scripts/sw1-steady-window-sweep.py`, `scripts/w2-reopen-prominence-audit.py`,
  `scripts/eb1-estimator-bakeoff.py`, `scripts/eb1-arm-b-trackers.py`.

## 10. The one adoption candidate now on the table

Unlike SW-1, EB-1 produced something worth adopting later: **replacing
`calculate_tempo`'s median of consecutive gaps with an all-pairs or comb
period estimate.** It is worth 12 rows on the gating set and cuts
between-levels rows from 21 to 7.

It is a **logic change** under ADR-015's zero-regression gate, it moves
scored outcomes, and it therefore needs its own pre-registration, its own
increment, and an owner re-bless. **It is not commissioned.** Do not bundle
it with the prior-table ablation (§7) — one bounded change per session
(rule 6), and bundling them would make it impossible to say which one
worked.
