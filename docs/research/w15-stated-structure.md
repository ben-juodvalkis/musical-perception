# W15 — the stated-structure channel

**Date:** 2026-09-02 · **Status:** PROPOSED, REPORTED-ONLY · **Branch:**
`agent/marathon` · Commissioned 2026-08-31 (W0 §4), ratified 2026-09-01.
Artifacts: `scripts/w15-stated-structure.py`,
`docs/research/w15-stated-structure.json`,
`tests/test_w15_stated_structure.py`. Pre-registration and prediction
scorecard: [RESEARCH-LOG.md](RESEARCH-LOG.md), 2026-09-02.

> **Headline.** The hypothesis was that the teacher announces the meter out
> loud and we are throwing that away. He does not. Across all 52 frozen
> transcripts the channel emits **zero** beats-per-bar claims, because the
> corpus contains no sentence of that shape. What he *does* announce, 24
> times on 15 clips, is **lengths and repetitions** — how many counts a step
> lasts, how many eight-count phrases a section runs, how many more times to
> do it. Typing those correctly is easy (**30 of 31** audited candidates,
> against a hand gold set). Reading any of them as a bar length is wrong
> **8 times in 11**. W15 fails its own verdict rule and closes as a meter
> channel; the typed length claims are the residue worth keeping.

## 1. What was built

A read-only parser over `evals/traces/*/whisper.json`. It generates a
candidate at every numeral **not inside an ascending count run of length
≥ 3** (the teacher counting is not the teacher stating a quantity), then
types each candidate by an enumerated rule. Each emitted claim records the
rule that fired, so the disambiguation is auditable rather than implicit.

Gates nothing. Wired into no pipeline path (Standing Lesson 9). No file
under `evals/` and no file under `src/musical_perception/` is touched.

### 1.1 The typing rules

| rule | frame | type | why |
|---|---|---|---|
| R7 | `N beat(s) to/per a/the bar` | `beats_per_bar` | the only route to a meter claim |
| R3 | `N count(s)` (also `in N count(s)`) | `step_duration` | **a count is always a beat** (owner ruling), so this is a duration in beats, not a bar length |
| R4 | `N eight(s)` | `phrases` | a length in 8-count phrases — a grouping rung (ADR-017), not a bar count |
| R2 | `N bar(s)` | `bars` | musical bar, but only when a numeral quantifies it |
| R2-reject | determiner + spatial frame owns `bar` | abstain | in a ballet class the barre is furniture |
| R1 | `N more time(s)` | `repetitions` | |
| R1-reject | bare `this/last/each/same time` | abstain | not a quantity |
| R6 | `N more` | `repetitions` | |
| R5 | `in`/`at`/`on` + `N` at a clause boundary | `entry_point` | the count to come in on |
| R5-reject | the same frame quantifying a noun, or continuing an ascent | abstain | "on two legs"; "in, two, three" |
| R8 | anything else numeral-bearing | abstain | |

### 1.2 Declared deviation from the pre-registered vocabulary

The condition named four types (`beats-per-bar | repetitions | bars |
unknown`). The corpus forces three more — `step_duration`, `phrases`,
`entry_point` — and **forcing them into the four is the mis-typing the
workstream exists to prevent.** The clearest case: folding `phrases` into
`bars` reads a "two eights" announcement as *two bars* when it means
sixteen counts, which in 4/4 is *four* bars. The fold to the pre-registered
vocabulary is computed and reported anyway (`by_folded` in the JSON), so
the condition is scored as written; the finer types are what the channel
actually emits.

## 2. What the corpus contains

52 clips, all owner-verified. 219 candidates; **195 abstain**; 24 typed
claims on 15 clips.

| type | claims | clips |
|---|---|---|
| `beats_per_bar` | **0** | 0 |
| `bars` | **0** | 0 |
| `repetitions` | 11 | 9 |
| `step_duration` | 6 | 5 |
| `phrases` | 5 | 4 |
| `entry_point` | 2 | 2 |

**F1 — the meter is never stated.** Not once in 52 clips. This is the
finding, and it is about the material, not the parser: no sentence of the
form "N beats to the bar", "it's in three-four", "count it in six" occurs
anywhere in the corpus. A second meter channel built on declarative speech
has nothing to read.

**F2 — the `bar` homonym is a hazard for a keyword parser, not for this
one.** `bar`/`barre` occurs **13 times** and is furniture **13 times**. The
homonym gate (R2) therefore rejected **zero** candidates — a numeral-anchored
parser never proposes them. A keyword-anchored parser would have proposed
all 13, and two of them sit immediately *before* a numeral, so a `bar N`
pattern would have emitted two confident, entirely spurious bar lengths.
The design choice — anchor on the numeral, not the unit — is what made the
gate unnecessary, and that is worth more than the gate.

**F3 — what he announces is length, and it is consistent.** All five
`phrases` claims are on clips whose `counts` truth is divisible by 8 (48,
48, 96, 96). All six `step_duration` claims are "N counts <step>": a step's
length in beats. This is the eight-count phrase showing up in *speech*, the
same rung W2 found in the *timing* channel at lag 8. Two independent
channels, one grouping level, no bar in either.

## 3. The naive reading, re-derived on this corpus

The pre-registered bar — "beat the W0-2026-08-31 baseline of 1-of-3" — is
stale: two of that baseline's three rows were retired on 2026-09-01 and the
barre-1 rows are gone. The naive reading is therefore re-derived here, on
tonight's 52 clips: **read the spoken number as the bar grouping**, on every
candidate whose unit is `count(s)`, `bar(s)` or `eight(s)`, plus every bare
`in/at N`.

| clip | naive reads | truth numerator | |
|---|---|---|---|
| `barre6-ballonne-demo` | 3 (`in_N`) | 3 | **agrees** |
| `barre6-frappe-demo` | 4 (`N_counts`) | 4 | **agrees** |
| `barre6-frappe-take1` | 4 (`N_counts`) | 4 | **agrees** |
| `barre6-ballonne-take1` | 2 (`N_eights`) | 4 | wrong |
| `barre6-ballonne-take2` | 2 (`N_eights`) | 4 | wrong |
| `barre6-degage-take1` | 2 (`N_eights`) | 4 | wrong |
| `barre6-degage-take2` | 2 (`N_eights`) | 4 | wrong |
| `barre6-plie-take1` | 4 (`in_N_counts`) | 3 | wrong |
| `barre6-plie-take2` | 4 (`N_counts`) | 3 | wrong |
| `barre6-rond-de-jambe-demo` | 1 (`in_N`) | 3 | wrong |
| `rig-mixed-4-4-104-quantities` | 6 (`N_counts`) | 4 | wrong |

**3 of 11 (0.273)**, identical on the first-claim and any-claim
aggregations. Near the stale 1-of-3 rate, and nowhere near consumable.

**F4 — the plié clips are the trap, and they are the common case.** Both
plié takes are 3/4 exercises in which the teacher says the port de bras
takes *four counts*. The number is correct, the sentence is true, and
reading it as the bar length gives 4 against a truth of 3. `rig-mixed` is
the same shape at 6-vs-4. The naive reading is not noisy — it is
**systematically** wrong wherever a step's length differs from the bar,
which is most of the time.

**F5 — even the charitable reading tops out at 5 of 15.** Read *any* typed
claim's value as the bar grouping — the strongest form of the hypothesis,
and one the channel could not implement, since choosing which claims to
read that way is exactly the disambiguation — and it agrees on 5 of the 15
firing clips (0.333). Three of those five agreements are a `step_duration`
of 4 or a `repetitions` of 4 on a 4/4 clip: the right number for the wrong
reason, on the most common meter in the corpus.

## 4. Type precision

Hand gold set: all 24 emitted claims plus all 7 rule-level rejections, 31
audited, labelled by reading each candidate's context in the frozen
transcript. **30 correct, precision 0.968.**

The single disagreement is `barre6-rond-de-jambe-demo` at t=51.0. The
teacher says "one more on one"; R6 correctly types the first numeral as a
repetition, and R5 then types the *second* numeral as a separate
entry-point claim. It is the object of the first claim, not a second
announcement. Double-counting of this kind is the failure mode a
frame-based parser has, and it is left in rather than patched, because
patching it after seeing it is how a 0.968 becomes a 1.000 that means
nothing.

**Under-generation, disclosed.** Three genuine announcements are abstained
on, all by rules working as pre-registered:
- `also eight` (`barre6-releve-finish-take1`) — a length claim where the
  numeral *is* the unit; no frame covers it.
- `one last time` (2 clips) — rule 1 as committed treats `last` as a
  determiner and abstains. Semantically it is a repetition.
- **the count-in.** `seven and eight` opens a phrase on **6 clips** and is
  the corpus's most frequent structural announcement. It is not
  numeral-plus-frame, so no rule sees it. It is also the one un-parsed
  pattern that carries real information — it says the counting runs to
  eight — and it is parked as a backlog note rather than added tonight,
  because adding a frame after seeing which frames would have helped is not
  pre-registration.

## 5. Verdict

The pre-registered rule: W15 continues only if the channel emits a
correctly-typed claim informative about meter on **≥ 2 clips where tier-1
`meter_triple` is currently wrong** (34 such clips exist, so opportunity is
not the constraint).

**Strict: 0 rows.** No meter-bearing claim is emitted anywhere, on any
clip, wrong meter or right.
**Charitable: 4 rows** where a typed claim's value happens to equal the
truth numerator on a currently-wrong clip — three of them `step_duration`
or `repetitions` coincidences on 4/4, one the `entry_point` 3 on the 3/4
ballonné demo. The channel has no way to know which four.

**W15 closes as a meter channel.** It is not a parser failure and not a
corpus-size problem: the sentences do not exist. The residue worth keeping
is F3 — a working, tested, abstaining length-and-repetition channel whose
claims agree with the counts labels — and F2's design lesson.

## 6. What follows

1. **No pipeline change.** Nothing here is wired in, and nothing should be
   on this evidence.
2. **Backlog: the count-in frame.** `seven and eight` on 6 clips, plus
   `also eight` and `one last time`. It reads the *phrase*, not the bar,
   which is the rung the corpus actually carries — and it is testable
   against the `counts` labels that already exist. Cheap; pre-register it
   separately.
3. **Backlog, from the residue and outside W15's scope:** in the 3/4 clips
   the teacher's count-position callouts come in `two, three` doublets
   where the 4/4 clips carry `two, three, four`. That is a *counting*
   signal, not a declarative one, and it points at the meter that the
   declarative channel does not. Worth a sizing pass before anyone builds
   it.
4. **For W5.** The stated-structure evidence that exists is grouping-rung
   evidence (`phrases`, `step_duration`), which is what ADR-017's factored
   representation has a place for. It is weak, sparse (15 of 52 clips) and
   should enter as a low-weight observation on the grouping ladder, never
   as a meter vote.
