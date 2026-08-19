# State of play — 2026-08-19

**For:** the owner. **By:** an agent session on `agent/state-of-play`, after reading
the charter, ADR-016, the annotation convention, the blessed baseline and the whole
research log. Plain language; no pipeline or eval files touched.

**Labels.** **✓** = I checked it myself against the files — opened the grids, read the
blessed numbers, read the scorer, and re-ran the test suite and scoring suite on a
synced `main` (`034d226`). **(relayed)** = repeated from the log, not independently
confirmed. The log has misreported its own bookkeeping twice, so the label matters.

## 1. What this is trying to build

Imagine a rehearsal pianist who never needs to be told the tempo. The teacher marks
a combination — "five, six, seven, eight… tendu, close…" or just "da DA da da" — and
the pianist already knows the speed, the meter (in two, three, four, a lilting
six-eight), whether the movement is felt in twos or threes under the beat, and how
long the phrase is. This project is the **ears** of that accompanist: it listens to
the marking and writes down, in a fixed form, what a good pianist would have
understood. It plays nothing; a later part of the system would.

The first version found the beat by transcribing the teacher's *words* and timing
them. It worked in exactly one situation — counting numbers, in four, over one eight
— and fell apart on step names, vocables, quiet or legato marking, and anything not
in four. Two weeks ago the owner and the loop reset the plan (ADR-016): treat the
**voice as a drum** — find beats from the sound itself (vowel starts), read meter
from which beats are stressed, and let every kind of evidence (sound, words, the
kind of exercise, later gestures) argue it out in one joint judgement that also knows
how unsure it is. And: *build the measuring tools first; nothing is believed until it
passes a human-verified test.* That is what the fortnight has been.

## 2. Where it stands today

**What genuinely works:**

- **Ground truth exists.** All 30 practice clips have a beat grid; on 28 the owner
  personally placed or confirmed every beat (802 beats). Two were declined — one the
  owner himself could not parse, and the pliés demo, where the pulse is not in the
  audio at all, only in knowing pliés are slow. ✓ counted the files and flags.
- **The acoustic beat detector beats the word-based one decisively on studio
  marking.** Over the 28 verified clips it finds about **83 in 100** voiced beats
  (words: 45) and about **87 in 100** of its firings land on a real beat (words: 51).
  On step-name marking — what real teachers use, where words were hopeless — it
  finds **72 in 100**, up from 35, improving on 12 of 13 clips (one tie). On the one
  vocables clip: 14 of 16 beats, where the word system found 1. ✓ read the committed
  result files; did not re-run the experiment.
- **That win is flattered, and the log says so.** 25 of 28 grids were made by the
  owner *correcting the detector's own suggestions*, so wherever he kept a
  suggestion the detector gets credit for "finding" a beat it proposed. The
  anchor-free evidence is the three clips annotated from scratch against video:
  there it finds **56, 69 and 69 in 100** beats (from 49, 67, 39), and **half to
  three-quarters of its firings are not beats.** ✓ per-clip table. Quote those as the
  honest magnitude: clear win on clean studio marking, modest on real teaching video.
- **The discipline is real.** A convention ratified before annotation began; two
  quality checks that caught four real export errors (relayed); a grid format that
  can now tag silences and free-time stretches (33 silent-beat gaps, 6 free-time
  spans, 4 clips ✓); the owner's repeatability measured — *which* events are beats
  was perfectly repeatable, *where* they sit wobbles ~25 ms (relayed).
- **Tests and scoring pass today** on synced `main`: 213 tests green, "no outcome
  changes vs baseline." ✓ ran both.

**What the system a user would run actually does — unchanged.** The acoustic
detector is **not wired into the program** (✓ the main pipeline file never
references it). The user-facing numbers are the blessed baseline: **tempo right
about six times in ten** (17 of 30; 12 wrong, 1 declined); **meter, tempo and feel
all right together about four in ten** (11 of 29); **phrase length right 12 of 28**,
7 abstentions. When the teacher counts *numbers*: tempo 12 of 14, meter 10 of 14.
With *step names*: tempo 5 of 14, meter **1 of 13**. ✓ baseline file. One trap: tempo
"rose" 0.571→0.586 on 14 August because a wrong label was corrected (the grid and
the owner's ear said ~102, the label said 130) — not because anything improved. ✓

**Still hope:** meter from accent; the joint judgement; asking the language model
several times and reading the spread; anything using video; any voice but the
owner's; any accompanied class. None built, none in the corpus.

## 3. The progress story: what two weeks bought and cost

**Bought (✓ on `main`):** charter, ledger and goal ladder; grid format, tap-assist
tool and stage-level scoring; 28 verified grids; the ratified convention with three
evidence-based owner rulings (explanation speech counts if the pulse was voiced;
unmarked talk defaults to free time; silent beats credited only at the later
inference stage, never in the beat score); the detector and its passed kill-test;
format 2 with tags, method metadata and quality checks; a nightly unattended runner,
armed then fixed; and the first weekly self-review, recovered from a log. Plus two
findings worth the price (relayed): the transcription engine silently **drops whole
rounds of eight** on 4 of 14 clean numbers clips and still scores green; and the
video clips, "easy" against machine-made references, became the hardest slice once a
human said where the beats were.

**Cost:** mostly the owner's time — two full annotation sessions and several
one-question-at-a-time service sessions. The first unattended night ran 20 minutes
and ~$12 and wrote nothing (one missing permission flag; fixed ✓; tonight's 02:00 run
is the first real test). The rung-2 gate had to be scrapped and rewritten because it
was pre-registered against machine-made grids that were wrong. The ledger misreported
two things. Three increments — tag format, grid tagging, runner launch — sit on
`main` **merged but reviewed by nobody**; merging is not blessing, and nothing about
those merges is an endorsement. And across the fortnight **not one number a user
would see has improved** — by design, instruments first, but say it in those words.

## 4. Paths from here

- **Build meter-from-accent next (W2), as planned.** As an *accuracy* step it can
  move **at most 2 of 30 rows**: the scorer credits meter only when tempo is also
  right, and of the seven wrong non-4/4 clips only two have the tempo right — the
  other five are 17–50% off on tempo, which no meter code fixes. ✓ read the scorer
  and per-clip tempos; the review's first draft said seven and corrected itself to
  two. Hence its recommendation: **re-scope W2 as an evidence step** — does the
  accent signal recover the right grouping on the nine non-4/4 clips against the
  verified grids? — feeding the joint judgement (W5), where meter is actually fixed.
  I think that is right; the alternative is pre-registering against a two-row ceiling
  and calling a two-row move progress.
- **The cheap fix (W2.5):** the detector's silence gate drops soft material; the one
  step-name clip it failed to improve is the quiet one (5 of 16 beats). No owner time.
- **Off-the-shelf comparison (W3):** standard beat-trackers on the same grids. No
  owner time, no API key; answers "is this hard, or are we just early?"
- **Grow the corpus (W4, capture):** the 8 remaining Barre-1 exercises, then other
  voices and an accompanied class. The only road to the completion targets, which
  assume **60+** verified clips against **28**. Costs owner annotation hours. (A third
  of Barre-1 is set aside on this machine, untouched — I did not open it.)
- **The big swing (W5):** owner-attended by rule; needs W2's evidence and, honestly,
  a bigger corpus to be judged on.
- **Or pause the unattended loop** until the three unreviewed increments are read —
  `main` is now the loop's instruction channel, and work lands on it faster than it
  is reviewed.

## 5. The optimistic case

The central bet was written as a kill-test before any code existed, predictions
first, and it **passed against human ground truth with margins several times the
requirement** — five of six predictions landed. The hardest case, vocables, went from
unusable to strong. Every remaining failure has a named cause and a named unbuilt
fix; the waltz clip even carries a measured, repeating stress pattern in its verified
grid — exactly what the meter step is built to read. The process is unusually honest:
anchoring disclosed, a review that corrected its own headline, negative results
treated as deliverables. The corpus — beat-annotated rhythmic speech — has no
published equal, so even a partial result is a contribution. If accent-meter works
and the joint judgement lands, step-name meter could go from one-in-thirteen to
something a pianist could use.

## 6. The pessimistic case

Nothing a user sees has moved; all of it is instrumentation and one component test.
The corpus is **28 clips, 26 in one voice** (the owner's), English, unaccompanied,
studio-recorded — and the detector was tuned and the grids anchored to *that* voice;
nobody knows how it does on a stranger. The completion targets assume a corpus that
does not exist and has no funded plan; the only queued material is one teacher's
video, the hardest slice. Two results are permanent: a slow 60 and a half-time 120
are identical in the pulse, and in the pliés demo the pulse is not in the audio — the
system must ask or abstain, and that product conversation has not happened. The
marathon machinery eats effort: three unreviewed merges, a $12 night that wrote
nothing, a ledger wrong twice. **Most likely failure:** the joint judgement proves
harder to build and debug than the rule-stack it replaces, on a corpus too small to
tell whether it helped, while owner annotation time runs out — and the project drifts
into perfecting its instruments. **Where it may be fooling itself:** quoting the
anchored +37-point step-name gain as detection quality; reading "merged" as
"progress"; targets written for 60 clips while holding 28; and the teaching-video
numbers, where half the detector's firings are not beats.

## 7. Questions that need the owner

**Owner-only:**
1. The first batch review: bless, amend or reject the three PROPOSED increments on
   `main` (W1 tag format since 08-16; the 08-18 grid work; the runner launch).
2. Rule on amendments A1–A6 — above all **A6** (W2 as an evidence step) and **A4**
   (completion targets vs. a 28-clip corpus; the targets are owner-editable).
3. `rig-numbers-2-4-120-clean` still says `accompanied: false` (✓ line 12) on the
   clip the owner heard music on; case files are agent-untouchable.
4. The vocables listen — the detector emitted nothing near 7.27 s and 9.70 s. Blocked
   on the Air for lack of MP3s; **the MP3s are on this machine** (✓), so it can be
   done here.
5. Two grids' prose BPM notes now disagree with their tags (adagio, 160-long): which
   artifact to correct.
6. Whether to spend annotation hours on Barre-1 now, and whether/when to capture
   other voices and an accompanied class — corpus decisions only the owner can fund.
7. Whether the nightly loop keeps running before the batch review.

**Merely unfinished (agents can do it):** W2.5 quiet floor; W3 baselines; W2 as
evidence; checking tonight's runner.

## 8. What I would do next

Pause the nightly loop for one owner session and spend it on the batch review and
rulings A6/A4, so `main` stops accumulating unreviewed work and the next increments
aim at the right target. Then send the unattended sessions at W3 and W2-as-evidence —
neither needs owner time or an API key — while the owner's scarce hours go to Barre-1
annotation, because the corpus, not the code, is the binding constraint on every
claim this project wants to make. Keep W5 parked until W2's evidence and something
like 40 verified clips exist; a big model judged on 28 clips cannot be told from luck.
