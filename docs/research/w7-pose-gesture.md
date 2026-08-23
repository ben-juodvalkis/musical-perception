# W7 — the pose/gesture pulse channel: a negative result

**Date:** 2026-08-23 · **Rung:** M / W7 · **Branch:** `agent/marathon`
**Status:** PROPOSED — negative result, owner review pending.
**Reproduce:** `python scripts/w7-pose-gesture-report.py`
(read-only over committed traces; no media, no models, no API key)

## The question

The Ballet Barre 1 batch contains takes where the teacher demonstrates to
piano and says almost nothing — six of the 22 clips carry ≤ 3 transcribed
words. On those clips the marker channel the whole pipeline rests on is
empty. W7 asks whether the dancer's *movement* carries a pulse that could
stand in.

No ground truth exists for these clips — their case files are BLOCKED on
W1.5 — so nothing here is an accuracy claim. Two things are answerable
without labels: does the gesture channel produce a periodic signal at all,
and does it land where the voice channel independently lands.

## Answer

**No, not as prototyped, and the failure is diagnosable rather than
mysterious.** Movement events extract cleanly on all 22 clips, but the
periodicity found in them is the *detector's own thinning scale*, not a
musical beat, and it agrees with the voice channel on zero clips.

| Prediction | Result | |
|---|---|---|
| **G1** extraction works, ≥ 1 event/s | 22/22 clips, median **2.76/s** | HIT |
| **G2** periodicity on ≥ 12/22 clips | **8/22** have ≥ 1 significant window; median per-clip coverage **0.00** | MISS |
| **G3** dominant period > 1.2 s (phrasal) | median **0.28 s** (211 BPM) | MISS |
| **G4** voice agreement < 50% | **0 of 7** clips agree at any metric level | HIT |
| **G5** coverage voice-independent | voice-less cov **0.00** vs voiced **0.05** | MISS (wrong direction) |
| **G6** module is inert | pytest green, `no outcome changes vs baseline` | HIT |

G4 was pre-registered as an honest low expectation and "hit" at zero,
which is not a success — a prediction that the channel would be weak was
correct, and calling that a hit would be scoring the thermometer instead
of the patient. The three misses are the content.

## Why the periodicity is an artifact, and how that was established

Every significant window's period sits between 163 and 240 BPM — far
above any plausible ballet tempo, and suspiciously close to the mean IOI
of the event detector itself (~0.36 s at an event rate of 2.76/s). A
post-hoc sweep (**post-hoc, not pre-registered**) settles it:

| min IOI | smoothing | clips with signal | median event rate | median period | implied BPM |
|---|---|---|---|---|---|
| 0.20 s | 0.06 s | 12 | 2.76/s | 0.29 s | 208 |
| 0.35 s | 0.12 s | 9 | 1.59/s | 0.45 s | 133 |
| 0.50 s | 0.20 s | 4 | 1.12/s | 0.61 s | 98 |

The detected period tracks the minimum-IOI parameter at **+0.10 s at every
setting**, and the number of clips carrying any signal *falls* — 12 → 9 →
4 — as the analysis scale approaches musical tempo. A real musical period
would sit still while the parameter moved, and would become easier to see,
not harder, as the detector stopped chopping it up. This one does neither.
The gesture channel, at this parameterization, is measuring itself.

*(The sweep above was run against the mid-session null and is retained as
the parameter-tracking evidence, which is a relationship between rows and
is unaffected by the null. Absolute significance counts elsewhere in this
document use the final null.)*

## Three nulls, two of them wrong — disclosed

The significance test went through three nulls this session. Both
rejections are recorded because each names a way this kind of test fails:

1. **Plain uniform.** The event detector enforces a minimum IOI, so the
   observations carry a constraint uniform draws lack; the test then
   reports the constraint. Symptom: every clip's best period pinned to the
   short edge of the candidate grid.
2. **Shuffled IOIs.** Permuting intervals is the *identity* on an
   isochronous train, so this null has exactly zero power against the one
   hypothesis the module exists to test. It scored a synthetic,
   perfectly-periodic input at p = 0.31. Caught by a unit test written as
   a positive control, not by inspection of the results — which is the
   argument for writing the positive control.
3. **Hard-core uniform** (adopted): the same number of events placed
   uniformly at random subject to the same minimum IOI. Shares the
   constraint, retains power against isochrony. Both controls pass.

The middle null was live when the first full results table was produced.
Its numbers were *more* favourable (12/22 clips rather than 8/22), so the
correction moved the result against the hypothesis.

## Secondary finding: `detection_rate` is not a usability signal

Fourteen of the 22 clips initially reported **zero** movement events. That
was a bug, not silence: undetected frames arrive as `NaN`, and a plain
median over them makes the event threshold `NaN`, after which no frame can
ever fall below it. Worth keeping is that `pose.npz`'s `detection_rate`
did not predict which clips were affected — one clip reporting
`detection_rate = 1.00` still carried 0.43% `NaN` landmarks, enough to
erase every event in it. Any consumer of these traces must check the
landmarks for gaps, not the summary field. Fixed and regression-tested
(`test_nan_gaps_do_not_erase_every_event`).

## What this does and does not license

It does **not** license "movement carries no beat". It licenses: *velocity
minima of torso-normalized limb speed, thinned at 0.2 s, do not carry
recoverable musical periodicity on this corpus, and the natural fix —
analysing at beat scale — makes the signal rarer rather than clearer.*

Recommendation, consistent with W2's: do not iterate this standalone. W2
found accent periodicity sitting at the count phrase rather than the bar,
and half its clips carrying no significant periodicity at any lag; W7
finds movement periodicity that dissolves under scale change. Both are
weak evidence channels that a joint posterior could still consume as
*votes* — which is W5's design — and neither survives being asked to carry
a tempo on its own. The specific next thing worth trying, if W7 is
revisited, is not a better peak-picker but a different event definition:
the exercise's own phrase structure (preparation, arrival, recovery) is
what a dancer places on the beat, and that is a segmentation problem, not
a periodicity problem.
