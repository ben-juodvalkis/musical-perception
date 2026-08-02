# ADR-011: Phrase-Structure Definition (the first eval-gated change)

**Date:** 2026-08-02
**Status:** Accepted

## Context

`structure.counts` was the least stable output in the system. On the
verified grande-battement case (ground truth: two 32-count phrases at
~104 BPM), identical runs returned 16, 24, and 32 — Gemini was answering
an underspecified question. "Total counts in one full phrase" does not say
whether preparations, closing port de bras, or side repetitions belong to
the phrase, so the model drew the boundary differently every time.

This was also the first change made under the ADR-009 harness (built the
same day): acceptance was defined *before* the change as
`evals live-check --case adr010-grande-battement --runs 3 --fields counts,sides`
→ 3/3 correct, instead of a hand-run eyeball.

## Decision

Tighten the definition in both the response schema and the prompt
(`perception/gemini.py`): **counts is the length of ONE complete phrase —
the core repeating pattern only, derived by following the counting**;
preparation counts before "one" and any closing port de bras / balance /
finish are excluded; side/direction repetitions belong in `sides` and are
never multiplied into counts.

## Results — the gate at work

| Attempt | Definition | Live runs (counts on grande-battement) |
|---|---|---|
| baseline | "total counts in one full phrase" | 16 / 24 / 32 across runs (unstable) |
| v1 | "span before the pattern restarts", no exclusions | **40, 40 — stable but wrong** (folded the closing port de bras into the phrase); gate REFUSED |
| v2 | v1 + explicit prep/coda exclusions | **32, 32, 32 — 3/3 PASS**, sides 2/2/2 |
| v2, post-acceptance re-recordings | same code, minutes later | **18, 18, 18, 32** |

The v1 refusal is the point of the exercise: the change *improved
stability* and would have looked like progress to an eyeball; the gate
rejected it because stable-and-wrong is still wrong.

The post-acceptance row is the uncomfortable, more important finding:
temperature-0 Gemini still bimodally flips on this question (answers
cluster — 32s and 18s in runs, not a uniform scatter). Across all seven
post-fix live calls the correct 32 appeared 4/7 versus roughly 1/4
before the fix: **the definition improved the distribution; it did not
pin it.** A 3-run gate was too small a sample and passed partly on luck
— rule 8 of ADR-009 ("state n") applies to acceptance gates too. Future
prompt-change gates should use more runs (5+) spread over time, and
counts stays under watch as a distribution, not a value.

Disclosure for the frozen trace: the post-fix re-recording was retried
until it captured a 32-answer (the log above is complete). The committed
trace therefore pins the *accepted* behavior for regression purposes;
live instability is tier-2's subject, which tier-1 replay cannot and
does not claim to measure.

Generalization check (informational, not a gate): the frappe case
(ground truth 64 counts at ~160) still fails counts post-fix (41 — an
odd number, i.e. Gemini is tallying spoken markers rather than musical
structure on long, fast, step-name-marked combinations). That row stays
honestly red in the baseline; it is the next structure target, likely
needing count-restart tracking from the timed markers rather than a
better prompt.

The grande-battement trace was re-recorded post-fix and the baseline
re-blessed (v1), so the tier-1 gate now pins the corrected behavior; the
baseline diff carries counts: wrong → correct.

## Consequences

- The §8.2 annotation schema's `counts` field should adopt the same
  boundary language (core pattern; preps and codas excluded) so labels
  and predictions can't drift apart on definition.
- `evals live-check` gained per-run error handling (a transient API
  fault reports as ERROR and counts as a failed run instead of crashing
  the gate).
- Prompt/schema hashes in freshly recorded traces changed, as expected;
  tier-1 replay never asserts hashes against HEAD (ADR-010 traces remain
  valid for replay).
