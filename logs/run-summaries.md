# Nightly run summaries

One entry per unattended `scripts/air-nightly.sh` run: outcome, turns,
duration, cost, and the agent's own closing message. Written by the run that
produced it, **committed by the following night's run**, so the newest entry
here is normally one night ahead of what is on `main`.

This is the committed record. The raw stream-json transcript lives at
`logs/agent-nightly.log` on the runner and is deliberately gitignored — see
`docs/research/agent-environment.md` operating notes for why.

*(The 2026-08-19 entry below was backfilled by hand from that run's log, which
is how the runner's permission failure was found.)*

## run 2026-08-19T09:00:04Z · main 034d226

- **outcome:** success, shell exit 0
- **turns:** 107 · **duration:** 20.2 min · **cost:** $11.98

**Agent's closing message:**

> The hook and I agree on the facts; we disagree only on whether an unsatisfiable condition should keep a session running. It shouldn't, so this is my last message.
> 
> Everything actionable is above. The one-sentence version: the nightly runner started correctly, could not write a single file because `scripts/air-nightly.sh:32` passes no permission mode, and the rung-0 checklist item meant to catch that was blind to it by specifying an interactive dry-run — so W0's review is complete and unfiled, and one line plus a headless test run recovers it.

*Raw transcript is gitignored at `logs/agent-nightly.log` on the runner.*

## run 2026-08-20T09:00:05Z · main 7a98067

- **outcome:** success, shell exit 0
- **turns:** 55 · **duration:** 11.6 min · **cost:** $4.69

**Agent's closing message:**

> …fied.** One complete increment (W2 = rung 3), evidence by full command output in this transcript: pytest 222 passed / 3 skipped; `evals run --suite tier0,tier1,stage1` → `no outcome changes vs baseline`; `git diff --stat main` → 6 files, 1224 insertions, 0 deletions, nothing under `evals/cases/`, `evals/traces/`, `evals/grids/`, or `src/musical_perception/evals/`, `evals/baseline.json` untouched.
> 
> **Rung M's own completion — owner-only, now recorded as such.** The charter reserves it to a meta-rung report co-signed by the owner with a multifaceted ablation table across at least three evidence channels. A session cannot supply that, and the charter says so explicitly. The one-line ledger note is filed, and the 45-turn per-session bound is reached.
> 
> The substantive output for your review remains the W2 negative result: accent periodicity in this corpus sits at the count phrase (lag 8), not the bar, with half the clips carrying no significant periodicity at any lag; and a salience clock cannot separate 2/4 from 4/4 (r=0.90) or 3/4 from 6/8 (r=0.93) on any data whatsoever. Recommendation is to fold accent periodicity into W5 as one observation channel rather than iterate it standalone.

*Raw transcript is gitignored at `logs/agent-nightly.log` on the runner.*
