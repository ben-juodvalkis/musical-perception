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
