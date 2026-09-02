---
description: Run the owner-attended lateral review (docs/research/lateral-review-protocol.md) in this checkout
---
You are running the LATERAL REVIEW for the musical-perception research
loop. Ben (the owner, a dance musician, not a coder) started this by hand
and will join once you have briefed him. This is a high-effort session:
read deeply, think widely, do not rush the reading to get to the ideas.

First, bring this checkout up to date: `git fetch origin main` and confirm
`main` is at `origin/main` (if it is not, say so and stop). Then read
docs/research/lateral-review-protocol.md in full and follow it exactly;
it is the source of truth. If it is missing, stop and say so.

Then: branch agent/lateral-YYYY-MM-DD from origin/main and do the
write-and-commit probe; the unattended read; the anomalies and the
assumption ledger; the outward scan (this machine has full internet and
may have a browser tool — use them, and read the papers that matter in
full); the ideas, the graveyard check and the ranked question pool; then
ONE plain-language briefing (per the "Talking to Ben about this work"
section of CLAUDE.md) ending in exactly ONE question — never a numbered
list — and END YOUR TURN there. Then one question per turn, each chosen
after his answer. When he is done, write the memo and the ledger entry,
commit, push, and open the draft PR or hand him the compare link, per the
protocol.

It commissions nothing; touches no file under src/, evals/, scripts/ or
tests/; never edits the charter; never enumerates the Barre 1 video
directory (this machine is where held-out material lives — the
containment rules apply in full); no live model calls; stop after 80
turns.
