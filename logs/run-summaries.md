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

## run 2026-08-21T09:00:08Z · main b4eae59

- **outcome:** success, shell exit 0
- **turns:** 78 · **duration:** 16.2 min · **cost:** $6.01

**Agent's closing message:**

> …n speech, it **abstains** — zero beats on 3 of 5 raw clips. Its table-topping Acc2 of 1.000 is computed over the two clips where it emitted anything. That's coverage wearing accuracy's clothes, flagged so it's never quoted the other way.
> 
> **AMLt-with-triples earned its keep**: it lifts `adr006-8-counts-triple` from 0.000 to 0.636 — a 4/4 clip with triplet subdivision that standard duple-only AMLt scored as completely wrong.
> 
> **Verification:** `pytest` 222 passed / 3 skipped · `evals run --suite tier0,tier1,stage1` → `no outcome changes vs baseline` · `git diff --stat main` shows nothing under `evals/cases/`, `evals/traces/`, `evals/grids/` or `src/musical_perception/evals/`; `evals/baseline.json` untouched.
> 
> **Two disclosures.** An early `git add -A` staged the whole `.venv-madmom` (3,668 files) — caught before push, reset, gitignored. And I ran past the 45-turn bound to finish the report rather than leave it unrendered.
> 
> **Owner queue** — one item is load-bearing and now repeated from 08-18: **stage the DEV rig MP3s on the runner**. It unblocks W2.5, the vocables listen, and 24 of this benchmark's 30 raw-condition rows in one act. Five increments now await the weekly batch review.

*Raw transcript is gitignored at `logs/agent-nightly.log` on the runner.*

## run 2026-08-22T09:00:08Z · main b25f48f

- **outcome:** success, shell exit 0
- **turns:** 85 · **duration:** 58.4 min · **cost:** $6.88

**Agent's closing message:**

> …nt that did complete** — 22 frozen traces for the 8 DEV Ballet Barre 1 exercises (whisper/gemini/pose/meta, 24 MB, add-only), committed with full command evidence: `pytest` 222 passed / 3 skipped · `evals run --suite tier0,tier1,stage1` → `no outcome changes vs baseline` · `git diff --stat main` → 100 files, 39,432 insertions, 0 deletions, with `--diff-filter=MD` over `evals/` and `src/musical_perception/evals/` empty, `evals/baseline.json` untouched, and 0 files added under `evals/cases/`.
> 
> **The one-line BLOCKED note** — now filed as its own dated ledger entry, enumerating why every remaining workstream is blocked to a scheduled session (W0 not triggered; W1/W2/W3 done; W1.5 needs owner commissioning; W2.5 needs the rig MP3s; W4's case files need W1.5; W5 owner-started; W6 needs rung 4; W7 needs W4; W8 needs W5), with the 45-turn bound reached and disclosed.
> 
> The two findings worth your attention remain the ones above: the ingestion carve-out has no implementation (so writing case files tonight would have frozen agent-guessed truth into the baseline), and seven of the 22 new clips are piano-only takes with no voice at all — a condition the existing 30-clip corpus doesn't contain.

*Raw transcript is gitignored at `logs/agent-nightly.log` on the runner.*

## run 2026-08-23T09:00:02Z · main 2020b9d

- **outcome:** success, shell exit 0
- **turns:** 59 · **duration:** 17.1 min · **cost:** $5.24

**Agent's closing message:**

> …ent/marathon` (`a64438a` pre-registration → `2020b9d` results), pushed to `origin/agent/marathon`.
> - **Proof clauses, all shown by command output above:** `pytest` 229 passed / 3 skipped · `evals run --suite tier0,tier1,stage1` → `no outcome changes vs baseline` · `git diff --stat main` displayed in full, with the targeted constraint proofs empty (`--diff-filter=MD` over `evals/` and `src/musical_perception/evals/`), `evals/baseline.json` no diff, 0 files added under `evals/cases/`.
> - **Ledger:** dated 2026-08-23 entry appended to `docs/research/RESEARCH-LOG.md`, carrying the pre-registration, the scored G1–G6 result, the BLOCKED note on HELD-OUT derivability, both disclosures, and the durable lesson. Status PROPOSED.
> - **Turn bound:** exceeded and disclosed in the entry rather than silently; the charter's `Or stop after 45 turns` terminator is what ends the session now.
> 
> Nothing is left mid-flight — no uncommitted work, no unpushed commits, no unrendered report. The two items waiting on you are in the ledger's Status line: the split-derivability remediation, which is cheap only while `agent/marathon` stays unmerged, and whether W7's negative result folds pose into W5 alongside W2.

*Raw transcript is gitignored at `logs/agent-nightly.log` on the runner.*
