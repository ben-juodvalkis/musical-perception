# Agent Environment — running the loop in the cloud (and locally)

**Date:** 2026-08-09 · Companion to the [agent charter](agent-charter.md).
This is the rung-0 checklist plus the operating notes for scheduled cloud
sessions.

## Where sessions run

- **Cloud (Claude Code on the web / remote sessions).** Each session runs
  in an ephemeral container that clones the repo fresh from `main`, works,
  pushes a branch, and is discarded. Anything not committed does not
  exist — which is the charter's memory discipline, enforced by physics.
- **Local (`claude` CLI on the owner's machine).** Same loop, run as
  `claude -p "/goal <rung condition>" --output-format stream-json --verbose`
  in the repo root. Local is required for anything touching SEALED data
  and is the fallback whenever cloud data logistics lag.
- **Hybrid (recommended):** cloud for the grind (implementation rungs,
  baselines, meta-review — all DEV-only), local for the human parts
  (recording, grid verification, SEALED scoring, blessing).

## Rung-0 checklist (owner, one-time)

1. **Repo privacy.** Confirm the GitHub repo is private **before** any
   recording is committed. Recordings of identifiable people (the teacher
   videos) do not go in this repo regardless — see Data below.
2. **DEV audio into the repo.** The `.gitignore` now carries an exception
   for `audio/rig/*.mp3`. From the machine holding the recordings:
   `git add audio/rig/*.mp3 && git commit && git push`. (~30 small MP3s;
   this is what lets a cloud session hear anything.)
3. **Video staging.** Teacher video goes to private storage the
   environment can reach — Git LFS on this private repo (mind the ~1 GB
   free tier) or a private bucket pulled by the setup script via a secret
   URL. Sealed clips: neither — owner's machine only.
4. **Cloud environment configuration** (claude.ai/code → environment):
   - Env var `GEMINI_API_KEY` (secret). Needed only for live-perception
     work (rung 5, trace ingestion); trace-replay rungs run without it.
   - Network policy: allow package installs (PyPI) and, for
     live-perception rungs, `generativelanguage.googleapis.com`; the
     meta-rung benefits from general web access for literature checks.
   - Setup script: `bash scripts/cloud-setup.sh` (below); add `--live`
     only for ingestion/rung-5 sessions.
5. **Dry run.** One supervised interactive cloud session executing rung 1
   end-to-end before any schedule is created. Fix friction here, not at
   2am.

## Setup script

`scripts/cloud-setup.sh`:

- default: `pip install -e ".[dev]"` — everything trace-replay rungs and
  the stage-1 acoustic work need (numpy, librosa/parselmouth via the
  prosody extra if listed, pytest).
- `--live`: adds the heavy extras (`.[all,dev]`) for Whisper/Gemini/pose —
  note Whisper model weights (~1.5 GB for `large-v3-turbo`) download per
  fresh container, so **batch ingestion work into dedicated sessions**
  rather than paying that cost nightly.

## Scheduling (after the dry run passes)

A cloud routine fires nightly and creates a fresh session with this
standing prompt (the charter carries the moving state, so the prompt never
changes):

```
Read docs/research/agent-charter.md and docs/research/RESEARCH-LOG.md.
Execute the CURRENT RUNG exactly per its condition in the charter,
treating that condition as your completion contract: keep working until
every proof clause is demonstrably satisfied in your own command outputs,
or the rung's turn/effort bound is reached, or you are BLOCKED per the
charter's boot sequence. Push only to the rung's agent/* branch; never
touch main, evals/cases, evals/traces, evals/baseline.json, or the
scorer code; append the dated RESEARCH-LOG.md entry before finishing.
If the current rung is marked owner-only or already PROPOSED and awaiting
blessing, write nothing but a one-line ledger note and stop.
```

Completion notifications (push/email) on, so a finished or blocked run
reaches the owner's phone. In interactive sessions (web UI or local CLI
v2.1.139+), the same contract is set directly with `/goal <condition>`
pasted from the charter, paired with auto mode for unattended tool calls.

Guard in the prompt's last sentence matters: it makes the schedule safe to
leave running even when the owner hasn't blessed yesterday's work — the
loop idles politely instead of stacking unreviewed changes.

## Operating notes

- **Steering:** cloud sessions accept messages mid-run; interrupt, redirect,
  or archive from the web UI or phone.
- **Parallelism:** rung 6 (baselines) is deliberately independent — run it
  in a second session alongside any pipeline rung. Racing two designs of
  one rung = two sessions, two branches, the eval decides.
- **Cost bounds:** the turn clause in every condition is the budget
  ceiling; long unattended sessions consume plan usage at the configured
  model's rate.
- **Failure modes to expect:** environment install friction (fix in the
  setup script, not ad hoc), madmom-era tool rot in rung 6 (isolated
  venv per [Review 4 §(d)](review-4-tools-baselines.md)), and containers
  reclaiming uncommitted work (the conditions require committing —
  trust the contract, not the container).
- **SEALED discipline in the cloud:** sealed material has no repo
  presence, so a cloud session cannot leak it into the loop even by
  accident. Milestone SEALED scoring runs locally, and its numbers enter
  the ledger via an owner-written entry.
