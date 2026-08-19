# Agent Environment — runners for the loop

**Date:** 2026-08-09 (updated same day: runner decision recorded) ·
Companion to the [agent charter](agent-charter.md).

## The runner decision

Three machines, three roles. Decided 2026-08-09 (see the
[ledger](RESEARCH-LOG.md) entry):

| Machine | Role | Why |
|---|---|---|
| **The Air** (always-on 16 GB MacBook Air) | **Primary runner** — nightly rung execution, ingestion, rung-6 baselines | Data locality (recordings live there, no repo/LFS/bucket logistics); persistent state (model weights download once, the finicky MIR-tool venvs get built once); free compute; fanless throttling just means slower, never failed |
| **Owner's main machine** | **SEALED vault + blessing desk** | Sealed clips exist *only* here — never on the Air, never in the repo — so the DEV/SEALED separation stays physical on every runner. Milestone SEALED scoring and `evals bless` happen here |
| **Cloud sessions** | **Overflow** — burst parallelism (racing two designs of a rung; rung 6 alongside a pipeline rung), clean-room reproducibility checks | Ephemeral fresh-clone containers; zero ops; but they pay the data-logistics and per-session-setup tax the Air doesn't |

One expectation to keep straight: **the runner is the hands, not the
brain.** Claude Code on the Air still calls the hosted Claude models —
plan usage is identical to cloud. What moves on-device is execution:
files, tools, tests, evals, model weights.

A note on cloud secrets, since it was a worry: cloud environments support
env-var secrets (`GEMINI_API_KEY` etc.) and they work fine. The genuine
cloud frictions are *state*, not secrets — fresh clones don't contain
gitignored media, and every container re-pays environment setup.

## Air runner setup (one-time)

1. **Account.** A dedicated macOS user account for the agent (the cheap
   sandbox for autonomous work on a personal machine); at minimum a
   dedicated directory. Install Claude Code; note the absolute path from
   `which claude` — launchd jobs don't inherit your shell PATH.
2. **Repo + env.** Clone the (private) repo; run
   `bash scripts/cloud-setup.sh --live` once. Unlike cloud, this persists:
   Whisper weights, pip env, and later the isolated rung-6 tool venvs all
   survive between runs.
3. **Data.** Sync the DEV recordings to `audio/rig/` (matching the
   `evals/traces/*/meta.json` media paths). **DEV only — sealed clips
   never touch this machine.** Committing the DEV MP3s to the private
   repo (the `.gitignore` exception exists) is still recommended once,
   for provenance and to unlock cloud overflow — but the Air path does
   not require it to start.
4. **Power.** Keep it plugged in; `sudo pmset -c sleep 0` so the machine
   never sleeps on AC (display can sleep); the nightly wrapper also runs
   under `caffeinate -i`.
5. **Schedule.** `scripts/air-nightly.sh` is the nightly entry point
   (fresh `main`, then one goal-driven run appended to
   `~/musical-perception-agent.log`). Load it via launchd:

   `~/Library/LaunchAgents/com.musical-perception.nightly.plist`:

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
     "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0"><dict>
     <key>Label</key><string>com.musical-perception.nightly</string>
     <key>ProgramArguments</key><array>
       <string>/bin/bash</string>
       <string>/Users/AGENT_USER/musical-perception/scripts/air-nightly.sh</string>
     </array>
     <key>StartCalendarInterval</key>
     <dict><key>Hour</key><integer>2</integer><key>Minute</key><integer>0</integer></dict>
     <key>StandardOutPath</key><string>/tmp/mp-nightly.out</string>
     <key>StandardErrorPath</key><string>/tmp/mp-nightly.err</string>
   </dict></plist>
   ```

   `launchctl load ~/Library/LaunchAgents/com.musical-perception.nightly.plist`
   — the desktop app's scheduled tasks are an equivalent alternative.
6. **Interactive runs** (dry runs, supervised rungs): open the repo and
   paste the rung's `/goal` condition from the charter directly, paired
   with auto mode.

## The standing contract (runner-agnostic)

Scheduled runs never edit their invocation — the charter carries the
moving state. The same contract works as a `/goal` (Air, headless or
interactive) and as a cloud routine prompt:

```
/goal The CURRENT RUNG named at the top of docs/research/agent-charter.md
is complete per that rung's own condition as written in the charter —
every proof clause demonstrated by command output in this transcript, the
constraints verified (git diff --stat main shown; work only on the rung's
agent/* branch; evals/cases, evals/traces, evals/baseline.json and the
scorer code untouched), and a dated entry appended to
docs/research/RESEARCH-LOG.md — OR the rung's stated turn bound is
reached — OR the current rung is owner-only or awaiting blessing, in
which case a one-line ledger note saying so is the entire deliverable.
Begin by reading the charter and the ledger's Standing Lessons.
```

The final clause is the idle guard: the schedule is safe to leave running
when yesterday's work hasn't been blessed — the loop notes it and stops
instead of stacking unreviewed changes. The boot-sequence requirement
("begin by reading the charter") is what puts the rung's full condition
into the transcript where the goal evaluator can judge it.

## Cloud overflow setup (when parallel sessions are wanted)

- DEV audio must be in the repo for cloud clones to hear it (the
  `.gitignore` exception covers `audio/rig/*.mp3` only — probe-verified).
- Environment config: `GEMINI_API_KEY` as a secret (needed only for
  live-perception work); network policy allowing PyPI and, for rung 5 /
  ingestion, `generativelanguage.googleapis.com`; setup script
  `bash scripts/cloud-setup.sh` (add `--live` only for ingestion
  sessions — weights re-download per container, so batch that work).
- A cloud routine firing nightly with the standing contract above as its
  prompt, completion notifications on. Create it only after the Air loop
  has a working rhythm; overflow parallelism is its job, not primary
  execution.

## Local models policy

Decided with the runner decision, under the project's accuracy-first
posture:

- **In now — specialized nets** (all comfortable on the Air): Whisper
  (already local), **speech SSL front-ends (WavLM / DistilHuBERT)** — the
  singing-voice beat-tracking literature's finding that speech
  representations are the right front-end for sparse vocal rhythm
  ([Review 4 §(b)](review-4-tools-baselines.md)), Silero VAD, MediaPipe.
- **Deferred — general local LLMs (7–27B) in the semantic channel.** By
  the charter's own posture this is a cost optimization at the expense of
  accuracy, i.e., premature; practically, a 27B model at 4-bit is
  ~14–16 GB of weights and does not fit a 16 GB Air alongside anything
  else. The eval harness — not intuition — decides when a cheaper model
  matches the frontier one.
- **Rung-5 backlog note:** an audio-native small model (Qwen2-Audio
  class) as a cheap third ensemble vote — worth measuring when rung 5
  runs, not worth prioritizing.
- **Graduation condition:** at roughly 140+ annotated clips, fine-tuning
  a small model on the corpus becomes the real local-model play (the
  Vision 10 P2 posture converting the dataset into a model).

## Rung-0 checklist (owner, one-time)

**Path A — the Air (primary, start here):**
1. Repo privacy confirmed (before any recording is ever committed).
2. Air runner set up per the section above (account, env, data synced,
   power, schedule loaded but not yet enabled).
3. SEALED vault: sealed material staged on the main machine only.
4. One supervised interactive dry-run of rung 1 on the Air. Fix friction
   here, not at 2am. Then enable the schedule.
5. **A HEADLESS write probe — amended 2026-08-19, and the amendment is the
   point.** Step 4 as originally written is *structurally blind to the
   failure it exists to prevent*: an interactive session gets permissions
   from a human answering prompts, so it can never detect a headless-only
   permission failure. The 2026-08-19 nightly run proved it — the schedule
   fired correctly, the agent read everything and completed W0 across 107
   turns, and it could not write a single byte, because permission mode is
   **per-session and inherits nothing**; `claude -p` with no
   `--permission-mode` starts in `default`, where every write blocks on a
   human who is not there. Before enabling any schedule, run the probe
   below **headlessly, with the exact flags the wrapper uses**, and confirm
   the file exists afterwards:

   ```bash
   claude -p 'Write the word ok to /tmp/mp-write-probe.txt, then stop.' --permission-mode auto && cat /tmp/mp-write-probe.txt
   ```

   If that file is missing, the schedule will run all night and commit
   nothing. Generalized: **never accept an interactive test as evidence
   about a non-interactive runner.** Test the runner in the mode it runs in.

**Path B — cloud overflow (later, optional):**
5. DEV MP3s committed and pushed; cloud environment configured; one
   supervised cloud dry-run; then the overflow routine.

## Operating notes

- **Steering the Air:** nightly runs are fire-and-forget (log +
  branch + ledger are the record); supervised runs are just interactive
  sessions. **Two records, moved 2026-08-19:** the raw transcript is
  `logs/agent-nightly.log` **inside the repo but gitignored** — in the repo
  so a session can read why its predecessor failed (it cannot read outside
  its working directory, which is why the 2026-08-19 failure needed a human
  to diagnose), gitignored because it runs ~1 MB per run, can quote personal
  teacher-video speech, and a directory listing inside it would encode the
  HELD-OUT split by absence. The committed record is
  `logs/run-summaries.md`: outcome, turns, duration, cost and the agent's
  closing message per run, published by the *following* night's run (one
  night's lag, by design — it commits after the pull so the tree is clean
  before the agent works). Read the summary from anywhere; read the raw log
  on the runner.
- **Air failure modes:** macOS updates rebooting the machine (settings →
  defer major updates), launchd PATH issues (absolute `claude` path in
  the wrapper), Wi-Fi drops mid-push (the wrapper's git operations fail
  loudly into the log; the next night self-heals since work is
  committed locally on the rung branch).
- **Cost bounds:** the turn clause in every rung condition is the budget
  ceiling — same plan usage whether the hands are the Air or a cloud
  container.
- **Video staging:** teacher video is personal data — private storage
  only (the Air and the vault), never this repo; ingestion sessions read
  it locally.
- **SEALED discipline:** sealed clips have no presence on the Air or in
  the repo, so no agent session — local or cloud — can leak them into
  the loop even by accident. Milestone SEALED numbers enter the ledger
  via an owner-written entry.
