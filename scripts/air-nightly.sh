#!/usr/bin/env bash
# Nightly agent run on the Air runner (see docs/research/agent-environment.md).
# Loaded by launchd (com.musical-perception.nightly); safe to run manually.
# Starts from fresh main — the charter's CURRENT RUNG pointer carries the
# moving state, so this script never changes between rungs.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

# The raw transcript lives INSIDE the repo but is gitignored (logs/ is ignored
# except run-summaries.md). Two reasons it moved here from $HOME on 2026-08-19:
# a session cannot read outside its working directory, so under the old path an
# agent could not inspect why its predecessor failed; and the raw stream must
# never be committed — ~1 MB per run, it can quote personal teacher-video speech,
# and a directory listing in it would encode the HELD-OUT split by absence,
# which the charter keeps off this repository.
mkdir -p logs
LOG="${REPO}/logs/agent-nightly.log"
SUMMARY="${REPO}/logs/run-summaries.md"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"   # launchd: set absolute path via env or edit here

{
  echo "=== nightly run $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  git fetch origin main
  git checkout main
  git pull --ff-only origin main
} >>"$LOG" 2>&1

# Publish the PREVIOUS run's summary, after the pull so the tree is clean before
# the agent works and there is no race with its own commits. One night's lag by
# design. DISCLOSURE (charter rule 1): this pushes main, which the charter
# reserves to the owner. It carries only logs/run-summaries.md — a machine's
# record of its own runs, never research work — and is pending owner
# ratification under rule 9.
if [ -n "$(git status --porcelain -- "$SUMMARY" 2>/dev/null)" ]; then
  {
    git add "$SUMMARY"
    git commit -m "logs: nightly run summary (automated)"
    git push origin main
  } >>"$LOG" 2>&1 || echo "summary publish failed (non-fatal)" >>"$LOG"
fi

read -r -d '' STANDING_CONTRACT <<'EOF' || true
/goal The CURRENT RUNG named at the top of docs/research/agent-charter.md
is complete per that rung's own condition as written in the charter -
every proof clause demonstrated by command output in this transcript, the
constraints verified (git diff --stat main shown; work only on the rung's
agent/* branch; evals/cases, evals/traces, evals/baseline.json and the
scorer code untouched), and a dated entry appended to
docs/research/RESEARCH-LOG.md - OR the rung's stated turn bound is
reached - OR the current rung is owner-only or awaiting blessing, in
which case a one-line ledger note saying so is the entire deliverable.
Begin by reading the charter and the ledger's Standing Lessons.
EOF

# --permission-mode auto is LOAD-BEARING, not a convenience. Permission mode is
# per-session and inherits nothing from any interactive session: without it a
# headless run starts in "default", where every write waits on a human who is not
# there. The 2026-08-19 run burned 107 turns completing W0 and could not write one
# byte. Never remove this flag; verify it with the headless probe in
# docs/research/agent-environment.md's rung-0 checklist.
RUN_START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
caffeinate -i "$CLAUDE_BIN" -p "$STANDING_CONTRACT" --model opus \
  --permission-mode auto \
  --output-format stream-json --verbose >>"$LOG" 2>&1
RUN_EXIT=$?
set -e

echo "=== run finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >>"$LOG"

# Append this run's summary: the few facts worth reading from anywhere, without
# the transcript. Committed by tomorrow's run (see the publish block above).
RUN_START="$RUN_START" RUN_EXIT="$RUN_EXIT" LOG="$LOG" SUMMARY="$SUMMARY" \
python3 - <<'PY' >>"$LOG" 2>&1 || echo "summary write failed (non-fatal)" >>"$LOG"
import json, os, subprocess

log, summary = os.environ["LOG"], os.environ["SUMMARY"]
started, exit_code = os.environ["RUN_START"], os.environ["RUN_EXIT"]

result = None
with open(log, errors="replace") as fh:
    for line in fh:
        line = line.strip()
        if not line.startswith("{") or '"type":"result"' not in line:
            continue
        try:
            result = json.loads(line)
        except ValueError:
            pass

head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                      capture_output=True, text=True).stdout.strip()

if result is None:
    body = (f"- **outcome:** NO RESULT EVENT — the run died before finishing "
            f"(shell exit {exit_code}). Read `logs/agent-nightly.log`.\n")
else:
    mins = (result.get("duration_ms") or 0) / 60000
    cost = result.get("total_cost_usd")
    cost_txt = f"${cost:.2f}" if isinstance(cost, (int, float)) else "n/a"
    flag = " (is_error)" if result.get("is_error") else ""
    body = (f"- **outcome:** {result.get('subtype')}{flag}, shell exit {exit_code}\n"
            f"- **turns:** {result.get('num_turns')} · "
            f"**duration:** {mins:.1f} min · **cost:** {cost_txt}\n")
    text = (result.get("result") or "").strip().replace("\r", "")
    tail = text[-1200:]
    if len(text) > 1200:
        tail = "…" + tail
    body += f"\n**Agent's closing message:**\n\n> " + tail.replace("\n", "\n> ") + "\n"

entry = (f"\n## run {started} · main {head}\n\n{body}"
         f"\n*Raw transcript is gitignored at `logs/agent-nightly.log` on the runner.*\n")

with open(summary, "a") as fh:
    fh.write(entry)
print("run summary appended")
PY
