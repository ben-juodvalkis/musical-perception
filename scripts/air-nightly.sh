#!/usr/bin/env bash
# Nightly agent run on the Air runner (see docs/research/agent-environment.md).
# Loaded by launchd (com.musical-perception.nightly); safe to run manually.
# Starts from fresh main — the charter's CURRENT RUNG pointer carries the
# moving state, so this script never changes between rungs.
set -euo pipefail

# The whole body lives in a function invoked on the last line so bash parses
# the entire file before executing any of it. This script switches branches
# while it runs; without the wrapper, a checkout that changes this file's
# bytes leaves a running bash resuming at a byte offset into different
# content (2026-08-24 hardening, found while fixing the checkout race below).
nightly() {

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
# The PREVIOUS run's summary waits here — untracked, gitignored — until the
# publish step folds it into run-summaries.md on a fresh main. Until 2026-08-24
# it waited as an uncommitted edit to run-summaries.md itself, a TRACKED file:
# when a session also left HEAD on a branch whose committed copy of that file
# lagged main's, the next run's `git checkout main` refused to overwrite the
# edit and set -e killed the whole night five seconds in (the silent 08-24
# slot). No tracked file is left dirty between runs any more.
PENDING="${REPO}/logs/pending-summary.md"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"   # launchd: set absolute path via env or edit here

{
  echo "=== nightly run $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  git fetch origin main

  # Re-entrancy guard (2026-08-24): the tree must be clean before the branch
  # switch, and nothing a previous run left behind may be destroyed.
  # (a) A summary tail still sitting uncommitted in the tracked file — the old
  #     mechanism's state, or a night whose publish commit failed — is lifted
  #     into $PENDING. The append-only invariant is checked byte-for-byte;
  #     anything that is not a pure append is stashed, never guessed at.
  if [ -n "$(git status --porcelain -- "$SUMMARY")" ]; then
    git show HEAD:logs/run-summaries.md > "${SUMMARY}.base"
    BASE_BYTES=$(( $(wc -c < "${SUMMARY}.base") ))
    if head -c "$BASE_BYTES" "$SUMMARY" | cmp -s - "${SUMMARY}.base"; then
      tail -c +"$(( BASE_BYTES + 1 ))" "$SUMMARY" >> "$PENDING"
      git checkout -- "$SUMMARY"
    else
      git stash push -m "nightly: non-append run-summaries.md edit, preserved for the owner" -- "$SUMMARY"
    fi
    rm -f "${SUMMARY}.base"
  fi
  # (b) Any other leftover tracked changes are stashed with a dated message —
  #     recoverable evidence of an anomaly (git stash list), not a reason the
  #     night dies. Untracked files (media staging) are never touched.
  git stash push -m "nightly $(date -u +%Y-%m-%d): leftover uncommitted state, preserved for the owner"

  git checkout main
  git pull --ff-only origin main
} >>"$LOG" 2>&1

# Publish the PREVIOUS run's summary, after the pull so the tree is clean before
# the agent works and there is no race with its own commits. One night's lag by
# design. Failure is convergent, not fatal: if the push fails (Wi-Fi drop) the
# commit stays local and rides out with the next successful push of main; if
# the commit itself fails, guard (a) lifts the appended tail straight back into
# $PENDING tomorrow. DISCLOSURE (charter rule 1, ratified narrowly at B3,
# 2026-08-24): this pushes main carrying only logs/run-summaries.md — a
# machine's record of its own runs, never research work.
if [ -s "$PENDING" ]; then
  {
    cat "$PENDING" >> "$SUMMARY"
    rm -f "$PENDING"
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
agent/* branch; no existing file under evals/cases/, evals/traces/, or
evals/baseline.json modified, and no scorer code touched outside a
declared EVAL-CHANGE workstream), and a dated entry appended to
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

# Append this run's summary — to $PENDING, never to the tracked file: the
# tracked file is only touched by the publish step above, moments before its
# own commit, so no uncommitted edit to a tracked file survives this script.
# Committed by tomorrow's run (see the publish block above).
RUN_START="$RUN_START" RUN_EXIT="$RUN_EXIT" LOG="$LOG" PENDING="$PENDING" \
python3 - <<'PY' >>"$LOG" 2>&1 || echo "summary write failed (non-fatal)" >>"$LOG"
import json, os, subprocess

log, pending = os.environ["LOG"], os.environ["PENDING"]
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

with open(pending, "a") as fh:
    fh.write(entry)
print("run summary appended to pending (published by tomorrow's run)")
PY

}
nightly "$@"
