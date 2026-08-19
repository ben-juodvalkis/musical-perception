#!/usr/bin/env bash
# Nightly agent run on the Air runner (see docs/research/agent-environment.md).
# Loaded by launchd (com.musical-perception.nightly); safe to run manually.
# Starts from fresh main — the charter's CURRENT RUNG pointer carries the
# moving state, so this script never changes between rungs.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="${HOME}/musical-perception-agent.log"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"   # launchd: set absolute path via env or edit here

{
  echo "=== nightly run $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  git fetch origin main
  git checkout main
  git pull --ff-only origin main
} >>"$LOG" 2>&1

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
caffeinate -i "$CLAUDE_BIN" -p "$STANDING_CONTRACT" --model opus \
  --permission-mode auto \
  --output-format stream-json --verbose >>"$LOG" 2>&1

echo "=== run finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >>"$LOG"
