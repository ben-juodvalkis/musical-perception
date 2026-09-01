#!/usr/bin/env bash
# Environment setup for agent sessions (see docs/research/agent-environment.md).
# Default: trace-replay + stage-1 acoustic work (no heavy models).
# --live: adds Whisper/Gemini/pose extras for ingestion and rung-5 sessions
#         (downloads model weights; batch such work into dedicated sessions).
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRAS="dev"
if [[ "${1:-}" == "--live" ]]; then
  EXTRAS="all,dev"
fi

# One interpreter throughout: cloud images ship a system pip that refuses
# to upgrade itself (Debian-managed, no RECORD file) and a `pytest` on
# PATH that belongs to a different Python — both broke this script as a
# cloud-environment setup script on 2026-09-01.
python -m pip install --upgrade pip >/dev/null 2>&1 || echo "pip self-upgrade skipped (system pip)"
python -m pip install -e ".[${EXTRAS}]"
python -c "import musical_perception; print('musical_perception import OK')"
python -m pytest -q --collect-only >/dev/null && echo "pytest collection OK"
