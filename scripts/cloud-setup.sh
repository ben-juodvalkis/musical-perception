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

python -m pip install --upgrade pip >/dev/null
pip install -e ".[${EXTRAS}]"
python -c "import musical_perception; print('musical_perception import OK')"
pytest -q --collect-only >/dev/null && echo "pytest collection OK"
