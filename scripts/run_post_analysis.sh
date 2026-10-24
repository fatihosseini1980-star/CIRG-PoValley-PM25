#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-work}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python "$ROOT/src/event_analysis.py" --root "$WORKSPACE"
python "$ROOT/src/moving_block_bootstrap.py" --root "$WORKSPACE"
python "$ROOT/src/sensitivity_203.py" --root "$WORKSPACE"
