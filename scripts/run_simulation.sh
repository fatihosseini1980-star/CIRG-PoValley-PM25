#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CIRG_SIM_OUT="$ROOT/results/simulation"
export CIRG_SIM_REPS="${CIRG_SIM_REPS:-50}"
export CIRG_SIM_SHIFT_REPS="${CIRG_SIM_SHIFT_REPS:-50}"
python "$ROOT/src/simulation.py"
python "$ROOT/src/simulation_shift.py"
