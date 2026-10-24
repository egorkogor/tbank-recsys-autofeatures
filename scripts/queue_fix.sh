#!/usr/bin/env bash
# Review fixes (24.10): exact equal-count window branches for KION / X5. Starts after the session queue.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
step() { local name=$1; shift; echo "$(date +%H:%M:%S) start $name"; "$@" > "logs/$name.log" 2>&1; echo "$(date +%H:%M:%S) done  $name"; }
until grep -q "session queue finished" logs/queue_session.log 2>/dev/null; do sleep 30; done
TBREC_DATASET=kion step kion_run_r1_window_k33_val $PY scripts/run_branch.py r1_window_k33 --stage val
TBREC_DATASET=x5   step x5_run_r1_window_k25_val   $PY scripts/run_branch.py r1_window_k25 --stage val
echo "$(date +%H:%M:%S) fix queue finished"
