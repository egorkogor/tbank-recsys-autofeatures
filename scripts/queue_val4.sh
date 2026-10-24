#!/usr/bin/env bash
# Val-stage queue, part 4: equal-count branches (56 = B0's aggregates) and the full stack. Starts after part 3.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
step() { local name=$1; shift; echo "$(date +%H:%M:%S) start $name"; "$@" > "logs/$name.log" 2>&1; echo "$(date +%H:%M:%S) done  $name"; }

until grep -q "queue3 finished" logs/queue_val3.log 2>/dev/null; do sleep 30; done
step run_r1_decay_k56_val  $PY scripts/run_branch.py r1_decay_k56 --stage val
step run_r1_window_k56_val $PY scripts/run_branch.py r1_window_k56 --stage val
step run_full_val          $PY scripts/run_branch.py full --stage val
echo "$(date +%H:%M:%S) queue4 finished"
