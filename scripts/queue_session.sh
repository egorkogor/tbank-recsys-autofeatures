#!/usr/bin/env bash
# Streaming extension (reference point, not the main comparison); starts after the mood queue.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
step() { local name=$1; shift; echo "$(date +%H:%M:%S) start $name"; "$@" > "logs/$name.log" 2>&1; echo "$(date +%H:%M:%S) done  $name"; }
until grep -q "mood queue finished" logs/queue_mood.log 2>/dev/null; do sleep 30; done
step session_test     $PY -m pytest -q -p no:cacheprovider tests/test_generic.py -k session
step session_build    $PY scripts/build_session.py
step run_b0_sess_val  $PY scripts/run_branch.py b0_sess --stage val
step run_r1_sess_val  $PY scripts/run_branch.py r1_sess --stage val
echo "$(date +%H:%M:%S) session queue finished"
