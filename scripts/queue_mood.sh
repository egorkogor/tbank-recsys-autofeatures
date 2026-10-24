#!/usr/bin/env bash
# Listener-state proxies (Yambda); starts after the master queue (one heavy job at a time).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
step() { local name=$1; shift; echo "$(date +%H:%M:%S) start $name"; "$@" > "logs/$name.log" 2>&1; echo "$(date +%H:%M:%S) done  $name"; }
until grep -q "master queue finished" logs/queue_master.log 2>/dev/null; do sleep 30; done
step mood_clusters $PY scripts/build_mood.py clusters
step mood_test     $PY -m pytest -q -p no:cacheprovider tests/test_generic.py -k mood
step mood_build    $PY scripts/build_mood.py build
step run_b0_ctx_mood_val $PY scripts/run_branch.py b0_ctx_mood --stage val
echo "$(date +%H:%M:%S) mood queue finished"
