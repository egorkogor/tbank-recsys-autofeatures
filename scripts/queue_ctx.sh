#!/usr/bin/env bash
# Yambda request-time context: hand-made (b0_ctx) vs searched (r1_ctx). Starts after val queue part 4.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
step() { local name=$1; shift; echo "$(date +%H:%M:%S) start $name"; "$@" > "logs/$name.log" 2>&1; echo "$(date +%H:%M:%S) done  $name"; }

until grep -q "queue4 finished" logs/queue_val4.log 2>/dev/null; do sleep 30; done
step ctx_b0           $PY scripts/build_context.py b0
step run_b0_ctx_val   $PY scripts/run_branch.py b0_ctx --stage val
step ctx_grammar      $PY scripts/build_context.py grammar
step ctx_search       $PY scripts/build_context.py search
step ctx_names        $PY scripts/build_context.py names
step run_r1_ctx_val   $PY scripts/run_branch.py r1_ctx --stage val
echo "$(date +%H:%M:%S) ctx queue finished"
