#!/usr/bin/env bash
# Yambda val queue, part 5: stability checks of the search - placebo (shuffled labels) and stability gate.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
step() { local name=$1; shift; echo "$(date +%H:%M:%S) start $name"; "$@" > "logs/$name.log" 2>&1; echo "$(date +%H:%M:%S) done  $name"; }

until grep -q "ctx queue finished" logs/queue_ctx.log 2>/dev/null; do sleep 30; done
step search_r1_decay_placebo $PY scripts/search_r1.py --grid decay --placebo
step search_r1_decay_stable  $PY scripts/search_r1.py --grid decay --stable 3
step feat_r1_decay_stable    $PY scripts/build_features.py names --names-file artifacts/r1_decay_stable3_selected.json --out r1_decay_stable
step run_r1_decay_stable_val $PY scripts/run_branch.py r1_decay_stable --stage val
echo "$(date +%H:%M:%S) queue5 finished"
