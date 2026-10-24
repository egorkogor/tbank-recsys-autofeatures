#!/usr/bin/env bash
# Val-stage queue: one heavy job at a time (RAM is shared with other programs on the machine).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
SEL_DAYS=274,277,280,283,286,287,289,291,293
step() { local name=$1; shift; echo "$(date +%H:%M:%S) start $name"; "$@" > "logs/$name.log" 2>&1; echo "$(date +%H:%M:%S) done  $name"; }

until [ -f artifacts/features/grammar_decay.cost.json ] && [ -f artifacts/objective.json ]; do sleep 20; done
step search_r1_decay  $PY scripts/search_r1.py --grid decay
step run_b0_val       $PY scripts/run_branch.py b0 --stage val
step feat_r1_decay    $PY scripts/build_features.py names --names-file artifacts/r1_decay_selected.json --out r1_decay
step run_r1_decay_val $PY scripts/run_branch.py r1_decay --stage val
step feat_grammar_window $PY scripts/build_features.py grammar --grid window --days $SEL_DAYS --users-frac 0.3 --val-half A
step search_r1_window $PY scripts/search_r1.py --grid window
step feat_r1_window   $PY scripts/build_features.py names --names-file artifacts/r1_window_selected.json --out r1_window
step run_r1_window_val $PY scripts/run_branch.py r1_window --stage val
step ials_val         $PY scripts/first_stage_ials.py --stage val
step run_b0_ials_val  $PY scripts/run_branch.py b0_ials --stage val
echo "$(date +%H:%M:%S) queue finished"
