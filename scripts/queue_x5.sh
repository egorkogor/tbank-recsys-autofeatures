#!/usr/bin/env bash
# X5 val-stage queue: the same protocol and scripts as Yambda and KION.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 TBREC_DATASET=x5
step() { local name=$1; shift; echo "$(date +%H:%M:%S) start $name"; "$@" > "logs/x5_$name.log" 2>&1; echo "$(date +%H:%M:%S) done  $name"; }

until grep -q "kion queue finished" logs/queue_kion.log 2>/dev/null; do sleep 30; done
step feat_b0 $PY scripts/kion_features.py b0
step feat_grammar_decay $PY scripts/kion_features.py grammar --grid decay
step choose_objective  $PY scripts/choose_objective.py
step trivial_ials      $PY scripts/run_branch.py trivial_ials_score --stage val
step trivial_pop       $PY scripts/run_branch.py trivial_pop_rank --stage val
step trivial_hist      $PY scripts/run_branch.py trivial_hist_rank --stage val
step run_base_only_val $PY scripts/run_branch.py base_only --stage val
step run_b0_val        $PY scripts/run_branch.py b0 --stage val
step search_r1_decay   $PY scripts/search_r1.py --grid decay
step feat_r1_decay     $PY scripts/kion_features.py names --names-file artifacts/x5/r1_decay_selected.json --out r1_decay
step run_r1_decay_val  $PY scripts/run_branch.py r1_decay --stage val
step feat_grammar_window $PY scripts/kion_features.py grammar --grid window
step search_r1_window  $PY scripts/search_r1.py --grid window
step feat_r1_window    $PY scripts/kion_features.py names --names-file artifacts/x5/r1_window_selected.json --out r1_window
step run_r1_window_val $PY scripts/run_branch.py r1_window --stage val
step run_r1_decay_k25_val  $PY scripts/run_branch.py r1_decay_k25 --stage val
step run_r1_window_k25_val $PY scripts/run_branch.py r1_window_k25 --stage val
step compare_val_B     $PY scripts/compare.py --stage val --part val_B --base b0 b0 base_only r1_decay r1_window r1_decay_k25 r1_window_k25 trivial_ials_score trivial_pop_rank trivial_hist_rank
echo "$(date +%H:%M:%S) x5 queue finished"
