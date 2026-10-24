#!/usr/bin/env bash
# All remaining val-stage steps, strictly one heavy job at a time (RAM is shared with other programs).
# A failed step waits 3 minutes and retries (up to 3 attempts): memory pressure from other programs is transient.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/experiment
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
step() {
  local name=$1; shift
  for attempt in 1 2 3; do
    echo "$(date +%H:%M:%S) start $name (attempt $attempt)"
    if "$@" > "logs/$name.log" 2>&1; then echo "$(date +%H:%M:%S) done  $name"; return 0; fi
    echo "$(date +%H:%M:%S) FAIL  $name: $(tail -1 logs/$name.log | cut -c1-150)"; sleep 180
  done
  echo "$(date +%H:%M:%S) GIVE UP $name"; return 1
}
Y() { TBREC_DATASET=yambda step "$@"; }
K() { local n=$1; shift; TBREC_DATASET=kion step "kion_$n" "$@"; }
X() { local n=$1; shift; TBREC_DATASET=x5 step "x5_$n" "$@"; }

# --- Yambda ---
Y run_b0_perseus_val      $PY scripts/run_branch.py b0_perseus --stage val
Y r5_round3               $PY scripts/r5_eval.py --round 3 --proposals logs/experiment/r5_round3_proposals.json
Y build_r5                $PY scripts/build_r5.py
Y run_r1_r5_val           $PY scripts/run_branch.py r1_r5 --stage val
Y run_r1_decay_k56_val    $PY scripts/run_branch.py r1_decay_k56 --stage val
Y run_r1_window_k56_val   $PY scripts/run_branch.py r1_window_k56 --stage val
Y run_full_val            $PY scripts/run_branch.py full --stage val
Y ctx_b0                  $PY scripts/build_context.py b0
Y run_b0_ctx_val          $PY scripts/run_branch.py b0_ctx --stage val
Y ctx_grammar             $PY scripts/build_context.py grammar
Y ctx_search              $PY scripts/build_context.py search
Y ctx_names               $PY scripts/build_context.py names
Y run_r1_ctx_val          $PY scripts/run_branch.py r1_ctx --stage val
Y search_r1_decay_placebo $PY scripts/search_r1.py --grid decay --placebo
# --- KION (from the window search on) ---
K search_r1_window        $PY scripts/search_r1.py --grid window
K feat_r1_window          $PY scripts/kion_features.py names --names-file artifacts/kion/r1_window_selected.json --out r1_window
K run_r1_window_val       $PY scripts/run_branch.py r1_window --stage val
K run_r1_decay_k33_val    $PY scripts/run_branch.py r1_decay_k33 --stage val
K run_r1_window_k33_val   $PY scripts/run_branch.py r1_window_k33 --stage val
# --- X5 ---
X feat_b0                 $PY scripts/kion_features.py b0
X feat_grammar_decay      $PY scripts/kion_features.py grammar --grid decay
X choose_objective        $PY scripts/choose_objective.py
X trivial_ials            $PY scripts/run_branch.py trivial_ials_score --stage val
X trivial_pop             $PY scripts/run_branch.py trivial_pop_rank --stage val
X trivial_hist            $PY scripts/run_branch.py trivial_hist_rank --stage val
X run_base_only_val       $PY scripts/run_branch.py base_only --stage val
X run_b0_val              $PY scripts/run_branch.py b0 --stage val
X search_r1_decay         $PY scripts/search_r1.py --grid decay
X feat_r1_decay           $PY scripts/kion_features.py names --names-file artifacts/x5/r1_decay_selected.json --out r1_decay
X run_r1_decay_val        $PY scripts/run_branch.py r1_decay --stage val
X feat_grammar_window     $PY scripts/kion_features.py grammar --grid window
X search_r1_window        $PY scripts/search_r1.py --grid window
X feat_r1_window          $PY scripts/kion_features.py names --names-file artifacts/x5/r1_window_selected.json --out r1_window
X run_r1_window_val       $PY scripts/run_branch.py r1_window --stage val
X run_r1_decay_k25_val    $PY scripts/run_branch.py r1_decay_k25 --stage val
X run_r1_window_k25_val   $PY scripts/run_branch.py r1_window_k25 --stage val
# --- Yambda, last: stability gate ---
Y search_r1_decay_stable  $PY scripts/search_r1.py --grid decay --stable 3
Y feat_r1_decay_stable    $PY scripts/build_features.py names --names-file artifacts/r1_decay_stable3_selected.json --out r1_decay_stable
Y run_r1_decay_stable_val $PY scripts/run_branch.py r1_decay_stable --stage val
echo "$(date +%H:%M:%S) master queue finished"
