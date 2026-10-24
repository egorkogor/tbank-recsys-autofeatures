#!/usr/bin/env bash
# Overnight analysis after the single OOT run: slices, guardrails, SHAP, scaling, val-B tables (tie-aware NDCG).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
run() { local name=$1; shift; echo "$(date +%H:%M:%S) start $name"; "$@" > "logs/$name.log" 2>&1 && echo "$(date +%H:%M:%S) done  $name" || echo "$(date +%H:%M:%S) FAIL  $name"; }
until grep -q "oot all finished" logs/queue_oot_all.log 2>/dev/null; do sleep 60; done
MAIN="b0 r1_decay r1_window_k56 full b0_audio b0_perseus b0_ctx r3_perseus trivial_seen_before"
run an_slices_oot      $PY scripts/analyze.py slices $MAIN --stage oot --part eval
run an_guard_oot       $PY scripts/analyze.py guardrails b0 r1_decay full b0_ctx r3_perseus trivial_seen_before --stage oot --part eval
run an_shap_oot        $PY scripts/analyze.py shap b0 r1_decay full --stage oot
run an_slices_valB     $PY scripts/analyze.py slices $MAIN --stage val --part val_B
run an_guard_valB      $PY scripts/analyze.py guardrails b0 r1_decay full b0_ctx r3_perseus trivial_seen_before --stage val --part val_B
run scaling_build      $PY scripts/scaling.py build
run scaling_learn      $PY scripts/scaling.py learn
Y="trivial_item_lp_rate trivial_seen_before trivial_user_artist_lp b0 r1_decay r1_window r1_decay_k56 r1_window_k56 r1_r5 r1_decay_stable b0_ials b0_audio b0_perseus b0_trunc256 full r3_perseus r3_perseus_typed b0_ctx r1_ctx b0_ctx_mood b0_sess r1_sess"
run valB_yambda        $PY scripts/compare.py --stage val --part val_B --base b0 --out artifacts/final_val_B.md $Y
run valB_kion  env TBREC_DATASET=kion $PY scripts/compare.py --stage val --part val_B --base b0 --out artifacts/kion/final_val_B.md trivial_ials_score trivial_pop_rank base_only b0 r1_decay r1_window r1_decay_k33 r1_window_k33
run valB_x5    env TBREC_DATASET=x5   $PY scripts/compare.py --stage val --part val_B --base b0 --out artifacts/x5/final_val_B.md trivial_ials_score trivial_pop_rank trivial_hist_rank base_only b0 r1_decay r1_window r1_decay_k25 r1_window_k25
run hyp_valB_yambda    $PY scripts/hypotheses.py --stage val --part val_B
echo "$(date +%H:%M:%S) after-oot finished"
