#!/usr/bin/env bash
# The single OOT run for the three domains. Run once, after every val-stage step is done and the code is
# committed: freeze (manifest with hashes) -> every frozen branch through the gate -> Perseus -> hypotheses.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
log() { echo "$(date +%H:%M:%S) $*"; }

Y_BRANCHES="trivial_item_lp_rate trivial_seen_before trivial_user_artist_lp b0 r1_decay r1_window r1_decay_k56 r1_window_k56 r1_r5 r1_decay_stable b0_ials b0_audio b0_perseus b0_trunc256 full r3_perseus b0_ctx r1_ctx b0_ctx_mood"
K_BRANCHES="trivial_ials_score trivial_pop_rank base_only b0 r1_decay r1_window r1_decay_k33 r1_window_k33"
X_BRANCHES="trivial_ials_score trivial_pop_rank trivial_hist_rank base_only b0 r1_decay r1_window r1_decay_k25 r1_window_k25"

set -e
TBREC_DATASET=yambda $PY scripts/freeze.py $Y_BRANCHES
TBREC_DATASET=kion   $PY scripts/freeze.py $K_BRANCHES
TBREC_DATASET=x5     $PY scripts/freeze.py $X_BRANCHES
set +e
log "frozen"

for ds in yambda kion x5; do
  TBREC_DATASET=$ds bash scripts/queue_oot.sh > logs/queue_oot_$ds.log 2>&1 && log "done  oot $ds" || log "FAIL  oot $ds"
done
TBREC_DATASET=yambda $PY scripts/score_r3_oot.py > logs/oot_yambda_r3_perseus.log 2>&1 && log "done  r3_perseus oot" || log "FAIL  r3_perseus oot"

TBREC_DATASET=yambda $PY scripts/hypotheses.py --stage oot > logs/hyp_oot_yambda.log 2>&1
TBREC_DATASET=kion   $PY scripts/hypotheses.py --stage oot > logs/hyp_oot_kion.log 2>&1
TBREC_DATASET=x5     $PY scripts/hypotheses.py --stage oot > logs/hyp_oot_x5.log 2>&1
TBREC_DATASET=yambda $PY scripts/compare.py --stage oot --base b0 $Y_BRANCHES > logs/cmp_oot_yambda.log 2>&1
TBREC_DATASET=kion   $PY scripts/compare.py --stage oot --base b0 $K_BRANCHES > logs/cmp_oot_kion.log 2>&1
TBREC_DATASET=x5     $PY scripts/compare.py --stage oot --base b0 $X_BRANCHES > logs/cmp_oot_x5.log 2>&1
log "oot all finished"
