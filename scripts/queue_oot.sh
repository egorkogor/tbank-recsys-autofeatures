#!/usr/bin/env bash
# The single OOT run for TBREC_DATASET (default yambda). Needs <ART>/freeze.json; branches are read from it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
step() { local name=$1; shift; echo "$(date +%H:%M:%S) start $name"; "$@" > "logs/$name.log" 2>&1; echo "$(date +%H:%M:%S) done  $name"; }

ARTDIR=artifacts; [ "${TBREC_DATASET:-yambda}" = yambda ] || ARTDIR=artifacts/$TBREC_DATASET
test -f $ARTDIR/freeze.json
BRANCHES=$($PY -c "import json; print(' '.join(json.load(open('$ARTDIR/freeze.json'))['branches']))")
if [ "${TBREC_DATASET:-yambda}" = yambda ] && echo " $BRANCHES " | grep -qE " (b0_ials|full) "; then step ials_oot $PY scripts/first_stage_ials.py --stage oot; fi
for b in $BRANCHES; do
  case $b in
    r3_perseus*) ;;  # scored by the Perseus adapter after the freeze (local experiment logs)
    *) step "oot_${TBREC_DATASET:-yambda}_$b" $PY scripts/run_branch.py "$b" --stage oot ;;
  esac
done
echo "$(date +%H:%M:%S) oot finished"
