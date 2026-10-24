#!/usr/bin/env bash
# The single OOT run (Yambda). Needs artifacts/freeze.json; branches are read from it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
step() { local name=$1; shift; echo "$(date +%H:%M:%S) start $name"; "$@" > "logs/$name.log" 2>&1; echo "$(date +%H:%M:%S) done  $name"; }

test -f artifacts/freeze.json
BRANCHES=$($PY -c "import json; print(' '.join(json.load(open('artifacts/freeze.json'))['branches']))")
if echo "$BRANCHES" | grep -qE "ials|full"; then step ials_oot $PY scripts/first_stage_ials.py --stage oot; fi
for b in $BRANCHES; do
  case $b in
    r3_perseus*) ;;  # scored by the Perseus adapter after the freeze (local experiment logs)
    *) step "oot_$b" $PY scripts/run_branch.py "$b" --stage oot ;;
  esac
done
echo "$(date +%H:%M:%S) oot finished"
