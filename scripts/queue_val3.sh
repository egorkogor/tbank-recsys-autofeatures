#!/usr/bin/env bash
# Val-stage queue, part 3: audio, agent round 1, stacking with Perseus. Starts after part 2.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/experiment
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
step() { local name=$1; shift; echo "$(date +%H:%M:%S) start $name"; "$@" > "logs/$name.log" 2>&1; echo "$(date +%H:%M:%S) done  $name"; }

until grep -q "queue2 finished" logs/queue_val2.log 2>/dev/null; do sleep 30; done
step audio_prepare    $PY scripts/build_audio.py prepare
step audio_build      $PY scripts/build_audio.py build
step tests            $PY -m pytest -q -p no:cacheprovider tests
step run_b0_audio_val $PY scripts/run_branch.py b0_audio --stage val
step r5_round1        $PY scripts/r5_eval.py --round 1 --proposals logs/experiment/r5_round1_proposals.json
until [ -f logs/experiment/perseus_stage3_last.md ]; do sleep 30; done
step run_b0_perseus_val $PY scripts/run_branch.py b0_perseus --stage val
echo "$(date +%H:%M:%S) queue3 finished"
