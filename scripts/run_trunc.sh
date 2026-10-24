#!/usr/bin/env bash
# B0 on truncated user history; N = history length of the chosen Perseus config (artifacts/perseus_len.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
N=$(cat artifacts/perseus_len.txt 2>/dev/null || echo 512)
.venv/Scripts/python.exe scripts/build_features.py b0_trunc --user-limit "$N"
.venv/Scripts/python.exe scripts/run_branch.py "b0_trunc$N" --stage val
